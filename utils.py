"""Utilitas audio preprocessing dan inferensi emosi."""

from __future__ import annotations

import io
import tempfile
from pathlib import Path
from typing import Any

import librosa
import numpy as np
import pandas as pd
import soundfile as sf
import torch
import torch.nn.functional as F
import torchaudio

from model import WavLMSERModel

TARGET_SAMPLE_RATE = 16000
MAX_DURATION_SECONDS = 4.0
MIN_DURATION_SECONDS = 0.35
MAX_SAMPLES = int(TARGET_SAMPLE_RATE * MAX_DURATION_SECONDS)
MIN_SAMPLES = int(TARGET_SAMPLE_RATE * MIN_DURATION_SECONDS)
SILENCE_TRIM_TOP_DB = 30

LABEL2ID = {
    "netral": 0,
    "senang": 1,
    "sedih": 2,
    "marah": 3,
    "takut": 4,
    "jijik": 5,
}

ID2LABEL = {idx: label for label, idx in LABEL2ID.items()}


def _load_via_soundfile(path: str) -> tuple[torch.Tensor, int]:
    """Fallback decode WAV/FLAC via soundfile."""
    data, sample_rate = sf.read(path, dtype="float32", always_2d=True)
    waveform = torch.from_numpy(data.T).float()
    if waveform.ndim == 1:
        waveform = waveform.unsqueeze(0)
    return waveform, int(sample_rate)


def _load_via_librosa(path: str) -> tuple[torch.Tensor, int]:
    """Fallback decode MP3/audio umum via librosa (lazy import)."""
    import librosa

    audio_np, sample_rate = librosa.load(path, sr=None, mono=False)
    waveform = torch.from_numpy(audio_np).float()
    if waveform.ndim == 1:
        waveform = waveform.unsqueeze(0)
    return waveform, int(sample_rate)


def _decode_audio_file(path: str) -> tuple[torch.Tensor, int]:
    """torchaudio → librosa → soundfile."""
    try:
        return torchaudio.load(path)
    except Exception:
        try:
            return _load_via_librosa(path)
        except Exception:
            return _load_via_soundfile(path)


def load_audio(file: io.BytesIO | str | Path) -> tuple[torch.Tensor, int]:
    """Muat audio dari file upload Streamlit (.wav / .mp3)."""
    if isinstance(file, (str, Path)):
        path = str(file)
        waveform, sample_rate = _decode_audio_file(path)
    else:
        suffix = ".wav"
        if hasattr(file, "name") and file.name:
            suffix = Path(file.name).suffix or suffix

        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
            tmp.write(file.getvalue())
            tmp_path = tmp.name

        try:
            waveform, sample_rate = _decode_audio_file(tmp_path)
        finally:
            Path(tmp_path).unlink(missing_ok=True)

    if waveform.ndim == 1:
        waveform = waveform.unsqueeze(0)
    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)

    return waveform, int(sample_rate)


def _trim_silence(y: np.ndarray) -> np.ndarray:
    """Potong hening di awal/akhir (30 dB), dibuang jika hasil terlalu pendek.

    Guard `len(yt) >= MIN_SAMPLES` mencegah audio lirih (misal rekaman
    mikrofon bervolume rendah) hilang total akibat trim yang terlalu agresif.
    Identik dengan `load_waveform()` pada pipeline/ver4-ser-pipeline.ipynb.
    """
    trimmed, _ = librosa.effects.trim(y, top_db=SILENCE_TRIM_TOP_DB)
    if len(trimmed) >= MIN_SAMPLES:
        return trimmed
    return y


def _peak_normalize(y: np.ndarray) -> np.ndarray:
    """Normalisasi amplitudo puncak ke [-1.0, 1.0]. Skip jika sinyal nyaris hening."""
    peak = float(np.max(np.abs(y))) if len(y) else 0.0
    if peak > 1e-5:
        return y / peak
    return y


def _fix_length_eval(y: np.ndarray) -> np.ndarray:
    """Ambil dari awal jika > MAX_SAMPLES, right zero-pad jika lebih pendek.

    Mode eval WAJIB mengambil segmen dari awal agar konsisten dengan
    `fix_length(mode='eval')` pada pipeline v4.
    """
    if len(y) > MAX_SAMPLES:
        y = y[:MAX_SAMPLES]
    elif len(y) < MAX_SAMPLES:
        y = np.pad(y, (0, MAX_SAMPLES - len(y)), mode="constant")
    return y.astype(np.float32)


def preprocess_audio(
    audio: torch.Tensor,
    sample_rate: int,
) -> tuple[torch.Tensor, dict[str, Any]]:
    """Preprocessing SER sesuai kontrak v4 (pipeline Cell 37).

    Urutan wajib: mono -> resample 16kHz -> sanitasi NaN -> trim silence
    30dB -> peak normalization -> crop dari awal/zero-pad ke MAX_SAMPLES.
    Menyimpang dari urutan ini membuat input inferensi berbeda dari
    kondisi training checkpoint v4.
    """
    if audio.ndim == 1:
        audio = audio.unsqueeze(0)
    if audio.shape[0] > 1:
        audio = audio.mean(dim=0, keepdim=True)

    original_duration = audio.shape[-1] / sample_rate

    y = audio.squeeze(0).numpy().astype(np.float32)
    if sample_rate != TARGET_SAMPLE_RATE:
        y = librosa.resample(
            y,
            orig_sr=sample_rate,
            target_sr=TARGET_SAMPLE_RATE,
        )
    y = np.nan_to_num(y, nan=0.0, posinf=0.0, neginf=0.0)
    y = _trim_silence(y)
    y = _peak_normalize(y)

    trimmed = len(y) > MAX_SAMPLES
    y = _fix_length_eval(y)

    duration_after = len(y) / TARGET_SAMPLE_RATE
    info = {
        "original_sample_rate": sample_rate,
        "target_sample_rate": TARGET_SAMPLE_RATE,
        "original_duration_sec": round(original_duration, 3),
        "processed_duration_sec": round(duration_after, 3),
        "trimmed": trimmed,
        "max_duration_sec": MAX_DURATION_SECONDS,
    }
    return torch.from_numpy(y), info


def get_audio_info(file: io.BytesIO | str | Path) -> dict[str, Any]:
    """Ambil metadata audio tanpa preprocessing penuh."""
    if hasattr(file, "name") and file.name:
        filename = Path(file.name).name
    elif isinstance(file, (str, Path)):
        filename = Path(file).name
    else:
        filename = "unknown"
    waveform, sample_rate = load_audio(file)
    duration_sec = round(waveform.shape[-1] / sample_rate, 3)
    channels = 1 if waveform.ndim == 1 else waveform.shape[0]

    return {
        "filename": filename,
        "sample_rate": sample_rate,
        "duration_sec": duration_sec,
        "channels": channels,
        "num_samples": int(waveform.shape[-1]),
    }


def get_waveform_envelope(file: io.BytesIO | str | Path, num_points: int = 400) -> pd.DataFrame:
    """Downsample waveform ke envelope min/max per bucket untuk visualisasi ringan (tanpa matplotlib)."""
    waveform, sample_rate = load_audio(file)
    y = waveform.squeeze(0).numpy()
    total_samples = len(y)
    if total_samples == 0:
        return pd.DataFrame({"Puncak": [], "Lembah": []})

    bucket_size = max(1, total_samples // num_points)
    peaks, troughs = [], []
    for i in range(0, total_samples, bucket_size):
        chunk = y[i : i + bucket_size]
        peaks.append(float(chunk.max()))
        troughs.append(float(chunk.min()))

    duration = total_samples / sample_rate
    time_axis = np.linspace(0, duration, len(peaks))
    return pd.DataFrame({"Detik": time_axis, "Puncak": peaks, "Lembah": troughs}).set_index("Detik")


def get_transcription_waveform(file: io.BytesIO | str | Path) -> "np.ndarray":
    """Ambil waveform mono 16 kHz PENUH (tanpa potong) untuk transkrip STT."""
    waveform, sample_rate = load_audio(file)
    if sample_rate != TARGET_SAMPLE_RATE:
        waveform = torchaudio.functional.resample(waveform, sample_rate, TARGET_SAMPLE_RATE)
    return waveform.squeeze(0).numpy().astype("float32")


def transcribe_audio(asr_pipeline: Any, waveform: "np.ndarray", language: str = "indonesian") -> str:
    """Transkrip audio ke teks menggunakan pipeline Whisper (ASR)."""
    output = asr_pipeline(
        {"raw": waveform, "sampling_rate": TARGET_SAMPLE_RATE},
        generate_kwargs={
            "language": language,
            "task": "transcribe",
            "temperature": (0.0, 0.2, 0.4, 0.6, 0.8, 1.0),
            "compression_ratio_threshold": 2.4,
            "logprob_threshold": -1.0,
            "no_speech_threshold": 0.6,
            # condition_on_prev_tokens=False (bukan default True ala CLI OpenAI): mencegah
            # satu segmen yang berhalusinasi/nge-loop meracuni konteks segmen berikutnya.
            "condition_on_prev_tokens": False,
        },
        chunk_length_s=30,
    )
    return str(output.get("text", "")).strip()


def transcribe_audio_segments(
    asr_pipeline: Any, waveform: "np.ndarray", language: str = "indonesian"
) -> tuple[str, list[dict[str, Any]]]:
    """Transkrip audio + timestamp per-segmen (untuk analisis emosi per-segmen)."""
    output = asr_pipeline(
        {"raw": waveform, "sampling_rate": TARGET_SAMPLE_RATE},
        generate_kwargs={
            "language": language,
            "task": "transcribe",
            "temperature": (0.0, 0.2, 0.4, 0.6, 0.8, 1.0),
            "compression_ratio_threshold": 2.4,
            "logprob_threshold": -1.0,
            "no_speech_threshold": 0.6,
            "condition_on_prev_tokens": False,
        },
        chunk_length_s=30,
        return_timestamps=True,
    )
    text = str(output.get("text", "")).strip()
    segments = [
        {"text": str(chunk["text"]).strip(), "start": float(chunk["timestamp"][0]), "end": float(chunk["timestamp"][1])}
        for chunk in output.get("chunks", [])
        if chunk.get("timestamp") and chunk["timestamp"][0] is not None and chunk["timestamp"][1] is not None
    ]
    return text, segments


def slice_waveform(waveform: "np.ndarray", start: float, end: float) -> "np.ndarray":
    """Potong waveform mono 16kHz berdasarkan rentang waktu (detik)."""
    start_sample = max(0, int(start * TARGET_SAMPLE_RATE))
    end_sample = min(len(waveform), int(end * TARGET_SAMPLE_RATE))
    return waveform[start_sample:end_sample]


STT_FALLBACK_MESSAGE = "Transkripsi sementara tidak tersedia. Silakan coba lagi nanti."

_whisper_pipeline: Any | None = None
_whisper_config_key: tuple[str, str] | None = None


def create_whisper_pipeline(model_name: str, device_name: str) -> Any:
    """Buat pipeline Whisper ASR yang dipreload dan dicache oleh Streamlit."""
    from transformers import pipeline

    device = 0 if device_name == "cuda" else -1
    return pipeline(
        "automatic-speech-recognition",
        model=model_name,
        device=device,
    )


def get_whisper_pipeline(
    model_name: str,
    device_name: str,
    *,
    loader: Any | None = None,
) -> Any:
    """Lazy-load pipeline Whisper; pertama kali dipanggil saat user minta transkrip."""
    global _whisper_pipeline, _whisper_config_key

    if loader is not None:
        return loader(model_name, device_name)

    key = (model_name, device_name)
    if _whisper_pipeline is None or _whisper_config_key != key:
        _whisper_pipeline = create_whisper_pipeline(model_name, device_name)
        _whisper_config_key = key
    return _whisper_pipeline


def safe_transcribe(
    file: io.BytesIO | str | Path,
    model_name: str,
    device_name: str,
    language: str = "indonesian",
    *,
    pipeline_loader: Any | None = None,
) -> tuple[str, bool]:
    """Transkrip audio ke teks. Tidak pernah raise; mengembalikan (teks, sukses)."""
    try:
        if hasattr(file, "seek"):
            file.seek(0)
        waveform = get_transcription_waveform(file)
        asr = get_whisper_pipeline(model_name, device_name, loader=pipeline_loader)
        text = transcribe_audio(asr, waveform, language)
        cleaned = str(text).strip()
        if not cleaned:
            return "Tidak ada ucapan yang terdeteksi.", True
        return cleaned, True
    except Exception:
        return STT_FALLBACK_MESSAGE, False


def safe_transcribe_segments(
    file: io.BytesIO | str | Path,
    model_name: str,
    device_name: str,
    language: str = "indonesian",
    *,
    pipeline_loader: Any | None = None,
) -> tuple[str, list[dict[str, Any]], bool]:
    """Transkrip + segmen waktu. Tidak pernah raise; mengembalikan (teks, segmen, sukses)."""
    try:
        if hasattr(file, "seek"):
            file.seek(0)
        waveform = get_transcription_waveform(file)
        asr = get_whisper_pipeline(model_name, device_name, loader=pipeline_loader)
        text, segments = transcribe_audio_segments(asr, waveform, language)
        cleaned = text.strip()
        if not cleaned:
            return "Tidak ada ucapan yang terdeteksi.", [], True
        return cleaned, segments, True
    except Exception:
        return STT_FALLBACK_MESSAGE, [], False


def predict_emotion(
    model: WavLMSERModel,
    processor: Any,
    waveform: torch.Tensor,
    device: torch.device | str,
) -> dict[str, Any]:
    """Jalankan inferensi emosi pada waveform 1D yang sudah dipreprocess.

    Argumen processor WAJIB identik dengan pemanggilan feature_extractor
    pada pipeline/ver4-ser-pipeline.ipynb. Tanpa
    `return_tensors="pt"`, hasil BatchFeature berisi list numpy, bukan
    tensor, sehingga `.to(device)` melempar AttributeError.
    """
    device = torch.device(device)

    inputs = processor(
        [waveform.numpy()],
        sampling_rate=TARGET_SAMPLE_RATE,
        padding=True,
        truncation=True,
        max_length=MAX_SAMPLES,
        return_attention_mask=True,
        return_tensors="pt",
    )
    input_values = inputs["input_values"].to(device)
    attention_mask = inputs.get("attention_mask")
    if attention_mask is not None:
        attention_mask = attention_mask.to(device)
    else:
        attention_mask = torch.ones_like(input_values, dtype=torch.long, device=device)

    with torch.no_grad():
        logits = model(input_values, attention_mask=attention_mask)

    probabilities = F.softmax(logits, dim=-1).squeeze(0).cpu().numpy()
    predicted_id = int(np.argmax(probabilities))
    confidence = float(probabilities[predicted_id])

    prob_df = pd.DataFrame(
        {
            "Emosi": [ID2LABEL[i] for i in range(len(probabilities))],
            "Probabilitas": probabilities,
            "Persentase (%)": probabilities * 100,
        }
    ).sort_values("Probabilitas", ascending=False)

    return {
        "predicted_label": ID2LABEL[predicted_id],
        "predicted_id": predicted_id,
        "confidence": confidence,
        "probabilities": probabilities,
        "logits": logits.squeeze(0).cpu().numpy(),
        "probabilities_df": prob_df,
    }


def summarize_prediction(result: dict) -> dict:
    """Ringkas prediksi untuk tampilan ranking & margin."""
    prob_df = result["probabilities_df"]
    top_pct = float(prob_df.iloc[0]["Persentase (%)"])

    second_label = None
    second_pct = 0.0
    if len(prob_df) > 1:
        second_label = str(prob_df.iloc[1]["Emosi"])
        second_pct = float(prob_df.iloc[1]["Persentase (%)"])

    margin_pp = top_pct - second_pct

    if margin_pp >= 20:
        separation = "Pemisahan kuat dari emosi lain"
    elif margin_pp >= 10:
        separation = "Pemisahan cukup jelas dari emosi lain"
    else:
        separation = "Pemisahan tipis — emosi lain masih dekat"

    return {
        "top_label": result["predicted_label"],
        "top_pct": top_pct,
        "second_label": second_label,
        "second_pct": second_pct,
        "margin_pp": margin_pp,
        "separation": separation,
        "num_classes": len(prob_df),
    }
