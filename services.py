"""Layanan ML (Model Loading & Inference) dengan sistem cache Streamlit."""

import gc
import json
from pathlib import Path

import pandas as pd
import torch
import streamlit as st

try:
    from transformers import AutoFeatureExtractor as FeatureExtractor
except ImportError:
    try:
        from transformers import Wav2Vec2FeatureExtractor as FeatureExtractor
    except ImportError:
        from transformers.models.wav2vec2.feature_extraction_wav2vec2 import Wav2Vec2FeatureExtractor as FeatureExtractor

from model import load_model
from utils import MIN_DURATION_SECONDS, TARGET_SAMPLE_RATE, load_audio, preprocess_audio, predict_emotion, slice_waveform
from config import SER_BACKBONE

MODELS_DIR = Path(__file__).resolve().parent / "models"
DATASET_METADATA_PATH = MODELS_DIR / "metadata_split_v4.csv"
MODEL_CONFIG_PATH = MODELS_DIR / "config_v4.json"
TEST_EVALUATION_PATH = MODELS_DIR / "evaluasi_test_v4.csv"
LABEL_NOISE_PATH = MODELS_DIR / "label_noise_candidates.csv"
TRAINING_HISTORY_PATH = MODELS_DIR / "history_v4.json"
TRAINING_CURVE_IMAGE_PATH = MODELS_DIR / "kurva_training_v4.png"


@st.cache_resource(show_spinner="Memuat model WavLM...")
def load_ser_model(device_name: str):
    device = torch.device(device_name)
    model = load_model(device=device)
    return model, device


@st.cache_resource(show_spinner="Memuat feature extractor...")
def load_feature_extractor():
    return FeatureExtractor.from_pretrained(SER_BACKBONE)


@st.cache_resource(show_spinner="Memuat Whisper (STT)...")
def load_whisper_lazy(model_name: str, device_name: str):
    """Cache pipeline Whisper agar hanya dimuat satu kali per proses aplikasi."""
    from utils import create_whisper_pipeline
    return create_whisper_pipeline(model_name, device_name)


@st.cache_data(show_spinner=False)
def load_dataset_metadata() -> pd.DataFrame | None:
    """Metadata split dataset training (~19rb baris) — dicache karena dipakai sidebar & dashboard."""
    if not DATASET_METADATA_PATH.exists():
        return None
    return pd.read_csv(DATASET_METADATA_PATH)


@st.cache_data(show_spinner=False)
def load_model_metrics() -> dict | None:
    """Metrik training model (akurasi, epoch terbaik, dst) dari config_v4.json."""
    if not MODEL_CONFIG_PATH.exists():
        return None
    with open(MODEL_CONFIG_PATH) as f:
        return json.load(f)


@st.cache_data(show_spinner=False)
def load_test_evaluation() -> pd.DataFrame | None:
    """Hasil prediksi per sampel test (true vs pred) — untuk confusion matrix & metrik per kelas."""
    if not TEST_EVALUATION_PATH.exists():
        return None
    return pd.read_csv(TEST_EVALUATION_PATH)


@st.cache_data(show_spinner=False)
def load_label_noise_candidates() -> pd.DataFrame | None:
    """Kandidat sampel dengan label berpotensi bising, hasil audit training."""
    if not LABEL_NOISE_PATH.exists():
        return None
    return pd.read_csv(LABEL_NOISE_PATH)


@st.cache_data(show_spinner=False)
def load_training_history() -> dict | None:
    """Kurva loss/akurasi train & val per epoch, lr schedule, dan stage unfreezing."""
    if not TRAINING_HISTORY_PATH.exists():
        return None
    with open(TRAINING_HISTORY_PATH) as f:
        return json.load(f)


def check_model_ready(device_name: str) -> tuple[bool, str | None]:
    try:
        load_ser_model(device_name)
        load_feature_extractor()
        return True, None
    except FileNotFoundError as exc:
        return False, str(exc)
    except RuntimeError as exc:
        return False, str(exc)
    except Exception as exc:
        return False, f"Gagal memuat model: {exc}"


def run_prediction(uploaded_file, device_name: str) -> tuple[dict, dict]:
    with st.spinner("Menganalisis pola emosi dari audio..."):
        model, device = load_ser_model(device_name)
        processor = load_feature_extractor()
        uploaded_file.seek(0)
        waveform, sample_rate = load_audio(uploaded_file)
        processed_waveform, preprocess_info = preprocess_audio(waveform, sample_rate)
        result = predict_emotion(model, processor, processed_waveform, device)
        gc.collect()
    return result, preprocess_info


def run_segment_predictions(
    waveform_16k, segments: list[dict], device_name: str, max_segments: int = 20
) -> list[dict]:
    """Jalankan prediksi emosi per-segmen transkrip, reuse pipeline preprocessing & model yang sama."""
    model, device = load_ser_model(device_name)
    processor = load_feature_extractor()

    results = []
    with st.spinner("Menganalisis emosi per-segmen..."):
        for seg in segments[:max_segments]:
            if seg["end"] - seg["start"] < MIN_DURATION_SECONDS:
                continue
            chunk = slice_waveform(waveform_16k, seg["start"], seg["end"])
            processed_waveform, _ = preprocess_audio(
                torch.from_numpy(chunk).unsqueeze(0), TARGET_SAMPLE_RATE
            )
            result = predict_emotion(model, processor, processed_waveform, device)
            results.append({**seg, "result": result})
        gc.collect()
    return results
