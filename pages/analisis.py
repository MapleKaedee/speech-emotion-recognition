"""Halaman analisis emosi Speech Emotion Recognition berbasis WavLM."""

from __future__ import annotations

import gc
import hashlib
import traceback
from datetime import datetime
from pathlib import Path

import pandas as pd
import streamlit as st
import torch

from config import (
    IS_CLOUD,
    ENABLE_STT,
    WHISPER_MODEL,
    WHISPER_LANGUAGE,
    MAX_CLOUD_PREDICTIONS,
    MAX_RECORD_DURATION_SECONDS,
    MIN_RECORD_DURATION_SECONDS,
    MODEL_DISPLAY_PATH,
)
from services import (
    check_model_ready,
    run_prediction,
    run_segment_predictions,
    load_whisper_lazy,
)
from utils import (
    ID2LABEL,
    LABEL2ID,
    get_audio_info,
    get_transcription_waveform,
    get_waveform_envelope,
    safe_transcribe,
    safe_transcribe_segments,
    summarize_prediction,
)

from components.css import inject_custom_css
from components.recorder import record_audio
from components.ui import (
    render_empty_state,
    render_metadata_card,
    render_section_header,
    render_top3_cards,
    render_probability_bars,
    render_result_card,
    render_transcript_card,
    render_segment_timeline,
    render_waveform_chart,
    render_export_buttons,
    render_history_list,
    format_file_size,
)


def _audio_key(audio_file, source: str) -> tuple:
    """Identitas cache berbasis hash byte, bukan nama file.

    Rekaman mikrofon dapat memiliki nama/ukuran yang sama antar sesi,
    sehingga name+size tidak cukup untuk invalidasi cache prediksi.
    """
    audio_file.seek(0)
    digest = hashlib.sha256(audio_file.getvalue()).hexdigest()
    audio_file.seek(0)
    return (source, digest)


def _clear_prediction_cache() -> None:
    st.session_state.pop("prediction_cache", None)
    st.session_state.pop("prediction_file_key", None)


def _reset_cloud_session() -> None:
    st.session_state.pop("cloud_prediction_count", None)
    _clear_prediction_cache()


def main() -> None:
    inject_custom_css()
    device_name = "cuda" if torch.cuda.is_available() else "cpu"
    model_ready, model_error = check_model_ready(device_name)

    render_section_header(
        "Analisis Emosi",
        "Kontrol Ada di Panel Samping",
        "Pilih sumber audio dan jalankan analisis lewat sidebar.",
    )
    if not model_ready:
        st.error(
            "Model gagal dimuat sehingga prediksi tidak dapat dijalankan.\n\n"
            f"{model_error or 'Periksa file checkpoint di folder models/'}"
        )

    cloud_limit_reached = IS_CLOUD and st.session_state.get("cloud_prediction_count", 0) >= MAX_CLOUD_PREDICTIONS

    audio_file = None
    audio_info: dict | None = None
    display_name = "rekaman-mikrofon.wav"
    metadata_failed = False
    want_segments = False
    predict_clicked = False
    reanalyze_clicked = False

    with st.sidebar:
        render_section_header(
            "Langkah 1", "Sumber Audio", "Unggah file .wav/.mp3 atau rekam langsung dari mikrofon."
        )

        source_mode = st.radio(
            "Sumber audio",
            options=["Unggah File", "Rekam Mikrofon"],
            horizontal=True,
            label_visibility="collapsed",
        )

        source_label = "upload"
        if source_mode == "Unggah File":
            audio_file = st.file_uploader(
                "Pilih file audio",
                type=["wav", "mp3"],
                label_visibility="collapsed",
                help="Format yang didukung: .wav dan .mp3",
            )
            source_label = "upload"
        else:
            st.caption(
                f"Rekaman otomatis berhenti di {MAX_RECORD_DURATION_SECONDS:.0f} detik. "
                f"Minimal {MIN_RECORD_DURATION_SECONDS:.1f} detik."
            )
            audio_file = record_audio(
                max_seconds=MAX_RECORD_DURATION_SECONDS,
                min_seconds=MIN_RECORD_DURATION_SECONDS,
                key="recorded_audio",
            )
            source_label = "record"

        if audio_file is None:
            _reset_cloud_session()
        else:
            file_key = _audio_key(audio_file, source_label)
            if st.session_state.get("prediction_file_key") != file_key:
                _clear_prediction_cache()
                st.session_state["prediction_file_key"] = file_key

            display_name = (
                Path(audio_file.name).name if getattr(audio_file, "name", None) else "rekaman-mikrofon.wav"
            )

            try:
                audio_file.seek(0)
                audio_info = get_audio_info(audio_file)
                audio_info["filename"] = display_name
            except Exception as exc:
                metadata_failed = True
                st.error(
                    "File audio tidak dapat diproses. "
                    "Coba gunakan file .wav atau .mp3 dengan durasi pendek dan kualitas suara jelas.\n\n"
                    f"Detail teknis: {type(exc).__name__}: {exc}"
                )

            if not metadata_failed:
                render_metadata_card(
                    filename=audio_info["filename"],
                    duration_sec=audio_info["duration_sec"],
                    sample_rate=audio_info["sample_rate"],
                    channels=audio_info["channels"],
                    file_size=format_file_size(getattr(audio_file, "size", None)),
                )

                render_section_header(
                    "Langkah 2", "Pratinjau Audio", "Pastikan audio dapat diputar sebelum melakukan prediksi."
                )
                audio_file.seek(0)
                st.audio(audio_file)
                audio_file.seek(0)
                render_waveform_chart(get_waveform_envelope(audio_file))

                if IS_CLOUD:
                    used = st.session_state.get("cloud_prediction_count", 0)
                    if used >= MAX_CLOUD_PREDICTIONS:
                        st.info(
                            f"Mode cloud: maksimal **{MAX_CLOUD_PREDICTIONS} analisis per sesi** "
                            "(batas RAM server gratis). "
                            "**Refresh halaman (F5)** lalu upload file baru untuk analisis berikutnya."
                        )

                if ENABLE_STT:
                    want_segments = st.checkbox(
                        "Analisis emosi per-segmen (berdasarkan transkrip)",
                        help=(
                            "Membagi audio jadi beberapa segmen berdasarkan timestamp transkrip Whisper, "
                            "lalu menjalankan model emosi per segmen (bukan cuma satu label untuk seluruh audio)."
                        ),
                    )

                st.markdown("<div style='height:0.5rem'></div>", unsafe_allow_html=True)
                predict_clicked = st.button(
                    "Analisis Emosi",
                    type="primary",
                    use_container_width=True,
                    disabled=not model_ready or cloud_limit_reached,
                )
                if st.session_state.get("prediction_cache") is not None and not cloud_limit_reached:
                    reanalyze_clicked = st.button(
                        "Analisis Ulang", use_container_width=True, disabled=not model_ready
                    )

        render_history_list(st.session_state.get("prediction_history", []))

    if audio_file is None or metadata_failed:
        render_empty_state()
        return

    want_stt = ENABLE_STT
    should_predict = model_ready and (predict_clicked or reanalyze_clicked)
    if should_predict and reanalyze_clicked:
        _clear_prediction_cache()

    if should_predict and not st.session_state.get("prediction_cache"):
        try:
            audio_file.seek(0)
            result, preprocess_info = run_prediction(audio_file, device_name)
            st.session_state["prediction_cache"] = {
                "result": result,
                "preprocess_info": preprocess_info,
                "want_stt": want_stt,
                "want_segments": want_segments,
            }
            if IS_CLOUD:
                st.session_state["cloud_prediction_count"] = (
                    st.session_state.get("cloud_prediction_count", 0) + 1
                )
            st.session_state["session_analysis_count"] = st.session_state.get("session_analysis_count", 0) + 1
            history = st.session_state.setdefault("prediction_history", [])
            history.insert(
                0,
                {
                    "filename": display_name,
                    "label": result["predicted_label"],
                    "confidence": result["confidence"],
                    "time": datetime.now().strftime("%H:%M:%S"),
                },
            )
            del history[10:]
        except FileNotFoundError as exc:
            st.error(f"File model tidak ditemukan.\n\n{exc}")
            return
        except RuntimeError as exc:
            error_text = str(exc)
            if "tidak cocok" in error_text.lower() or "checkpoint" in error_text.lower():
                st.error(f"Checkpoint tidak cocok dengan arsitektur model.\n\n{error_text}")
            elif "HuggingFace" in error_text or "pretrained" in error_text.lower():
                st.error(
                    "Gagal memuat model HuggingFace. "
                    "Periksa koneksi internet untuk unduhan pertama kali.\n\n"
                    f"{error_text}"
                )
            else:
                st.error(f"Terjadi kesalahan saat memuat model.\n\n{error_text}")
            return
        except Exception as exc:
            st.error(
                "File audio tidak dapat diproses. "
                "Coba gunakan file .wav atau .mp3 dengan durasi pendek dan kualitas suara jelas.\n\n"
                f"Detail teknis: {type(exc).__name__}: {exc}"
            )
            with st.expander("Traceback lengkap (debug)"):
                st.code(traceback.format_exc(), language="python")
            return

    cache = st.session_state.get("prediction_cache")
    if cache is None:
        render_empty_state()
        return

    result = cache["result"]
    preprocess_info = cache["preprocess_info"]
    want_stt = cache.get("want_stt", False)
    want_segments = cache.get("want_segments", False)

    render_section_header("Hasil", "Ringkasan Emosi")

    summary = summarize_prediction(result)
    render_result_card(summary)

    if preprocess_info["trimmed"]:
        st.warning(
            f"Audio dipotong menjadi maksimal {preprocess_info['max_duration_sec']} detik "
            f"untuk analisis emosi (durasi asli: {preprocess_info['original_duration_sec']} dtk). "
            "Transkrip tetap memakai audio penuh."
        )

    if want_stt:
        if "transcript" not in cache:
            with st.spinner("Mentranskrip ucapan ke teks (Whisper)..."):
                audio_file.seek(0)
                if want_segments:
                    transcript, raw_segments, stt_ok = safe_transcribe_segments(
                        audio_file,
                        WHISPER_MODEL,
                        device_name,
                        WHISPER_LANGUAGE,
                        pipeline_loader=load_whisper_lazy,
                    )
                    cache["raw_segments"] = raw_segments
                else:
                    transcript, stt_ok = safe_transcribe(
                        audio_file,
                        WHISPER_MODEL,
                        device_name,
                        WHISPER_LANGUAGE,
                        pipeline_loader=load_whisper_lazy,
                    )
                cache["transcript"] = transcript
                cache["stt_ok"] = stt_ok
                gc.collect()
        render_transcript_card(cache["transcript"])
        if not cache.get("stt_ok", True):
            st.caption(
                "Catatan: transkripsi Whisper tidak tersedia sementara. "
                "Analisis emosi tetap berjalan normal."
            )

        raw_segments = cache.get("raw_segments")
        if raw_segments:
            if "segment_results" not in cache:
                audio_file.seek(0)
                waveform_16k = get_transcription_waveform(audio_file)
                cache["segment_results"] = run_segment_predictions(waveform_16k, raw_segments, device_name)
            segment_results = cache["segment_results"]
            if segment_results:
                render_segment_timeline(segment_results)
                if len(segment_results) < len(raw_segments):
                    st.caption(
                        f"Menampilkan {len(segment_results)} dari {len(raw_segments)} segmen terdeteksi "
                        "(segmen sangat pendek dilewati atau dibatasi maks 20 segmen)."
                    )
            else:
                st.info("Tidak ada segmen yang cukup panjang untuk dianalisis per-segmen.")

    st.markdown("#### Top 3 Emosi")
    render_top3_cards(result["probabilities_df"])

    st.markdown("#### Confidence Semua Kelas")
    render_probability_bars(result["probabilities_df"], highlight=result["predicted_label"])

    render_export_buttons(result, cache.get("transcript"), display_name)

    with st.expander("Detail Teknis"):
        st.markdown("**Probabilitas Semua Kelas**")
        display_df = result["probabilities_df"][["Emosi", "Persentase (%)"]].copy()
        display_df["Persentase (%)"] = display_df["Persentase (%)"].map(lambda x: f"{x:.2f}%")
        st.dataframe(display_df, use_container_width=True, hide_index=True)

        st.markdown("**Mapping Label**")
        mapping_df = pd.DataFrame(
            [{"ID": idx, "Label": label} for label, idx in sorted(LABEL2ID.items(), key=lambda x: x[1])]
        )
        st.dataframe(mapping_df, use_container_width=True, hide_index=True)

        st.markdown("**Raw Logits**")
        logits_df = pd.DataFrame(
            {
                "Emosi": [ID2LABEL[i] for i in range(len(result["logits"]))],
                "Logit": result["logits"],
            }
        )
        st.dataframe(logits_df, use_container_width=True, hide_index=True)

        st.markdown("**Probabilitas (Raw)**")
        prob_raw_df = pd.DataFrame(
            {
                "Emosi": [ID2LABEL[i] for i in range(len(result["probabilities"]))],
                "Probabilitas": result["probabilities"],
            }
        )
        st.dataframe(prob_raw_df, use_container_width=True, hide_index=True)

        st.markdown("**Path Model**")
        st.code(MODEL_DISPLAY_PATH, language=None)


main()