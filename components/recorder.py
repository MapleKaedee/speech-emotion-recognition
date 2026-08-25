"""Custom component perekam mikrofon dengan auto-stop durasi maksimum & validasi durasi minimum.

st.audio_input bawaan Streamlit tidak punya API untuk menghentikan rekaman otomatis
(murni dikontrol browser). Component "no-build" ini (index.html statis, tanpa
React/npm) mengimplementasikan protokol postMessage Streamlit secara manual supaya
MediaRecorder di browser bisa di-auto-stop lewat JS timer, lalu di-decode ulang jadi
WAV asli di sisi browser (Web Audio API) sebelum dikirim ke Python -- karena
MediaRecorder cuma bisa menghasilkan WebM/Opus, dan pipeline utils.load_audio() di
project ini tidak bisa membaca format itu tanpa ffmpeg/torchcodec.
"""

import base64
import io
from pathlib import Path

import streamlit as st
import streamlit.components.v1 as components

_component = components.declare_component(
    "mic_recorder", path=str(Path(__file__).parent / "recorder_frontend")
)


def record_audio(max_seconds: float, min_seconds: float, key: str | None = None) -> io.BytesIO | None:
    """Render tombol rekam, kembalikan buffer WAV siap pakai atau None."""
    result = _component(max_seconds=max_seconds, min_seconds=min_seconds, key=key)
    if not result:
        return None

    if result.get("error") == "too_short":
        st.error(
            f"Rekaman terlalu pendek ({result['duration_ms'] / 1000:.2f} dtk). "
            f"Minimal {min_seconds:.1f} detik — silakan ulangi rekam."
        )
        return None
    if result.get("error") == "processing_failed":
        st.error("Gagal memproses rekaman mikrofon. Coba rekam ulang.")
        return None

    audio_base64 = result.get("audio_base64")
    if not audio_base64:
        return None

    _, b64data = audio_base64.split(",", 1)
    buf = io.BytesIO(base64.b64decode(b64data))
    buf.name = "rekaman-mikrofon.wav"
    buf.size = buf.getbuffer().nbytes
    return buf