"""Entry point aplikasi Streamlit — konfigurasi global & navigasi antar halaman."""

from __future__ import annotations

import streamlit as st
import torch

from components.css import inject_custom_css
from config import ENABLE_STT, WHISPER_MODEL

st.set_page_config(
    page_title="Speech Emotion Recognition",
    page_icon="🎙️",
    layout="wide",
    initial_sidebar_state="expanded",
)
inject_custom_css()


def preload_whisper() -> None:
    """Siapkan Whisper saat startup agar user tidak terkena cold start STT."""
    if not ENABLE_STT:
        return

    from services import load_whisper_lazy

    device_name = "cuda" if torch.cuda.is_available() else "cpu"
    try:
        load_whisper_lazy(WHISPER_MODEL, device_name)
    except Exception as exc:
        st.warning(
            "Whisper belum berhasil dipreload. Fitur transkripsi akan mencoba lagi "
            f"saat digunakan. Detail: {type(exc).__name__}: {exc}"
        )


preload_whisper()

pages = [
    st.Page("pages/home.py", title="Home", icon=":material/home:", default=True),
    st.Page("pages/analisis.py", title="Analisis Emosi", icon=":material/mic:"),
    st.Page("pages/model.py", title="Model", icon=":material/psychology:"),
    st.Page("pages/dataset.py", title="Dataset", icon=":material/dataset:"),
]
pg = st.navigation(pages, position="top")
pg.run()
