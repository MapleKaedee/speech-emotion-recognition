"""Halaman Home — landing page, status sistem, dan aktivitas sesi berjalan."""

from __future__ import annotations

from collections import Counter

import torch
import streamlit as st

from config import EMOTION_COLORS, EMOTION_ICONS, IS_CLOUD, MAX_CLOUD_PREDICTIONS
from services import check_model_ready
from utils import ID2LABEL
from components.css import inject_custom_css
from components.ui import (
    render_activity_list,
    render_empty_state,
    render_hero,
    render_quota_card,
    render_section_header,
    render_status_card,
    render_status_grid,
)


def main() -> None:
    inject_custom_css()
    render_hero()

    device_name = "cuda" if torch.cuda.is_available() else "cpu"
    model_ready, model_error = check_model_ready(device_name)
    device_label = "GPU (CUDA)" if device_name == "cuda" else "CPU"

    render_section_header("Status Sistem", "Kesiapan Layanan Saat Ini")
    status_cards = [
        render_status_card("🧠", "Status Model", "Siap" if model_ready else "Gagal", ok=model_ready),
        render_status_card("💻", "Perangkat", device_label),
    ]
    render_status_grid(status_cards)
    if IS_CLOUD:
        used = st.session_state.get("cloud_prediction_count", 0)
        st.markdown(render_quota_card(used, MAX_CLOUD_PREDICTIONS), unsafe_allow_html=True)
    if not model_ready and model_error:
        st.error(model_error)

    emotion_items = "".join(
        f'<div class="sidebar-emotion-item" style="--emotion-color:{EMOTION_COLORS[label]};">'
        f'<span class="sidebar-emotion-emoji">{EMOTION_ICONS[label]}</span>'
        f"<span>{label}</span>"
        f'<span class="sidebar-emotion-id">{idx}</span></div>'
        for idx in sorted(ID2LABEL)
        for label in [ID2LABEL[idx]]
    )
    st.markdown(
        f"""
        <div class="sidebar-stat-grid">
            <div class="sidebar-stat-card"><div class="sidebar-stat-icon">🧠</div>
            <div class="sidebar-stat-label">Backbone</div>
            <div class="sidebar-stat-value">WavLM Base Plus</div></div>
            <div class="sidebar-stat-card"><div class="sidebar-stat-icon">🎭</div>
            <div class="sidebar-stat-label">Kelas</div>
            <div class="sidebar-stat-value">6 Emosi</div></div>
            <div class="sidebar-stat-card"><div class="sidebar-stat-icon">📁</div>
            <div class="sidebar-stat-label">Format</div>
            <div class="sidebar-stat-value">.wav / .mp3</div></div>
        </div>
        <div class="sidebar-section-label">Peta Emosi</div>
        <div class="sidebar-emotion-grid">{emotion_items}</div>
        """,
        unsafe_allow_html=True,
    )

    render_section_header("Aktivitas Sesi Ini", "Riwayat Analisis pada Sesi Berjalan")
    history = st.session_state.get("prediction_history", [])
    if not history:
        render_empty_state(
            icon="📭",
            title="Belum ada analisis yang dijalankan pada sesi ini.",
            desc="Riwayat akan muncul di sini setelah kamu menjalankan analisis pertama.",
        )
    else:
        total_count = st.session_state.get("session_analysis_count", len(history))
        top_emotion, _ = Counter(entry["label"] for entry in history).most_common(1)[0]
        activity_cards = [
            render_status_card("📊", "Total Analisis Sesi Ini", str(total_count)),
            render_status_card(
                EMOTION_ICONS.get(top_emotion, "🎭"),
                "Emosi Terbanyak",
                top_emotion.capitalize(),
            ),
        ]
        render_status_grid(activity_cards)
        render_activity_list(history)

    st.markdown("<div style='height:0.5rem'></div>", unsafe_allow_html=True)
    link1, link2 = st.columns(2)
    with link1:
        st.page_link("pages/analisis.py", label="Mulai Analisis →", icon=":material/mic:")
    with link2:
        st.page_link("pages/model.py", label="Lihat detail Model →", icon=":material/psychology:")


main()  