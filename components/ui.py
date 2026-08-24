"""Komponen UI murni untuk merender elemen antarmuka."""

import html
import json
from pathlib import Path

import streamlit as st
import pandas as pd
from config import EMOTION_ICONS, EMOTION_COLORS


def format_file_size(size_bytes: int | None) -> str:
    if not size_bytes:
        return "—"
    if size_bytes < 1024:
        return f"{size_bytes} B"
    if size_bytes < 1024 * 1024:
        return f"{size_bytes / 1024:.1f} KB"
    return f"{size_bytes / (1024 * 1024):.2f} MB"


def render_section_header(step: str, title: str = "", desc: str | None = None) -> None:
    title_html = f'<div class="section-title">{html.escape(title)}</div>' if title else ""
    desc_html = f'<p class="section-desc">{html.escape(desc)}</p>' if desc else ""
    st.markdown(
        f"""
        <div class="section-card">
            <div class="section-step">{html.escape(step)}</div>
            {title_html}
            {desc_html}
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_transcript_card(text: str) -> None:
    if text:
        body = f'<div class="transcript-text">"{html.escape(text)}"</div>'
    else:
        body = (
            '<div class="transcript-empty">Tidak ada ucapan yang terdeteksi '
            "(audio mungkin tanpa kata-kata yang jelas).</div>"
        )
    st.markdown(
        f"""
        <div class="transcript-card">
            <div class="transcript-head">📝 Transkrip Ucapan (Speech-to-Text)</div>
            {body}
        </div>
        """,
        unsafe_allow_html=True,
    )


def _format_timestamp(seconds: float) -> str:
    total = int(round(seconds))
    return f"{total // 60:02d}:{total % 60:02d}"


def render_segment_timeline(segments: list[dict]) -> None:
    """Timeline emosi per-segmen transkrip (opt-in, hasil dari run_segment_predictions)."""
    render_section_header("Per-Segmen", "Emosi Sepanjang Transkrip")
    for seg in segments:
        label = seg["result"]["predicted_label"]
        confidence = seg["result"]["confidence"] * 100
        icon = EMOTION_ICONS.get(label, "🎭")
        accent = EMOTION_COLORS.get(label, "#a3a3a3")
        time_range = f"{_format_timestamp(seg['start'])}–{_format_timestamp(seg['end'])}"
        text = html.escape(seg["text"] or "(tanpa teks)")
        st.markdown(
            f"""
            <div class="segment-card" style="--emotion-color:{accent};">
                <div class="segment-time">{time_range}</div>
                <div class="segment-text">"{text}"</div>
                <div class="segment-emotion-row">
                    <span>{icon} <span style="text-transform:capitalize;">{label}</span></span>
                    <span>{confidence:.1f}%</span>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )


def render_hero() -> None:
    st.markdown(
        """
        <div class="hero-card">
            <div class="hero-title">Speech Emotion Recognition</div>
            <p class="hero-subtitle">
                Unggah atau rekam suara, dengarkan preview, lalu sistem akan memprediksi emosi
                dominan sekaligus menampilkan transkrip teks dari audio.
            </p>
            <span class="hero-badge">WavLM + Whisper STT</span>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_empty_state(
    icon: str = "🎧",
    title: str = "Belum ada audio yang tersedia.",
    desc: str = "Unggah file .wav/.mp3 atau rekam langsung dari mikrofon.",
) -> None:
    st.markdown(
        f"""
        <div class="empty-state">
            <div class="empty-icon">{icon}</div>
            <div class="empty-title">{html.escape(title)}</div>
            <div class="empty-desc">{html.escape(desc)}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_status_card(icon: str, label: str, value: str, ok: bool = True, full_width: bool = False) -> str:
    """Kartu status dengan aksen warna ok/fail, dipakai di grid dash-status-grid."""
    state = "ok" if ok else "fail"
    width_class = " dash-status-card--full" if full_width else ""
    return (
        f'<div class="dash-status-card {state}{width_class}">'
        f'<div class="dash-status-icon">{icon}</div>'
        f'<div class="dash-status-body">'
        f'<div class="dash-status-label">{html.escape(label)}</div>'
        f'<div class="dash-status-value">{html.escape(value)}</div>'
        f"</div></div>"
    )


def render_status_grid(cards: list[str]) -> None:
    st.markdown(f'<div class="dash-status-grid">{"".join(cards)}</div>', unsafe_allow_html=True)


def render_quota_card(used: int, limit: int) -> str:
    """Kartu kuota cloud dengan progress bar, sengaja full-width untuk memutus grid rata."""
    pct = min(100.0, used / limit * 100) if limit else 0.0
    return (
        '<div class="dash-status-card ok dash-status-card--full dash-quota-card">'
        '<div class="dash-status-icon">☁️</div>'
        '<div class="dash-status-body" style="flex:1;">'
        '<div class="dash-status-label">Kuota Cloud Sesi Ini</div>'
        f'<div class="dash-status-value">{used}/{limit} prediksi</div>'
        '<div class="prob-bar-wrap" style="margin-top:0.45rem;">'
        f'<div class="prob-bar-fill" style="width:{pct:.0f}%; background:#fafafa;"></div>'
        "</div></div></div>"
    )


def render_activity_list(history: list[dict]) -> None:
    """Riwayat aktivitas sesi dengan aksen warna per emosi, pengganti st.dataframe generik."""
    rows = "".join(
        f'<div class="dash-activity-row" style="--emotion-color:{EMOTION_COLORS.get(entry["label"], "#a3a3a3")};">'
        f'<span class="dash-activity-emoji">{EMOTION_ICONS.get(entry["label"], "🎭")}</span>'
        f'<div class="dash-activity-main">'
        f'<div class="dash-activity-label">{html.escape(entry["label"])}</div>'
        f'<div class="dash-activity-meta">{html.escape(entry["time"])} · {html.escape(entry["filename"])}</div>'
        "</div>"
        f'<div class="dash-activity-conf">{entry["confidence"] * 100:.1f}%</div>'
        "</div>"
        for entry in history
    )
    st.markdown(f'<div class="dash-activity-list">{rows}</div>', unsafe_allow_html=True)


def render_metadata_card(
    filename: str,
    duration_sec: float,
    sample_rate: int,
    channels: int,
    file_size: str,
) -> None:
    """Kartu metadata audio, ditata 2x2 karena dipanggil dari kolom sidebar yang sempit."""
    render_section_header("Metadata Audio")
    st.markdown(f'<div class="meta-label">Nama File</div><div class="meta-value">{filename}</div>', unsafe_allow_html=True)
    r1c1, r1c2 = st.columns(2)
    with r1c1:
        st.markdown(f'<div class="meta-label">Durasi</div><div class="meta-value">{duration_sec} dtk</div>', unsafe_allow_html=True)
    with r1c2:
        st.markdown(f'<div class="meta-label">Sample Rate</div><div class="meta-value">{sample_rate} Hz</div>', unsafe_allow_html=True)
    r2c1, r2c2 = st.columns(2)
    with r2c1:
        st.markdown(f'<div class="meta-label">Channel</div><div class="meta-value">{channels}</div>', unsafe_allow_html=True)
    with r2c2:
        st.markdown(f'<div class="meta-label">Ukuran File</div><div class="meta-value">{file_size}</div>', unsafe_allow_html=True)


def render_waveform_chart(envelope_df: pd.DataFrame) -> None:
    if envelope_df.empty:
        return
    st.markdown('<div class="meta-label" style="margin-top:0.75rem;">Bentuk Gelombang</div>', unsafe_allow_html=True)
    # ponytail: st.line_chart dengan height kecil (dicoba 100/140) membuat area plot
    # kolaps ke 0px (chrome legenda+axis menghabiskan seluruh tinggi) -- terverifikasi
    # lewat Playwright, bukan cuma di sidebar. 220 adalah titik aman terkecil yang
    # masih menyisakan area plot terlihat; naikkan lagi kalau butuh chart lebih pendek.
    st.line_chart(envelope_df, height=220, color=["#fafafa", "#737373"])


def render_history_list(history: list[dict]) -> None:
    """Daftar riwayat prediksi sesi berjalan (opt-in, hanya dirender kalau history tidak kosong)."""
    if not history:
        return
    items = "".join(
        f'<div class="sidebar-history-item">'
        f'<span class="sidebar-history-emoji">{EMOTION_ICONS.get(entry["label"], "🎭")}</span>'
        f'<div class="sidebar-history-body">'
        f'<div class="sidebar-history-label">{html.escape(entry["label"])} · {entry["confidence"] * 100:.0f}%</div>'
        f'<div class="sidebar-history-meta">{html.escape(entry["time"])} · {html.escape(entry["filename"])}</div>'
        f"</div></div>"
        for entry in history
    )
    st.markdown(
        f'<div class="sidebar-section-label">Riwayat Sesi</div>'
        f'<div class="sidebar-history-list">{items}</div>',
        unsafe_allow_html=True,
    )


def render_top3_cards(prob_df: pd.DataFrame) -> None:
    top3 = prob_df.head(3).reset_index(drop=True)
    cols = st.columns(3)
    for i, col in enumerate(cols):
        if i >= len(top3):
            break
        row = top3.iloc[i]
        emotion = row["Emosi"]
        pct = row["Persentase (%)"]
        col.markdown(
            f"""
            <div class="top3-card">
                <div class="top3-rank">#{i + 1}</div>
                <div class="top3-emoji">{EMOTION_ICONS.get(emotion, "🎭")}</div>
                <div class="top3-label">{emotion}</div>
                <div class="top3-conf-label">Confidence</div>
                <div class="top3-pct">{pct:.1f}%</div>
            </div>
            """,
            unsafe_allow_html=True,
        )


def render_probability_bars(prob_df: pd.DataFrame, highlight: str | None = None) -> None:
    for _, row in prob_df.iterrows():
        emotion = row["Emosi"]
        pct = float(row["Persentase (%)"])
        color = EMOTION_COLORS.get(emotion, "#a3a3a3")
        weight = "700" if emotion == highlight else "500"
        st.markdown(
            f"""
            <div class="prob-row-label">
                <span style="font-weight:{weight}; text-transform:capitalize;">
                    {EMOTION_ICONS.get(emotion, "")} {emotion}
                </span>
                <span style="font-weight:650; color:#fafafa;">{pct:.1f}%</span>
            </div>
            <div class="prob-bar-wrap">
                <div class="prob-bar-fill" style="width:{pct:.1f}%; background:{color};"></div>
            </div>
            """,
            unsafe_allow_html=True,
        )


def render_export_buttons(result: dict, transcript: str | None, filename: str) -> None:
    """Tombol unduh hasil prediksi sebagai CSV atau JSON."""
    prob_df = result["probabilities_df"][["Emosi", "Persentase (%)"]]
    payload = {
        "filename": filename,
        "predicted_label": result["predicted_label"],
        "confidence": result["confidence"],
        "probabilities": {
            row["Emosi"]: round(float(row["Persentase (%)"]), 2) for _, row in prob_df.iterrows()
        },
        "transcript": transcript,
    }

    c1, c2 = st.columns(2)
    with c1:
        st.download_button(
            "Unduh CSV",
            data=prob_df.to_csv(index=False).encode("utf-8"),
            file_name=f"ser_hasil_{Path(filename).stem}.csv",
            mime="text/csv",
            use_container_width=True,
        )
    with c2:
        st.download_button(
            "Unduh JSON",
            data=json.dumps(payload, ensure_ascii=False, indent=2).encode("utf-8"),
            file_name=f"ser_hasil_{Path(filename).stem}.json",
            mime="application/json",
            use_container_width=True,
        )


def render_result_card(summary: dict) -> None:
    label = summary["top_label"]
    confidence = summary["top_pct"] / 100
    icon = EMOTION_ICONS.get(label, "🎭")
    accent = EMOTION_COLORS.get(label, "#a3a3a3")

    margin_html = ""
    if summary["second_label"]:
        margin_html = (
            f'<div class="result-margin">+{summary["margin_pp"]:.1f} p.p. dari '
            f'{summary["second_label"]} (#2 · {summary["second_pct"]:.1f}%)</div>'
        )

    st.markdown(
        f"""
        <div class="result-card" style="border-color: {accent}44;">
            <div class="result-inner">
                <div class="result-dominance">Emosi Dominan · Tertinggi dari {summary["num_classes"]} kelas</div>
                <div class="result-emoji">{icon}</div>
                <p class="result-label">{label}</p>
                <div class="result-conf-label">Skor Tertinggi</div>
                <div class="result-confidence">{summary["top_pct"]:.1f}%</div>
                <p class="result-rank-note">{summary["separation"]}</p>
                {margin_html}
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )