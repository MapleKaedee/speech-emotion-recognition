"""Halaman detail Dataset — distribusi sampel, eksplorasi data, kualitas label."""

from __future__ import annotations

import altair as alt
import streamlit as st

from config import EMOTION_COLORS
from services import load_dataset_metadata, load_label_noise_candidates
from utils import ID2LABEL
from components.css import inject_custom_css
from components.ui import render_section_header

LABELS_ORDER = [ID2LABEL[i] for i in sorted(ID2LABEL)]
SPLIT_COLORS = {"train": "#fafafa", "val": "#a3a3a3", "test": "#525252"}


def _id_number(n: int) -> str:
    return f"{n:,}".replace(",", ".")


def main() -> None:
    inject_custom_css()
    st.markdown(
        """
        <div class="hero-card">
            <div class="hero-title">Dataset</div>
            <p class="hero-subtitle">Distribusi lengkap, eksplorasi sampel, dan kualitas label dataset training.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    df = load_dataset_metadata()
    if df is None:
        st.warning("File metadata dataset (models/metadata_split_v4.csv) tidak ditemukan.")
        return

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Sampel", _id_number(len(df)))
    c2.metric("Jumlah Sumber", df["sumber"].nunique())
    c3.metric("Jumlah Bahasa", df["bahasa"].nunique())
    c4.metric("Jumlah Kelas", len(LABELS_ORDER))

    render_section_header("Distribusi", "Sampel per Kategori")

    col1, col2 = st.columns(2)
    with col1:
        st.caption("Per Kelas Emosi")
        emo_df = df["emosi"].value_counts().rename_axis("Emosi").reset_index(name="Jumlah")
        emo_chart = (
            alt.Chart(emo_df)
            .mark_bar(cornerRadiusTopLeft=6, cornerRadiusTopRight=6)
            .encode(
                x=alt.X("Emosi:N", sort=LABELS_ORDER, title=None),
                y=alt.Y("Jumlah:Q"),
                color=alt.Color(
                    "Emosi:N",
                    scale=alt.Scale(domain=list(EMOTION_COLORS.keys()), range=list(EMOTION_COLORS.values())),
                    legend=None,
                ),
                tooltip=["Emosi", "Jumlah"],
            )
            .properties(height=300)
        )
        st.altair_chart(emo_chart, use_container_width=True)
    with col2:
        st.caption("Per Split (Train / Val / Test)")
        split_df = df["split"].value_counts().reindex(["train", "val", "test"]).rename("Jumlah")
        st.bar_chart(split_df, height=300)

    col3, col4 = st.columns(2)
    with col3:
        st.caption("Per Sumber Dataset")
        sumber_df = df["sumber"].value_counts().rename("Jumlah")
        st.bar_chart(sumber_df, height=280)
    with col4:
        st.caption("Per Bahasa")
        bahasa_df = df["bahasa"].value_counts().rename("Jumlah")
        st.bar_chart(bahasa_df, height=280)

    st.caption("Kelas Emosi × Split — cek keseimbangan kelas antar split")
    cross_df = df.groupby(["emosi", "split"]).size().reset_index(name="Jumlah")
    cross_chart = (
        alt.Chart(cross_df)
        .mark_bar()
        .encode(
            x=alt.X("emosi:N", title=None, sort=LABELS_ORDER),
            xOffset=alt.XOffset("split:N", sort=["train", "val", "test"]),
            y=alt.Y("Jumlah:Q"),
            color=alt.Color(
                "split:N",
                scale=alt.Scale(domain=list(SPLIT_COLORS.keys()), range=list(SPLIT_COLORS.values())),
                legend=alt.Legend(title="Split"),
            ),
            tooltip=["emosi", "split", "Jumlah"],
        )
        .properties(height=320)
    )
    st.altair_chart(cross_chart, use_container_width=True)

    st.caption("Distribusi Bobot Sample (class-balancing weight)")
    hist_chart = (
        alt.Chart(df)
        .mark_bar(color="#fafafa")
        .encode(
            x=alt.X("bobot:Q", bin=alt.Bin(maxbins=30), title="Bobot"),
            y=alt.Y("count():Q", title="Jumlah Sampel"),
            tooltip=[alt.Tooltip("count():Q", title="Jumlah")],
        )
        .properties(height=260)
    )
    st.altair_chart(hist_chart, use_container_width=True)

    render_section_header("Eksplorasi", "Cari Sampel Data")

    fc1, fc2, fc3 = st.columns(3)
    emosi_filter = fc1.selectbox("Emosi", ["Semua"] + LABELS_ORDER)
    sumber_filter = fc2.selectbox("Sumber", ["Semua"] + sorted(df["sumber"].unique().tolist()))
    split_filter = fc3.selectbox("Split", ["Semua", "train", "val", "test"])

    filtered = df
    if emosi_filter != "Semua":
        filtered = filtered[filtered["emosi"] == emosi_filter]
    if sumber_filter != "Semua":
        filtered = filtered[filtered["sumber"] == sumber_filter]
    if split_filter != "Semua":
        filtered = filtered[filtered["split"] == split_filter]

    st.caption(f"{_id_number(len(filtered))} baris cocok (menampilkan maks 200)")
    st.dataframe(
        filtered[["path", "emosi", "sumber", "bahasa", "split"]].head(200),
        use_container_width=True,
        hide_index=True,
    )

    render_section_header("Kualitas Data", "Kandidat Label Bising")

    noise_df = load_label_noise_candidates()
    if noise_df is not None:
        st.metric("Kandidat Label Bising", _id_number(len(noise_df)))
        st.caption(
            "Sampel dengan prediksi model tidak sesuai label asli & confidence rendah saat audit training "
            "— dipertimbangkan sebagai noise, bukan langsung dihapus."
        )
        display_noise = noise_df.copy()
        display_noise["confidence"] = display_noise["confidence"].map(lambda x: f"{x * 100:.1f}%")
        st.dataframe(display_noise, use_container_width=True, hide_index=True)
    else:
        st.info("File label_noise_candidates.csv tidak ditemukan.")


main()