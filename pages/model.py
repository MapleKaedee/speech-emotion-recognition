"""Halaman detail Model — arsitektur, kurva training, confusion matrix, metrik per kelas."""

from __future__ import annotations

import altair as alt
import pandas as pd
import streamlit as st
import torch
from sklearn.metrics import classification_report

from config import MODEL_DISPLAY_PATH, SER_BACKBONE
from services import (
    MODELS_DIR,
    TEST_EVALUATION_PATH,
    TRAINING_CURVE_IMAGE_PATH,
    TRAINING_HISTORY_PATH,
    load_model_metrics,
    load_ser_model,
    load_test_evaluation,
    load_training_history,
)
from utils import ID2LABEL
from components.css import inject_custom_css
from components.ui import render_section_header

LABELS_ORDER = [ID2LABEL[i] for i in sorted(ID2LABEL)]


def _id_number(n: int) -> str:
    return f"{n:,}".replace(",", ".")


def _render_architecture() -> None:
    render_section_header("Arsitektur", "Alur Model")
    st.code(
        "WavLM Base Plus (microsoft/wavlm-base-plus, hidden=768)\n"
        "        │\n"
        "        ▼\n"
        "Attentive Statistics Pooling (weighted mean + std -> 1536-dim)\n"
        "        │\n"
        "        ▼\n"
        "LayerNorm -> Dropout(0.30) -> Linear(1536,256) -> GELU -> Dropout(0.225) -> Linear(256,6)\n"
        "        │\n"
        "        ▼\n"
        "   6 kelas emosi (softmax)",
        language=None,
    )


def _render_training_curves(history: dict) -> None:
    render_section_header("Training", "Kurva Akurasi & Loss per Epoch")

    epochs = list(range(1, len(history.get("la", [])) + 1))
    colors = ["#fafafa", "#737373"]

    col1, col2 = st.columns(2)
    with col1:
        st.caption("Akurasi")
        acc_df = pd.DataFrame({"Epoch": epochs, "Train": history["la"], "Validasi": history["va"]}).melt(
            "Epoch", var_name="Set", value_name="Akurasi"
        )
        acc_chart = (
            alt.Chart(acc_df)
            .mark_line(point=True)
            .encode(
                x=alt.X("Epoch:O"),
                y=alt.Y("Akurasi:Q", axis=alt.Axis(format=".0%")),
                color=alt.Color("Set:N", scale=alt.Scale(range=colors), legend=alt.Legend(title=None)),
                tooltip=["Epoch", "Set", alt.Tooltip("Akurasi:Q", format=".2%")],
            )
            .properties(height=280)
        )
        st.altair_chart(acc_chart, use_container_width=True)
    with col2:
        st.caption("Loss")
        loss_df = pd.DataFrame({"Epoch": epochs, "Train": history["ll"], "Validasi": history["vl"]}).melt(
            "Epoch", var_name="Set", value_name="Loss"
        )
        loss_chart = (
            alt.Chart(loss_df)
            .mark_line(point=True)
            .encode(
                x=alt.X("Epoch:O"),
                y=alt.Y("Loss:Q"),
                color=alt.Color("Set:N", scale=alt.Scale(range=colors), legend=alt.Legend(title=None)),
                tooltip=["Epoch", "Set", alt.Tooltip("Loss:Q", format=".3f")],
            )
            .properties(height=280)
        )
        st.altair_chart(loss_chart, use_container_width=True)

    stages = history.get("stage", [])
    if stages:
        transitions = []
        prev = None
        for i, s in enumerate(stages, start=1):
            if s != prev:
                transitions.append(f"Epoch {i}: {s}")
                prev = s
        st.caption("Gradual unfreezing — " + " → ".join(transitions))


def _render_confusion_matrix(eval_df: pd.DataFrame) -> None:
    render_section_header("Evaluasi Test", "Confusion Matrix")

    cm = pd.crosstab(eval_df["emosi_true"], eval_df["emosi_pred"]).reindex(
        index=LABELS_ORDER, columns=LABELS_ORDER, fill_value=0
    )
    cm_long = cm.reset_index().melt(id_vars="emosi_true", var_name="emosi_pred", value_name="Jumlah")
    threshold = cm_long["Jumlah"].max() / 2

    heatmap = (
        alt.Chart(cm_long)
        .mark_rect()
        .encode(
            x=alt.X("emosi_pred:N", title="Prediksi", sort=LABELS_ORDER),
            y=alt.Y("emosi_true:N", title="Aktual", sort=LABELS_ORDER),
            color=alt.Color("Jumlah:Q", scale=alt.Scale(scheme="blues"), legend=None),
            tooltip=["emosi_true", "emosi_pred", "Jumlah"],
        )
    )
    text = (
        alt.Chart(cm_long)
        .mark_text(baseline="middle", fontSize=13)
        .encode(
            x=alt.X("emosi_pred:N", sort=LABELS_ORDER),
            y=alt.Y("emosi_true:N", sort=LABELS_ORDER),
            text="Jumlah:Q",
            color=alt.condition(alt.datum.Jumlah > threshold, alt.value("white"), alt.value("black")),
        )
    )
    st.altair_chart((heatmap + text).properties(height=380), use_container_width=True)

    st.caption("Metrik per Kelas")
    report = classification_report(
        eval_df["emosi_true"], eval_df["emosi_pred"], labels=LABELS_ORDER, output_dict=True, zero_division=0
    )
    report_df = pd.DataFrame(report).T.loc[LABELS_ORDER]
    report_df = report_df.rename(
        columns={"precision": "Precision", "recall": "Recall", "f1-score": "F1-Score", "support": "Support"}
    )
    report_df[["Precision", "Recall", "F1-Score"]] = report_df[["Precision", "Recall", "F1-Score"]].round(3)
    report_df["Support"] = report_df["Support"].astype(int)
    st.dataframe(report_df, use_container_width=True)


def main() -> None:
    inject_custom_css()
    st.markdown(
        """
        <div class="hero-card">
            <div class="hero-title">Model</div>
            <p class="hero-subtitle">Arsitektur, performa training, dan evaluasi model WavLM SER v4.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    metrics = load_model_metrics() or {}
    device_name = "cuda" if torch.cuda.is_available() else "cpu"
    try:
        model, _ = load_ser_model(device_name)
        param_label = f"{sum(p.numel() for p in model.parameters()) / 1e6:.1f} Jt"
    except Exception:
        param_label = "—"

    test_acc = metrics.get("test_acc")
    val_acc = metrics.get("best_val_acc")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Akurasi Test", f"{test_acc * 100:.1f}%" if test_acc is not None else "—")
    c2.metric("Akurasi Val Terbaik", f"{val_acc * 100:.1f}%" if val_acc is not None else "—")
    c3.metric("Epoch Terbaik", metrics.get("best_epoch", "—"))
    c4.metric("Total Parameter", param_label)

    _render_architecture()

    history = load_training_history()
    if history:
        _render_training_curves(history)
    elif TRAINING_CURVE_IMAGE_PATH.exists():
        render_section_header("Training", "Kurva Akurasi & Loss per Epoch")
        st.image(
            str(TRAINING_CURVE_IMAGE_PATH),
            caption="Kurva training v4 (gambar statis — data JSON mentah tidak tersedia)",
            width="stretch",
        )
    else:
        training_curve_path = MODELS_DIR / "kurva_training_v4.png"
        if training_curve_path.exists():
            st.image(
                str(training_curve_path),
                caption="Kurva training model v4",
                use_container_width=True,
            )
            st.caption(
                f"{TRAINING_HISTORY_PATH.name} belum tersedia. "
                "Grafik ditampilkan dari artefak kurva training v4."
            )
        else:
            st.info(
                f"File {TRAINING_HISTORY_PATH.name} dan artefak kurva training v4 "
                "tidak ditemukan."
            )

    eval_df = load_test_evaluation()
    if eval_df is not None:
        _render_confusion_matrix(eval_df)
    else:
        st.info(
            f"File {TEST_EVALUATION_PATH.name} tidak ditemukan. "
            "Confusion matrix tidak dapat ditampilkan."
        )

    with st.expander("Konfigurasi Training"):
        st.caption(f"Backbone: {SER_BACKBONE}")
        st.caption(f"Dropout: {metrics.get('dropout', '—')} · Label Smoothing: {metrics.get('label_smoothing', '—')}")
        st.caption(
            f"Mixup alpha: {metrics.get('mixup_alpha', '—')} · Mixup prob: {metrics.get('mixup_prob', '—')}"
        )
        st.caption(f"Kandidat label bising terdeteksi saat training: {metrics.get('noise_candidates_count', '—')}")

        bobot = metrics.get("cremad_bobot_per_kelas")
        if bobot:
            st.markdown("**Bobot Kelas (CREMA-D)**")
            bobot_df = pd.DataFrame(sorted(bobot.items(), key=lambda x: -x[1]), columns=["Emosi", "Bobot"])
            st.dataframe(bobot_df, use_container_width=True, hide_index=True)

        st.markdown("**Path Checkpoint**")
        st.code(MODEL_DISPLAY_PATH, language=None)


main()
