"""Definisi arsitektur model WavLM SER, download checkpoint, dan load model."""

from __future__ import annotations

from pathlib import Path

import torch
import torch.nn as nn
from huggingface_hub import hf_hub_download
from transformers import AutoConfig, AutoModel

PRETRAINED_MODEL = "microsoft/wavlm-base-plus"
HIDDEN_SIZE = 768
NUM_LABELS = 6
DEFAULT_DROPOUT = 0.30  # Harus identik dengan arsitektur pipeline v4

BASE_DIR = Path(__file__).resolve().parent
MODEL_REPO_ID = "elnathzzz/wavlm-ser-multilingual"
MODEL_FILENAME = "ser_wavlm_v4_best.pt"
MODEL_PATH = BASE_DIR / "models" / MODEL_FILENAME


class AttentionPooling(nn.Module):
    """Attentive statistics pooling: weighted mean + weighted std (output hidden*2).

    Arsitektur ini WAJIB identik dengan `AttentiveStatsPooling` pada
    pipeline/ver4-ser-pipeline.ipynb (sumber kebenaran v4). Checkpoint dilatih dengan
    fitur [weighted_mean, weighted_std], bukan [attn_pooled, plain_mean].
    Nama atribut `self.attn` dipertahankan agar key state_dict tetap
    kompatibel dengan checkpoint (backward-compat loading).
    """

    def __init__(self, hidden_size: int = HIDDEN_SIZE, dropout: float = DEFAULT_DROPOUT):
        super().__init__()
        self.attn = nn.Sequential(
            nn.LayerNorm(hidden_size),
            nn.Linear(hidden_size, 128),
            nn.Tanh(),
            nn.Dropout(dropout),
            nn.Linear(128, 1),
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        scores = self.attn(hidden_states).squeeze(-1)
        if attention_mask is not None:
            # -1e4 (bukan -inf) untuk mencegah NaN saat seluruh frame dalam
            # satu sample termasking (softmax(-inf) -> NaN pada edge case).
            scores = scores.masked_fill(attention_mask == 0, -1e4)
        weights = torch.softmax(scores, dim=-1).unsqueeze(-1)

        mean = torch.sum(weights * hidden_states, dim=1)
        variance = torch.sum(weights * (hidden_states - mean.unsqueeze(1)) ** 2, dim=1)
        std = torch.sqrt(variance.clamp(min=1e-6))

        return torch.cat([mean, std], dim=-1), weights.squeeze(-1)


class WavLMSERModel(nn.Module):
    """Model SER berbasis WavLM dengan attentive statistics pooling (mean + std)."""

    def __init__(
        self,
        num_labels: int = NUM_LABELS,
        dropout: float = DEFAULT_DROPOUT,
        pretrained_model: str = PRETRAINED_MODEL,
    ):
        super().__init__()
        self.config = AutoConfig.from_pretrained(pretrained_model)
        self.backbone = AutoModel.from_pretrained(pretrained_model)
        hidden_size = int(self.config.hidden_size)
        self.pooling = AttentionPooling(hidden_size, dropout * 0.5)
        self.classifier = nn.Sequential(
            nn.LayerNorm(hidden_size * 2),
            nn.Dropout(dropout),
            nn.Linear(hidden_size * 2, 256),
            nn.GELU(),
            nn.Dropout(dropout * 0.75),
            nn.Linear(256, num_labels),
        )

    def forward(
        self,
        input_values: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        outputs = self.backbone(input_values, attention_mask=attention_mask)
        hidden_states = outputs.last_hidden_state

        frame_mask = None
        if attention_mask is not None:
            frame_mask = self.backbone._get_feature_vector_attention_mask(
                hidden_states.shape[1],
                attention_mask,
            )

        # AttentionPooling sudah mengembalikan fitur 1536-dim (attentive
        # stats: weighted mean + weighted std). Jangan concat mean_pooled
        # tambahan, itu akan mengubah dimensi classifier dan menyimpang
        # dari kontrak checkpoint v4.
        pooled_features, _attn_weights = self.pooling(hidden_states, frame_mask)
        return self.classifier(pooled_features)


def _strip_module_prefix(state_dict: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    if not any(key.startswith("module.") for key in state_dict):
        return state_dict
    return {key.removeprefix("module."): value for key, value in state_dict.items()}


def _extract_state_dict(checkpoint: object) -> dict[str, torch.Tensor]:
    if isinstance(checkpoint, dict):
        for key in ("model_state_dict", "model_state", "state_dict"):
            if key in checkpoint and isinstance(checkpoint[key], dict):
                return checkpoint[key]
        tensor_items = {
            key: value
            for key, value in checkpoint.items()
            if isinstance(value, torch.Tensor)
        }
        if tensor_items:
            return tensor_items
        raise ValueError(
            "Checkpoint dictionary tidak berisi state_dict yang dikenali. "
            "Key yang diharapkan: 'model_state_dict', 'model_state', atau 'state_dict'."
        )
    if isinstance(checkpoint, nn.Module):
        return checkpoint.state_dict()
    raise ValueError(
        f"Format checkpoint tidak didukung: {type(checkpoint).__name__}. "
        "Harap gunakan state_dict langsung atau dictionary berisi key model."
    )


def ensure_model_downloaded(model_path: Path | str | None = None) -> Path:
    """Unduh checkpoint v4 dari Hugging Face Hub jika belum ada secara lokal."""
    path = Path(model_path) if model_path is not None else MODEL_PATH
    path.parent.mkdir(parents=True, exist_ok=True)

    if path.exists():
        return path

    print(f"Mengunduh {MODEL_FILENAME} dari Hugging Face: {MODEL_REPO_ID}")
    try:
        downloaded = hf_hub_download(
            repo_id=MODEL_REPO_ID,
            filename=MODEL_FILENAME,
            local_dir=str(path.parent),
            local_dir_use_symlinks=False,
        )
    except TypeError:
        downloaded = hf_hub_download(
            repo_id=MODEL_REPO_ID,
            filename=MODEL_FILENAME,
            local_dir=str(path.parent),
        )
    except Exception as exc:
        raise RuntimeError(
            f"Gagal mengunduh checkpoint {MODEL_FILENAME} dari Hugging Face "
            f"({MODEL_REPO_ID}). Pastikan repo publik atau token HF tersedia."
        ) from exc

    downloaded_path = Path(downloaded)
    if downloaded_path != path and downloaded_path.exists():
        downloaded_path.replace(path)
    if not path.exists():
        raise RuntimeError(f"Unduhan model gagal. File tidak ditemukan: {path}")

    return path


def load_model(
    model_path: str | Path | None = None,
    device: torch.device | str = "cpu",
    dropout: float = DEFAULT_DROPOUT,
) -> WavLMSERModel:
    """Muat model WavLM SER (auto-download checkpoint jika perlu).

    Pemanggilan yang didukung:
    - load_model()
    - load_model(device="cuda")
    - load_model(model_path, device)
    - load_model(model_path=model_path, device=device)
    """
    device = torch.device(device)
    resolved_path = ensure_model_downloaded(model_path)

    try:
        model = WavLMSERModel(dropout=dropout)
    except Exception as exc:
        raise RuntimeError(
            f"Gagal memuat backbone HuggingFace '{PRETRAINED_MODEL}'. "
            "Pastikan koneksi internet tersedia untuk unduhan pertama kali."
        ) from exc

    try:
        checkpoint = torch.load(resolved_path, map_location=device, weights_only=False)
    except FileNotFoundError as exc:
        raise FileNotFoundError(f"File model tidak ditemukan: {resolved_path}") from exc
    except Exception as exc:
        raise RuntimeError(
            f"Gagal membaca checkpoint: {resolved_path}. Error: {exc}"
        ) from exc

    state_dict = _strip_module_prefix(_extract_state_dict(checkpoint))

    try:
        model.load_state_dict(state_dict, strict=True)
    except RuntimeError as exc:
        model_keys = set(model.state_dict().keys())
        ckpt_keys = set(state_dict.keys())
        missing = sorted(model_keys - ckpt_keys)
        unexpected = sorted(ckpt_keys - model_keys)
        raise RuntimeError(
            "Checkpoint tidak cocok dengan arsitektur model.\n"
            f"- Key hilang ({len(missing)}): {missing[:5]}{'...' if len(missing) > 5 else ''}\n"
            f"- Key tidak terduga ({len(unexpected)}): {unexpected[:5]}{'...' if len(unexpected) > 5 else ''}\n"
            f"Detail: {exc}"
        ) from exc

    model.to(device)
    model.eval()
    return model
