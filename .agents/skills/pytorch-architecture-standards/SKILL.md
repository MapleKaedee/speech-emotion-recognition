---
name: pytorch-architecture-standards
description: Architecture standards for PyTorch Speech Emotion Recognition models based on WavLM backbone, Attentive Stats Pooling, LayerNorm, GELU activation, and gradient checkpointing.
---

# PyTorch Architecture Standards Skill

Dokumen ini mendefinisikan aturan arsitektur PyTorch untuk model Speech Emotion Recognition berbasis `microsoft/wavlm-base-plus`.

## Struktur Arsitektur Model (`WavLMSER`)

### 1. Pretrained Backbone
- **Model Source**: Wajib `microsoft/wavlm-base-plus`.
- **Hidden Size**: `768` (dimensi tersembunyi `wavlm-base-plus`).
- **Gradient Checkpointing**: Aktifkan `model.backbone.gradient_checkpointing_enable()` jika `USE_GRAD_CKPT=True` untuk menghemat VRAM GPU.

### 2. Attentive Stats Pooling (`AttentiveStatsPooling`)
- Mengkompresi sekuens fitur $N$ frame menjadi 1 vektor ringkas berukuran $1536$ ($768 \times 2$).
- Menghitung dua komponen terbobot:
  1. **Weighted Mean ($\mu$)**: `torch.sum(weight * x, dim=1)`
  2. **Weighted Standard Deviation ($\sigma$)**: `torch.sqrt(var.clamp(min=1e-6))`
- Hasil penggabungan: `torch.cat([mean, std], dim=-1)` dengan ukuran output 1536.

### 3. Classifier Head & Aktivasi
Struktur sekuensial classifier wajib mematuhi skema berikut:

```python
nn.Sequential(
    nn.LayerNorm(1536),           # Input 1536 (768 * 2) dari AttentiveStatsPooling
    nn.Dropout(0.30),             # Dropout awal 0.30
    nn.Linear(1536, 256),         # Proyeksi linier ke 256
    nn.GELU(),                    # Wajib GELU (bukan ReLU) untuk menyelaraskan v7
    nn.Dropout(0.225),            # Dropout 0.30 * 0.75
    nn.Linear(256, 6)             # Proyeksi ke 6 kelas emosi
)
```

### 4. Skema Unfreezing Gradual
- **Freeze Initial Epochs**: Epoch 1-2 bekukan seluruh backbone WavLM (`freeze_all_backbone`), hanya latih pooling dan classifier head.
- **Gradual Unfreeze**: Epoch 3+ buka 6 layer teratas backbone WavLM (`unfreeze_last_layers(model, last_n=6)`). Dilarang melakukan full unfreeze 12/24 layer untuk mencegah *catastrophic forgetting*.

### 5. Pengambilan Checkpoint & Inisialisasi State Dict
- Ekstraksi `state_dict` pada `model.py` wajib mendukung pemuatan dari `torch.load()` checkpoint Stage 1 (`ser_wavlm_v7_best.pt`) dan Stage 2 (`ser_wavlm_v7_stage2_best.pt`).
- Penamaan kunci layer classifier harus sinkron antara notebook `ser-augmemted.ipynb` dan `model.py`.

## Verifikasi Kepatuhan
Jalankan verifikasi sintaks dan struktur tensor:
```bash
python -c "import torch; from model import WavLMSERModel; m = WavLMSERModel(); x = torch.randn(2, 64000); out, _ = m(x); print(out.shape)"
# Output wajib: torch.Size([2, 6])
```
