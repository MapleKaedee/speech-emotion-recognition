---
name: audio-pipeline-engineering
description: Rules and mandatory specs for speech audio signal processing including 16 kHz mono sampling, 30 dB silence trimming, peak normalization, 4.0s pad/crop, and SpecAugment.
---

# Audio Pipeline Engineering Skill

Dokumen ini berisi standar wajib untuk pemrosesan sinyal audio dalam sistem Speech Emotion Recognition (SER).

## Standar Pemrosesan Sinyal Audio (Audio Physics Rules)

### 1. Sampling Rate & Channel
- **Sampling Rate**: Wajib `16,000 Hz` (16 kHz).
  - Alasan: Sesuai teorema Nyquist-Shannon ($f_{sample} \ge 2 \times f_{max}$), 16 kHz mencakup frekuensi ucapan manusia hingga 8 kHz yang berisi 100% informasi spektral emosi.
  - Model `microsoft/wavlm-base-plus` dilatih khusus pada audio 16 kHz.
- **Channel**: Wajib `Mono` (1 channel).
  - Jika audio stereo, gabungkan dengan rerata kanal (`y.mean(axis=0)`).

### 2. Silence Trimming & Normalisasi
- **Silence Trimming**: Wajib memotong hening di awal dan akhir sinyal menggunakan `librosa.effects.trim(y, top_db=30)`.
  - Hanya potong jika durasi setelah trim >= 0.35 detik (`MIN_SAMPLES = 5,600`).
- **Peak Normalization**: Wajib menormalisasi amplitudo puncak ke skala `[-1.0, 1.0]`.
  - Formula: `y = y / np.max(np.abs(y))` jika `np.max(np.abs(y)) > 1e-5`.

### 3. Padding & Cropping Durasi Audio
- **Max Duration**: `4.0 detik` (`MAX_SAMPLES = 64,000`).
- **Cropping (Audio > 4.0s)**:
  - Mode Train: Ambil potongan 4.0 detik acak (`np.random.randint(0, len(y) - MAX_SAMPLES + 1)`).
  - Mode Eval/Inferensi: Ambil 4.0 detik tepat di tengah (`(len(y) - MAX_SAMPLES) // 2`).
- **Padding (Audio < 4.0s)**: Pad angka 0 di sebelah kanan hingga mencapai 64,000 sampel.

### 4. SpecAugment (Offline & Online)
- **Time Masking Probability**: `0.30`.
- **Max Time Mask Duration**: Maksimal 8% dari total sampel (`int(0.08 * MAX_SAMPLES) = 5,120` sampel).
- **Pitch Shift Offline**: Gunakan `n_steps = +2` dan `n_steps = -2` saja. Dilarang menggunakan `pitch_up1` karena mengurangi variansi riil data.

## Verifikasi Kepatuhan
Setiap fungsi preprocessing wajib diverifikasi dengan pemeriksaan bentuk tensor `(batch_size, 64000)` dan range data `[-1.0, 1.0]`.
