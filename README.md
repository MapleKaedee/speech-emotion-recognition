# Speech Emotion Recognition

## Identitas Proyek

Proyek ini dibuat oleh:

1. Farros Rifantiarno Ramadhani, `A11.2024.15694`
2. Devin Abiyyu Pandu Pratama, `A11.2024.15980`
3. Nuur Wahid Saifullah, `A11.2024.15974`

Link aplikasi: [emotionrecognitionspeech.streamlit.app](https://emotionrecognitionspeech.streamlit.app/)

## Deskripsi

Aplikasi web Speech Emotion Recognition (SER) untuk mendeteksi emosi dari audio ucapan. Sistem menggunakan model WavLM base-plus yang telah dilatih pada beberapa dataset emosi, kemudian menggabungkannya dengan Whisper untuk transkripsi suara dan analisis emosi per segmen.

Label emosi yang digunakan: `netral`, `senang`, `sedih`, `marah`, `takut`, dan `jijik`.

## Fitur Utama

- Mengunggah file audio `.wav` atau `.mp3`.
- Merekam audio langsung dari mikrofon.
- Mendeteksi satu emosi utama dari audio.
- Menampilkan probabilitas keenam kelas emosi.
- Menampilkan top 3 emosi dan confidence score.
- Menampilkan waveform dan metadata audio.
- Mentranskripsikan audio penuh menggunakan `openai/whisper-small`.
- Menganalisis emosi berdasarkan timestamp dari transkripsi Whisper.
- Menampilkan dashboard dataset dan evaluasi model jika artefaknya tersedia.
- Mencache model dengan `st.cache_resource` agar tidak dimuat ulang pada setiap interaksi.

## Model dan Hugging Face

Model SER utama adalah checkpoint v4:

```text
Checkpoint lokal:
models/ser_wavlm_v4_best.pt

Repository Hugging Face:
elnathzzz/wavlm-ser-multilingual

File checkpoint Hugging Face:
ser_wavlm_v4_best.pt

Backbone:
microsoft/wavlm-base-plus
```

Aplikasi menggunakan strategi local-first:

1. Aplikasi mencari `models/ser_wavlm_v4_best.pt` terlebih dahulu.
2. Jika file lokal tidak tersedia, aplikasi mengunduh `ser_wavlm_v4_best.pt` dari repository Hugging Face.
3. File yang diunduh disimpan di folder `models/` dan digunakan kembali oleh proses berikutnya.

Checkpoint dimuat menggunakan `strict=True` agar ketidakcocokan antara tensor checkpoint dan arsitektur model langsung terdeteksi.

## Arsitektur Model SER

1. WavLM base-plus.
2. Attentive Statistics Pooling dengan weighted mean dan weighted standard deviation.
3. `LayerNorm`.
4. Dropout `0.30`.
5. Linear layer berukuran 256.
6. Aktivasi `GELU`.
7. Dropout `0.225`.
8. Linear layer menuju enam kelas emosi.

Implementasi Streamlit diselaraskan dengan `pipeline/ver4-ser-pipeline.ipynb` pada backbone, pooling, classifier, feature extractor, attention mask, dan kontrak preprocessing.

## Kontrak Preprocessing SER

Preprocessing aplikasi disamakan dengan preprocessing inferensi pada pipeline v4.

Urutan proses:

1. Audio diubah menjadi mono.
2. Audio di-resample ke `16.000 Hz` menggunakan `librosa`.
3. Nilai NaN dan infinity disanitasi menjadi nol.
4. Silence trimming dilakukan dengan batas `30 dB`.
5. Peak normalization dilakukan jika sinyal tidak hening.
6. Audio lebih panjang dari 4 detik dipotong dari awal.
7. Audio lebih pendek dari 4 detik diberi zero-padding di sebelah kanan.
8. Hasil akhir selalu berukuran `64.000` sampel.

Kontrak 4 detik hanya berlaku untuk model SER. Audio untuk Whisper tidak dipotong menjadi 4 detik.

## Transkripsi dan Analisis Segmen Whisper

Whisper memakai model `openai/whisper-small` dan dikonfigurasi untuk Bahasa Indonesia.

Proses STT:

1. Audio didekode dan dijadikan mono.
2. Audio di-resample ke `16.000 Hz`.
3. Audio penuh dikirim ke pipeline automatic speech recognition.
4. Bahasa transkripsi ditetapkan ke Bahasa Indonesia.
5. Mode per segmen meminta timestamp dari Whisper.
6. Segmen dengan durasi kurang dari `0,35` detik dilewati.
7. Jumlah segmen dibatasi maksimal 20 untuk menjaga waktu dan resource inferensi.

Whisper dipreload ketika aplikasi mulai dan dicache dengan `st.cache_resource`. Akibatnya, penantian download model terjadi saat startup deployment, bukan ketika user pertama kali meminta analisis per segmen.

## Pipeline Training v4

Notebook utama training:

```text
pipeline/ver4-ser-pipeline.ipynb
```

Versi yang dapat dijalankan di Kaggle:

[Buka notebook SER Pipeline di Kaggle](https://www.kaggle.com/code/elnathh/ser-pipeline)

Panduan penjelasan per cell:

```text
pipeline/penjelasan_notebook_v4.md
```

Pipeline v4 mencakup:

- EDA dataset dan pemeriksaan kualitas audio.
- Silence trimming dan analisis durasi.
- Split train, validation, dan test.
- Augmentasi offline untuk sumber data yang ditargetkan.
- WavLM base-plus.
- Attentive Statistics Pooling.
- Focal Loss.
- Class-specific loss multiplier.
- Gradual unfreezing backbone.
- Noise detection berbasis confidence.
- Stage 2 fine-tuning dengan filtering noise tepercaya.
- Evaluasi test eksplisit.
- Classification report dan confusion matrix aktual.
- Export checkpoint dan artefak evaluasi.

Augmentasi training tidak dijalankan pada audio inferensi. Augmentasi hanya digunakan untuk memperkaya data latih.

## Kelas Emosi

| ID | Label |
|---:|---|
| 0 | `netral` |
| 1 | `senang` |
| 2 | `sedih` |
| 3 | `marah` |
| 4 | `takut` |
| 5 | `jijik` |

## Struktur Direktori

```text
ser-streamlit-app/
├── app.py
├── config.py
├── model.py
├── services.py
├── utils.py
├── requirements.txt
├── AGENTS.md
├── docs/
│   ├── tdd_changes_tracker.md
│   └── walkthrough.md
├── pages/
│   ├── analisis.py
│   ├── dashboard.py
│   ├── dataset.py
│   ├── home.py
│   └── model.py
├── components/
│   ├── css.py
│   ├── recorder.py
│   └── ui.py
├── models/
│   ├── ser_wavlm_v4_best.pt
│   ├── feature_extractor/
│   ├── config_v4.json
│   ├── evaluasi_test_v4.csv
│   ├── metadata_split_v4.csv
│   ├── confusion_matrix_v4.png
│   └── kurva_training_v4.png
└── pipeline/
    ├── ver4-ser-pipeline.ipynb
    └── penjelasan_notebook_v4.md
```

## Instalasi Lokal

### 1. Clone repository

```bash
git clone https://github.com/Elnathz/speech-emotion-detection.git
cd speech-emotion-detection
```

### 2. Buat virtual environment

Windows PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

Linux atau macOS:

```bash
python -m venv .venv
source .venv/bin/activate
```

### 3. Instal dependency

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### 4. Jalankan Streamlit

```bash
python -m streamlit run app.py
```

Aplikasi lokal tersedia di `http://localhost:8501`.

## Deployment Streamlit

Aplikasi dapat dijalankan melalui Streamlit Community Cloud menggunakan repository GitHub sebagai sumber deployment.

Hal yang perlu diperhatikan:

- Checkpoint SER lokal dipakai terlebih dahulu jika tersedia.
- Jika checkpoint lokal tidak tersedia, aplikasi mengunduhnya dari Hugging Face Hub.
- Whisper small dipreload ketika aplikasi mulai.
- Model yang sudah dimuat dicache dan dipakai bersama oleh user pada instance yang sama.
- Jika Community Cloud melakukan hibernasi atau reboot, proses pemuatan model dapat terjadi kembali.
- Cron job eksternal dapat mengirim request berkala untuk menjaga traffic, tetapi tidak menjamin aplikasi bebas hibernasi atau masalah resource.

## Troubleshooting

### Checkpoint tidak ditemukan

Pastikan file berikut tersedia:

```text
models/ser_wavlm_v4_best.pt
```

Jika file tidak ada, pastikan server dapat mengakses Hugging Face Hub.

### Gagal mengunduh dari Hugging Face

Periksa koneksi internet. Jika repository Hugging Face dibuat private, tambahkan token Hugging Face pada environment deployment.

### Startup sangat lama

Startup dapat memerlukan waktu karena WavLM dan Whisper small harus dimuat. Tunggu proses selesai dan periksa log deployment sebelum melakukan refresh berulang.

### Aplikasi kehabisan RAM

Whisper small dan WavLM membutuhkan resource besar. Reboot aplikasi untuk membersihkan memory sementara. Jika masalah tetap terjadi, gunakan deployment dengan resource lebih besar atau pisahkan layanan STT dari aplikasi SER.

### Analisis per segmen tidak menampilkan hasil

Periksa apakah Whisper menghasilkan timestamp. Segmen yang lebih pendek dari `0,35` detik memang dilewati oleh aplikasi.

## Verifikasi Teknis

```powershell
python -m py_compile app.py model.py utils.py services.py
python -c "import model; import utils; import services; print('[OK] import berhasil')"
```

Checkpoint harus dapat dimuat secara ketat dengan arsitektur model v4. Preprocessing SER harus menghasilkan tensor dengan bentuk `(64000,)`.

## Riwayat Versi Singkat

- v1: baseline WavLM dan pipeline dasar.
- v2: EDA lebih lengkap, focal loss, class-specific weighting, dan noise filtering.
- v3: pengembangan lanjutan dan checkpoint lineage.
- v4: model utama yang dipakai Streamlit, dengan evaluasi final dan preprocessing tervalidasi.
- v5: eksperimen standalone dari WavLM base-plus, bukan model utama aplikasi.

## Referensi Model dan Standar Commit

Model backbone menggunakan `microsoft/wavlm-base-plus` dari Hugging Face. Checkpoint SER hasil fine-tuning tersedia pada repository `elnathzzz/wavlm-ser-multilingual`.

Standar commit repository menggunakan Conventional Commits dengan tipe `feat`, `fix`, `refactor`, `docs`, `style`, `test`, dan `chore`.
