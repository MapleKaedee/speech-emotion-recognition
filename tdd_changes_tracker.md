# TDD Changes Tracker

## [2026-08-24] Preload Whisper Small untuk Analisis Transkrip dan Segmen

### Rincian Perubahan

1. Whisper dikunci ke `openai/whisper-small` pada lokal maupun cloud agar kualitas transkripsi konsisten.
2. Pipeline Whisper dipreload saat startup aplikasi dan tetap disimpan dengan `st.cache_resource`.
3. Audio STT tetap memakai waveform penuh, mono, dan resample ke 16 kHz. Audio tidak dipotong 4 detik karena batas 4 detik hanya berlaku untuk inferensi SER.
4. Analisis per segmen tetap menggunakan timestamp Whisper, melewati segmen di bawah 0.35 detik, dan membatasi maksimal 20 segmen.

### Dampak dan Batasan

- Cold start berpindah ke startup deployment. User tidak lagi memicu download Whisper saat pertama kali meminta analisis segmen.
- Startup memerlukan waktu dan RAM lebih besar karena Whisper small sekitar 967 MB.
- Jika preload gagal karena koneksi atau resource, aplikasi tetap berjalan dan akan mencoba memuat ulang saat fitur STT digunakan.

## [2026-08-24] Sinkronisasi Streamlit dengan Model Hugging Face v4

### File Terdampak

- [model.py](model.py)
- [utils.py](utils.py)
- [services.py](services.py)
- [config.py](config.py)
- [requirements.txt](requirements.txt)
- [walkthrough.md](walkthrough.md)

### Rincian Perubahan

1. Streamlit sekarang mengunduh `ser_wavlm_v4_best.pt` dari repo Hugging Face `elnathzzz/wavlm-ser-multilingual` menggunakan `huggingface_hub`.
2. Arsitektur inferensi tetap memakai WavLM base-plus, attentive statistics pooling, dan classifier v4 yang kompatibel dengan checkpoint.
3. Preprocessing inferensi disamakan dengan `pipeline/ver4-ser-pipeline.ipynb`: mono, resample 16 kHz, sanitasi NaN, trim silence 30 dB, peak normalization, crop dari awal ke 4 detik, lalu zero-padding.
4. Namespace metrik Streamlit diarahkan ke artefak v4. Jika artefak v4 belum tersedia di `models/`, panel metrik akan tetap kosong tanpa menghambat inferensi.
5. Ketergantungan Google Drive dan `gdown` dihapus dari alur pemuatan checkpoint.
6. Backbone Streamlit memakai `AutoConfig` dan `AutoModel` seperti pipeline v4, dengan dropout pooling `DROPOUT * 0.5`.
7. Resampling inferensi memakai `librosa.resample` agar mengikuti implementasi `librosa.load(..., sr=16000)` pada pipeline v4.

### Dampak dan Batasan

- Unduhan pertama memerlukan akses ke Hugging Face Hub. Repo private memerlukan token Hugging Face.
- Checkpoint v4 harus cocok dengan arsitektur attentive statistics pooling. `load_state_dict(strict=True)` tetap digunakan untuk mendeteksi mismatch.
- Perubahan crop dari center ke awal dan resampling dengan librosa mengubah input inferensi agar konsisten dengan pipeline v4.

## [2026-08-24] Revisi Pipeline v5 Menjadi Standalone dari WavLM Base

### File Terdampak

- [pipeline/ver5-ser-pipeline.ipynb](pipeline/ver5-ser-pipeline.ipynb)
- [tdd_changes_tracker.md](tdd_changes_tracker.md)

### Rincian Perubahan Skenario dan Logic

1. Notebook `pipeline/ver5-ser-pipeline.ipynb` dipertahankan sebagai pipeline v5, tetapi seluruh ketergantungan pada checkpoint v4 dan pipeline sebelumnya dihapus.
2. Model v5 selalu diinisialisasi dari `microsoft/wavlm-base-plus`, sehingga notebook dapat dipresentasikan dan dijalankan dari awal tanpa input model eksperimen sebelumnya.
3. Resep v4 dipertahankan sebagai referensi: EDA, split, augmentasi Indonesia, arsitektur WavLM base-plus, focal loss, dropout 0.30, gradual unfreezing, evaluasi, dan export package.
4. Stage 2 v5 sekarang berjalan walaupun deteksi noise menghasilkan nol kandidat. Jika tidak ada kandidat, tidak ada sampel yang dihapus dan seluruh data train tetap digunakan untuk fine-tuning.
5. History Stage 1 dan Stage 2 tetap disimpan terpisah di `history_v5.json`. Kurva training Stage 1 dan Stage 2 juga diekspor sebagai file berbeda.
6. Metadata package v5 mencatat mode `standalone_from_microsoft/wavlm-base-plus`, tidak memiliki `source_checkpoint`, dan mencatat lineage `WavLM base-plus -> v5 Stage 1 -> v5 Stage 2`.
7. Output eksekusi lama pada notebook dibersihkan. Notebook harus dijalankan ulang dari awal di Kaggle sebelum metrik v5 digunakan.

### Dampak dan Batasan

- Model v4 tetap menjadi champion utama untuk presentasi berdasarkan hasil evaluasi yang sudah tersedia, yaitu test accuracy 85.52%.
- Perubahan ini menjaga kesamaan struktur dengan v4, tetapi angka v5 tidak dijamin sama karena v5 tidak memakai checkpoint v4.
- Training Kaggle belum dijalankan setelah revisi standalone. Package v5 yang lama tidak lagi merepresentasikan source code terbaru sampai notebook dijalankan ulang.

## [2026-08-24] Promosi Checkpoint Stage 2 v3 menjadi Input Pipeline v4

### File Terdampak

- [pipeline/ser-pipeline.ipynb](pipeline/ser-pipeline.ipynb)
- [pipeline/ser_wavlm_v3_best.pt](pipeline/ser_wavlm_v3_best.pt)
- [tdd_changes_tracker.md](tdd_changes_tracker.md)

### Rincian Perubahan Skenario dan Logic

1. Checkpoint lokal hasil Stage 2 yang sebelumnya bernama `ser_wavlm_v7_stage2_best.pt` diganti menjadi `ser_wavlm_v3_best.pt`. Isi tensor dan metadata checkpoint tidak diubah.
2. `PRETRAINED_CHECKPOINT_CANDIDATES` sekarang mencari checkpoint v3 pada path Kaggle Input dan path lokal `pipeline/ser_wavlm_v3_best.pt`.
3. Fallback otomatis ke checkpoint output saat ini dihapus. Jika checkpoint v3 tidak ditemukan, pipeline v4 melatih dari `microsoft/wavlm-base-plus` dan mencetak status tersebut.
4. Namespace output modeling diganti dari v3 menjadi v4, mencakup checkpoint Stage 1, checkpoint Stage 2, log training, split metadata, evaluasi test, heatmap, konfigurasi, dan package.
5. Lineage eksperimen dicatat sebagai `v3 Stage 2 input -> v4 Stage 1 -> v4 Stage 2`.

### Dampak dan Batasan

- Pipeline v4 adalah continuation fine-tuning dari checkpoint Stage 2 v3. Ini bukan eksperimen independen dari bobot WavLM base.
- Path lokal Windows tidak tersedia otomatis di Kaggle. Checkpoint v3 harus diunggah atau dilampirkan sebagai Kaggle Input agar kandidat path Kaggle dapat menemukannya.
- Training Kaggle belum dijalankan setelah perubahan. Angka output yang tersimpan di notebook masih merupakan hasil run v3 sebelumnya dan tidak boleh disebut sebagai hasil v4.

## [2026-08-10] Penyelarasan Arsitektur Model & Preprocessing dengan Notebook Training (`ser-augmemted.ipynb`)

### File Terdampak

- [model.py](file:///c:/Kuliah/Semester%20Antara/PM/EMOTION-DETECTION/ser-streamlit-app/model.py)
- [utils.py](file:///c:/Kuliah/Semester%20Antara/PM/EMOTION-DETECTION/ser-streamlit-app/utils.py)
- [pipeline/ser-augmemted.ipynb](file:///c:/Kuliah/Semester%20Antara/PM/EMOTION-DETECTION/ser-streamlit-app/pipeline/ser-augmemted.ipynb)

### Rincian Perubahan Skenario & Logic:

1. **Penyelarasan Classifier Non-Linearity (`model.py`)**:
   - **Sebelum:** Menggunakan `nn.ReLU()` pada dua layer classifier.
   - **Sesudah:** Diubah menjadi `nn.GELU()` dan disesuaikan posisi `nn.Dropout(dropout * 0.75)` agar **100% identik** dengan kelas `WavLMSER` pada notebook training v7 (`ser-augmemted.ipynb`).
   - **Alasan:** Perbedaan fungsi aktivasi antara training vs inferensi dapat menyebabkan penurunan performa prediksi saat memuat `ser_wavlm_v7_best.pt`.

2. **Koreksi Durasi Maksimum Audio (`utils.py`)**:
   - **Sebelum:** `MAX_DURATION_SECONDS = 8.0`
   - **Sesudah:** `MAX_DURATION_SECONDS = 4.0`
   - **Alasan:** Pada notebook `ser-augmemted.ipynb` Cell 4, `MAX_SECONDS` dikonfigurasi kaku pada **4.0 detik** (`MAX_SAMPLES = 64,000`). Menyesuaikan durasi inferensi ke 4.0s menjamin bentuk tensor input identik dengan kondisi saat model v7 dilatih.

## [2026-08-10] Verifikasi Path Dataset & Early Exit pada Notebook Training (`ser-augmemted.ipynb`)

### File Terdampak

- [ser-augmemted.ipynb](file:///c:/Kuliah/Semester%20Antara/PM/EMOTION-DETECTION/ser-streamlit-app/pipeline/ser-augmemted.ipynb)

### Rincian Perubahan Skenario & Logic:

1. **Verifikasi Jalur Multi-Kandidat & Early Exit (Cell 2 Konfigurasi Global)**:
   - **Sebelum:** Path dataset `KAGGLE_INDOWAVE`, `KAGGLE_RAVDESS`, `KAGGLE_EMODB`, `KAGGLE_CREMAD` di-hardcode ke `/kaggle/input/...` tanpa dilakukan pengecekan keberadaannya di awal. Jika dijalankan di lingkungan lokal/Colab, kode baru akan mendeteksi kegagalan di cell 3 setelah sempat memproses fungsi-fungsi lain.
   - **Sesudah:** Ditambahkan struktur `DATASET_CANDIDATES` untuk setiap dataset yang mencakup path Kaggle, Colab, dan lokal (`RAW_DIR` atau `./pipeline/Dataset`). Setiap dataset diperiksa dengan `os.path.isdir()`, mencetak status terdeteksi (dengan ikon status), dan jika tidak ada 1 pun dataset yang ditemukan (`found_count == 0`), eksekusi cell langsung dihentikan menggunakan `sys.exit()`.
   - **Alasan:** Mencegah notebook berjalan sia-sia sampai pertengahan cell jika dataset belum dimuat/di-mount di sistem.

## [2026-08-10] Pembersihan Emoji Dekoratif & Standarisasi Log Profesional (`ser-augmemted.ipynb`)

### File Terdampak

- [ser-augmemted.ipynb](file:///c:/Kuliah/Semester%20Antara/PM/EMOTION-DETECTION/ser-streamlit-app/pipeline/ser-augmemted.ipynb)

### Rincian Perubahan Skenario & Logic:

1. **Pembersihan Emoji Dekoratif & Simbol Non-Standard**:
   - **Sebelum:** Penggunaan emoji visual secara masif pada judul markdown, header, dan perintah print (`🔍`, `✅`, `❌`, `🛑`, `📌`, `🖥️`, `📂`, `📊`, `🔄`, `🔧`, `⏹`, `📉`, `🔁`, `🟡`, `📥`).
   - **Sesudah:** Mengubah seluruh output log dan header markdown menjadi string teks terstruktur standar berbasis tag profesional seperti `[OK]`, `[NOT FOUND]`, `[ERROR]`, `[CHECK]`, `[NOTE]`, `[STATS]`, `[CONFIG]`, `[BEST]`.
   - **Alasan:** Memenuhi standar Nol AI Slop proyek, meningkatkan keterbacaan log di terminal tanpa ketergantungan render font emoji, dan mencegah masalah encoding cp1252 pada lingkungan Windows.

## [2026-08-10] Pembentukan Custom Skill ML & Matriks Alur Kerja Wajib (`AGENTS.md`)

### File Terdampak

- [AGENTS.md](file:///c:/Kuliah/Semester%20Antara/PM/EMOTION-DETECTION/ser-streamlit-app/AGENTS.md)
- [.agents/skills/audio-pipeline-engineering/SKILL.md](file:///c:/Kuliah/Semester%20Antara/PM/EMOTION-DETECTION/ser-streamlit-app/.agents/skills/audio-pipeline-engineering/SKILL.md)
- [.agents/skills/pytorch-architecture-standards/SKILL.md](file:///c:/Kuliah/Semester%20Antara/PM/EMOTION-DETECTION/ser-streamlit-app/.agents/skills/pytorch-architecture-standards/SKILL.md)
- [.agents/skills/zero-ai-slop/SKILL.md](file:///c:/Kuliah/Semester%20Antara/PM/EMOTION-DETECTION/ser-streamlit-app/.agents/skills/zero-ai-slop/SKILL.md)

### Rincian Perubahan Skenario & Logic:

1. **Pembuatan 3 Custom Skill ML & Governance**:
   - `audio-pipeline-engineering`: Mengunci parameter pemrosesan sinyal suara 16 kHz mono, trim silence 30 dB, peak normalization, 4.0s pad/crop (64,000 sampel), dan SpecAugment.
   - `pytorch-architecture-standards`: Mengunci standar arsitektur PyTorch WavLM base-plus, Attentive Stats Pooling (output 1536), classifier LayerNorm -> Dropout -> Linear -> GELU -> Dropout -> Linear, dan gradual unfreezing 6 layer.
   - `zero-ai-slop`: Mengunci aturan larangan karakter em-dash, larangan emoji dekoratif, larangan basa-basi AI, dan mempertahankan penamaan emosi Bahasa Indonesia.
2. **Pembentukan Matriks Alur Kerja Skill 5-Fase pada `AGENTS.md`**:
   - Menjadikan eksekusi skill wajib berurutan (Fase 1 Hygiene -> Fase 2 Ideation & Planning -> Fase 3 Implementation -> Fase 4 Debugging -> Fase 5 Verification & Progress Tracking).

## [2026-08-12] Fitur Rekam Mikrofon Langsung (Live Record) via `st.audio_input`

### File Terdampak

- [app.py](file:///D:/Code/speech-emotion-recognition/app.py)
- [components/ui.py](file:///D:/Code/speech-emotion-recognition/components/ui.py)
- [requirements.txt](file:///D:/Code/speech-emotion-recognition/requirements.txt)

### Rincian Perubahan Skenario & Logic:

1. **Sumber Audio Ganda (`app.py`)**:
   - **Sebelum:** Satu-satunya sumber input adalah `st.file_uploader` (.wav/.mp3).
   - **Sesudah:** Ditambahkan `st.radio` pemilih sumber ("Unggah File" / "Rekam Mikrofon"). Mode rekam memakai `st.audio_input` native Streamlit (bukan dependency pihak ketiga seperti `streamlit-webrtc`), karena hasilnya berupa `UploadedFile` WAV yang langsung kompatibel dengan `load_audio()` di `utils.py` tanpa perubahan pipeline.
   - **Alasan:** Kebutuhan "rekam lalu analisis" (bukan streaming real-time), sehingga widget native cukup dan tidak menambah kompleksitas WebRTC/buffering.

2. **Cache Key Berbasis Hash, Bukan Nama/Ukuran File (`app.py`)**:
   - **Sebelum:** `_upload_key()` memakai `(file.name, file.size)`. Dua rekaman mikrofon berbeda berpotensi memiliki nama dan ukuran identik, sehingga cache prediksi bisa salah tidak ter-invalidasi.
   - **Sesudah:** `_audio_key()` memakai `(source, sha256(file.getvalue()))`. Hash byte menjamin setiap audio unik terdeteksi sebagai input baru, terlepas dari nama file.

3. **Unifikasi Variabel Audio (`app.py`)**:
   - Variabel `uploaded_file` diganti menjadi `audio_file` di seluruh alur (metadata, preview, prediksi SER, transkrip Whisper) agar satu jalur kode menangani kedua sumber tanpa duplikasi logic.

4. **Copy UI Netral Sumber (`components/ui.py`)**:
   - `render_hero()` dan `render_empty_state()` diperbarui agar tidak hanya menyebut "unggah", tetapi juga opsi rekam mikrofon.

5. **Kenaikan Minimum Versi Streamlit (`requirements.txt`)**:
   - **Sebelum:** `streamlit>=1.28.0`.
   - **Sesudah:** `streamlit>=1.41.0`.
   - **Alasan:** `st.audio_input` baru stabil dan general availability pada Streamlit v1.40.0, dengan perbaikan bug penamaan file unik pada v1.41.0. Versi lama tidak menjamin API ini tersedia.

### Catatan Non-Fungsional

- Tidak ada perubahan pada `model.py`, arsitektur WavLM, atau `MAX_DURATION_SECONDS` (tetap 4.0 detik). Preprocessing v7 tidak tersentuh.
- Perekaman mikrofon browser memerlukan konteks aman (`localhost` atau HTTPS) dan izin mikrofon dari pengguna.

## [2026-08-12] Perbaikan Bug Kritis: Crash Inferensi & Penyimpangan Arsitektur dari Checkpoint v7

### Konteks

Ditemukan saat pengguna mencoba menganalisis hasil rekam mikrofon dan menerima pesan "File audio tidak dapat diproses". Investigasi menunjukkan bug ini juga terjadi pada mode unggah file, jadi bukan regresi dari fitur live record, melainkan bug pre-existing di `utils.py` dan `model.py` yang baru terpicu.

### File Terdampak

- [model.py](file:///D:/Code/speech-emotion-recognition/model.py)
- [utils.py](file:///D:/Code/speech-emotion-recognition/utils.py)
- [app.py](file:///D:/Code/speech-emotion-recognition/app.py)

### Rincian Perubahan Skenario & Logic:

1. **Bug Crash: `predict_emotion()` Tanpa `return_tensors="pt"` (`utils.py`)**:
   - **Sebelum:** `processor(waveform.numpy(), sampling_rate=TARGET_SAMPLE_RATE)` tanpa `return_tensors="pt"`. `Wav2Vec2FeatureExtractor` mengembalikan `BatchFeature` berisi list numpy, bukan tensor PyTorch. Pemanggilan `inputs["input_values"].to(device)` melempar `AttributeError: 'list' object has no attribute 'to'`.
   - **Sesudah:** Argumen dilengkapi persis sesuai pemanggilan `feature_extractor` pada `ser-augmemted.ipynb` Cell 15 (fungsi inferensi): `processor([waveform.numpy()], sampling_rate=TARGET_SAMPLE_RATE, padding=True, truncation=True, max_length=MAX_SAMPLES, return_attention_mask=True, return_tensors="pt")`.
   - **Alasan:** `except Exception` generik di `app.py` membungkam `AttributeError` ini menjadi pesan "File audio tidak dapat diproses", menyesatkan pengguna karena file audio sebenarnya valid (metadata dan pratinjau tampil normal).

2. **Bug Arsitektur: `AttentionPooling` Tidak Sesuai Checkpoint v7 (`model.py`)**:
   - **Sebelum:** `AttentionPooling.forward()` mengembalikan tensor tunggal (weighted mean saja). `WavLMSERModel.forward()` lalu meng-concat hasil itu dengan `mean_pooled` (rata-rata aritmetika tanpa bobot) menjadi fitur 1536-dim: `[attn_pooled, mean_pooled]`.
   - **Sesudah:** `AttentionPooling.forward()` diubah menjadi attentive statistics pooling identik dengan `AttentiveStatsPooling` di notebook: menghitung `weighted_mean` dan `weighted_std` dari attention weights yang sama, mengembalikan `torch.cat([mean, std], dim=-1)` sebagai tuple `(pooled, attn_weights)`. `WavLMSERModel.forward()` menyerahkan hasil ini langsung ke `classifier` tanpa concat tambahan.
   - **Alasan:** Checkpoint `ser_wavlm_v7_best.pt` dilatih dengan fitur `[weighted_mean, weighted_std]`. Karena nama atribut (`self.pooling`, `self.attn`) kebetulan identik antara implementasi lama dan checkpoint, `load_state_dict(strict=True)` lolos tanpa error meskipun semantik 768 dimensi kedua sama sekali berbeda (`plain_mean` vs `weighted_std`). Ini adalah silent correctness bug: aplikasi berjalan dan mengeluarkan prediksi tanpa exception, tetapi akurasi menyimpang dari `test_acc 0.7746` yang tercatat di `models/config_v7.json` karena separuh fitur classifier menerima distribusi input yang salah.

3. **Penyelarasan `masked_fill` dan `DEFAULT_DROPOUT` (`model.py`)**:
   - **Sebelum:** `masked_fill(attention_mask == 0, float("-inf"))` dan `DEFAULT_DROPOUT = 0.35`.
   - **Sesudah:** `masked_fill(attention_mask == 0, -1e4)` (mencegah `softmax(-inf)` menghasilkan NaN pada edge case seluruh frame termasking) dan `DEFAULT_DROPOUT = 0.30` (identik dengan `models/config_v7.json`).

4. **Preprocessing Audio Menyimpang dari `load_waveform` + `fix_length(mode='eval')` (`utils.py`)**:
   - **Sebelum:** `preprocess_audio()` hanya melakukan mono-mix, resample, dan crop dari awal (`audio[..., :MAX_SAMPLES]`) untuk audio > 4 detik. Audio < 4 detik tidak di-pad sama sekali, dibiarkan lebih pendek dari `MAX_SAMPLES`.
   - **Sesudah:** Ditambahkan fungsi `_trim_silence()`, `_peak_normalize()`, `_fix_length_eval()` yang mereplikasi urutan wajib notebook v7: sanitasi NaN (`np.nan_to_num`) -> silence trim 30dB dengan guard `len(trimmed) >= MIN_SAMPLES` (5600 sampel) -> peak normalization ke `[-1.0, 1.0]` -> center crop (bukan crop dari awal) untuk audio > 4 detik, atau right zero-pad ke tepat 64000 sampel untuk audio < 4 detik.
   - **Alasan:** Rekaman mikrofon umumnya berdurasi di bawah 4 detik, sehingga sebelumnya selalu masuk ke jalur tensor berukuran variabel (tidak pernah tepat 64000 sampel) yang tidak pernah dilihat model saat training. Constants baru: `MIN_DURATION_SECONDS = 0.35`, `MIN_SAMPLES = 5600`, `SILENCE_TRIM_TOP_DB = 30`.

5. **Traceback Exception Ditampilkan ke Pengguna (`app.py`)**:
   - **Sebelum:** `except Exception:` generik menampilkan pesan tetap "File audio tidak dapat diproses" tanpa detail, menyembunyikan root cause.
   - **Sesudah:** Pesan error menyertakan `type(exc).__name__` dan `str(exc)`, ditambah `st.expander` berisi `traceback.format_exc()` untuk debugging pada blok prediksi (`app.py` fungsi `main()`, penanganan hasil `run_prediction`).

### Verifikasi

- `python -m py_compile app.py components\ui.py components\sidebar.py components\css.py services.py config.py utils.py model.py` -> berhasil, tanpa error sintaks.
- Verifikasi fungsional dijalankan di virtual environment sementara (`.venv_test`, dihapus setelah pengujian, tidak tercatat di git):
  - Unit test preprocessing murni (`_trim_silence`, `_peak_normalize`, `_fix_length_eval`, sanitasi NaN) untuk kasus audio pendek, panjang, hening total: PASS, semua output tepat 64000 sampel, tanpa NaN.
  - Unit test `AttentionPooling` dan `WavLMSERModel` forward pass dengan backbone `microsoft/wavlm-base-plus` asli: shape pooled `(1, 1536)`, shape logits `(1, 6)`, tanpa NaN, termasuk edge case partial mask.
  - Test `predict_emotion()` end-to-end dengan `AutoFeatureExtractor` asli: tidak lagi melempar `AttributeError`, probabilitas softmax menjumlah ke 1.0.
  - Test pipeline penuh `preprocess_audio() -> predict_emotion()` untuk simulasi rekaman mikrofon pendek (1.5 detik, 48kHz) dan unggah panjang (6 detik, 16kHz): keduanya PASS tanpa error.

### Dampak yang Perlu Diketahui

- Hasil prediksi akan **berubah** dibanding perilaku sebelumnya untuk audio yang sama, karena perbaikan pooling mengubah fitur yang masuk ke classifier secara signifikan (perbaikan ini dimaksudkan, karena perilaku lama salah secara matematis terhadap checkpoint).
- Belum ada verifikasi akurasi terhadap `models/evaluasi_test_v7.csv` untuk mengonfirmasi angka `test_acc 0.7746` tercapai kembali. Perlu pengujian batch terpisah sebelum mengklaim paritas penuh dengan hasil training.

## [2026-08-19] Integrasi Checkpoint WavLM v8 (Dinamakan v7_best.pt) ke Repositori Lokal
- **Tanggal**: 19 Agustus 2026
- **Konteks**: Model baru hasil eksperimen notebook pipeline/ver2-ser-pipeline.ipynb (Focal Loss gamma=2.0, CREMA-D weight normalization, Hard drop noise filtering) diintegrasikan ke repositori utama.
- **Rincian Perubahan**:
  - Mengganti seluruh isi direktori models/ dengan *package* rilis terbaru dari pipeline/ser_wavlm_v7_package/ (termasuk ser_wavlm_v7_best.pt, config_v7.json, dan file log evaluasi).
  - Melakukan *load test* inferensi via model.py untuk mengonfirmasi kompabilitas penuh arsitektur baru.
  - Performa terukur pada data uji: **80.90% Accuracy** (	est_acc: 0.8090).
  - Tidak ada perubahan struktural kode PyTorch. model.py sudah selaras sepenuhnya dengan desain evaluasi (AttentiveStatsPooling memproduksi 1536 tensor dimensi gabungan).
### [UPDATE] Modifikasi Training Pipeline v8
- **Target**: Peningkatan metrik untuk kelas minoritas (Sedih, Senang, Takut, Jijik) dan dataset IndoWave.
- **File Diubah**: pipeline/ver2-ser-pipeline.ipynb
- **Perubahan**:
  1. **Noise Filter Eksklusi**: Menambahkan proteksi pada Stage 2 Hard Drop agar kelas rentan (sedih, senang, takut, jijik) TIDAK dibuang meskipun model tidak yakin, mencegah recall anjlok.
  2. **Augmentasi Spesifik Kelas**: Menambahkan augmentasi offline (Pitch Shift & Time Stretch) spesifik hanya untuk kelas sedih, senang, takut, dan jijik dari dataset CREMA-D.
  3. **Oversampling IndoWaveSentiment**: Melakukan duplikasi dataset IndoWave sebanyak 4x sebelum split untuk memaksa model lebih banyak melihat data suara bahasa Indonesia selama training.

## [2026-08-22] Perbaikan Validitas Eksperimen dan Target Metrik di pipeline/ver3-ser-pipeline.ipynb

### Konteks

Audit penuh terhadap pipeline/ver2-ser-pipeline.ipynb dan pipeline/ver3-ser-pipeline.ipynb menemukan bahwa cell Stage 1 dan Stage 2 sebelumnya hanya menyiapkan checkpoint, dataset, dan optimizer. Loop training tidak dijalankan. Karena itu, metrik yang terlihat tidak dapat dianggap sebagai hasil eksperimen v8 yang baru.

### File Terdampak

- pipeline/ver3-ser-pipeline.ipynb
- tdd_changes_tracker.md
- walkthrough.md

### Perubahan Skenario dan Logic

1. Menambahkan loop training Stage 1 dan fine-tuning Stage 2 yang benar-benar menyimpan checkpoint.
2. Memperbaiki pemisahan learning rate berdasarkan nama backbone, bukan token wavlm yang tidak ada pada nama parameter model.
3. Memperbaiki unfreeze_last_layers() agar mengambil core melalui _get_core().
4. Menonaktifkan Mixup pada baseline karena helper mixup_batch() dan mixup_loss() tidak tersedia. Baseline harus dapat dieksekusi tanpa simbol yang hilang.
5. Memperbaiki perhitungan Focal Loss agar pt dihitung dari probabilitas kelas yang tidak terdistorsi oleh class weight.
6. Menambahkan multiplier loss terkontrol untuk menaikkan peluang recall sedih dan senang, serta mengurangi bias prediksi takut dan jijik.
7. Mempertahankan augmentasi offline hanya untuk sumber Indonesia, yaitu IndoWave dan E-SERAVD. Augmentasi CREMA-D tidak menjadi bagian baseline karena CREMA-D sudah merupakan sumber dominan dan masalah utamanya adalah ambiguitas label, bukan kekurangan jumlah file.
8. Menghapus oversampling IndoWave 4x tambahan setelah augmentasi. Augmentasi lima varian sudah menjadi koreksi distribusi yang cukup untuk baseline.
9. Mengubah hard drop menjadi konservatif. Kandidat harus memiliki confidence minimal 0.95, berasal dari CREMA-D, dan label asalnya hanya boleh netral atau marah. Kelas target recall sedih, senang, takut, dan jijik dilindungi dari hard drop.
10. Menentukan checkpoint berdasarkan gabungan macro-F1, recall sedih, recall senang, precision takut, dan precision jijik, bukan akurasi keseluruhan saja.
11. Menambahkan tabel distribusi augmentasi serta metrik precision, recall, F1 per sumber dan kelas, termasuk pasangan error terbesar.

### Verifikasi

- JSON notebook valid setelah perubahan.
- Semua cell Python yang dapat diparse berhasil melewati ast.parse.
- Cell khusus Kaggle dan Hugging Face ditandai sebagai environment-dependent dan belum dieksekusi lokal.
- Training Kaggle belum dijalankan, sehingga tidak ada klaim kenaikan metrik.

### Keputusan yang Belum Boleh Dianggap Terbukti

- Target precision takut dan jijik di atas 80%.
- Target recall sedih dan senang di atas 80%.
- Keunggulan hard drop dibanding tanpa hard drop.
- Keunggulan Focal Loss dibanding Cross Entropy.

Semua klaim tersebut harus dibuktikan melalui run Kaggle dengan split, seed, checkpoint, dan dataset yang sama.
