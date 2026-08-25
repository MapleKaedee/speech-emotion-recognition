# Panduan Mendetail `pipeline/ver4-ser-pipeline.ipynb`

Dokumen ini menjelaskan notebook training Speech Emotion Recognition (SER) berbasis WavLM v4 dengan bahasa yang ditujukan untuk pembaca yang belum terbiasa dengan machine learning.

Notebook yang dibahas:

```text
pipeline/ver4-ser-pipeline.ipynb
```

Label output proyek selalu memakai Bahasa Indonesia: `netral`, `senang`, `sedih`, `marah`, `takut`, dan `jijik`.

## Cara membaca notebook

Notebook terdiri dari dua jenis cell:

- Cell Markdown berisi judul, konteks, dan penjelasan.
- Cell Code berisi instruksi Python yang benar-benar dijalankan.

Jalankan cell dari atas ke bawah. Banyak cell memakai variabel yang dibuat oleh cell sebelumnya. Jika langsung menjalankan cell evaluasi tanpa menjalankan konfigurasi, pemuatan data, dan training, variabel seperti `model`, `test_loader`, `FINAL_MODEL_PATH`, `true_all`, atau `pred_all` belum tentu tersedia.

Notebook v4 memiliki 83 cell, dari cell `000` sampai `082`. Urutan dokumentasi ini mengikuti urutan cell tersebut.

## Gambaran besar pipeline

```text
Dataset audio
    -> parser nama file atau folder
    -> metadata gabungan
    -> EDA dan pemeriksaan kualitas
    -> split train, validation, test
    -> augmentasi offline data train Indonesia
    -> silence trimming, normalisasi, crop atau padding 4 detik
    -> feature extractor WavLM
    -> WavLM base-plus
    -> Attentive Stats Pooling
    -> classifier enam kelas
    -> training Stage 1
    -> deteksi kandidat label noise
    -> fine-tuning Stage 2
    -> evaluasi test
    -> export checkpoint dan package
    -> prediksi audio baru
```

Model menerima waveform mono dengan sample rate 16.000 Hz dan panjang maksimum 4 detik, atau 64.000 sampel. WavLM mengubah waveform menjadi fitur suara, pooling merangkum fitur sepanjang waktu, dan classifier menghasilkan enam logit emosi.

## Status lineage v4

v4 bukan selalu training independen dari nol. Cell konfigurasi mencari checkpoint Stage 2 v3. Jika checkpoint ditemukan, bobot tersebut dipakai sebagai inisialisasi sebelum Stage 1 v4. Jika tidak ditemukan, model memakai bobot pretrained `microsoft/wavlm-base-plus`.

Lineage yang mungkin terjadi:

```text
Checkpoint v3 Stage 2 -> Stage 1 v4 -> Stage 2 v4 -> checkpoint final v4
```

atau:

```text
WavLM base-plus pretrained -> Stage 1 v4 -> Stage 2 v4 -> checkpoint final v4
```

Cell 000 juga memperingatkan bahwa output tersimpan bisa berasal dari run v3 sebelumnya. Angka metrik tidak boleh disebut sebagai hasil v4 sebelum notebook dijalankan ulang dari awal di Kaggle.

## Peta singkat cell

| Cell | Bagian | Fungsi utama |
|---|---|---|
| 000 | Identitas v4 | Menjelaskan lineage dan target eksperimen |
| 001-006 | Environment | Instalasi dan import library |
| 007-009 | Konfigurasi | Path, dataset, label, audio, hyperparameter |
| 010-014 | Parser dataset | Mengubah lima dataset menjadi metadata seragam |
| 015-030 | EDA dan cleanup | Memeriksa distribusi, durasi, file rusak, SNR, bobot |
| 031-034 | Split dan augmentasi | Membuat train, validation, test tanpa leakage |
| 035-040 | Preprocessing dan loader | Menyiapkan waveform, feature extractor, batch |
| 041-043 | Arsitektur model | WavLM, pooling, classifier |
| 044-046 | Optimisasi | Focal Loss, class weight, AdamW, scheduler |
| 047-051 | Stage 1 | Training awal dan checkpoint terbaik |
| 052-054 | Noise detection | Mencari kandidat label yang mencurigakan |
| 055-062 | Stage 2 dan MLOps | Fine-tuning, diagnosis, upload Hugging Face |
| 063-071 | Evaluasi | Kurva, test report, heatmap, error analysis |
| 072-074 | Export | Membuat package model v4 |
| 075-082 | Inference | Prediksi audio baru dan link download |

# Penjelasan berdasarkan urutan cell

## Cell 000. Identitas dan perubahan v4

Cell ini adalah pengantar eksperimen. Ia menjelaskan perubahan dari versi sebelumnya dan memberi peringatan tentang output lama.

Perubahan penting v4:

1. Checkpoint awal v4 adalah checkpoint Stage 2 v3 jika file tersebut tersedia.
2. Fallback ke output v4 lama dihapus agar rerun Kaggle tidak melanjutkan file yang tertinggal secara diam-diam.
3. Checkpoint, log, heatmap, dan package diberi namespace v4.
4. Evaluasi final mencakup train, validation, test, metrik macro, confusion matrix, dan pasangan error.
5. Augmentasi offline tetap diarahkan ke sumber data Indonesia pada baseline ini.

Target recall atau precision 80 persen yang ditulis di cell ini adalah sasaran eksperimen, bukan hasil yang dijamin. Hasil nyata harus dibaca dari evaluasi test.

CREMA-D tidak diberi augmentasi offline pada baseline v4. Notebook menganggap masalah utamanya adalah ambiguitas label dan perbedaan sumber, bukan sekadar jumlah file. Jika augmentasi CREMA-D ingin diuji, jadikan itu eksperimen ablation terpisah.

## Cell 001. Judul instalasi dan import

Cell Markdown ini membuka bagian persiapan environment.

## Cell 002. Penjelasan instalasi library

Library yang dijelaskan cell ini memiliki fungsi berikut:

- `transformers`: WavLM dan feature extractor.
- `accelerate`: dukungan pengelolaan training model.
- `librosa`: membaca dan mengolah sinyal audio.
- `soundfile`: membaca dan menulis file audio.
- `tqdm`: progress bar.
- `scikit-learn`: split data dan metrik.
- `seaborn` dan `matplotlib`: grafik.

## Cell 003. Deteksi environment dan instalasi

Kode memeriksa apakah notebook berjalan di Kaggle, Colab, atau lokal. Setelah itu, kode menjalankan `pip install` untuk library yang diperlukan.

Cell ini memiliki efek samping karena mengubah environment Python. `check=False` membuat proses tidak langsung berhenti ketika pip gagal. Jadi pesan `Install selesai` tidak selalu membuktikan semua library berhasil dipasang. Jika terjadi error import, log instalasi harus diperiksa.

## Cell 004. Instalasi ulang

Cell ini mengulangi logika cell 003. Ia tidak menambah proses baru dan kemungkinan tersisa dari pengembangan notebook. Untuk presentasi, cukup jelaskan bahwa dua cell tersebut memastikan library tersedia, dengan catatan cell 004 bersifat duplikat.

## Cell 005. Deskripsi import modul

Cell Markdown ini menjelaskan bahwa modul berikutnya dipakai untuk numerik, visualisasi, dataset, pemodelan PyTorch, split data, dan metrik.

## Cell 006. Import library dan pemilihan device

Kode mengimpor library utama lalu memilih `cuda` jika GPU tersedia, atau `cpu` jika tidak tersedia.

WavLM memiliki banyak parameter sehingga GPU sangat membantu. Jika GPU tidak ada, training tetap dapat dipahami secara konsep, tetapi waktunya jauh lebih lama. Ketika CUDA tersedia, notebook mencetak nama GPU dan VRAM serta mengaktifkan TF32 untuk operasi tertentu pada GPU yang kompatibel.

## Cell 007. Judul konfigurasi global v4

Cell ini menandai awal konfigurasi eksperimen.

## Cell 008. Deskripsi konfigurasi

Cell Markdown ini memperkenalkan path data, sample rate, durasi audio, seed, dan parameter training.

## Cell 009. Path, dataset, audio, label, bobot, dan hyperparameter

Ini adalah cell yang paling banyak menentukan perilaku pipeline.

### Direktori kerja

Notebook membuat:

- `data/raw` untuk data mentah lokal.
- `data/augmented` untuk file augmentasi offline.
- `data/processed` untuk data hasil proses jika diperlukan.
- `models` untuk checkpoint.
- `logs` untuk riwayat, grafik, CSV evaluasi, dan kandidat noise.

Di Kaggle, `BASE_DIR` diarahkan ke `/kaggle/working`. Di lokal, `BASE_DIR` memakai direktori kerja notebook.

### Dataset yang dicari

Notebook mencari lima sumber:

1. IndoWaveSentiment.
2. RAVDESS.
3. EmoDB.
4. CREMA-D.
5. E-SERAVD.

Setiap sumber memiliki beberapa kandidat path. Notebook memakai kandidat pertama yang benar-benar ada. Jika tidak ada dataset sama sekali, eksekusi dihentikan agar training tidak berjalan dengan data kosong.

### Backbone

```python
PRETRAINED_MODEL = 'microsoft/wavlm-base-plus'
```

Backbone adalah model pretrained yang sudah mempelajari pola umum suara dari data besar. Varian `base-plus` dipilih karena lebih realistis untuk GPU Kaggle T4 atau P100 dibandingkan WavLM large.

Pretrained tidak berarti model sudah mengenal enam emosi proyek. WavLM menyediakan representasi umum, sedangkan pooling dan classifier harus dilatih untuk tugas SER.

### Kontrak audio

| Parameter | Nilai | Makna |
|---|---:|---|
| Sample rate | 16.000 Hz | 16.000 sampel per detik |
| Durasi maksimum | 4,0 detik | Audio dipotong atau diberi padding |
| Panjang maksimum | 64.000 sampel | 16.000 x 4 |
| Durasi minimum | 0,35 detik | File lebih pendek dibuang pada cleanup |
| SNR minimum diagnostik | 3 dB | Untuk laporan EDA, bukan filter utama |

Mono berarti audio hanya memiliki satu kanal. Audio stereo akan diproses menjadi satu waveform.

### Urutan label

```text
0 netral
1 senang
2 sedih
3 marah
4 takut
5 jijik
```

`LABEL2IDX` mengubah nama menjadi angka. `IDX2LABEL` melakukan kebalikannya saat hasil ditampilkan. Urutan ini harus konsisten antara training, checkpoint, aplikasi, dan dokumentasi.

### Bobot sumber dan CREMA-D

`BOBOT_SUMBER` memberi bobot dasar untuk sumber data. CREMA-D diberi koreksi per kelas berdasarkan nilai agreement yang dicatat notebook. Nilai agreement rendah membuat bobot metadata lebih rendah.

Namun jangan salah menjelaskan bagian ini. Kolom `df['bobot']` dihitung dan divisualisasikan, tetapi tidak diteruskan sebagai sample weight ke loss. Training aktual memakai class weight berdasarkan frekuensi dan `CLASS_LOSS_MULTIPLIER` pada cell 046. Jadi bobot per baris di cell ini adalah metadata dan bahan EDA, bukan bobot loss per file.

### Split, seed, dan augmentasi

```python
VAL_SPLIT = 0.15
TEST_SPLIT = 0.15
SEED = 42
```

Seed membantu membuat proses acak lebih mudah direproduksi. Ia tidak menjamin angka identik pada semua GPU dan versi library.

Mode augmentasi offline yang disiapkan:

- `pitch_up2`: pitch naik dua semitone.
- `pitch_down2`: pitch turun dua semitone.
- `stretch_slow`: ucapan diperlambat.
- `stretch_fast`: ucapan dipercepat.
- `noise_light`: noise ringan ditambahkan.

### Hyperparameter utama

| Parameter | Nilai | Kegunaan |
|---|---:|---|
| Batch size | 6 | Audio per batch |
| Gradient accumulation | 2 | Update optimizer setiap dua batch |
| Epoch Stage 1 | 22 | Batas maksimum training awal |
| Freeze epochs | 2 | Dua epoch awal hanya head yang dilatih |
| Unfreeze last N | 6 | Enam layer backbone terakhir dibuka |
| Backbone LR | 0,00001 | Update kecil untuk pretrained backbone |
| Head LR | 0,0003 | Update lebih besar untuk head |
| Weight decay | 0,01 | Regularisasi AdamW |
| Dropout | 0,30 | Regularisasi classifier |
| Focal gamma | 2,0 | Fokus pada contoh sulit |
| Label smoothing | 0,0 | Tidak aktif pada baseline |
| Mixup | `False` | Tidak digunakan pada baseline |
| SpecAugment | `True` | Time masking ringan pada train |
| AMP | `True` | Mixed precision jika GPU tersedia |
| Gradient checkpointing | `True` | Mengurangi penggunaan VRAM |

### Checkpoint dan output

Output utama:

- `ser_wavlm_v4_best.pt` untuk checkpoint Stage 1 terbaik.
- `ser_wavlm_v4_stage2_best.pt` untuk checkpoint Stage 2 terbaik.
- `history_v4.json` untuk riwayat training.
- `metadata_split_v4.csv` untuk metadata split.

Input checkpoint mencari `ser_wavlm_v3_best.pt` pada Kaggle Input, path pipeline lokal, atau direktori model. Jika tidak ditemukan, model kembali ke bobot pretrained WavLM.

Kode masih menulis metadata gabungan dengan nama `metadata_gabungan_v7.csv`. Itu adalah nama file lama yang tertinggal, bukan bukti model menjadi v7. Path checkpoint dan konfigurasi yang aktif tetap v4.

## Cell 010. Judul parser dataset

Bagian ini mengubah lima dataset dengan format nama file atau folder berbeda menjadi satu metadata seragam.

## Cell 011. Deskripsi mapping label

Sebelum audio dilatih, kode label asli harus diterjemahkan ke label proyek. Angka yang sama pada dua dataset belum tentu memiliki arti yang sama, sehingga mapping wajib dilakukan per sumber.

## Cell 012. Mapping dan fungsi parser

Fungsi `_buat_baris` membuat kolom `path`, `emosi`, `label`, `sumber`, `bahasa`, dan `file`.

### IndoWaveSentiment

Parser mengambil bagian kedua dari nama file yang dipisahkan tanda hubung. Mapping notebook:

- `01` menjadi `netral`.
- `02` menjadi `senang`.
- `03` menjadi `None` dan tidak dipakai.
- `04` menjadi `jijik`.
- `05` menjadi `sedih`.

Menurut dokumentasi dataset IndoWaveSentiment, kode `03` sebenarnya adalah kelas surprise atau kejutan, sedangkan `05` adalah disappointed atau kecewa. Notebook sengaja tidak memasukkan kelas tersebut karena target proyek hanya enam kelas dan tidak memiliki label `kejutan` atau `kecewa`. Lihat [dokumentasi format nama IndoWaveSentiment](https://pmc.ncbi.nlm.nih.gov/articles/PMC11647155/).

### RAVDESS

Parser mengambil bagian ketiga nama file. Kode `01` dan `02` dipetakan ke `netral`, `03` ke `senang`, `04` ke `sedih`, `05` ke `marah`, `06` ke `takut`, `07` ke `jijik`, sedangkan `08` diabaikan.

### EmoDB

Parser mengambil karakter indeks keenam dari nama file. Karakter tersebut dicocokkan dengan mapping EmoDB. Label sumber yang tidak dipakai di-set ke `None`.

### CREMA-D

Parser memecah nama file dengan underscore dan membaca token seperti `ANG`, `DIS`, `FEA`, `HAP`, `NEU`, dan `SAD`. Token tersebut menjadi `marah`, `jijik`, `takut`, `senang`, `netral`, dan `sedih`.

### E-SERAVD

Parser membaca nama folder label dari path. Awalan angka dan pemisah dibuang agar folder seperti `01_angry` dapat dicocokkan ke mapping.

Parser tidak mendengarkan audio untuk mengetahui emosi. Ia hanya membaca metadata yang sudah ditulis pada nama file atau folder.

## Cell 013. Deskripsi loading metadata

Cell ini menjelaskan bahwa semua parser akan dijalankan, path yang rusak disaring, lalu hasilnya digabung menjadi satu DataFrame.

## Cell 014. Penggabungan dataset dan bobot

Kode menjalankan parser untuk setiap dataset yang ditemukan. Hasil non-kosong digabung, lalu duplikat path dibuang.

Setelah itu, kode menghitung bobot dasar sumber dan koreksi per kelas CREMA-D. Output yang dicetak meliputi:

- Total file unik sebelum augmentasi.
- Jumlah file per sumber.
- Jumlah file per emosi.
- Rata-rata bobot CREMA-D per emosi.
- Proporsi data Indonesia.

Metadata disimpan sebagai CSV agar proses parsing dapat diaudit tanpa membaca ulang semua audio.

## Cell 015. Judul EDA sebelum training

EDA atau Exploratory Data Analysis adalah pemeriksaan kondisi data sebelum model dilatih. Tujuannya menemukan masalah yang tidak terlihat hanya dari jumlah total file.

## Cell 016. EDA 1, distribusi kelas per sumber

Cell ini menjelaskan grafik jumlah file per emosi dan per sumber dataset.

Pemeriksaan ini penting karena model dapat terlihat memiliki akurasi tinggi hanya karena sering menebak kelas mayoritas. Dengan memisahkan sumber, kita juga dapat melihat apakah satu dataset mendominasi training atau hanya menyumbang kelas tertentu.

## Cell 017. Grafik distribusi kelas

Kode membuat count plot dari `df`, memakai label angka sebagai sumbu kelas dan warna berbeda untuk sumber dataset. Tabel silang di bawah grafik menampilkan angka secara numerik.

Cara membaca grafik:

- Batang tinggi berarti jumlah file lebih banyak.
- Jika sebuah kelas hampir hanya berasal dari satu sumber, model dapat belajar ciri khas sumber, bukan emosi.
- Ketimpangan kelas menjadi alasan penggunaan class weight pada cell 046.

## Cell 018. EDA 2, durasi dan outlier

Cell ini menjelaskan bahwa durasi diperiksa setelah silence trimming. Batas minimalnya 0,35 detik dan batas maksimumnya 4 detik.

## Cell 019. Scanning durasi dan file rusak

Untuk setiap file, kode:

1. Membaca audio pada sample rate 16.000 Hz.
2. Menyimpan durasi mentah ke `durasi_raw_sec`.
3. Membuang silence awal dan akhir dengan `top_db=30`.
4. Menyimpan durasi setelah trim ke `durasi_sec`.
5. Mencatat file yang gagal dibaca di `corrupt_files`.

File di bawah 0,35 detik kemungkinan terlalu pendek untuk membawa informasi emosi yang cukup. File di atas 4 detik tidak langsung dibuang karena loader akan memotongnya ketika menjadi input model.

Histogram membantu melihat seberapa sering batas 4 detik memotong audio. Ini penting ketika menjelaskan trade-off antara konteks suara dan penggunaan VRAM.

## Cell 020. EDA 3, file corrupt dan nilai kosong

Cell ini memberi judul untuk pemeriksaan file yang rusak, tidak terbaca, missing, atau tidak memiliki energi.

## Cell 021. Daftar file corrupt

Kode mencetak jumlah file yang gagal dibaca dan maksimal sepuluh contoh path. File belum dibuang di sini. Penghapusan dilakukan setelah seluruh pemeriksaan awal selesai pada cell 030.

## Cell 022. EDA 4, Signal-to-Noise Ratio

SNR atau Signal-to-Noise Ratio adalah perkiraan perbandingan kekuatan sinyal terhadap noise. Nilai dB yang lebih tinggi biasanya berarti sinyal ucapan lebih dominan.

## Cell 023. Estimasi SNR pada sampel

Notebook mengambil maksimal 400 file secara acak, lalu memperkirakan SNR dari energi total dan sebagian energi terendah waveform.

Hasilnya dipakai untuk histogram dan jumlah sampel di bawah 3 dB. Ini adalah pemeriksaan diagnostik, bukan filter utama. Perhitungan tersebut bukan pengukuran studio yang sempurna karena notebook tidak mempunyai rekaman noise terpisah sebagai ground truth.

## Cell 024. EDA 5, bobot per kelas dan sumber

Cell ini menjelaskan grafik dan tabel bobot rata-rata berdasarkan sumber dan label emosi. Bobot membantu melihat strategi koreksi dataset secara visual.

## Cell 025. Grafik dan tabel bobot

Kode membuat grouped bar chart dan pivot table dari kolom `df['bobot']`.

Penting untuk presentasi: grafik ini menunjukkan rancangan bobot metadata, tetapi cell training tidak meneruskan bobot per baris tersebut ke loss. Loss aktual menggunakan class weight hasil `compute_class_weight` yang dimodifikasi multiplier kelas.

## Cell 026. EDA 6, confusion matrix aktual

Cell ini menegaskan bahwa confusion matrix tidak boleh dibuat dari angka manual. Heatmap resmi baru tersedia setelah model menghasilkan prediksi pada test set.

## Cell 027. Placeholder confusion matrix

Kode hanya mencetak bahwa confusion matrix manual sudah dihapus dari EDA dan heatmap aktual dibuat pada evaluasi akhir. Cell ini bukan hasil evaluasi model.

## Cell 028. Cara membaca confusion matrix

Confusion matrix memiliki:

- Sumbu Y sebagai label aktual.
- Sumbu X sebagai prediksi model.
- Diagonal sebagai prediksi benar.
- Sel di luar diagonal sebagai pasangan emosi yang tertukar.

Contoh, baris `sedih` dan kolom `senang` menunjukkan jumlah audio yang sebenarnya `sedih` tetapi diprediksi `senang`. Ini menurunkan recall `sedih`.

## Cell 029. EDA 7, cleanup dataset

Cell ini membuka pembersihan berdasarkan kriteria quality assurance.

## Cell 030. Membuang file terlalu pendek dan corrupt

Kode membuang file dengan durasi setelah trim kurang dari 0,35 detik. File corrupt biasanya memiliki durasi `-1` dan ikut terhapus karena berada di bawah batas minimum.

File lebih panjang dari 4 detik tetap dipakai. File tersebut akan dipotong oleh `load_waveform` ketika masuk ke model. Ringkasan cell mencetak jumlah file awal, file dibuang, file yang akan dipotong, dan data siap pakai.

## Cell 031. Aturan split data

Cell ini menegaskan aturan anti data leakage. Split wajib dilakukan sebelum augmentasi.

Data leakage terjadi jika varian augmentasi dari file yang sama masuk ke train dan test. Dalam kondisi tersebut, test bukan lagi simulasi data baru sehingga akurasi bisa terlalu optimistis.

## Cell 032. Split train, validation, dan test

Kode memakai data original yang sumbernya tidak berakhiran `_aug`. Pembagiannya:

- Train: 70 persen.
- Validation: 15 persen.
- Test holdout: 15 persen.

`stratify=df_orig['label']` menjaga proporsi kelas tetap mendekati data awal.

Peran masing-masing split:

- Train dipakai untuk memperbarui bobot model.
- Validation dipakai untuk memilih checkpoint dan memantau overfitting.
- Test hanya dipakai di akhir untuk mengukur generalisasi.

## Cell 033. Augmentasi offline train only

Cell ini menetapkan augmentasi offline hanya untuk data latih Bahasa Indonesia. Validation dan test harus tetap asli agar perbandingan hasil valid.

## Cell 034. Membuat augmentasi Indonesia

Kode memproses sumber `indowavesentiment` dan `eseravd` yang ada di `df_train_orig`. Untuk setiap file, lima mode dapat menghasilkan maksimal lima file baru.

Augmentasi yang dilakukan:

- Pitch naik dan turun mengubah tinggi suara.
- Time stretch mengubah kecepatan ucapan.
- Noise ringan membuat model tidak terlalu bergantung pada rekaman yang sangat bersih.

File baru disimpan ke `data/augmented`, diberi sumber dengan suffix `_aug`, lalu digabung hanya ke `df_train`.

Cell ini juga mencetak jumlah augmentasi per sumber dan per emosi. Jadi pengguna dapat melihat berapa file yang benar-benar dibuat, bukan hanya berapa mode yang dikonfigurasi.

CREMA-D, RAVDESS, dan EmoDB tidak diberi augmentasi offline pada baseline. Selain itu, train masih dapat menerima augmentasi ringan secara acak saat `__getitem__` pada cell 037. Validation dan test tidak menerima augmentasi acak tersebut.

## Cell 035. Judul audio loading dan SpecAugment

Bagian ini menyiapkan waveform agar semua contoh memiliki bentuk yang kompatibel dengan WavLM.

## Cell 036. Kontrak preprocessing waveform

Cell Markdown ini menyebut empat operasi utama:

1. Silence trimming.
2. Peak normalization.
3. Crop jika terlalu panjang.
4. Zero-padding jika terlalu pendek.

## Cell 037. Audio loader, augmentasi runtime, dan Dataset

### `load_waveform`

Fungsi ini membaca audio sebagai mono pada 16 kHz. Nilai NaN dan infinity diubah menjadi nol. Silence awal dan akhir dipotong dengan `top_db=30`. Waveform kemudian dinormalisasi berdasarkan amplitudo puncak.

Jika file gagal dibaca, fungsi mengembalikan array nol minimal agar batch tidak langsung rusak. Ini adalah pengaman terakhir, bukan pengganti validasi data pada EDA.

### `fix_length`

Jika audio lebih panjang dari 64.000 sampel:

- Mode `train` memilih titik awal secara acak.
- Mode `eval` mengambil potongan dari awal.

Jika audio lebih pendek, zero-padding ditambahkan sampai 64.000 sampel.

Konsekuensinya, audio lebih panjang dari 4 detik dapat dilihat dari potongan berbeda pada epoch berbeda ketika training. Evaluasi tetap konsisten mengambil bagian awal.

### `spec_augment` dan `augment_light`

`augment_light` secara acak dapat melakukan:

- Perubahan gain antara 0,80 dan 1,20.
- Penambahan noise kecil.
- Pergeseran waveform dengan `np.roll`.
- Time masking jika SpecAugment aktif.

Pada notebook ini SpecAugment bekerja langsung pada waveform dengan mengosongkan segmen waktu. Ia bukan masking pada spectrogram mel seperti implementasi SpecAugment yang lebih umum. Komentar kode menyebut smooth edge fade, tetapi operasi aktual mengisi segmen dengan nol secara langsung.

### `SERWaveformDataset`

Dataset PyTorch mengembalikan waveform, label angka, dan metadata path, emosi, sumber, serta bahasa. Augmentasi runtime hanya aktif ketika mode dataset adalah `train`.

## Cell 038. Judul DataLoader dan Mixup

Judul masih menyebut Mixup, tetapi konfigurasi v4 menetapkan `USE_MIXUP = False` dan cell 040 tidak menjalankan Mixup. Jadi Mixup tidak boleh disebut aktif pada hasil v4.

## Cell 039. Deskripsi feature extractor dan collate

Cell ini menjelaskan bahwa waveform dari beberapa contoh akan disusun menjadi batch tensor yang dapat diterima WavLM.

## Cell 040. Feature extractor dan DataLoader

`AutoFeatureExtractor.from_pretrained(MODEL_SOURCE)` mengambil aturan input sesuai WavLM. `collate_fn` melakukan hal berikut:

1. Mengambil waveform setiap item batch.
2. Mengubah label menjadi tensor integer.
3. Menyimpan metadata.
4. Memanggil feature extractor pada sample rate 16 kHz.
5. Mengaktifkan padding, truncation, dan attention mask.

Terdapat tiga DataLoader:

- `train_loader` mengacak data dan menggunakan mode train.
- `val_loader` tidak mengacak data dan menggunakan mode eval.
- `test_loader` tidak mengacak data dan menggunakan mode eval.

Notebook tidak memakai `WeightedRandomSampler`. Alasannya mencegah koreksi distribusi dilakukan dua kali karena class weight sudah digunakan pada loss.

# Arsitektur model secara mendalam

## Cell 041. Judul WavLM dan Attentive Stats Pooling

Cell ini menandai awal pembangunan model.

## Cell 042. Deskripsi arsitektur

Arsitektur terdiri dari tiga bagian besar:

```text
Waveform
    -> WavLM base-plus
    -> Attentive Stats Pooling
    -> classifier head
    -> enam logit emosi
```

## Cell 043. Implementasi arsitektur PyTorch

### 1. WavLM base-plus sebagai backbone

`self.backbone = AutoModel.from_pretrained(model_source)` memuat WavLM pretrained.

Backbone mengubah urutan waveform menjadi urutan representasi fitur suara. Setiap posisi mewakili potongan waktu yang sudah diproses oleh jaringan neural. Representasi ini dapat menangkap pola energi, intonasi, ritme, dan artikulasi yang berguna untuk klasifikasi emosi.

Backbone tidak langsung menghasilkan enam label. Ia menghasilkan `last_hidden_state`, yaitu tensor fitur untuk banyak posisi waktu.

Untuk WavLM base-plus, ukuran fitur tersembunyi yang digunakan notebook adalah 768. Jika panjang urutan fitur adalah `T`, bentuk tensor kira-kira:

```text
[batch, T, 768]
```

Nilai `T` bukan jumlah sampel waveform karena WavLM melakukan feature encoding dan downsampling.

### 2. Feature mask dan attention mask

Audio yang lebih pendek dari 4 detik mendapat zero-padding. Padding bukan bagian dari ucapan dan seharusnya tidak ikut dihitung sebagai informasi emosi.

`_feature_mask` mengubah attention mask pada level waveform menjadi mask pada level fitur WavLM. Posisi padding kemudian dapat dikeluarkan dari perhitungan pooling.

### 3. Attentive Stats Pooling

Pooling mengubah fitur berurutan sepanjang waktu menjadi satu vektor tetap. Ini dibutuhkan karena classifier menerima ukuran input konsisten, sementara panjang urutan fitur bersifat temporal.

Layer pooling terdiri dari:

```text
LayerNorm(768)
    -> Linear(768, 128)
    -> Tanh
    -> Dropout(0,15)
    -> Linear(128, 1)
    -> score attention
```

Kegunaan tiap bagian:

- `LayerNorm` menstabilkan skala fitur sebelum attention dihitung.
- `Linear(768, 128)` memproyeksikan fitur besar ke ruang attention lebih kecil.
- `Tanh` memberi non-linearitas agar hubungan fitur dan kepentingan waktu tidak hanya lurus.
- `Dropout(0,15)` membantu attention tidak bergantung pada satu jalur fitur ketika training.
- `Linear(128, 1)` menghasilkan satu skor untuk setiap posisi waktu.
- `softmax` mengubah skor menjadi bobot yang jumlahnya satu sepanjang waktu.

Jika mask tersedia, posisi padding diberi skor sangat negatif sehingga hampir tidak mendapat bobot setelah softmax.

#### Weighted mean

Mean adalah rata-rata fitur dengan bobot attention. Bagian suara yang dianggap penting oleh model memberi kontribusi lebih besar.

Mean dapat merangkum karakteristik umum ucapan, seperti kecenderungan energi dan warna representasi suara.

#### Weighted standard deviation

Standard deviation mengukur seberapa banyak fitur berubah sepanjang waktu. Perubahan ini berguna untuk membedakan suara yang stabil dari suara dengan dinamika kuat, seperti perubahan tekanan atau intonasi.

Mean berukuran 768 dan standard deviation juga berukuran 768. Keduanya digabung:

```text
768 mean + 768 standard deviation = 1536 fitur
```

Inilah alasan output pooling berukuran 1536 meskipun hidden size WavLM hanya 768.

### 4. Classifier head

Classifier menerima vektor hasil pooling berukuran 1536.

```text
LayerNorm(1536)
    -> Dropout(0,30)
    -> Linear(1536, 256)
    -> GELU
    -> Dropout(0,225)
    -> Linear(256, 6)
```

Kegunaan tiap layer:

- `LayerNorm(1536)` menstabilkan vektor gabungan mean dan standard deviation.
- `Dropout(0,30)` mematikan sebagian aktivasi secara acak supaya head tidak menghafal data train.
- `Linear(1536, 256)` memadatkan informasi menjadi representasi keputusan yang lebih kecil.
- `GELU` memberi non-linearitas yang halus dan umum dipakai pada transformer.
- `Dropout(0,225)` memberi regularisasi tambahan sebelum output.
- `Linear(256, 6)` menghasilkan enam logit, satu untuk setiap kelas emosi.

Logit belum berupa persentase. Softmax baru digunakan ketika hasil prediksi ingin ditampilkan sebagai probabilitas.

### 5. Freeze dan gradual unfreezing

`freeze_all_backbone` membekukan seluruh parameter WavLM, tetapi pooling dan classifier tetap dilatih. Ini membuat head belajar menyesuaikan diri tanpa langsung mengubah representasi pretrained secara agresif.

`unfreeze_last_layers` tetap membekukan sebagian besar backbone dan membuka enam layer encoder terakhir. Layer terakhir biasanya lebih dekat dengan tugas downstream karena memproses representasi tingkat tinggi.

Jadwal v4:

- Epoch 1 dan 2: backbone dibekukan, pooling dan classifier dilatih.
- Setelah epoch 2: enam layer backbone terakhir dibuka, pooling dan classifier tetap dilatih.

Gradual unfreezing mengurangi risiko catastrophic forgetting, yaitu kondisi ketika fine-tuning terlalu besar merusak kemampuan representasi pretrained.

### 6. Gradient checkpointing

Gradient checkpointing menghemat VRAM dengan tidak menyimpan semua aktivasi intermediate. Saat backward pass, beberapa aktivasi dihitung ulang. Trade-off-nya, komputasi sedikit lebih lambat, tetapi model lebih mungkin berjalan pada GPU dengan VRAM terbatas.

### 7. DataParallel

Jika lebih dari satu GPU terdeteksi, model dibungkus dengan `nn.DataParallel`. Batch dibagi ke beberapa GPU dan hasilnya digabung. Pada Kaggle satu GPU biasanya menjadi kondisi umum, sehingga bagian ini tidak selalu aktif.

## Cell 044. Judul loss dan optimizer

Cell ini membuka bagian optimisasi, yaitu cara model menghitung kesalahan dan memperbarui parameter.

## Cell 045. Deskripsi loss, Mixup, AdamW, dan scheduler

Markdown cell ini menyebut beberapa komponen, tetapi konfigurasi aktual harus dibaca bersama cell 046. Mixup tidak aktif pada baseline v4.

## Cell 046. Class weight, Focal Loss, optimizer, dan scheduler

### Class weight

`compute_class_weight(class_weight='balanced')` menghitung bobot berdasarkan frekuensi kelas. Kelas yang jarang mendapat bobot lebih besar sehingga kesalahan pada kelas tersebut memberi kontribusi lebih besar ke loss.

Setelah itu, bobot dikalikan:

```text
[1,00, 1,15, 1,30, 1,00, 0,80, 0,80]
```

Dengan urutan label notebook:

- `netral`: 1,00.
- `senang`: 1,15.
- `sedih`: 1,30.
- `marah`: 1,00.
- `takut`: 0,80.
- `jijik`: 0,80.

Setelah dikalikan, bobot dinormalisasi agar rata-ratanya satu.

### Focal Loss

Cross Entropy biasa memberi penalti berdasarkan probabilitas kelas benar. Focal Loss menambahkan faktor yang mengecilkan kontribusi contoh yang sudah sangat mudah dan memberi fokus relatif pada contoh sulit.

Secara konsep:

```text
focal loss = (1 - p_benar)^gamma x cross entropy x bobot kelas
```

Jika `p_benar` tinggi, faktor tersebut kecil. Jika model ragu atau salah, faktor lebih besar. `gamma=2.0` mengatur kekuatan fokus pada contoh sulit.

`LABEL_SMOOTHING = 0.0`, sehingga label smoothing tidak aktif pada baseline v4 walaupun class `FocalLoss` mendukungnya.

### AdamW dan differential learning rate

Parameter dibagi menjadi dua kelompok:

- Backbone: learning rate `1e-5`.
- Pooling dan classifier: learning rate `3e-4`.

Head perlu learning rate lebih besar karena harus belajar menyesuaikan representasi untuk tugas baru. Backbone pretrained dijaga dengan learning rate kecil agar tidak rusak.

AdamW juga memakai `weight_decay=0.01` sebagai regularisasi.

### Cosine scheduler, warmup, dan AMP

Cosine scheduler mengubah learning rate bertahap mengikuti bentuk kosinus. Warmup 10 persen dari total langkah memberi waktu bagi optimizer untuk mulai lebih aman.

AMP memakai mixed precision jika CUDA tersedia. Ini dapat menghemat VRAM dan mempercepat operasi tertentu.

## Cell 047. Judul training Stage 1

Stage 1 adalah training utama pada data train setelah preprocessing dan augmentasi.

## Cell 048. Deskripsi utilitas training

Cell ini menjelaskan fungsi akurasi, pemindahan tensor ke device, AMP, dan schedule freeze.

## Cell 049. Fungsi training dan evaluasi dasar

Fungsi `akurasi` menghitung proporsi prediksi yang sama dengan label. `to_device_inputs` memindahkan tensor input ke CPU atau GPU. `autocast_ctx` memilih mixed precision ketika aktif.

`train_epoch` dan `eval_epoch` menjalankan satu putaran DataLoader. Training melakukan backward dan optimizer update, sedangkan evaluasi memakai `torch.no_grad()` agar tidak menyimpan graph dan lebih hemat memori.

Gradient accumulation membagi loss dengan `GRAD_ACCUM_STEPS`, lalu melakukan optimizer step setiap dua batch. Ini mensimulasikan batch efektif lebih besar tanpa langsung membutuhkan VRAM untuk batch besar.

Gradient clipping membatasi norm gradient pada 1,0 untuk mencegah update ekstrem.

## Cell 050. Deskripsi loop Stage 1

Checkpoint terbaik dipilih berdasarkan monitor validasi, bukan hanya training accuracy.

## Cell 051. Load checkpoint dan loop Stage 1

`load_checkpoint_weights` membaca beberapa bentuk checkpoint dan menyesuaikan prefix `module.` jika model memakai DataParallel. Loading memakai `strict=True`, sehingga struktur checkpoint harus cocok dengan arsitektur model v4.

Jika checkpoint v3 ditemukan, bobotnya dimuat. Jika tidak, notebook menggunakan bobot WavLM base-plus.

Loop Stage 1 berjalan maksimal 22 epoch. Setiap epoch:

1. Menentukan apakah backbone dibekukan atau enam layer terakhir dibuka.
2. Menjalankan training.
3. Menjalankan validation.
4. Menghitung macro F1, macro precision, macro recall, recall `sedih`, recall `senang`, precision `takut`, dan precision `jijik`.
5. Menggabungkan metrik tersebut menjadi `monitor`.
6. Menyimpan checkpoint jika monitor meningkat minimal `MIN_DELTA`.
7. Menghentikan training jika tidak ada peningkatan selama enam epoch.

Monitor yang dipakai:

```text
0,40 macro F1
+ 0,20 recall sedih
+ 0,20 recall senang
+ 0,10 precision takut
+ 0,10 precision jijik
```

Artinya pemilihan checkpoint tidak hanya mengejar akurasi keseluruhan. Metrik kelas yang menjadi fokus diberi pengaruh langsung.

Pada akhir Stage 1, checkpoint terbaik dimuat kembali dan sementara ditetapkan sebagai `FINAL_MODEL_PATH`. Nilai ini dapat diganti oleh Stage 2.

## Cell 052. Judul deteksi label noise

Label noise adalah label yang kemungkinan tidak sesuai dengan isi audio. Model tidak dapat membuktikan otomatis bahwa label salah, sehingga hasil tahap ini disebut kandidat noise.

## Cell 053. Deskripsi inferensi pasca-Stage 1

Kandidat dicari setelah model awal memiliki kemampuan membedakan pola. Gagasannya, jika model sangat yakin terhadap kelas lain tetapi label metadata berbeda, file tersebut layak diperiksa.

## Cell 054. Deteksi kandidat label noise

Kode membuat prediksi pada `df_train_orig`, bukan data augmentasi. File menjadi kandidat jika:

1. Prediksi berbeda dari label asli.
2. Confidence prediksi minimal 0,95.

Hasil menyimpan path, sumber, label asli, prediksi, dan confidence. Daftar ini diekspor ke `label_noise_candidates.csv`.

Confidence tinggi bukan bukti absolut label salah. Model juga dapat sangat yakin tetapi salah. Karena itu, Stage 2 memakai aturan hard drop yang konservatif.

Ada dua parameter konfigurasi yang mudah membingungkan:

- `NOISE_CONF_THRESHOLD = 0.40`.
- `NOISE_DOWNWEIGHT = 0.3`.

Kedua variabel tersebut bukan aturan utama yang dipakai fungsi deteksi dan Stage 2 aktual. Kode memakai `HARD_NOISE_CONF_THRESH = 0.95` dan hard drop terbatas. Jangan menjelaskan seolah semua kandidat di bawah 0,40 otomatis diturunkan bobotnya.

## Cell 055. Judul Stage 2

Judul Markdown menyebut noise downweighting. Perilaku kode aktual harus dibaca pada cell 057.

## Cell 056. Deskripsi filtering konservatif

Cell ini menjelaskan hard drop kandidat CREMA-D yang sangat yakin dan berasal dari label yang dianggap lebih tepercaya.

## Cell 057. Hard drop dan fine-tuning Stage 2

Kode hanya memilih:

- Sumber `cremad`.
- Confidence minimal 0,95.
- Label asli `netral` atau `marah`.

Kelas `sedih`, `senang`, `takut`, dan `jijik` tidak masuk daftar hard drop. Ini melindungi kelas target recall dan kelas yang sedang dipantau precision-nya.

Setelah path yang dipilih dibuang dari train, Stage 2:

1. Membuat Dataset dan DataLoader baru.
2. Memakai lima epoch.
3. Menurunkan learning rate menjadi 30 persen dari Stage 1.
4. Menggunakan validation yang sama.
5. Menyimpan checkpoint terbaik berdasarkan monitor metrik yang sama.

Jika checkpoint Stage 2 berhasil dibuat, file tersebut menjadi `FINAL_MODEL_PATH`. Jika tidak ada kandidat atau tidak ada checkpoint Stage 2, checkpoint Stage 1 tetap dipakai.

Secara teknis, ini bukan downweighting per file. Ini adalah hard removal selektif. Hal tersebut harus dijelaskan secara jujur saat presentasi.

## Cell 058. Deskripsi pemeriksaan checkpoint

Cell ini mengarahkan notebook untuk memuat checkpoint terbaik dan mencetak diagnosis perbedaan metrik train dan validation.

## Cell 059. Nama checkpoint Stage 1 dan final

Kode mencetak path checkpoint Stage 1 dan `FINAL_MODEL_PATH`, sehingga pengguna dapat mengetahui apakah hasil final berasal dari Stage 1 atau Stage 2.

## Cell 060. Diagnosis gap train-validation

Kode mengambil `train_acc` dan `val_acc` dari checkpoint, lalu menghitung:

```text
gap = train accuracy - validation accuracy
```

Interpretasi yang dipakai:

- Gap lebih dari 20 persen: peringatan overfitting.
- Gap lebih dari 10 persen: overfitting ringan.
- Selain itu: generalisasi dianggap lebih baik.

Ini hanya diagnosis sederhana. Gap kecil tidak menjamin model bagus, dan gap besar tidak memberi tahu penyebab tunggal. Evaluasi test dan metrik per kelas tetap diperlukan.

## Cell 061. Deskripsi upload Hugging Face

Cell ini membuka langkah MLOps opsional, yaitu mengirim checkpoint ke Hugging Face Hub.

## Cell 062. Upload checkpoint ke Hugging Face

Kode memasang `huggingface_hub`, mengambil `HF_TOKEN` dari Kaggle Secrets, lalu mengunggah checkpoint final ke:

```text
elnathzzz/wavlm-ser-multilingual
```

File yang dikirim diberi nama `ser_wavlm_v4_best.pt`.

Cell ini memiliki efek samping dan membutuhkan koneksi internet. Jika tujuan hanya memahami atau menguji training, cell ini tidak perlu dijalankan. Untuk deployment, checkpoint di Hugging Face tetap memerlukan arsitektur `WavLMSER`, preprocessing, dan dependency yang sama.

## Cell 063. Judul evaluasi dan visualisasi

Bagian ini menilai model setelah checkpoint final dipilih.

## Cell 064. Deskripsi kurva training

Kurva train dan validation dipakai untuk membaca konvergensi dan potensi overfitting.

## Cell 065. Kurva akurasi dan loss

Kode membaca `riwayat` dan membuat dua grafik:

- Akurasi latih dan validasi setiap epoch.
- Loss latih dan validasi setiap epoch.

File disimpan sebagai `kurva_training_v4.png`.

Perlu diperhatikan bahwa `riwayat` diisi oleh loop Stage 1. Cell ini tidak menambahkan riwayat Stage 2 ke grafik. Karena itu, judul grafik menyebut Stage 1 dan jangan menjelaskannya sebagai riwayat lengkap dua tahap.

Cara membaca grafik:

- Train naik dan validation ikut naik: model masih belajar dengan generalisasi yang cukup baik.
- Train naik tetapi validation berhenti atau turun: indikasi overfitting.
- Loss validation makin jauh di atas train: model semakin tidak cocok pada data yang belum dilihat.

## Cell 066. Deskripsi evaluasi holdout test

Cell ini memperkenalkan test set. Test berbeda dari validation karena tidak dipakai untuk memilih checkpoint selama training.

## Cell 067. Classification report dan test accuracy

Kode memuat `FINAL_MODEL_PATH`, menjalankan inference pada `test_loader`, lalu menghasilkan:

- `true_all`: label aktual test.
- `pred_all`: prediksi model.
- `test_acc`: akurasi test eksplisit.
- Classification report.

Test accuracy dihitung dengan:

```python
test_acc = accuracy_score(true_all, pred_all)
```

Jadi test accuracy memang ada dan ditampilkan pada cell ini. Perbedaannya:

- Train accuracy mengukur data yang dipakai untuk update bobot.
- Validation accuracy membantu memilih checkpoint.
- Test accuracy mengukur generalisasi pada holdout yang tidak dipakai untuk dua keputusan tersebut.

Classification report menampilkan precision, recall, F1-score, dan support per kelas. Accuracy keseluruhan tidak cukup untuk mengetahui kelas mana yang masih gagal.

## Cell 068. Deskripsi confusion matrix

Cell ini menjelaskan bahwa heatmap dibuat dari prediksi aktual, bukan angka manual.

## Cell 069. Heatmap dan ringkasan metrik final

Kode membuat confusion matrix dari `true_all` dan `pred_all`. `cm_pct` menormalisasi setiap baris berdasarkan jumlah label aktual pada kelas tersebut.

Karena normalisasi per baris, nilai diagonal pada satu baris dapat dibaca sebagai recall kelas tersebut. Misalnya diagonal baris `sedih` menunjukkan proporsi audio `sedih` yang berhasil dikenali sebagai `sedih`.

Heatmap juga menampilkan jumlah mentah dalam tanda kurung. Persentase tanpa support dapat menyesatkan, terutama pada kelas yang jumlahnya kecil.

Cell ini mencetak secara eksplisit:

- Checkpoint final.
- Epoch terbaik.
- Train accuracy.
- Validation accuracy.
- Test accuracy.
- Gap train-validation.
- Macro precision.
- Macro recall.
- Macro F1.

Macro metric menghitung rata-rata antar kelas dengan bobot kelas sama. Ini berguna pada dataset tidak seimbang karena kelas besar tidak boleh sepenuhnya menutupi kelas kecil.

## Cell 070. Deskripsi DataFrame evaluasi

Cell ini menjelaskan bahwa setiap prediksi test disimpan agar error analysis dapat dilakukan pada level file.

## Cell 071. CSV evaluasi, sumber, bahasa, dan pasangan error

Kode membuat `eval_df` berisi path, label aktual, prediksi, nama emosi, sumber, bahasa, dan kolom `benar`.

CSV disimpan sebagai `evaluasi_test_v4.csv`.

Laporan tambahan:

### Akurasi per sumber

Menjawab apakah model lebih baik pada IndoWaveSentiment, E-SERAVD, RAVDESS, EmoDB, atau CREMA-D.

### Akurasi per bahasa

Membandingkan metadata bahasa `id`, `en`, dan `de`. Ini bukan pengukuran kemampuan bahasa yang sempurna, tetapi membantu melihat perbedaan domain.

### CREMA-D per emosi

Bagian ini penting untuk memeriksa apakah kelas `sedih` pada CREMA-D memang menjadi sumber masalah.

### Precision, recall, dan F1 per sumber dan kelas

Metrik dihitung terpisah untuk setiap sumber. Nilai macro keseluruhan bisa terlihat baik, tetapi kelas tertentu pada sumber tertentu tetap lemah.

### Pasangan error terbesar

Pasangan seperti `sedih -> senang` menunjukkan label aktual dan prediksi model. Tabel ini membantu membedakan:

- Recall rendah: banyak file kelas aktual keluar ke kelas lain.
- Precision rendah: banyak prediksi suatu kelas ternyata berasal dari kelas lain.

## Cell 072. Judul export package

Bagian ini mengumpulkan hasil penting ke satu folder dan file ZIP.

## Cell 073. Deskripsi isi package

Cell ini menjelaskan bahwa package berisi checkpoint, feature extractor, konfigurasi, grafik, metadata, dan hasil evaluasi.

## Cell 074. Membuat package v4

Kode membuat folder `download_package_v4`, lalu menyalin:

- Checkpoint final sebagai `ser_wavlm_v4_best.pt`.
- Konfigurasi feature extractor.
- `history_v4.json` jika tersedia.
- `metadata_split_v4.csv` jika tersedia.
- Kurva training.
- Confusion matrix.
- CSV evaluasi test.
- Daftar kandidat label noise jika tersedia.

`config_v4.json` mencatat versi, model source, daftar emosi, durasi, dropout, status Mixup, lineage checkpoint, metrik test, bobot CREMA-D, jumlah kandidat noise, epoch terbaik, validation accuracy, dan test accuracy.

Folder tersebut dikompres menjadi `ser_wavlm_v4_package.zip`.

Catatan penting: package berisi bobot dan konfigurasi, tetapi bukan seluruh runtime Python. Untuk inference tetap dibutuhkan class `WavLMSER`, preprocessing, dependency, dan kode aplikasi yang sesuai.

## Cell 075. Judul inference audio baru

Bagian ini memperagakan penggunaan model terlatih pada file audio baru.

## Cell 076. Deskripsi wrapper inference

Wrapper menerima path audio dan mengembalikan emosi teratas, confidence, serta seluruh probabilitas kelas.

## Cell 077. Fungsi `prediksi_emosi`

Alurnya harus sama dengan preprocessing training:

1. Membaca audio.
2. Silence trimming dan normalisasi melalui `load_waveform`.
3. Crop atau padding menjadi 4 detik.
4. Feature extraction pada 16 kHz.
5. Menambahkan attention mask jika belum tersedia.
6. Menjalankan model pada mode evaluasi.
7. Mengubah logit menjadi probabilitas dengan softmax.
8. Mengambil kelas dengan probabilitas terbesar.

Hasil yang dikembalikan:

- `emosi`: label Bahasa Indonesia.
- `keyakinan`: probabilitas kelas teratas.
- `semua_prob`: probabilitas keenam kelas.

Model juga mengembalikan `attn`, tetapi fungsi ini tidak memvisualisasikannya. Attention pooling tetap menjadi bagian internal model.

Fungsi `tampilkan_hasil` mencetak emosi, keyakinan, ranking probabilitas, dan waveform jika path tersedia.

## Cell 078. Deskripsi simulasi inference

Cell ini menjelaskan demonstrasi prediksi memakai salah satu file dari `df_test`.

## Cell 079. Menjalankan contoh prediksi

Kode memilih baris pertama dari `df_test`, mencetak sumber dan label aktual, lalu memanggil `prediksi_emosi`. Karena file berasal dari test set, hasilnya dapat dibandingkan dengan label aktual.

Satu file hanya demonstrasi. Ia tidak cukup untuk menyimpulkan kualitas keseluruhan model.

## Cell 080. Ringkasan konfigurasi v4

Ringkasan cell ini:

- Backbone: WavLM base-plus.
- Durasi input: 4 detik.
- Pooling: Attentive Stats Pooling dengan output 1536.
- Classifier: LayerNorm, Dropout 0,30, GELU, Dropout 0,225.
- Training: Stage 1 dengan gradual unfreezing, lalu Stage 2 jika syarat noise terpenuhi.
- Augmentasi offline: IndoWaveSentiment dan E-SERAVD pada train saja.
- CREMA-D: tidak diberi augmentasi offline pada baseline.
- Input checkpoint: checkpoint Stage 2 v3 jika ditemukan.
- Output: checkpoint, metrik, heatmap, error analysis, dan package v4.

## Cell 081. Deskripsi FileLink

Cell ini menjelaskan tombol download interaktif pada notebook Jupyter atau Kaggle.

## Cell 082. Membuat link package

`FileLink` membuat link ke `ser_wavlm_v4_package.zip`. Link hanya berfungsi jika file tersebut berada pada lokasi kerja yang dapat diakses notebook.

# Istilah metrik untuk pemula

Misalkan model memprediksi kelas `sedih`.

## Precision

Dari semua audio yang diprediksi sebagai `sedih`, berapa banyak yang benar-benar `sedih`?

Precision rendah berarti model terlalu sering memberi label `sedih` kepada audio dari kelas lain.

## Recall

Dari semua audio yang sebenarnya `sedih`, berapa banyak yang berhasil ditemukan model?

Recall rendah berarti banyak audio `sedih` terlewat dan diprediksi sebagai emosi lain.

## F1-score

F1-score menggabungkan precision dan recall. Nilai ini berguna jika model harus tepat ketika memilih suatu kelas sekaligus tidak melewatkan terlalu banyak anggota kelas tersebut.

## Support

Support adalah jumlah contoh aktual pada kelas tersebut. Metrik dari kelas dengan support kecil dapat berubah besar hanya karena beberapa file.

# Cara menjelaskan arsitektur saat presentasi

Gunakan urutan berikut:

1. Audio diseragamkan menjadi mono, 16 kHz, dan maksimal 4 detik.
2. WavLM base-plus bertindak sebagai representasi suara pretrained yang mengubah waveform menjadi fitur.
3. WavLM menghasilkan fitur pada banyak titik waktu, sehingga Attentive Stats Pooling merangkum bagian yang paling relevan.
4. Mean menangkap karakteristik rata-rata, sedangkan standard deviation menangkap perubahan dinamika suara.
5. Kedua statistik digabung menjadi 1536 fitur.
6. Classifier mengecilkan fitur ke 256 dimensi, lalu menghasilkan enam logit emosi.
7. Focal Loss memberi fokus relatif pada contoh sulit dan class weight membantu menghadapi ketidakseimbangan kelas.
8. Stage 1 melatih model dengan gradual unfreezing.
9. Stage 2 melakukan filtering CREMA-D yang sangat konservatif lalu fine-tuning dengan learning rate kecil.
10. Test accuracy dan classification report dibaca setelah checkpoint final dipilih.

# Batasan yang harus dijelaskan secara jujur

1. v4 dapat melanjutkan checkpoint v3, sehingga bukan selalu training independen dari bobot dasar WavLM.
2. Output tersimpan dapat berasal dari run lama sampai notebook dijalankan ulang.
3. SNR pada EDA adalah estimasi, bukan label kualitas audio yang sempurna.
4. `df['bobot']` divisualisasikan tetapi tidak dipakai sebagai bobot loss per file.
5. Mixup tidak aktif pada baseline v4.
6. Stage 2 melakukan hard drop selektif, bukan downweighting umum untuk semua kandidat noise.
7. Kurva training v4 menyimpan riwayat Stage 1, bukan gabungan lengkap Stage 1 dan Stage 2.
8. Test accuracy adalah metrik final, tetapi satu angka tidak cukup untuk menjelaskan kelas `sedih`, `senang`, `takut`, atau `jijik`.
9. Package ZIP membawa checkpoint dan konfigurasi, tetapi runtime inference tetap membutuhkan kode aplikasi dan dependency yang sesuai.

Dengan batasan ini, presentasi dapat menjelaskan fungsi, alasan, dan konsekuensi setiap keputusan tanpa mengklaim perilaku yang tidak dilakukan kode.
