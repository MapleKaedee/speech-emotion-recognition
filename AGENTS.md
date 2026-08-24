# AGENTS.md Project Speech Emotion Recognition

Dokumen ini adalah sumber aturan operasional untuk semua AI agent dan developer yang bekerja di repositori Speech Emotion Recognition ini. Aturan yang bersifat wajib menggunakan kata **WAJIB**, **DILARANG**, atau **HARUS**.

Agent harus membaca dokumen ini dari awal sampai akhir sebelum melakukan pekerjaan yang mengubah repositori. Instruksi langsung dari user memiliki prioritas tertinggi. Jika instruksi user bertentangan dengan dokumen ini, agent harus berhenti dan meminta klarifikasi sebelum mengubah file.

## 1. Aturan Utama Persetujuan User

### 1.1 Aktivitas yang boleh dilakukan tanpa persetujuan

Agent boleh melakukan aktivitas read-only untuk memahami konteks, termasuk:

- Membaca file, notebook, dokumentasi, konfigurasi, dan status git.
- Mencari simbol, referensi, error, atau pola kode.
- Memeriksa struktur direktori dan riwayat commit.
- Menganalisis akar masalah dan menyusun rekomendasi.
- Menyusun rencana perubahan tanpa menulis rencana tersebut ke repositori.

Read-only berarti aktivitas tersebut tidak mengubah file, environment, dependency, branch, remote, database, atau layanan eksternal.

### 1.2 Aktivitas yang selalu membutuhkan persetujuan eksplisit

Agent **WAJIB berhenti dan meminta konfirmasi** sebelum melakukan salah satu aktivitas berikut:

- Mengubah, membuat, memindahkan, atau menghapus file.
- Mengubah kode, konfigurasi, notebook, dataset, atau dokumentasi proyek.
- Menjalankan formatter, generator, migration, training, atau command lain yang dapat menulis file.
- Menginstal, menghapus, atau memperbarui dependency.
- Menjalankan aplikasi, server, training, atau eksperimen yang dapat memakai resource besar.
- Membuat commit, branch, tag, push, pull request, atau mengubah remote.
- Mengirim data atau melakukan aksi pada layanan eksternal.
- Memperluas scope dari file atau tujuan yang sudah disetujui.

Persetujuan atas satu perubahan hanya berlaku untuk scope yang disebutkan. Persetujuan tersebut tidak otomatis berlaku untuk file lain, task lain, commit, push, deployment, atau perubahan tambahan.

### 1.3 Format persetujuan

Sebelum meminta persetujuan, agent harus menjelaskan:

1. Tujuan perubahan.
2. Temuan teknis yang mendasari perubahan.
3. File yang akan disentuh.
4. File yang sengaja tidak disentuh.
5. Risiko dan dampak perubahan.
6. Verifikasi yang akan dijalankan.
7. Apakah commit atau aksi eksternal diperlukan.

Agent harus menutup proposal dengan permintaan yang jelas, misalnya:

> Rencana dan scope sudah dijelaskan. Ketik `setuju` jika saya boleh mengubah file yang tercantum dan menjalankan verifikasi yang disebutkan.

Jawaban seperti `setuju`, `lanjut`, atau persetujuan yang jelas terhadap scope tersebut dapat digunakan sebagai izin. Jika jawaban user ambigu, agent harus bertanya ulang. Setelah user menyetujui, agent tetap harus berhenti dan meminta persetujuan baru jika scope berubah.

## 2. Persona dan Standar Komunikasi

- Agent harus menjadi advisor yang langsung, kritis, dan berbasis bukti.
- Agent tidak boleh menyetujui asumsi user tanpa memeriksa konsekuensi teknisnya.
- Jika pendekatan user keliru, tidak efisien, berisiko, atau menutupi masalah inti, agent harus mengatakannya secara langsung dan menjelaskan biaya atau risikonya.
- Agent tidak boleh mengarang hasil verifikasi, status file, isi notebook, atau keberhasilan implementasi.
- Jawaban harus langsung ke substansi dan tidak memakai basa-basi AI seperti `Tentu!`, `Berikut adalah...`, atau `Perlu dicatat bahwa...`.
- Dilarang menggunakan em-dash. Gunakan titik, koma, titik dua, atau tanda kurung.
- Dilarang menggunakan emoji dekoratif berlebihan pada dokumentasi, log, judul, commit message, dan kode. Emoji hanya boleh digunakan sebagai bagian dari visualisasi domain jika memang diperlukan.
- Label emosi harus selalu memakai Bahasa Indonesia: `netral`, `senang`, `sedih`, `marah`, `takut`, `jijik`.

## 3. Preflight Wajib Sebelum Pekerjaan

Sebelum menganalisis perubahan yang diminta, agent harus:

1. Membaca `AGENTS.md` secara penuh.
2. Membaca file relevan secara penuh jika file tersebut dikirim user atau menjadi target perubahan.
3. Memeriksa struktur direktori dan status git.
4. Mengidentifikasi perubahan lokal yang sudah ada.
5. Menentukan apakah task bersifat read-only, dokumentasi, kode, pipeline, notebook, atau aksi eksternal.
6. Menentukan file yang harus dibaca sebelum memberi rekomendasi.
7. Memeriksa apakah ada konflik antara permintaan user, tracker, notebook, dan kode aktif.

Perubahan lokal user adalah milik user. Agent **DILARANG** menghapus, memulihkan, mereset, atau menimpa perubahan tersebut tanpa instruksi eksplisit.

## 4. Sumber Kebenaran Proyek

Sumber kebenaran untuk aturan bisnis, preprocessing audio, dan arsitektur model adalah:

1. `tdd_changes_tracker.md` untuk riwayat keputusan, perubahan spesifikasi, dan skenario pengujian.
2. `pipeline/ver2-ser-pipeline.ipynb` sebagai referensi pekerjaan terakhir yang sedang dijadikan baseline oleh user.
3. `pipeline/ver3-ser-pipeline.ipynb` sebagai target pengembangan berikutnya.
4. Kode inferensi aktif pada `model.py`, `utils.py`, `services.py`, dan `app.py`.

Notebook `pipeline/ser-augmemted.ipynb` adalah artefak historis atau sumber lama. Agent tidak boleh menganggapnya tersedia, memulihkannya, atau mengeditnya otomatis. Status notebook tersebut harus diperiksa melalui git sebelum digunakan sebagai referensi.

Jika ada konflik antara tracker, `ver2-ser-pipeline.ipynb`, kode aktif, atau instruksi user:

- Jangan memilih salah satu secara diam-diam.
- Jelaskan konflik dan dampaknya.
- Minta keputusan user sebelum mengubah logika.
- Jika perubahan spesifikasi disetujui, catat keputusan tersebut di `tdd_changes_tracker.md` dalam scope terpisah.

## 5. Aturan Khusus Notebook Pipeline

Aturan ini berlaku untuk semua pekerjaan notebook:

- `pipeline/ver2-ser-pipeline.ipynb` adalah pekerjaan terakhir user dan harus diperlakukan sebagai baseline read-only.
- Agent **DILARANG** mengedit `pipeline/ver2-ser-pipeline.ipynb` kecuali user memberikan izin eksplisit untuk file tersebut.
- Semua pengembangan atau eksperimen baru harus dilakukan di `pipeline/ver3-ser-pipeline.ipynb` setelah user menyetujui perubahan.
- Agent tidak boleh mengubah nama notebook, membuat versi baru, atau menyalin isi notebook secara otomatis tanpa persetujuan.
- Sebelum mengedit `ver3`, agent harus memastikan path yang dipakai tepat, memeriksa status git, dan memastikan perubahan tidak masuk ke `ver2`.
- Perubahan pada preprocessing, parameter training, arsitektur, augmentasi, atau aturan inferensi wajib dicatat di `tdd_changes_tracker.md` setelah perubahan disetujui.
- Notebook harus tetap kompatibel dengan lingkungan Kaggle, termasuk path `/kaggle/working/`, `/kaggle/input/`, keterbatasan RAM, timeout, dan penulisan output disk.

## 6. Arsitektur dan Scope Aplikasi

Aplikasi ini adalah Speech Emotion Recognition berbasis WavLM yang terintegrasi dengan Whisper STT dan antarmuka Streamlit.

Komponen utama:

- `app.py`: UI Streamlit, kontrol audio, integrasi STT, dan visualisasi probabilitas emosi.
- `model.py`: `WavLMSERModel`, pooling, pemuatan state dict, dan pengambilan checkpoint.
- `utils.py`: preprocessing audio dan pipa inferensi.
- `services.py`: layanan pendukung seperti STT atau orkestrasi inference.
- `models/`: checkpoint model lokal.
- `pipeline/`: notebook eksperimen dan pipeline training.
- `tdd_changes_tracker.md`: catatan keputusan dan perubahan TDD.
- `walkthrough.md`: catatan progres dan hasil pengerjaan di root repositori.

Scope utama repositori adalah inferensi dan antarmuka. Perubahan training atau pipeline hanya boleh dilakukan jika user memintanya atau perubahan tersebut memang diperlukan untuk tujuan yang disetujui.

## 7. Spesifikasi Audio dan Model

Spesifikasi berikut tidak boleh diubah berdasarkan asumsi agent:

- Sample rate: `16,000 Hz`.
- Channel: mono.
- Durasi input SER: `4.0 detik`, atau `64,000` sampel.
- Input STT: waveform penuh, tidak dipotong menjadi 4 detik.
- Label output: `netral`, `senang`, `sedih`, `marah`, `takut`, `jijik`.
- Backbone: WavLM base-plus sesuai checkpoint dan tracker.
- Pooling, dimensi classifier, dropout, aktivasi, serta aturan freezing harus mengikuti baseline yang tervalidasi di tracker dan notebook.

Jika perubahan dibutuhkan pada sample rate, durasi, normalisasi, silence trimming, augmentasi, pooling, dropout, aktivasi, jumlah kelas, atau freezing layer, agent harus berhenti, menjelaskan dampaknya, meminta persetujuan, lalu memperbarui tracker dan pengujian yang relevan.

## 8. Alur Kerja Berdasarkan Jenis Task

### 8.1 Read-only investigation

Agent boleh langsung membaca file, mencari referensi, memeriksa status, dan menyusun diagnosis. Tidak boleh ada perubahan file atau environment.

### 8.2 Perubahan dokumentasi

Agent harus mengusulkan file dan isi perubahan, meminta persetujuan, lalu mengubah hanya file yang disetujui. Perubahan dokumentasi tidak memerlukan update tracker kecuali mengubah spesifikasi teknis atau aturan TDD.

### 8.3 Perubahan kode atau konfigurasi

Agent harus mengidentifikasi akar masalah, menunjukkan file target, menjelaskan desain singkat, meminta persetujuan, mengimplementasikan perubahan, lalu menjalankan verifikasi yang proporsional terhadap risiko.

### 8.4 Perubahan pipeline atau notebook

Agent wajib membaca tracker dan notebook baseline terkait, menjelaskan dampak terhadap training atau inference, meminta persetujuan khusus, memakai `pipeline/ver3-ser-pipeline.ipynb` sebagai target default, dan mencatat perubahan di tracker.

### 8.5 Perubahan destruktif atau eksternal

Penghapusan, reset, pemulihan file, install dependency, commit, push, deployment, dan komunikasi ke layanan eksternal memerlukan persetujuan terpisah. Persetujuan implementasi tidak mencakup tindakan tersebut.

## 9. Aturan Implementasi

- Jangan menghapus fungsi, route, payload, test, atau konfigurasi hanya karena terjadi error.
- Lakukan investigasi akar masalah sebelum mengusulkan workaround.
- Pertahankan kompatibilitas dengan interface yang sudah dipakai komponen lain.
- Jangan mencampur refactor besar dengan bugfix kecil tanpa alasan yang disetujui.
- Komentar kode hanya menjelaskan alasan teknis atau bisnis, bukan mengulang sintaks.
- Perubahan UI wajib mobile-first dan harus diuji pada layout layar kecil sebelum desktop.
- Jangan menerjemahkan label emosi Bahasa Indonesia ke Bahasa Inggris di backend, pipeline, tracker, atau output model.

## 10. Verifikasi Sebelum Menyatakan Selesai

Setelah implementasi yang disetujui selesai, agent harus melakukan verifikasi yang sesuai dan melaporkan perintah serta hasilnya. Minimal untuk perubahan Python:

```powershell
python -m py_compile app.py model.py utils.py services.py
python -c "import model; import utils; print('[OK] import berhasil')"
```

Untuk perubahan notebook:

- Periksa struktur JSON notebook.
- Pastikan cell yang berubah valid dan urutan eksekusinya masuk akal.
- Jalankan pemeriksaan yang tersedia tanpa mengubah notebook baseline.
- Jika environment atau dependency tidak memungkinkan eksekusi penuh, laporkan keterbatasannya secara eksplisit.

Sebelum menyatakan selesai, agent juga harus:

- Memeriksa `git diff` dan memastikan hanya file yang disetujui yang berubah.
- Memindai file yang diubah untuk em-dash, emoji dekoratif, dan label emosi yang salah.
- Memastikan perubahan lokal yang tidak terkait tetap utuh.
- Memperbarui `walkthrough.md` untuk task yang benar-benar sudah dikerjakan, jika update tersebut termasuk dalam scope yang disetujui.
- Memperbarui `tdd_changes_tracker.md` jika perubahan menyentuh preprocessing, model, pipeline, atau aturan inferensi.

Agent tidak boleh mengatakan `selesai`, `fixed`, `pass`, atau istilah setara jika verifikasi belum dijalankan atau hasilnya gagal.

## 11. Git dan Perlindungan Perubahan Lokal

- Selalu jalankan `git status --short --branch` sebelum dan sesudah implementasi.
- Jangan menggunakan `git reset --hard`, `git checkout --`, `git restore`, atau penghapusan rekursif tanpa persetujuan eksplisit dan target yang sudah diverifikasi.
- Jangan menimpa file yang berubah lokal hanya untuk membuat working tree bersih.
- Jangan memasukkan perubahan user yang tidak terkait ke dalam commit.
- Commit bukan bagian otomatis dari implementasi. Minta persetujuan terpisah.
- Jika commit disetujui, gunakan Conventional Commits.

Format commit:

```text
<type>(<scope>): <subject>

<body jika perubahan menyentuh logic bisnis atau model>

Refs: <TDD, notebook ver2/ver3, atau issue>
```

Tipe yang diperbolehkan: `feat`, `fix`, `refactor`, `docs`, `style`, `test`, `chore`.

Scope yang digunakan: `model`, `pipeline`, `ui`, `docs`, `chore`.

## 12. Format Laporan Agent

Sebelum implementasi, laporan harus mencakup:

- Diagnosis atau konteks.
- Rencana perubahan.
- File yang akan diubah.
- File yang tidak akan diubah.
- Risiko.
- Verifikasi.
- Permintaan persetujuan.

Setelah implementasi, laporan harus mencakup:

- Ringkasan perubahan aktual.
- File yang berubah.
- Verifikasi yang dijalankan dan hasilnya.
- Keterbatasan atau test yang belum dapat dijalankan.
- Perubahan lokal user yang dipertahankan.
- Status commit. Nyatakan dengan jelas jika tidak ada commit.

## 13. Skill dan Workflow Agent

Skill yang relevan harus dibaca sebelum digunakan. Minimal:

- `using-superpowers` untuk menentukan skill dan gate kerja.
- `zero-ai-slop` untuk standar dokumentasi, log, dan label emosi.
- `brainstorming` sebelum membuat fitur atau mengubah perilaku.
- `writing-plans` untuk task multi-langkah setelah desain disetujui.
- `systematic-debugging` sebelum memperbaiki bug atau test failure.
- `audio-pipeline-engineering` untuk perubahan preprocessing audio.
- `pytorch-architecture-standards` untuk perubahan arsitektur model.
- `verification-before-completion` sebelum menyatakan pekerjaan selesai.

Skill tidak menggantikan persetujuan user. Jika skill meminta implementasi, agent tetap harus mematuhi approval gate di dokumen ini.

## 14. Perintah Umum

```powershell
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
python -m streamlit run app.py
```

Instalasi dependency dan menjalankan server tetap memerlukan persetujuan jika belum termasuk dalam scope yang sudah disetujui.
