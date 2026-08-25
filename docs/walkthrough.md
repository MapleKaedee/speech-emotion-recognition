# Walkthrough

## [2026-08-24] Peningkatan Sensitivitas Transkripsi Whisper

Transkripsi Whisper sebelumnya memakai waveform mono 16 kHz tanpa normalisasi level suara. Pada rekaman mikrofon yang pelan atau memiliki nilai non-finite, kondisi ini dapat membuat ucapan pendek lebih mudah salah dikenali. Konfigurasi decoding juga hanya menetapkan bahasa dan task.

Perbaikan dilakukan khusus pada jalur STT di `utils.py`. Audio penuh tetap dipertahankan, lalu nilai NaN dan infinity disanitasi, resampling dilakukan ke 16 kHz, dan peak normalization diterapkan tanpa crop 4 detik. SER tidak berubah karena kontrak preprocessing SER tetap terpisah.

Whisper sekarang memakai beam search dan parameter decoding yang lebih stabil untuk keluaran ambigu: temperature fallback, compression ratio threshold, logprob threshold, dan `condition_on_prev_tokens=False`. Konfigurasi ini digunakan baik untuk transkrip penuh maupun transkrip bertimestamp per segmen.

Perilaku diuji dengan tiga unit test menggunakan fake pipeline. Test memastikan waveform tetap penuh, hasil resampling finite dan ternormalisasi, serta parameter decoding sampai ke pipeline Whisper. Rekaman asli tidak tersedia, jadi hasil kalimat spesifik seperti "ini kok gitu sih" belum dapat dibandingkan langsung.

## [2026-08-24] Preload Whisper Small

- Whisper dikonfigurasi konsisten sebagai `openai/whisper-small`.
- Pipeline Whisper dipreload ketika aplikasi mulai dan dicache dengan `st.cache_resource`.
- Preprocessing STT memakai audio penuh, mono, dan 16 kHz. Audio tidak dipaksa menjadi 4 detik karena 4 detik hanya kontrak model SER.
- Analisis segmen memakai timestamp Whisper, melewati segmen kurang dari 0.35 detik, dan membatasi maksimal 20 segmen.

## [2026-08-24] Streamlit Menggunakan Checkpoint Hugging Face v4

### Perubahan

- `model.py` sekarang mengambil `ser_wavlm_v4_best.pt` dari `elnathzzz/wavlm-ser-multilingual` melalui Hugging Face Hub.
- `utils.py` menyamakan preprocessing inferensi dengan `pipeline/ver4-ser-pipeline.ipynb`: resample 16 kHz, trim silence 30 dB, peak normalization, crop dari awal ke 4 detik, dan zero-padding.
- `services.py` serta `config.py` tidak lagi menunjuk ke namespace checkpoint v7.
- `requirements.txt` memakai `huggingface_hub` dan tidak lagi memerlukan `gdown` untuk checkpoint SER.
- `model.py` memakai `AutoConfig` dan `AutoModel`, serta dropout pooling `DROPOUT * 0.5` seperti notebook v4.
- `utils.py` memakai `librosa.resample` agar resampling inferensi mengikuti preprocessing notebook v4.

### Verifikasi dan Batasan

- Arsitektur checkpoint tetap divalidasi dengan `strict=True` saat state dict dimuat.
- Artefak metrik v4 harus tersedia di folder `models/` agar dashboard metrik dapat menampilkannya.
- Unduhan model memerlukan akses jaringan ke Hugging Face saat checkpoint belum tersimpan lokal.

## [2026-08-13] Fix Bug Kritis: Iframe Recorder Membengkak Tanpa Henti

### Konteks

User tes manual fitur rekam (sesi sebelumnya) dan lapor: tidak ada preview hasil rekam, tidak ada tombol "Analisis Emosi", cuma tombol "Rekam" yang tersisa. Sesi sebelumnya sempat melihat gejala serupa lewat Playwright tapi keliru menyimpulkannya sebagai flakiness alat tes - ternyata bug nyata di komponen custom recorder.

### Root Cause (dikonfirmasi lewat pengukuran langsung, bukan tebakan)

1. Verifikasi protokol `components/recorder_frontend/index.html` terhadap source JS asli Streamlit (`ComponentInstance.*.js` di paket terpasang) -> implementasi `componentReady`/`setComponentValue`/`setFrameHeight` sudah 100% sesuai spec.
2. Diukur langsung atribut `height` elemen `<iframe>` komponen selama beberapa detik: **tumbuh tanpa henti**, ~2.756px -> 63.196px dalam 4.5 detik (~7000px per 0.5 detik).
3. Direproduksi & dikonfirmasi dalam isolasi: menghapus SATU baris `setFrameHeight()` yang dipanggil di dalam listener `"streamlit:render"` (event yang Streamlit kirim ke SETIAP komponen pada SETIAP rerun script) membuat tinggi iframe stabil sempurna.

**Mekanisme**: Streamlit mengirim `"streamlit:render"` ke komponen pada tiap rerun untuk sinkronisasi args/tema (perilaku normal). Kode lama memanggil `setFrameHeight()` di situ, dan melaporkan tinggi frame ke Streamlit rupanya memicu Streamlit merender ulang lagi -> loop umpan balik tak berujung, membengkak tiap siklus. Begitu iframe setinggi puluhan ribu piksel, seluruh layout sidebar rusak - persis gejala yang dilaporkan user, dan penyebab flakiness Playwright yang sempat salah didiagnosis sebagai masalah alat tes di sesi sebelumnya.

### Perbaikan

- `components/recorder_frontend/index.html`: hapus panggilan `setFrameHeight()` dari listener `"streamlit:render"` (komentar `ponytail:` ditambahkan menjelaskan kenapa).
- **Efek samping yang perlu diperbaiki juga**: setelah baris itu dihapus, ternyata satu-satunya pengukuran tinggi awal (di `window.load`) terjadi TERLALU DINI (sebelum browser selesai melukis tombol), jadi `scrollHeight` terukur 0 dan tombol jadi kepotong/nyaris tak terlihat (tinggi iframe cuma 4px). Diperbaiki dengan menunggu `requestAnimationFrame` ganda sebelum panggilan `setFrameHeight()` pertama, supaya browser sudah selesai melukis dulu.

### Verifikasi

- Pengukuran tinggi iframe berulang selama 4.5 detik setelah pindah ke mode "Rekam Mikrofon" (sebelum klik rekam) -> stabil di 72px, tidak tumbuh (sebelumnya tumbuh tanpa henti).
- Screenshot mengonfirmasi tombol "Rekam" tampil utuh dan proporsional, bukan kepotong.
- Playwright pada app sungguhan, diulang beberapa kali independen (browser baru tiap kali, meniru sesi user sungguhan): rekam -> auto-stop di 4 detik -> metadata, waveform, tombol "Analisis Emosi" muncul KONSISTEN (sebelumnya cuma kadang berhasil) -> klik "Analisis Emosi" -> hasil analisis lengkap tampil (mis. "Senang" 81.2%, 77.0% pada run berbeda).
- Path durasi minimum diuji ulang pada app sungguhan (bukan cuma prototipe standalone seperti sesi sebelumnya): rekam lalu stop manual di ~0.18 detik -> pesan "Rekaman terlalu pendek (0.18 dtk). Minimal 0.4 detik - silakan ulangi rekam." muncul dengan benar, konten utama tetap di empty-state (tidak lanjut ke metadata).
- Frame-detection Playwright yang sebelumnya sering gagal (20+ detik timeout) sekarang berhasil ditemukan segera pada mayoritas run setelah fix - mengonfirmasi bug height inilah akar penyebab flakiness yang sempat salah didiagnosis di sesi sebelumnya, bukan keterbatasan alat tes.

### Yang Tidak Diubah

- Logic auto-stop, validasi durasi minimum, encode WAV di sisi browser - semua sudah terbukti benar secara independen dari bug ini, tidak disentuh.
- `components/recorder.py`, `config.py`, `pages/analisis.py` - tidak ada perubahan.

### Pelajaran untuk Sesi Berikutnya

- Kalau menulis custom Streamlit component dengan protokol postMessage manual: **jangan pernah panggil `setFrameHeight()` di dalam listener `"streamlit:render"`** - itu event yang dikirim berulang pada tiap rerun, bukan event satu kali. Hanya panggil saat konten benar-benar berubah ukurannya (load awal, mulai/selesai aksi user), dan untuk pengukuran pertama, tunggu render browser selesai dulu (`requestAnimationFrame`) supaya tidak salah ukur 0.

## [2026-08-13] Custom Mic Recorder: Auto-Stop 4 Detik + Minimum 0.4 Detik

### Konteks

User minta aturan baru untuk rekam suara langsung di halaman Analisis: rekaman otomatis berhenti begitu mencapai 4 detik, dan durasi di bawah 0.4 detik harus mengulang rekam. `st.audio_input` (widget rekam bawaan Streamlit sebelumnya) murni dikontrol browser, tidak ada API Python untuk auto-stop. Dikonfirmasi ke user: mereka pilih "auto-stop asli di browser" meski disclose butuh effort lebih besar & kemungkinan dependency baru.

### Riset Dependency (sebelum membangun apa pun)

- `sound-streamlit` (py `AudioWidget`, ada param `min_duration`/`max_duration`) - didiskualifikasi: rilis terakhir Des 2023, repo GitHub 404, dan `requires_dist` mem-pin `streamlit==1.29.0` persis - akan bentrok dan berpotensi merusak Streamlit 1.61.1 yang dipakai project ini (navbar `position="top"`, Material Symbols, dll butuh versi lebih baru).
- `streamlit-mic-recorder` - dependency aman (`streamlit>=0.63`), tapi tidak ada parameter durasi maksimum sama sekali (dicek dari README & source GitHub langsung).
- Tidak ada package siap pakai yang aman + mendukung fitur ini.

### Solusi: Custom Component "No-Build"

`st.components.v1.declare_component` menunjuk ke `components/recorder_frontend/index.html` statis (HTML+JS biasa, TANPA React/npm/webpack) yang mengimplementasikan protokol `postMessage` Streamlit secara manual (`streamlit:componentReady`, `streamlit:setComponentValue`, `streamlit:setFrameHeight`). **Nol dependency Python/pip/sistem baru.**

**Masalah kritis yang ditemukan saat prototyping, dan cara mengatasinya**: `MediaRecorder` browser cuma bisa menghasilkan WebM/Opus, tidak bisa WAV langsung. Dites langsung: pipeline `utils.py::load_audio` (torchaudio → librosa → soundfile) gagal total membaca WebM di environment ini (tidak ada `ffmpeg`, `torchaudio` versi terpasang butuh `torchcodec` yang juga tidak ada). Solusi: decode + re-encode ke WAV asli 100% di sisi BROWSER pakai Web Audio API (`AudioContext.decodeAudioData` + penulisan header WAV manual di JS) sebelum dikirim ke Python - jadi Python selalu terima WAV asli, `utils.py` tidak perlu diubah sama sekali. Dites langsung: WAV hasil browser berhasil diproses oleh `utils.load_audio()`/`get_audio_info()` yang sudah ada, tanpa modifikasi.

### Perubahan

- `components/recorder_frontend/index.html` (baru): tombol rekam bergaya monokrom (warna disamakan manual dengan token project karena iframe terisolasi dari CSS utama), counter durasi berjalan + progress bar saat merekam, auto-stop via `setTimeout` di durasi maksimum (dikirim dari Python lewat `args`, bukan hardcode), penolakan durasi minimum, decode+encode WAV di `onstop`.
- `components/recorder.py` (baru): `record_audio(max_seconds, min_seconds, key)` - wrapper tipis yang decode base64 jadi `io.BytesIO` dengan `.name`/`.size` di-set manual, kompatibel penuh dengan interface yang sudah dipakai `pages/analisis.py` untuk `audio_file` (sama seperti `UploadedFile` dari `st.file_uploader`/`st.audio_input` sebelumnya) - jadi TIDAK perlu ubah `_audio_key`, `get_audio_info`, atau pipeline prediksi.
- `config.py`: tambah `MIN_RECORD_DURATION_SECONDS = 0.4`, sengaja terpisah dari `utils.MIN_DURATION_SECONDS` (0.35, constant preprocessing MODEL, bagian kontrak TDD v7 - tidak boleh diutak-atik tanpa proses AGENTS.md). Constant baru ini murni aturan UX rekam. Durasi maksimum reuse `utils.MAX_DURATION_SECONDS` (4.0) langsung, tidak duplikasi angka.
- `pages/analisis.py`: ganti pemanggilan `st.audio_input(...)` dengan `record_audio(max_seconds=MAX_DURATION_SECONDS, min_seconds=MIN_RECORD_DURATION_SECONDS, key="recorded_audio")`, tambah caption penjelasan aturan durasi.

### Verifikasi

- `python -m py_compile` untuk semua file baru/diubah -> bersih.
- `streamlit.testing.v1.AppTest` untuk kelima halaman termasuk Analisis dalam mode "Rekam Mikrofon" (custom component ter-render) -> tanpa exception.
- **Prototipe standalone** (sebelum integrasi, di luar project): dites lewat Playwright dengan Chromium sungguhan + fake mic device (`--use-fake-device-for-media-stream`) - auto-stop tepat di 4000ms terkonfirmasi by measurement, rekaman di bawah 400ms terkonfirmasi ditolak (`{error: "too_short", duration_ms: 164}`), WAV hasil decode berhasil diproses `utils.load_audio()` project tanpa modifikasi.
- **App sungguhan (setelah integrasi)**: dites lewat Playwright pada server lokal - klik rekam, tunggu lewat 4 detik tanpa stop manual -> rekaman berhenti sendiri, metadata + waveform + tombol muncul di sidebar, klik "Analisis Emosi" -> hasil analisis lengkap tampil di konten utama (emosi "Senang" 75.2%, transkrip STT berjalan) - audio dari mikrofon browser sungguhan berhasil diproses model end-to-end.
- Sempat menemukan flakiness murni di tooling tes (Playwright frame-detection kadang lambat mendeteksi iframe component, dan `page.screenshot()` beberapa kali menangkap frame basi/stale dibanding DOM sungguhan) - dikonfirmasi lewat `.inner_text()` (query DOM langsung, bukan screenshot visual) dan print debug server-side bahwa data SELALU benar di baliknya; ini keterbatasan alat verifikasi di environment sandbox ini, bukan bug aplikasi.

### Yang Tidak Diubah

- `utils.py` (pipeline decode/preprocess) - tidak disentuh sama sekali.
- `requirements.txt` - tidak ada dependency baru.
- Alur upload file (`st.file_uploader`) - cuma jalur rekam mikrofon yang berubah.

### Follow-up yang Disarankan

- Verifikasi durasi minimum (0.4 detik) di real app belum sempat direkam-ulang lewat Playwright karena flakiness tooling (sudah kuat divalidasi di prototipe standalone dengan kode identik) - kalau ingin extra yakin, coba manual di browser sungguhan.
- Kalau ke depan tombol rekam custom ini butuh polish visual lebih jauh (mis. animasi, ikon berbeda saat idle/recording), style-nya ada di `components/recorder_frontend/index.html`, terpisah dari `components/css.py` karena keterbatasan isolasi iframe.

## [2026-08-13] Rombak Dashboard: Dari Ringkasan Statis Jadi Status Operasional

### Konteks

User menilai `pages/dashboard.py` (distribusi dataset + kurva training/confusion matrix) redundant dengan `pages/dataset.py` dan `pages/model.py` yang sudah punya versi lengkapnya. Minta Dashboard diubah jadi halaman yang menunjukkan "kehidupan" aplikasi: status sistem, aktivitas analisis terbaru, dsb.

Dikonfirmasi ke user keterbatasan teknis: Streamlit tidak punya database, `st.session_state` murni per-sesi/per-tab (tidak dibagi antar pengguna). User memilih opsi ringan: "Aktivitas" cukup dari sesi berjalan saat ini (reuse `st.session_state["prediction_history"]` yang sudah dipakai sidebar Analisis), ditampilkan jujur sebagai "Aktivitas Sesi Ini" - bukan klaim aktivitas semua pengguna. Tidak menambah infrastruktur penyimpanan baru (sesuai juga dengan "Inference Only Scope" di AGENTS.md).

### Perubahan

- `pages/analisis.py`: tambah satu baris counter `st.session_state["session_analysis_count"]` di titik yang sama dengan `history.insert(...)`. Perlu karena `prediction_history` dibatasi 10 entri (`del history[10:]`), jadi tidak bisa dipakai untuk angka "Total Analisis Sesi Ini" yang akurat kalau user analisis >10 kali dalam satu sesi.
- `pages/dashboard.py`: rombak total. Dihapus: grafik distribusi dataset, gambar kurva training/confusion matrix, expander konfigurasi model (semua sudah ada di halaman Dataset/Model). Diganti:
  - **Status Sistem**: strip operasional ringkas (Model Siap/Gagal via `check_model_ready`, Perangkat CPU/GPU, plus kuota cloud sesi ini kalau `IS_CLOUD`) - sengaja dibuat lebih ringkas dari stat-grid Home supaya tidak terasa pengulangan, framing-nya "apakah sistem siap dipakai sekarang" bukan "tentang project".
  - **Aktivitas Sesi Ini**: kalau kosong, `st.info` + `st.page_link` ke Analisis (bukan reuse `render_empty_state()` karena copy-nya spesifik soal upload audio, tidak pas untuk konteks "belum ada aktivitas"). Kalau ada: 2 `st.metric` (Total Analisis Sesi Ini, Emosi Terbanyak lewat `collections.Counter`) + tabel `st.dataframe` dari `prediction_history` (Waktu/File/Emosi/Confidence). Pakai `st.dataframe` native, bukan reuse `render_history_list` dari `components/ui.py`, karena fungsi itu didesain khusus kolom sidebar sempit - tabel native lebih pas untuk konten utama yang lebar.
  - Quick links ke Model/Dataset dipertahankan (masih navigasi berguna, bukan duplikasi konten), icon-nya disamakan ke Material Symbols mengikuti gaya navbar baru.

### Verifikasi

- `python -m py_compile` untuk `pages/dashboard.py` dan `pages/analisis.py` -> bersih.
- `streamlit.testing.v1.AppTest` lewat `app.py` untuk kelima halaman -> tanpa exception, termasuk Dashboard versi kosong (session state kosong, seperti sesi baru).
- Playwright pada app sungguhan: jalankan 1 analisis nyata di halaman Analisis, buka Dashboard -> Status Model "Siap", Perangkat "GPU (CUDA)" (device sungguhan di environment ini), Total Analisis Sesi Ini "1", Emosi Terbanyak sesuai hasil prediksi, tabel aktivitas menampilkan waktu/nama file/emosi/confidence yang benar. Dicek juga versi kosong di sesi baru (browser context terpisah) -> pesan "Belum ada analisis..." + tombol "Mulai analisis pertama →" tampil rapi, tidak blank.

### Yang Tidak Diubah

- `pages/dataset.py`, `pages/model.py`, `pages/home.py` - tidak disentuh.
- Tidak ada penyimpanan/log persisten lintas sesi baru.
- Logic inferensi/prediksi.

### Follow-up yang Disarankan

- Kalau nanti user mau "aktivitas" benar-benar lintas sesi/pengguna (bukan cuma sesi browser saat ini), perlu penyimpanan persisten (file log atau DB ringan) - sudah didiskusikan tapi sengaja tidak dikerjakan sesi ini karena filesystem Streamlit Cloud gratis bisa reset saat redeploy.

## [2026-08-13] Navbar: Menu Individual Icon-Only dengan Material Symbols

### Konteks

User minta 4 perubahan pada navbar top hasil redesign sebelumnya: (1) hapus dropdown grup jadi item individual, (2) menu icon-only (teks disembunyikan), (3) icon elegan/simple yang selaras warna background monokrom (ganti dari emoji berwarna), (4) urutan Home, Analisa Suara, Dashboard, Model, Dataset.

### Riset & Keputusan Desain

- `st.navigation()` menerima `pages` sebagai list datar (bukan dict berkelompok) - dengan `position="top"`, list datar menghasilkan item individual tanpa dropdown sama sekali. Dikonfirmasi lewat Playwright sebelum implementasi (bukan asumsi dari dokumentasi saja).
- Streamlit 1.61.1 mendukung Material Symbols lewat `icon=":material/nama:"` (dikonfirmasi ada di `material_icon_names.py` paket terpasang). Icon ini SVG/font yang otomatis ikut warna teks CSS, jadi selaras tema monokrom tanpa aset baru: `home` (Home), `mic` (Analisis Emosi), `dashboard` (Dashboard), `psychology` (Model, ikon otak), `dataset` (Dataset).
- Teks label tiap item nav ada di `span[label]` terpisah dari `span[data-testid="stIconMaterial"]` (dicek lewat inspeksi DOM sungguhan) - bisa disembunyikan spesifik lewat CSS tanpa menyentuh icon-nya. `title=` di `st.Page()` tetap diisi (dipakai Streamlit sebagai identitas halaman), cuma disembunyikan visual.

### Perubahan

- `app.py`: `pages` diubah dari dict berkelompok jadi list datar 5 halaman, urutan Home/Analisis/Dashboard/Model/Dataset, semua icon jadi `:material/...:`.
- `components/css.py`: tambah rule `a[data-testid="stTopNavLink"] span[label] { display: none; }` untuk sembunyikan teks, styling warna icon pakai token yang sudah ada (`--text-tertiary` untuk item biasa, `--text-primary` untuk halaman aktif via `[aria-current="page"]`).

### Verifikasi

- `python -m py_compile` dan `streamlit.testing.v1.AppTest` (kelima halaman) -> bersih, tanpa exception.
- Playwright terhadap app sungguhan: 5 item nav individual terkonfirmasi (0 elemen dropdown), urutan icon terbaca `['home', 'mic', 'dashboard', 'psychology', 'dataset']` sesuai permintaan, klik tiap icon berhasil navigasi ke halaman yang benar, state aktif (icon putih di atas pill terang) beda dari state biasa (abu-abu).
- Sempat curiga ada bug render (icon "mic" terlihat seperti ikon panah-turun/tray di screenshot resolusi biasa) - ternyata cuma keterbatasan resolusi screenshot; di-crop & di-zoom pada `device_scale_factor=3` terkonfirmasi itu memang ikon mikrofon yang benar, bukan bug.

### Yang Tidak Diubah

- Emoji fungsional lain di aplikasi (`EMOTION_ICONS` di `config.py`, icon di kartu hasil analisis) - bukan bagian dari navigasi, di luar permintaan.
- Tidak ada tooltip/aria-label pengganti teks yang disembunyikan (di luar permintaan; Streamlit tidak menyediakan cara menambah atribut HTML kustom tanpa komponen custom).

### Follow-up yang Disarankan

- Cek tampilan navbar icon-only ini di breakpoint mobile/layar kecil (belum diuji viewport sempit di sesi ini, sementara AGENTS.md mensyaratkan mobile-first).

## [2026-08-13] Fix Waveform Chart Tidak Muncul + Verifikasi Visual dengan Playwright

### Konteks

User melaporkan grafik "Bentuk Gelombang" di sidebar Analisis tidak muncul, dan mengizinkan menambahkan browser headless ke environment kerja kalau memang dibutuhkan untuk verifikasi visual (redesign IA sebelumnya sempat menandai ini sebagai item yang tidak bisa diverifikasi karena tidak ada browser).

### Root Cause

Bukan bug logic Python maupun CSS proyek ini. Ditemukan lewat investigasi bertahap dengan Playwright (screenshot + inspeksi SVG langsung, dibandingkan di app bare tanpa CSS proyek sama sekali untuk isolasi):

- `utils.get_waveform_envelope()` sehat (data non-empty, rentang nilai wajar) - bukan di situ masalahnya.
- `st.line_chart(envelope_df, height=100, color=[...])` di `components/ui.py::render_waveform_chart` adalah akar masalahnya: pada Streamlit 1.61.1, memberi parameter `height` eksplisit yang kecil membuat area plot vertikal kolaps karena "chrome" (axis + legend) menghabiskan hampir seluruh tinggi yang dialokasikan. Dibuktikan dengan mengukur langsung koordinat Y pada SVG yang dirender: `height=100` -> rentang Y garis = 0px (benar-benar rata/tidak kelihatan), `height=140` (nilai SEBELUM redesign sesi ini) -> cuma ~11px dari 140px (nyaris tidak kelihatan juga), `height=220` -> ~36px rentang Y (jelas kelihatan), default (tanpa `height`) -> ~76px dari 350px.
- Bug ini sudah ada SEBELUM sesi redesign kemarin (height lama 140 sudah di ambang batas nyaris-tidak-kelihatan) - perubahan sesi kemarin (140 -> 100) mengubahnya dari "nyaris tidak kelihatan" jadi "benar-benar hilang".

### Perbaikan

- `components/ui.py::render_waveform_chart`: `height` dinaikkan dari `100` ke `220`, dengan komentar `ponytail:` menjelaskan kenapa nilai kecil tidak aman dan titik amannya di mana.

### Tooling Ditambahkan

- Playwright (`pip install playwright` + `playwright install chromium`) dipasang di `.venv` lokal, **bukan** di `requirements.txt` - murni alat verifikasi visual untuk sesi kerja, bukan dependency runtime aplikasi (lihat AGENTS.md 4.1, Inference Only Scope).
- Chromium butuh library sistem (`libnspr4.so` dkk.) yang tidak ada di Arch Linux secara default. `playwright install-deps` bawaan cuma support Debian/Ubuntu/Fedora (pakai `apt-get`, tidak ada di Arch). User menginstal manual lewat `pacman -S nss nspr at-spi2-core libcups libdrm mesa libxkbcommon libxcomposite libxdamage libxfixes libxrandr gtk3 pango cairo alsa-lib` di terminal asli mereka (sesi ini tidak punya TTY interaktif untuk prompt password `sudo`).

### Verifikasi

- Alur penuh diuji lewat Playwright: upload file audio sungguhan (file WAV sintetis dengan amplitude bervariasi, bukan nada murni konstan yang ternyata kasus degenerate) ke `st.file_uploader` di sidebar (Playwright bisa simulasikan upload file browser sungguhan, beda dengan `AppTest` yang tidak bisa), klik "Analisis Emosi", tunggu hasil.
- Konfirmasi visual: waveform (garis Puncak putih + Lembah abu-abu) tampil jelas di sidebar setelah perbaikan; screenshot sebelum vs sesudah dibandingkan langsung.
- Konfirmasi hasil analisis lengkap tampil di konten utama: result card, transkrip, Top 3 Emosi, confidence bars, tombol export, expander Detail Teknis - menutup item verifikasi "alur upload sampai hasil belum diverifikasi end-to-end" dari redesign sebelumnya.
- Sekalian ditutup: navbar top (grup Beranda/Analisis/Insight, termasuk dropdown "Analisis" yang perlu diklik dulu untuk membuka) dan layout halaman Home terkonfirmasi tampil benar lewat screenshot - item yang sebelumnya juga ditandai belum diverifikasi.
- Sempat salah duga ada bug kedua ("hasil analisis tidak muncul, tetap menampilkan empty-state walau audio sudah di-upload") - ternyata false alarm dari metodologi tes sendiri: `render_empty_state()` dipanggil dari 2 tempat berbeda dengan pesan yang SAMA (belum ada audio sama sekali, vs audio ada tapi belum di-klik Analisis), dan skenario tes awal memang belum pernah klik tombol "Analisis Emosi". Dikonfirmasi lewat debug print sementara di server (dihapus lagi setelah terbukti) yang menunjukkan `audio_file` sudah terisi benar di run yang bersangkutan. Dicatat sebagai potensi perbaikan copy kecil di masa depan (bedakan pesan "belum ada audio" vs "audio siap, klik Analisis Emosi"), bukan bug.
- `python -m py_compile` dan `streamlit.testing.v1.AppTest` (kelima halaman) dijalankan ulang setelah semua perubahan -> bersih, tanpa exception.

### Yang Tidak Diubah

- Tidak ada perubahan pada logic inferensi/model.
- `requirements.txt` tidak disentuh (Playwright sengaja tidak jadi dependency project).

### Follow-up yang Disarankan

- Pertimbangkan pesan empty-state yang berbeda untuk "belum ada audio" vs "audio sudah dipilih, klik Analisis Emosi di sidebar" supaya tidak membingungkan (ditemukan saat verifikasi sesi ini, bukan diminta user).
- Kalau ke depan mau QA visual jadi bagian rutin (bukan cuma ad-hoc), pertimbangkan skrip Playwright permanen di repo (di luar scope sesi ini per instruksi user).

## [2026-08-13] Redesign IA: Navbar, Sidebar Analisa Suara, dan Halaman Home

### Konteks

Lanjutan redesign UI. Permintaan user kali ini soal struktur navigasi, bukan visual/CSS: (1) menu sidebar diubah jadi navbar atas, (2) sidebar dipakai ulang sebagai panel kontrol analisa suara, (3) sidebar itu cuma muncul di halaman Analisis, (4) hasil analisis ditampilkan detail di halaman Analisis, (5) info statis (judul project, peta emosi, stat model) pindah ke halaman Home baru, (6) Home jadi landing page default. Dikonfirmasi ke user: seluruh alur input audio (bukan cuma kontrol inti) pindah ke sidebar, dan Riwayat Sesi tetap di sidebar analisis (bukan ke Home).

### Keputusan Desain

- `st.navigation(pages, position="top")` (Streamlit 1.61.1 mendukung ini) menggantikan menu sidebar bawaan. `st.sidebar` tetap container biasa terlepas dari `position`, dan hanya render kalau ada yang ditulis ke situ pada page run tersebut - jadi syarat "sidebar cuma muncul di Analisis" otomatis terpenuhi dengan hanya memanggil `st.sidebar` di `pages/analisis.py`, tanpa perlu trik tambahan.
- "Detail Teknis" (backbone/mode/checkpoint) dari sidebar lama didrop total, tidak dipindah ke Home - sudah jadi subset penuh dari expander "Konfigurasi Training" yang sudah ada di `pages/model.py`. Home cukup kasih `st.page_link` ke halaman Model.
- `render_hero()` dihapus dari halaman Analisis (dipakai di Home sebagai pemilik narasi intro), diganti heading ringan lewat `render_section_header()` yang sudah ada.
- Class CSS `sidebar-*` dipakai ulang apa adanya di halaman Home (tidak di-rename) - class tersebut cuma string CSS tanpa keterikatan ke container `st.sidebar` sungguhan, jadi aman dipakai di konten utama. Dibungkus `st.columns([1,2,1])` supaya tidak melebar penuh di layout wide.
- `render_metadata_card` diubah dari 4 kolom sejajar jadi 2x2, karena sekarang cuma dipanggil dari kolom sidebar sempit (~336px) - 4 kolom akan bikin nilai seperti "16000 Hz" wrap parah.
- Validasi desain (posisi navbar top, penempatan Detail Teknis, gotcha Streamlit) dilakukan lewat sub-agent Plan sebelum implementasi, bukan asumsi langsung.

### File yang Diubah

- `app.py`: tambah `st.Page("pages/home.py", ..., default=True)`, hapus `default=True` dari Analisis, `st.navigation(..., position="top")`, hapus import dan panggilan `render_sidebar` (sudah tidak ada).
- `pages/home.py` (baru): landing page - hero, stat grid, status model, peta emosi (port dari `sidebar.py` lama), CTA `st.page_link` ke Analisis dan Model.
- `components/sidebar.py`: dihapus. Konfirmasi lewat grep hanya diimpor dari `app.py`, tidak ada pemakai lain.
- `components/ui.py`: tambah `render_history_list(history)` (ekstraksi dari `sidebar.py` lama, murni render tanpa baca `st.session_state`), `render_metadata_card` jadi 2x2, `render_waveform_chart` height 140 -> 100.
- `pages/analisis.py`: restrukturisasi `main()` - alur input audio (pilih sumber, upload/rekam, metadata, preview, waveform, checkbox per-segmen, tombol Analisis Emosi/Analisis Ulang, riwayat sesi) dibungkus `with st.sidebar:`; blok jalankan prediksi dan seluruh hasil (result card, transkrip, segmen, top-3, confidence, export, detail teknis) tetap di konten utama. Kasus `audio_file is None` atau metadata gagal diproses: `render_empty_state()` dipindah ke KONTEN UTAMA (bukan sidebar), sesuai requirement.

### Yang Tidak Diubah

- Logic inferensi, preprocessing audio, arsitektur model (`utils.py::predict_emotion`, `model.py`, `services.py` tidak disentuh selain yang sudah ada).
- `pages/dashboard.py`, `pages/model.py`, `pages/dataset.py` - tidak ada perubahan kode, hanya terdampak otomatis oleh `position="top"` di level navigasi.
- Skema warna monokrom dan token CSS dari redesign sebelumnya.

### Verifikasi

- `python -m py_compile` untuk seluruh file yang diubah/dibuat plus `utils.py`, `services.py`, `config.py`, `model.py` -> tanpa error sintaks.
- `streamlit.testing.v1.AppTest` lewat `app.py` (mensimulasikan `st.navigation` asli) untuk kelima halaman (Home, Analisis, Dashboard, Model, Dataset) -> tanpa exception. Dicek juga jumlah elemen di `at.sidebar`: 0 di Home/Dashboard/Model/Dataset, terisi (radio + section header) di Analisis - mengonfirmasi sidebar memang cuma render saat halaman Analisis dibuka.
- Server Streamlit sungguhan dijalankan lokal (`.venv/bin/streamlit run app.py`), endpoint `/_stcore/health` mengembalikan `ok`, root page mengembalikan HTTP 200 - start-up bersih tanpa error dengan struktur navigasi baru.
- grep memastikan tidak ada referensi tersisa ke `components.sidebar`/`render_sidebar` yang sudah dihapus, dan `render_hero` cuma dipakai di `pages/home.py`.

### Yang Belum Diverifikasi

- Tampilan visual navbar grup ("Beranda"/"Analisis"/"Insight") dalam mode `position="top"` di browser sungguhan - environment kerja ini tidak punya headless browser/Playwright. `AppTest` mengonfirmasi tidak ada exception saat build navigasi, tapi tidak memvalidasi rendering visual dropdown grup.
- Alur upload file audio sungguhan lewat `st.file_uploader`/`st.audio_input` di dalam sidebar sempit sampai ke hasil prediksi penuh - `AppTest` tidak mendukung simulasi interaksi `file_uploader` dengan bytes nyata, jadi jalur ini hanya diverifikasi lewat pembacaan kode (logic dipindah tanpa diubah dari versi yang sudah terbukti jalan sebelumnya), bukan dijalankan end-to-end otomatis.
- Proporsi visual grid Peta Emosi/stat card saat dipindah dari kolom sidebar sempit ke halaman Home yang lebar (dibungkus `st.columns([1,2,1])`, tapi belum dicek visual).

### Follow-up yang Disarankan

- Uji manual di browser: cek navbar top, sidebar hanya muncul di Analisis, upload/rekam audio sampai hasil, dan tata letak Home di layar desktop maupun mobile (proyek ini mensyaratkan mobile-first per AGENTS.md 3.2, dan struktur navigasi top-nav + sidebar kontekstual ini belum pernah diuji di breakpoint kecil).

## [2026-08-13] Redesign UI Streamlit: Konsolidasi Token CSS & Komponen Section Header

### Konteks

Permintaan redesign tampilan Streamlit. Setelah eksplorasi kode, arsitektur komponen (`components/ui.py`, `css.py`, `sidebar.py`, `pages/*.py`) sudah cukup baik, tapi implementasi CSS-nya berantakan: warna/radius ditulis sebagai literal `rgba(...)` berulang di ~480 baris `components/css.py`, dan markup section header (step label + judul) diduplikasi manual di 4 halaman berbeda. Pengguna memutuskan lingkup redesign: fokus visual look & feel, tetap monokrom (dipoles bukan diganti skema warna), mencakup semua halaman.

### Keputusan Desain

- Tidak mengganti palet warna monokrom yang sudah ada, hanya menjadikannya konsisten lewat CSS custom properties (`:root`) di `components/css.py`, bukan membangun sistem token Python baru karena `EMOTION_COLORS`/`EMOTION_ICONS` di `config.py` memang harus tetap di Python (dipakai untuk inline style).
- Skala radius disederhanakan jadi 3 tingkat (`--radius-sm/md/lg`) menggantikan campuran 10-20px yang sebelumnya dipilih tidak konsisten antar komponen.
- Transisi hover ditambahkan pada card yang berperilaku seperti daftar (`section-card`, `top3-card`, `segment-card`, item sidebar), bukan pada `hero-card`/`result-card` yang statis.
- Inkonsistensi confusion matrix (PNG statis di `dashboard.py` vs Altair recompute di `model.py`) sengaja tidak disentuh karena itu masalah data/logic, bukan visual.
- Tidak membangun theming light/dark karena pengguna memilih tetap monokrom.

### File yang Diubah

- `components/css.py`: tambah blok `:root` token (surface, border, teks, radius, transisi), ganti literal berulang dengan `var(--token)`, tambah hover state.
- `components/ui.py`: tambah `render_section_header(step, title="", desc=None)`, escape teks dinamis (`transcript`, `segment text`) dengan `html.escape()` di `render_transcript_card()` dan `render_segment_timeline()`, pindahkan `summarize_prediction()` keluar (murni logic, bukan render).
- `utils.py`: terima `summarize_prediction()` dari `components/ui.py`.
- `pages/analisis.py`, `pages/dashboard.py`, `pages/model.py`: ganti markup `section-card` manual (masing-masing 2-3 lokasi) dengan pemanggilan `render_section_header()`.
- `pages/dataset.py`: hapus helper lokal `_section()`, pakai `render_section_header()` yang dibagi bersama.

### Yang Tidak Diubah

- Skema warna monokrom, `EMOTION_COLORS`/`EMOTION_ICONS` di `config.py`.
- Layout kolom (`st.columns`) di setiap halaman, jadi responsivitas mobile-first yang sudah ada tidak berubah; hover state hanya berlaku desktop dan tidak memengaruhi perangkat sentuh.
- Confusion matrix ganda (PNG di dashboard vs Altair di halaman model), logic inferensi, arsitektur model, pipeline audio.

### Verifikasi

- `python -m py_compile` untuk seluruh file yang diubah plus `app.py`, `config.py`, `model.py`, `services.py` -> tanpa error sintaks.
- `streamlit.testing.v1.AppTest` dijalankan lewat entry point `app.py` (mensimulasikan `st.navigation` asli, bukan impor file halaman langsung) untuk keempat halaman (Analisis, Dashboard, Model, Dataset) -> tanpa exception, markup `render_section_header()` ter-render dengan benar termasuk escaping karakter HTML.
- Server Streamlit dijalankan lokal (`.venv/bin/streamlit run app.py`) untuk memastikan proses start tanpa error sebelum verifikasi lewat AppTest.
- Ditemukan (tapi tidak diperbaiki, di luar lingkup redesign visual): `AppTest.from_file("pages/dashboard.py")` yang dipanggil langsung (tanpa lewat `app.py`) gagal dengan `ImportError` karena tabrakan nama modul antara `model.py` di root dan `pages/model.py`. Dikonfirmasi lewat `git stash` bahwa masalah ini sudah ada sebelum redesign ini dan tidak muncul saat navigasi lewat `app.py` yang sebenarnya.

### Follow-up yang Disarankan (Belum Dikerjakan)

- Uji visual manual di browser (screenshot sebelum/sesudah tiap halaman) karena environment kerja ini tidak punya headless browser/Playwright untuk verifikasi otomatis.
- Pertimbangkan menyatukan tabrakan nama `model.py`/`pages/model.py` yang ditemukan di atas, di luar lingkup task ini.

## [2026-08-12] Fitur Rekam Mikrofon Langsung (Live Record)

### Konteks

Branch `dev-eln`. Sebelumnya aplikasi hanya menerima audio melalui `st.file_uploader` (.wav/.mp3). Ditambahkan opsi rekam langsung dari mikrofon browser, lalu pengguna tetap menekan tombol "Analisis Emosi" secara manual (bukan inferensi otomatis saat rekaman selesai).

### Keputusan Desain

- Menggunakan `st.audio_input` native Streamlit, bukan dependency pihak ketiga (`streamlit-webrtc` dsb). Hasilnya adalah `UploadedFile` (subclass `BytesIO`) berformat WAV yang langsung kompatibel dengan `load_audio()` di `utils.py` tanpa modifikasi decoder.
- Tidak mengimplementasikan inferensi real-time/streaming. Kebutuhan yang diminta adalah pola "rekam, berhenti, preview, analisis", sesuai opsi 1 dari dua kemungkinan makna "live record" yang didiskusikan.
- Cache key prediksi diubah dari `(nama_file, ukuran_file)` menjadi `(sumber, sha256(bytes_audio))` karena dua rekaman mikrofon dapat memiliki nama dan ukuran identik, sehingga cache lama berisiko tidak ter-invalidasi dengan benar.

### File yang Diubah

- `app.py`: pemilih sumber (`st.radio`), integrasi `st.audio_input`, unifikasi variabel `uploaded_file` -> `audio_file`, fungsi `_audio_key()` menggantikan `_upload_key()`.
- `components/ui.py`: copy `render_hero()` dan `render_empty_state()` diperbarui agar netral terhadap sumber audio (unggah atau rekam).
- `requirements.txt`: `streamlit>=1.28.0` dinaikkan ke `streamlit>=1.41.0` karena `st.audio_input` baru general availability di v1.40.0 dengan perbaikan bug penting di v1.41.0.
- `tdd_changes_tracker.md`: dicatat sesuai kebijakan proyek untuk perubahan yang menyentuh alur input inferensi.

### Yang Tidak Diubah

- `model.py`, arsitektur `WavLMSERModel`, dan `services.py` (loading model/feature extractor).
- `utils.py`, termasuk `MAX_DURATION_SECONDS = 4.0`, target sample rate `16000 Hz`, dan pipeline STT Whisper.
- Label emosi Bahasa Indonesia (`netral`, `senang`, `sedih`, `marah`, `takut`, `jijik`).

### Verifikasi

- `python -m py_compile app.py components\ui.py components\sidebar.py components\css.py services.py config.py utils.py model.py` -> berhasil tanpa error sintaks.
- Verifikasi runtime penuh (`import streamlit`, `import utils`) tidak dapat dijalankan di environment kerja ini karena dependency (`streamlit`, `soundfile`, dll.) belum terpasang dan tidak ada virtualenv lokal (`.venv`) yang tersedia. Perlu dijalankan manual sebelum deployment:
  ```powershell
  .\.venv\Scripts\Activate.ps1
  pip install -r requirements.txt
  python -m streamlit run app.py
  ```

### Implikasi Operasional

- Mikrofon browser memerlukan konteks aman (`localhost` atau HTTPS). Pada deployment cloud tanpa HTTPS, tombol rekam akan gagal meminta izin mikrofon.
- Jika pengguna menolak izin mikrofon, mode "Unggah File" tetap tersedia sebagai fallback karena kedua mode independen melalui `st.radio`.

### Follow-up yang Disarankan (Belum Dikerjakan)

- Uji manual end-to-end dengan Streamlit terpasang: rekam audio, verifikasi metadata, jalankan prediksi, dan transkrip Whisper.
- Review responsivitas mobile untuk `st.radio` dan `st.audio_input` pada layar kecil, karena keduanya belum diverifikasi visual pada breakpoint mobile-first proyek ini.

## [2026-08-12] Perbaikan Bug Kritis: Crash Inferensi & Penyimpangan Arsitektur Pooling dari Checkpoint v7

### Konteks

Setelah fitur live record ditambahkan, pengguna melaporkan error "File audio tidak dapat diproses" saat menekan tombol Analisis Emosi. Investigasi lanjutan (bukan asumsi langsung) mengonfirmasi via tanya-jawab bahwa: (1) metadata dan pratinjau audio tampil normal sebelum error, artinya decode audio berhasil; (2) mode unggah file juga gagal dengan pesan sama. Kedua fakta ini mengeliminasi fitur live record sebagai penyebab, mengarah ke bug pre-existing di pipeline inferensi bersama.

### Root Cause yang Ditemukan

Tiga bug ditemukan lewat pembacaan kode dan perbandingan langsung dengan `pipeline/ser-augmemted.ipynb` (sumber kebenaran v7 per AGENTS.md):

1. **Crash `AttributeError`** di `utils.py` fungsi `predict_emotion()`. Pemanggilan `processor(...)` tanpa `return_tensors="pt"` mengembalikan list numpy, bukan tensor, sehingga `.to(device)` gagal. `except Exception` generik di `app.py` membungkam error asli menjadi pesan menyesatkan.

2. **Silent correctness bug** di `model.py` `AttentionPooling`. Implementasi lama mengembalikan `[attn_pooled, plain_mean]`, sedangkan checkpoint `ser_wavlm_v7_best.pt` dilatih dengan `[weighted_mean, weighted_std]` (attentive stats pooling). Karena nama atribut kebetulan sama, `load_state_dict(strict=True)` tidak pernah mendeteksi ketidakcocokan ini. Model tetap berjalan dan mengeluarkan angka, tapi separuh fitur classifier menerima distribusi input yang salah secara matematis.

3. **Preprocessing menyimpang dari kontrak v7** di `utils.py` `preprocess_audio()`. Tidak ada silence trim, peak normalization, atau zero-padding untuk audio pendek. Audio di bawah 4 detik (kasus umum untuk rekaman mikrofon) sebelumnya diloloskan dengan panjang tensor variabel, tidak pernah tepat 64000 sampel seperti kondisi training.

### Perbaikan yang Dilakukan

- `model.py`: `AttentionPooling.forward()` ditulis ulang menjadi attentive statistics pooling (weighted mean + weighted std), `WavLMSERModel.forward()` disesuaikan agar tidak lagi concat mean tambahan. `masked_fill` diganti dari `-inf` ke `-1e4` untuk mencegah NaN. `DEFAULT_DROPOUT` diselaraskan dari `0.35` ke `0.30`.
- `utils.py`: `predict_emotion()` dilengkapi argumen `padding`, `truncation`, `max_length`, `return_attention_mask`, `return_tensors="pt"` persis sesuai notebook. `preprocess_audio()` ditambah fungsi `_trim_silence()`, `_peak_normalize()`, `_fix_length_eval()` yang mereplikasi urutan wajib: sanitasi NaN -> trim 30dB (dengan guard panjang minimum) -> peak normalization -> center crop / right zero-pad ke tepat 64000 sampel.
- `app.py`: pesan error kini menyertakan `type(exc).__name__` dan pesan asli, ditambah expander traceback lengkap untuk debugging.

### Verifikasi yang Dilakukan

Karena `streamlit`, `torch`, dll. tidak terpasang di environment kerja utama, dibuat virtual environment sementara (`.venv_test`) khusus untuk pengujian, dihapus setelah selesai (tidak tercatat di git, tidak mengubah `requirements.txt` project):

1. Unit test murni numpy untuk `_trim_silence`, `_peak_normalize`, `_fix_length_eval`, dan sanitasi NaN, mencakup kasus audio pendek, audio panjang, audio hening total. Semua PASS, output selalu tepat 64000 sampel tanpa NaN.
2. Unit test `AttentionPooling` dan `WavLMSERModel.forward()` dengan backbone `microsoft/wavlm-base-plus` asli (bukan mock). Shape pooled `(1, 1536)`, shape logits `(1, 6)`, tanpa NaN, termasuk edge case partial attention mask.
3. Test `predict_emotion()` dengan `AutoFeatureExtractor` asli: berhasil tanpa `AttributeError`, softmax probabilitas menjumlah ke `1.0`.
4. Test pipeline penuh `preprocess_audio() -> predict_emotion()` untuk dua skenario: simulasi rekaman mikrofon pendek (1.5 detik, sample rate 48kHz, khas browser) dan simulasi unggah file panjang (6 detik, 16kHz). Keduanya PASS tanpa exception.

`python -m py_compile` juga dijalankan ulang untuk seluruh file yang diubah, berhasil tanpa error sintaks.

### Yang Belum Diverifikasi

- Akurasi aktual terhadap `models/evaluasi_test_v7.csv` untuk mengonfirmasi angka `test_acc 0.7746` di `models/config_v7.json` tercapai kembali setelah perbaikan pooling. Test yang dilakukan memakai waveform acak (random noise), bukan audio nyata dengan label ground truth, sehingga hanya membuktikan tidak ada crash dan shape/NaN benar, bukan membuktikan akurasi.
- Uji manual UI langsung (Streamlit berjalan) dengan rekaman mikrofon dan file unggah nyata.

### Dampak yang Perlu Diketahui

Hasil prediksi akan berbeda dibanding sebelum perbaikan ini, untuk audio yang sama persis. Ini disengaja karena perilaku lama salah secara matematis terhadap checkpoint yang dimuat. Jika ada laporan atau demo yang sudah terlanjur memakai output dari kode lama, angkanya tidak akan lagi cocok.

### Follow-up yang Disarankan

- Jalankan skrip evaluasi batch terhadap `models/evaluasi_test_v7.csv` atau subset test set asli untuk mengonfirmasi akurasi kembali ke kisaran `0.77`.
- Verifikasi manual di UI Streamlit dengan audio suara manusia nyata (bukan random noise), untuk kedua mode: unggah dan rekam mikrofon.
 
 
## Sesi: Refinement Training Pipeline v8
### Masalah:
- Recall untuk kelas **sedih** dan **senang** sangat rendah (dibawah 70%).
- Akurasi dataset **IndoWaveSentiment** sangat buruk karena tertelan oleh jumlah dataset CREMA-D.
### Solusi yang Diimplementasikan:
Modifikasi langsung pada resep training Kaggle (pipeline/ver2-ser-pipeline.ipynb):
1. **Selective Noise Filtering**: Mematikan filter hard-drop khusus untuk kelas sedih, senang, takut, dan jijik agar data valid tidak terbuang oleh model yang belum yakin.
2. **Class-Specific Augmentation**: Menambahkan pipeline augmentasi (pitch, time stretch) yang secara eksklusif membidik kelas lemah di CREMA-D.
3. **Extreme Oversampling**: Menduplikasi data latih IndoWaveSentiment sebanyak 4 kali lipat sebelum proses split.

## Sesi: Koreksi Pipeline Training dan Noise Handling di ver3

### Diagnosis

Audit terhadap notebook ver2 dan ver3 menemukan bahwa loop training Stage 1 dan Stage 2 belum ada. Notebook memuat checkpoint v7 dan menampilkan metrik lama, sehingga metrik tersebut belum membuktikan hasil eksperimen v8.

### Perubahan

- Target kerja dipindahkan ke pipeline/ver3-ser-pipeline.ipynb. Notebook ver2 tidak diubah.
- Augmentasi offline tetap hanya untuk IndoWave dan E-SERAVD karena keduanya lebih kecil. Augmentasi CREMA-D dikeluarkan dari baseline.
- Oversampling IndoWave 4x setelah augmentasi dihapus agar sumber kecil tidak menjadi dominan secara tidak wajar.
- Loop Stage 1 dan Stage 2 ditambahkan, termasuk penyimpanan checkpoint.
- Learning rate backbone diperbaiki agar parameter backbone.* benar-benar memakai learning rate backbone.
- Bug core pada gradual unfreezing diperbaiki.
- Mixup dinonaktifkan karena helper yang dipanggil tidak tersedia.
- Checkpoint dipilih berdasarkan gabungan macro-F1, recall sedih, recall senang, precision takut, dan precision jijik.
- Hard drop dibuat konservatif dengan confidence minimal 0.95 dan hanya melindungi label CREMA-D yang lebih tepercaya. Kelas target recall tidak dihapus.
- Tabel distribusi augmentasi dan metrik precision, recall, F1 per sumber serta kelas ditambahkan.

### Verifikasi

- JSON notebook valid.
- Cell Python yang dapat diparse valid secara AST.
- Training Kaggle belum dijalankan. Kenaikan metrik belum boleh diklaim.

### Catatan

Target precision takut dan jijik di atas 80% serta recall sedih dan senang di atas 80% harus dibuktikan melalui run Kaggle yang reproducible. Nilai dari screenshot, artefak lokal, dan output notebook sebelumnya berasal dari run yang berbeda dan tidak boleh dibandingkan langsung.

## Sesi: Dokumentasi Pipeline v9

### Perubahan

- Header paling atas `pipeline/ver3-ser-pipeline.ipynb` diubah dari v8 menjadi v9.
- Ditambahkan tabel perbandingan v8 ke v9 agar scope augmentasi, training loop, optimizer, hard drop, dan target metrik terdokumentasi langsung di notebook.
- Target v9 ditulis sebagai sasaran eksperimen, bukan klaim hasil. Recall `sedih` dan `senang`, serta precision `takut` dan `jijik`, ditargetkan minimal 80%.
- Riwayat v7 ke v8 dipertahankan sebagai konteks historis.

### Verifikasi

- JSON notebook valid.
- 33 cell Python dapat diparse dengan AST. Satu cell Kaggle yang memakai `!pip` dan `kaggle_secrets` dilewati dari pemeriksaan AST.
- Tidak ada perubahan pada `pipeline/ver2-ser-pipeline.ipynb` untuk subtask dokumentasi ini.
