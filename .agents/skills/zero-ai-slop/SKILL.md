---
name: zero-ai-slop
description: Mandates Zero AI Slop standards including strict ban of em-dashes, decorative emojis, AI boilerplate filler, and maintaining immutable Indonesian emotion labels across all docs, code, and logs.
---

# Zero AI Slop Skill

Dokumen ini mendefinisikan aturan ketat untuk menghilangkan segala bentuk "AI Slop" (gaya penulisan kaku, basa-basi AI, dan emoji dekoratif berlebihan) dari repositori ini.

## Aturan Mutlak Zero AI Slop

### 1. Larangan Em-Dash (`—`)
- **Dilarang keras menggunakan karakter em-dash (`—`)** di mana pun (dalam dokumentasi, komentar kode, log print, pesan commit, maupun jawaban ke pengguna).
- **Pengganti**: Gunakan tanda titik `.`, koma `,`, atau strip standar `-`.

### 2. Larangan Emoji Dekoratif Berlebihan
- **Dilarang menggunakan emoji dekoratif** pada judul dokumentasi, header markdown, commit message, dan log print terminal (`🔍`, `✅`, `❌`, `🛑`, `📌`, `🖥️`, `📂`, `📊`, `🔄`, `🔧`, `⏹`, `📉`, `🔁`, `🟡`, `📥`).
- **Pengganti Log**: Gunakan tag teks profesional berbasis huruf kapital dalam kurung siku seperti `[OK]`, `[NOT FOUND]`, `[ERROR]`, `[CHECK]`, `[NOTE]`, `[STATS]`, `[CONFIG]`, `[BEST]`.
- **Pengecualian Fungsional**: Emoji **hanya diperbolehkan** jika berfungsi sebagai bagian dari visualisasi UI/UX domain aplikasi (seperti `EMOJI_MAP = {'netral':'😐','senang':'😊','sedih':'😢','marah':'😡','takut':'😨','jijik':'🤢'}`).

### 3. Penamaan Label Emosi Tetap Bahasa Indonesia
- Label emosi dalam TDD, model, dataset, dan antarmuka **wajib** menggunakan Bahasa Indonesia:
  `['netral', 'senang', 'sedih', 'marah', 'takut', 'jijik']`.
- Dilarang mengonversi atau menerjemahkan label ini ke Bahasa Inggris (`neutral`, `happy`, `sad`, `angry`, `fearful`, `disgusted`) di dalam kode backend/pipeline.

### 4. Larangan Basa-Basi AI (No AI Boilerplate Filler)
- Dilarang membuka jawaban dengan kata-kata formal/kaku khas bot seperti:
  - "Tentu!"
  - "Berikut adalah..."
  - "Perlu dicatat bahwa..."
  - "Saya mengerti..."
- Langsung masuk ke poin teknis, analisis tajam, atau eksekusi perintah tanpa pengantar kosong.

### 5. Komentar Kode Efisien (Rationale, Not Syntax)
- Komentar kode hanya ditulis untuk menjelaskan **mengapa** (alasan teknis/bisnis), bukan merestatemen **apa** yang dilakukan sintaks.
  - *Buruk*: `# mengimpor os dan sys`
  - *Baik*: `# Gunakan sys.exit() untuk menghentikan sel jika dataset tidak terdeteksi`

## Verifikasi Kepatuhan
Sebelum mengklaim pekerjaan selesai, jalankan pemindaian karakter non-standard dan emoji pada file yang dimodifikasi.
