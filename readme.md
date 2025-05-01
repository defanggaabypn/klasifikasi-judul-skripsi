# ANALISIS PERBANDINGAN ALGORITMA K-NEAREST NEIGHBORS (KNN) DAN DECISION TREE BERDASARKAN HASIL SEMANTIC SIMILARITY JUDUL SKRIPSI DAN BIDANG KONSENTRASI

**STUDI KASUS: JURUSAN PENDIDIKAN TEKNOLOGI INFORMASI DAN KOMUNIKASI**

## Deskripsi Sistem

Sistem ini mengimplementasikan analisis perbandingan algoritma machine learning (KNN dan Decision Tree) untuk klasifikasi judul skripsi ke dalam berbagai bidang konsentrasi. Sistem ini menggunakan pendekatan semantic similarity berbasis embedding untuk memproses judul skripsi dalam bahasa Indonesia. Sistem ini dibangun dengan backend API Python untuk pemrosesan dan klasifikasi, serta frontend web PHP untuk interaksi pengguna.

## Aspek Kesesuaian Penelitian

### 1. Algoritma yang Digunakan
✅ Sistem telah mengimplementasikan kedua algoritma yang diteliti:
* K-Nearest Neighbors (KNN)
* Decision Tree

### 2. Semantic Similarity
✅ Sistem mengimplementasikan semantic similarity melalui:
* Penggunaan model IndoBERT untuk menghasilkan embedding vektor yang merepresentasikan makna semantik judul skripsi
* Perhitungan jarak/kemiripan antar embedding untuk menentukan similaritas antar judul

### 3. Bidang Konsentrasi
✅ Sistem dikembangkan untuk mengklasifikasikan judul skripsi ke dalam bidang konsentrasi yang berbeda (RPL, Jaringan, dan Multimedia) yang sesuai dengan program studi Pendidikan Teknologi Informasi dan Komunikasi.

### 4. Analisis Perbandingan
✅ Sistem melakukan analisis perbandingan antara dua algoritma melalui:
* Perhitungan dan visualisasi metrik performa (akurasi, presisi, recall, f1-score)
* Pembuatan confusion matrix untuk kedua model
* Perbandingan visual kinerja kedua algoritma

### 5. Studi Kasus
✅ Sistem berfokus pada judul skripsi di bidang Teknologi Informasi dan Komunikasi, yang sesuai dengan studi kasus yang diteliti.

## Fitur Utama

- Unggah file Excel yang berisi judul-judul skripsi
- Klasifikasi otomatis judul skripsi menggunakan KNN dan Decision Tree
- Representasi visual hasil klasifikasi dan perbandingan algoritma
- Pencarian judul serupa berdasarkan input yang diberikan
- Pelacakan riwayat semua prediksi
- Ekspor hasil ke berbagai format (PDF, Excel)
- Visualisasi detail metrik performa model

## Struktur Proyek
```text
project/
├── python-api/
│   ├── cache/
│   ├── models/
│   ├── uploads/
│   ├── venv/
│   ├── app.py
│   └── requirements.txt
├── web/
│   ├── assets/
│   │   └── css/
│   │       └── pdf-style.css
│   ├── clasess/
│   │   └── Exporter.php
│   ├── temp/
│   ├── uploads/
│   ├── vendor/
│   ├── composer.json
│   ├── composer.lock
│   ├── config.php
│   ├── databse_config.php
│   ├── delete_history.php        
│   ├── export.php
│   ├── history.php
│   ├── index.php
│   ├── install.php
│   ├── result.php
│   ├── save_upload.php
│   ├── visualisasi.php
│   └── .gitignore
├── eror.txt
├── setup_environment.bat
└── start-server.bat

## Persyaratan Sistem

**Backend Python**
- Python 3.8 atau lebih tinggi
- Flask
- PyTorch
- Scikit-learn
- Pandas
- NumPy
- Matplotlib
- Transformers (Hugging Face)
- PyMySQL
- CORS

**Frontend PHP**
- PHP 7.4 atau lebih tinggi
- MySQL
- Composer

## Instalasi

### Pengaturan Windows

1. Jalankan skrip pengaturan untuk membuat virtual environment dan menginstal dependensi:

    ```bash
    setup_environment.bat
    ```

2. Siapkan database MySQL:
    - Buat database baru bernama `skripsi_classification`
    - Impor skema SQL (tidak disertakan dalam repositori)

3. Konfigurasi koneksi database:
    - Edit `web/databse_config.php` dengan kredensial MySQL Anda

4. Mulai server:

    ```bash
    start-server.bat
    ```

### Pengaturan Manual

#### Backend Python

1. Buat virtual environment:

    ```bash
    cd python-api
    python -m venv venv
    source venv/bin/activate  # Di Windows: venv\Scripts\activate
    ```

2. Instal dependensi:

    ```bash
    pip install -r requirements.txt
    ```

3. Jalankan API Flask:

    ```bash
    python app.py
    ```

#### Frontend PHP

1. Instal dependensi PHP:

    ```bash
    cd web
    composer install
    ```

2. Konfigurasi web server Anda (Apache/Nginx) untuk melayani direktori web
3. Pastikan direktori uploads, temp, dan cache dapat ditulis oleh web server

## Cara Kerja Sistem

### Komponen Machine Learning

Sistem ini menggunakan dua algoritma machine learning untuk klasifikasi:

- **K-Nearest Neighbors (KNN)**: Mengklasifikasikan judul skripsi berdasarkan kemiripan dengan judul lainnya
- **Decision Tree**: Membuat pohon keputusan berdasarkan fitur-fitur judul skripsi

Kedua model menggunakan embedding IndoBERT untuk mengubah teks menjadi vektor numerik untuk pemrosesan.

### Alur Data

1. Pengguna mengunggah file Excel dengan judul skripsi (dan opsional kategori)
2. Backend Python memproses file dan:
   - Menghasilkan embedding untuk setiap judul menggunakan IndoBERT
   - Melatih model KNN dan Decision Tree pada data
   - Mengevaluasi performa model
   - Mengembalikan visualisasi dan metrik performa
3. Hasil ditampilkan pada antarmuka web
4. Prediksi disimpan dalam database untuk referensi di masa mendatang

### Endpoint API

- `/process` - Memproses file Excel dan melatih model
- `/predict` - Memprediksi kategori judul skripsi baru
- `/similar` - Mencari judul skripsi yang serupa
- `/template` - Mengunduh template Excel
- `/delete_prediction/<id>` - Menghapus prediksi
- `/get_predictions` - Mendapatkan semua prediksi
- `/get_prediction/<id>` - Mendapatkan detail prediksi tertentu
- `/get_predictions_by_upload/<id>` - Mendapatkan prediksi dari unggahan tertentu
- `/get_uploaded_files` - Mendapatkan daftar semua file yang diunggah

## Penggunaan

1. Buka antarmuka web di browser Anda
2. Unggah file Excel dengan judul skripsi atau gunakan template yang disediakan
3. Lihat hasil klasifikasi dan metrik performa perbandingan algoritma
4. Prediksi kategori untuk judul skripsi baru
5. Cari judul yang serupa
6. Ekspor hasil ke PDF atau Excel
7. Lihat riwayat semua prediksi

## Detail Teknis

### Model IndoBERT

Sistem ini menggunakan model `indobenchmark/indobert-base-p1` untuk menghasilkan embedding. Model ini secara khusus dilatih pada teks bahasa Indonesia, membuatnya sangat cocok untuk memproses judul skripsi dalam bahasa Indonesia.

### Sistem Caching

Untuk meningkatkan performa, sistem menyimpan embedding dalam cache untuk menghindari perhitungan ulang untuk teks yang sama.

### Struktur Database

Tabel utama dalam database meliputi:
- `thesis_titles` - Menyimpan judul skripsi dan kategorinya
- `categories` - Menyimpan informasi kategori
- `predictions` - Menyimpan hasil prediksi
- `model_performances` - Menyimpan metrik performa model
- `keyword_analysis` - Menyimpan hasil analisis kata kunci
- `uploaded_files` - Menyimpan informasi tentang file yang diunggah

## Pemecahan Masalah

Jika Anda mengalami masalah:

- Periksa file `eror.txt` untuk log kesalahan
- Pastikan semua direktori dapat ditulis oleh aplikasi
- Verifikasi pengaturan koneksi database di `databse_config.php`
- Pastikan API Python berjalan dan dapat diakses dari frontend PHP

## Kredit

Proyek ini menggunakan komponen utama berikut:

- IndoBERT oleh IndoNLP
- Flask untuk server API
- Scikit-learn untuk algoritma machine learning
- Hugging Face Transformers untuk pemrosesan NLP
