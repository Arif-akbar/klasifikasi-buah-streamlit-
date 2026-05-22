# Fruit Intel 2.0

Fruit Intel 2.0 adalah aplikasi klasifikasi kualitas buah berbasis Python dan Streamlit. Aplikasi ini menerima input gambar buah, lalu memprediksi jenis serta kondisi buah apakah masih segar atau sudah busuk.

Model utama yang digunakan pada aplikasi adalah CNN berbasis MobileNetV2. Proyek ini juga menyertakan model Random Forest sebagai pembanding atau eksperimen klasifikasi berbasis fitur gambar grayscale.

## Fitur

- Upload gambar buah melalui antarmuka web Streamlit.
- Klasifikasi 6 kelas buah:
  - Fresh Apple
  - Fresh Banana
  - Fresh Orange
  - Rotten Apple
  - Rotten Banana
  - Rotten Orange
- Menampilkan tingkat keyakinan prediksi model.
- Menampilkan histogram warna RGB dari gambar yang diunggah.
- Memberikan informasi sederhana apakah buah layak atau menunjukkan tanda pembusukan.

## Teknologi Yang Digunakan

- Python
- Streamlit
- TensorFlow / Keras
- MobileNetV2
- OpenCV
- NumPy
- Pandas
- Scikit-learn
- Joblib
- Plotly
- Pillow

## Struktur Proyek

```text
.
|-- app.py                 # Aplikasi utama Streamlit
|-- create_csv.py          # Script untuk membuat file CSV dari folder dataset
|-- modul.py               # Script training model CNN MobileNetV2
|-- evaluasi.py            # Script training dan evaluasi model Random Forest
|-- fruit_train.csv        # Data train hasil generate dari dataset
|-- fruit_test.csv         # Data test hasil generate dari dataset
|-- model_fruit_cnn.pkl    # Model CNN hasil training
|-- model_fruit_rf.pkl     # Model Random Forest hasil training
|-- dataset/               # Folder dataset gambar buah
|-- .gitignore
`-- README.md
```

## Format Dataset

Dataset diletakkan di dalam folder `dataset` dengan struktur berikut:

```text
dataset/
|-- train/
|   |-- freshapples/
|   |-- freshbanana/
|   |-- freshoranges/
|   |-- rottenapples/
|   |-- rottenbanana/
|   `-- rottenoranges/
`-- test/
    |-- freshapples/
    |-- freshbanana/
    |-- freshoranges/
    |-- rottenapples/
    |-- rottenbanana/
    `-- rottenoranges/
```

Setiap folder kelas berisi gambar dengan format `.jpg`, `.jpeg`, atau `.png`.

## Cara Menjalankan Proyek

1. Aktifkan virtual environment.

```bash
.venv\Scripts\activate
```

2. Install dependency yang dibutuhkan jika belum tersedia.

```bash
pip install streamlit numpy opencv-python joblib plotly pillow pandas scikit-learn tensorflow
```

3. Jalankan aplikasi Streamlit.

```bash
streamlit run app.py
```

4. Buka URL lokal yang muncul di terminal, biasanya:

```text
http://localhost:8501
```

5. Upload gambar buah, lalu klik tombol `MULAI DIAGNOSA AI`.

## Alur Training Model

Jika ingin membuat ulang dataset CSV dan melatih model dari awal, jalankan urutan berikut:

1. Generate file CSV dari folder dataset.

```bash
python create_csv.py
```

Script ini akan membuat:

- `fruit_train.csv`
- `fruit_test.csv`

2. Training model Random Forest.

```bash
python evaluasi.py
```

Output model:

```text
model_fruit_rf.pkl
```

3. Training model CNN MobileNetV2.

```bash
python modul.py
```

Output model:

```text
model_fruit_cnn.pkl
```

## Cara Kerja Aplikasi

Saat pengguna mengunggah gambar, aplikasi akan:

1. Membaca gambar menggunakan Pillow.
2. Mengubah gambar menjadi format RGB.
3. Menampilkan pratinjau gambar.
4. Menghitung histogram warna merah, hijau, dan biru.
5. Melakukan preprocessing gambar untuk model CNN:
   - resize ke ukuran `224 x 224`
   - normalisasi nilai piksel ke rentang `-1` sampai `1`
6. Melakukan prediksi menggunakan model CNN.
7. Menampilkan kelas hasil prediksi dan tingkat keyakinan model.

## Catatan

- File model `.pkl`, folder `.venv`, folder `dataset`, dan folder `__pycache__` sudah dimasukkan ke `.gitignore`.
- Jika aplikasi menampilkan peringatan file model tidak ditemukan, pastikan `model_fruit_cnn.pkl` dan `model_fruit_rf.pkl` tersedia di folder utama proyek.
- Model CNN menggunakan MobileNetV2 dengan bobot awal `imagenet`, sehingga training awal membutuhkan koneksi internet jika bobot belum tersedia di komputer.

## Author

Proyek UTS Pemrograman Python.
