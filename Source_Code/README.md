# DrowseGuard — Real-Time Drowsiness Detection

DrowseGuard adalah proyek Computer Vision yang kami buat untuk mata kuliah **Teknologi Berkembang B**. Ide awalnya sederhana: bagaimana kalau webcam laptop bisa membantu mengingatkan kita saat mata tertutup terlalu lama?

Melalui DrowseGuard, kondisi mata dipantau secara langsung menggunakan webcam. Sistem membedakan **mata terbuka (awake)** dan **mata tertutup (sleepy)** dengan model *Convolutional Neural Network* (CNN). Jika mata terdeteksi tertutup selama **minimal 2 detik**, aplikasi akan menampilkan peringatan dan membunyikan alarm. Karena itu, kedipan biasa tidak langsung dianggap sebagai tanda kantuk.

> **Catatan:** DrowseGuard masih berupa prototipe untuk keperluan pembelajaran. Peringatan didasarkan pada durasi mata tertutup, bukan diagnosis kondisi kantuk atau jaminan keselamatan saat berkendara.

## Apa saja yang bisa dilakukan?

- Memantau kondisi mata secara **real-time** melalui webcam.
- Menampilkan status **OPEN**, **CLOSED**, atau **FACE NOT FOUND**.
- Menghitung berapa lama mata terdeteksi tertutup.
- Menampilkan **Drowsiness Warning** dan memutar alarm ketika mata tertutup selama minimal 2 detik.
- Menampilkan pratinjau potongan kedua mata yang diproses oleh model CNN.
- Mengulang pemantauan ketika mata terbuka kembali, atau melalui tombol **Reset Monitoring** pada versi Gradio.

Kami menyediakan **dua versi aplikasi**: versi desktop dengan OpenCV, dan versi web lokal dengan Gradio. Versi Gradio digunakan sebagai antarmuka aplikasi akhir karena hasil pemantauan dapat dilihat dalam satu dashboard melalui browser.

## Cara kerja singkat

**Webcam → Deteksi lokasi mata (MediaPipe) → Potong & siapkan gambar mata → Prediksi EyeCNN → Hitung durasi mata tertutup → Peringatan dan alarm**

Model dilatih menggunakan **MRL Eye Dataset**. Sebelum diprediksi, gambar mata diubah menjadi grayscale, diseragamkan ukurannya ke 64 × 64 piksel, lalu dinormalisasi agar sesuai dengan input model. Prediksi kedua mata digabung dan dihaluskan supaya statusnya tidak mudah berubah akibat satu frame yang kurang jelas.

## Isi folder proyek

```text
B_Group01_OpenCV_Final_Submission/
│
├── B_Group01_OpenCV_Final_Report.pdf
├── B_Group01_OpenCV_Final_Slides.pdf
├── youtube_link.txt
│
├── Source_Code/
│   ├── main.py
│   ├── style.css
│   ├── eye_cnn_best.pth
│   ├── alarm.mp3
│   ├── predict_eye_webcam_custom_alarm.py
│   ├── train_eye_cnn.ipynb
│   ├── requirements.txt
│   ├── README_code.md
│   └── .vscode/                    (opsional, bisa dihapus)
│
├── Dataset_or_Sample_Input/
│   ├── kaggle_link.txt
│
└── Appendix/
    └── B1_Team_Contributions.pdf
```

Keterangan file utama:

| File | Kegunaan |
| --- | --- |
| `main.py` | Menjalankan DrowseGuard melalui dashboard Gradio di browser. |
| `style.css` | Mengatur tampilan dashboard agar lebih rapi dan nyaman dilihat. |
| `eye_cnn_best.pth` | Bobot model EyeCNN yang telah dilatih, sehingga tidak perlu training ulang untuk mencoba aplikasi. |
| `alarm.mp3` | Suara peringatan ketika mata tertutup terlalu lama. |
| `predict_eye_webcam_custom_alarm.py` | Versi awal aplikasi yang berjalan langsung pada jendela OpenCV desktop. |
| `train_eye_cnn.ipynb` | Notebook yang berisi proses pembacaan dataset, pelatihan, dan evaluasi model. |
| `requirements.txt` | Daftar library utama untuk menjalankan versi Gradio. |

## Cara menjalankan DrowseGuard

Pastikan **Python**, **webcam**, dan file yang dibutuhkan sudah tersedia. Kami menjalankan proyek ini menggunakan environment Python lokal. Jika memakai Conda, aktifkan environment yang sudah disiapkan terlebih dahulu.

### 1. Versi Gradio (utama)

Buka terminal di folder `Source_Code`, kemudian jalankan:

```bash
python -m pip install -r requirements.txt
python main.py
```

Setelah program berjalan, buka alamat lokal yang muncul di terminal (biasanya **http://127.0.0.1:7860**) melalui browser. Berikan izin akses webcam, lalu mulai pemantauan.

Di dashboard, kamu bisa melihat video webcam, status mata, durasi penutupan, status alarm, dan pratinjau kedua mata. Gunakan **Reset Monitoring** untuk mengulang pemantauan jika diperlukan.

### 2. Versi OpenCV desktop (alternatif)

Jika ingin mencoba versi desktop, instal `pygame` sebagai tambahan untuk suara alarm:

```bash
python -m pip install pygame
python predict_eye_webcam_custom_alarm.py
```

Pada versi ini, tekan **Q** untuk menutup aplikasi dan **T** untuk menguji suara alarm saat tidak ada peringatan aktif.

**Penting:** Simpan `main.py`, `style.css`, `eye_cnn_best.pth`, dan `alarm.mp3` di folder `Source_Code` yang sama. Jika ada file tersebut yang hilang, aplikasi Gradio tidak dapat dijalankan dengan benar.

## Dataset dan model

Kami menggunakan **MRL Eye Dataset**, yaitu dataset gambar mata yang terdiri dari dua kelompok: mata terbuka dan mata tertutup. Total data yang digunakan adalah **84.898 gambar**, dengan pembagian training, validation, dan testing. Model dilatih menggunakan PyTorch, lalu bobot terbaik disimpan dalam `eye_cnn_best.pth` untuk digunakan saat pemantauan webcam.

Sumber dataset:
- [MRL Eye Dataset (sumber asli)](https://mrl.cs.vsb.cz/eyedataset.html)
- [MRL Eye Dataset (versi Kaggle yang digunakan)](https://www.kaggle.com/datasets/akashshingha850/mrl-eye-dataset)

Perlu diperhatikan bahwa gambar pada dataset berasal dari kamera inframerah, sedangkan webcam laptop umumnya menggunakan citra biasa. Karena itu, hasil prediksi bisa dipengaruhi pencahayaan, posisi wajah, dan apakah mata terlihat jelas oleh kamera. Aplikasi juga belum dapat mendeteksi mata dengan baik ketika tertutup kacamata hitam.

## Anggota kelompok

Proyek ini dikerjakan oleh **Kelompok 01 — Teknologi Berkembang B**, Institut Teknologi Sepuluh Nopember (ITS).

| Nama | NRP |
| --- | --- |
| Maria Elvina Putri Damayanti | 5026241012 |
| I Gusti Made Sedana Yoga | 5026241037 |
| Farhan Fitran Maulana | 5026241153 |
| Muhammad Fairuz Ahnaf | 5026241169 |

---

DrowseGuard merupakan hasil proses belajar kami dalam menggabungkan pengolahan gambar, model CNN, serta aplikasi real-time. Proyek ini masih bisa dikembangkan lebih lanjut, terutama untuk meningkatkan kestabilan deteksi pada kondisi pencahayaan dan pengguna yang berbeda.
