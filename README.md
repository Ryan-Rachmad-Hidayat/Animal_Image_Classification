
# 🃏 Cards Image Classification

Proyek ini merupakan implementasi deep learning untuk klasifikasi gambar kartu menggunakan model **EfficientNetB3** dengan pendekatan transfer learning. Dataset yang digunakan mencakup berbagai jenis gambar kartu yang dikelompokkan dalam folder per kelas.

## 📁 Dataset
Dataset diklasifikasikan dalam struktur direktori:

```
cards-image-datasetclassification/
├── train/
│   ├── class1/
│   ├── class2/
│   └── ...
├── valid/
│   ├── class1/
│   ├── class2/
│   └── ...
├── test/
│   ├── class1/
│   └── ...
├── cards.csv          # Metadata dan label kategori
```

Setiap gambar akan diproses ulang ke ukuran **224x224 piksel** agar sesuai dengan input dari model **EfficientNetB3**.

## 🧠 Model Arsitektur

Model utama menggunakan arsitektur berikut:

- **EfficientNetB3** pretrained dari ImageNet, tanpa top layer
- **Global pooling**, diikuti oleh:
  - BatchNormalization
  - Dense(1024, ReLU) + Dropout(0.3)
  - Dense(128, ReLU) + Dropout(0.45)
  - Dense(kelas, Softmax)

## 🚀 Cara Menjalankan

1. Pastikan semua dependensi terpasang:

```bash
pip install -r requirements.txt
```

2. Jalankan notebook:

```bash
jupyter notebook Cards_Image_Classification.ipynb
```

Atau bisa juga menggunakan Kaggle Notebook, Google Colab, atau environment lain berbasis Jupyter.

## 📦 Dependensi

Disimpan dalam `requirements.txt`:

```txt
tensorflow>=2.11.0
numpy>=1.21.0
pandas>=1.3.0
opencv-python-headless>=4.5.5.64
tqdm>=4.64.0
scikit-learn>=1.0.2
```

## 📈 Hasil

Model dilatih selama 10 epoch dengan akurasi validasi yang meningkat signifikan. Evaluasi dilakukan menggunakan akurasi dan prediksi label untuk melihat kinerja klasifikasi multi-kelas.

## 📌 Catatan

- Gunakan GPU untuk mempercepat proses training (disarankan runtime dengan GPU di Colab/Kaggle).
- Model dapat disimpan ke `.h5` untuk keperluan deploy.
- Notebook ini dapat dikembangkan lebih lanjut dengan integrasi GradCAM, confusion matrix, atau deploy ke Streamlit/Gradio.

## 🧑‍💻 Kontributor
- Ryan Hidayat  
- Dibuat dengan ❤️ menggunakan TensorFlow dan EfficientNet
