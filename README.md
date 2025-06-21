
# Animals Image Classification (Custom CNN)

Proyek ini merupakan klasifikasi gambar hewan menggunakan **Convolutional Neural Network (CNN)** yang dibangun dari nol dengan Keras `Sequential` API. Dataset yang digunakan berasal dari Kaggle dan mencakup 3 kelas hewan yang berbeda.

## 📁 Dataset
Dataset dapat diunduh dari:
[Kaggle: Animal Image Dataset Classification](https://www.kaggle.com/datasets/borhanitrash/animal-image-classification-dataset)

Setiap gambar diresize ke ukuran **224x224 piksel**.

## 🧠 Model Arsitektur

Model CNN yang digunakan dibangun secara berurutan sebagai berikut:

```python
model = keras.models.Sequential([
    keras.layers.Rescaling(1./255, input_shape=[224, 224, 3]),  # Normalization
    Conv2D(32, (3, 3), activation='relu', padding='same'),
    MaxPooling2D((2, 2)),
    Conv2D(64, (3, 3), activation='relu', padding='same'),
    MaxPooling2D((2, 2)),
    Conv2D(128, (3, 3), activation='relu', padding='same'),
    MaxPooling2D((2, 2)),
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(len(classes), activation='softmax')  # Ensure the output size matches the number of classes
])
```

Model dikompilasi dengan:

```python
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
```

## 🚀 Cara Menjalankan

1. Pastikan semua dependensi terpasang:

```bash
pip install -r requirements.txt
```

2. Jalankan notebook:

```bash
jupyter notebook Animal_Image_Classification.ipynb
```

3. Pastikan dataset sudah diunduh dan disimpan dalam direktori yang sesuai seperti struktur di atas.

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

Model berhasil dilatih untuk mengklasifikasikan 53 kelas kartu dengan akurasi validasi yang meningkat secara bertahap selama pelatihan.

## 📌 Catatan

- Dataset memiliki jumlah kelas besar (53 kelas), sehingga pastikan cukup epoch dan augmentasi dilakukan.
- Gunakan GPU untuk mempercepat proses training.
- Dapat dikembangkan dengan visualisasi confusion matrix, Grad-CAM, dan evaluasi lainnya.

## 🧑‍💻 Kontributor
- Ryan Rachmad Hidayat  
- Dibuat dengan ❤️ menggunakan TensorFlow dan CNN
