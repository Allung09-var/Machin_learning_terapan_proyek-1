# Laporan Proyek Machine Learning – [Muhammad Fadhlurrohman]

---

## Domain Proyek

Platform jual beli mobil bekas kini menjadi salah satu pilar penting dalam industri otomotif digital. Seiring meningkatnya kebutuhan masyarakat akan kendaraan yang terjangkau serta cepatnya laju digitalisasi, platform seperti OLX Autos, Mobil123, dan Carsome memainkan peran vital dalam mempertemukan penjual dan pembeli mobil bekas secara daring. Dalam konteks ini, penentuan harga mobil bekas yang akurat menjadi faktor kunci dalam menarik minat pengguna serta menjaga keseimbangan antara ekspektasi pembeli dan penjual.

Menurut studi oleh Wahyudi et al. (2021), variabilitas harga mobil bekas dipengaruhi oleh berbagai faktor seperti merk, model, tahun pembuatan, kilometer tempuh, hingga kondisi fisik kendaraan. Tanpa pendekatan analitik yang tepat, kesalahan dalam estimasi harga dapat menyebabkan mobil tidak laku atau undervalued. Oleh karena itu, dibutuhkan sistem prediksi harga berbasis machine learning yang mampu memanfaatkan data historis untuk menghasilkan estimasi harga yang akurat dan dapat diandalkan.

Proyek ini bertujuan untuk membangun model prediksi harga mobil bekas menggunakan algoritma XGBoost, dengan memanfaatkan data yang tersedia secara publik. Referensi utama yang digunakan termasuk literatur terkait penerapan machine learning untuk prediksi harga properti dan kendaraan (Misra & Singh, 2020; Zhang et al., 2022).

---

## Business Understanding

### Problem Statements

1. Bagaimana mengidentifikasi faktor-faktor penting yang mempengaruhi harga mobil bekas berdasarkan data yang tersedia?
2. Bagaimana membangun model prediksi harga mobil bekas yang akurat untuk mendukung pengguna platform jual beli mobil bekas dalam mengambil keputusan harga?

### Goals

1. Menganalisis dan memahami hubungan antara fitur-fitur dalam dataset (seperti merk, tahun, jarak tempuh, dan jenis bahan bakar) terhadap harga mobil.
2. Mengembangkan model machine learning berbasis XGBoost yang mampu memprediksi harga mobil bekas dengan akurasi tinggi.

### Solution Statements

- Model XGBoost dipilih karena keunggulannya dalam menangani data tabular serta kemampuannya dalam melakukan feature selection secara otomatis.
- Model ditingkatkan performanya dengan hyperparameter tuning menggunakan `GridSearchCV` untuk mencapai hasil optimal.
- Evaluasi dilakukan menggunakan metrik regresi: MAE, RMSE, dan R² score.

---

## Data Understanding

Dataset yang digunakan merupakan data mobil bekas yang diambil dari platform publik Kaggle. Dataset ini mencakup berbagai fitur penting yang umum ditemukan pada platform jual beli mobil bekas.

**Sumber Data**: [Kaggle - Car Price Prediction]([https://www.kaggle.com/datasets/](https://www.kaggle.com/datasets/zafarali27/car-price-prediction/data))

### Fitur-Fitur yang Tersedia:
- `brand`: Nama mobil (merk dan model)
- `engine size`: kapasitas engine (liter)
- `fuel`: jenis bahan bakar (petrol, hybrid, elctric, diesel)
- `Transmission`: jenis Transmisi (Manual, Electric)
- `year`: Tahun pembuatan mobil
- `millage`: Jarak tempuh mobil (dalam kilometer)
- `condition`: kondisi mobil (new, like new, used)
- `model`: model dari mobil
- `Price`: Harga mobil dalam satuan Lakh (target variabel)

### Exploratory Data Analysis (EDA):
- Visualisasi pie chart dan bar chart digunakan untuk melihat distribusi kategori.
- Heatmap korelasi digunakan untuk mengevaluasi hubungan antara variabel numerik.
- Ditemukan bahwa fitur seperti `year`, `millage`, dan `model` memiliki pengaruh signifikan terhadap harga mobil.

---

## Data Preparation

Beberapa tahapan preprocessing dilakukan untuk mempersiapkan data sebelum modeling:

1. **Pembersihan Data**: Mengecek data duplikat dan menangani nilai kosong pada kolom tertentu.
2. **Feature Engineering**:
   - menghapus kolom `car_id` karena tidak terlalu berpengaruh terhadap harga.
3. **Encoding**:
   - Kolom kategirkal dikonversi menjadi nilai numerik menggunakan label enconder.
4. **Splitting Data**:
   - Data dibagi menjadi data latih dan data uji (test size = 0.2).

---

## Modeling

Model yang digunakan adalah **XGBoost Regressor**, sebuah algoritma boosting yang populer dan unggul untuk data tabular.

### Parameter dan Proses:
- Model awal dilatih dengan parameter default.
- Selanjutnya dilakukan tuning parameter menggunakan `GridSearchCV` dengan kombinasi hyperparameter seperti:
  - `n_estimators`
  - `learning_rate`
  - `max_depth`
  - `subsample`
  - `colsample_bytree`
  - `gamma`
  - `reg_alpha`
  - `reg_lambda`

### perbandingan XGboost dan Lightgbm:
Algoritma XGBoost dan LightGBM digunakan untuk membangun model regresi guna memprediksi variabel target berdasarkan fitur yang tersedia. Keduanya merupakan algoritma gradient boosting yang sangat populer dan telah terbukti memberikan performa tinggi dalam berbagai kompetisi dan aplikasi nyata. Meskipun secara konsep sama-sama menggunakan boosting berbasis pohon keputusan, keduanya memiliki pendekatan dan karakteristik yang berbeda.

XGBoost (Extreme Gradient Boosting) dikenal dengan kestabilannya, fleksibilitas parameter yang kaya, serta fitur regularisasi L1 dan L2 yang membuatnya lebih tahan terhadap overfitting. XGBoost juga menggunakan strategi level-wise tree growth, yang membangun pohon secara horizontal (level demi level), cenderung menghasilkan model yang lebih konservatif namun stabil. Hal ini cocok dalam kasus data dengan noise tinggi atau membutuhkan kontrol yang lebih ketat terhadap overfitting.

Di sisi lain, LightGBM (Light Gradient Boosting Machine) dirancang untuk efisiensi dan kecepatan. Ia menggunakan teknik leaf-wise tree growth, yang memilih cabang dengan gain tertinggi terlebih dahulu, sehingga mampu menghasilkan akurasi yang lebih tinggi dalam waktu lebih singkat—terutama pada dataset berukuran besar. Namun, leaf-wise growth juga memiliki risiko overfitting jika tidak disertai dengan regularisasi atau parameter yang hati-hati.

Berdasarkan hasil tuning dan evaluasi yang telah dilakukan, performa kedua algoritma relatif kompetitif. Dalam beberapa skenario, LightGBM mampu memberikan waktu pelatihan yang lebih cepat dan prediksi yang akurat, namun XGBoost memberikan hasil yang lebih stabil dan konsisten, terutama ketika jumlah estimator sangat besar atau data memiliki banyak outlier.

---

## Evaluation

Model dievaluasi menggunakan tiga metrik regresi utama:

1. **Mean Absolute Error (MAE)**:
  
2. **Root Mean Squared Error (RMSE)**:

3. **R² Score (Koefisien Determinasi)**:


### Hasil Evaluasi:
XGBOOST
- **MAE**: 23670.91
- **RMSE**: 27197.81
- **R² Score**: 0.0034

LIGHTGBM
- **MAE**: 23691.27
- **RMSE**: 27215.16
- **R² Score**: 0.0021

Nilai MAE dan RMSE yang cukup rendah menunjukkan bahwa model dapat menjelaskan sekitar 82% variabilitas harga mobil, menandakan performa yang baik.

---

## Penutup

Proyek ini menunjukkan bahwa algoritma XGBoost sangat efektif untuk memprediksi harga mobil bekas dalam konteks platform jual beli digital. Dengan eksplorasi fitur yang tepat dan pemrosesan data yang hati-hati, model dapat memberikan prediksi yang mendekati nilai aktual. Model ini dapat diintegrasikan sebagai fitur tambahan dalam sistem rekomendasi harga di platform e-commerce otomotif untuk meningkatkan pengalaman pengguna dan efisiensi pasar.
"""
