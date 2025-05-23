# Laporan Proyek Machine Learning - Haiqel Aziizul Hakeem

## Domain Proyek
Inflasi merupakan indikator ekonomi yang sangat penting bagi pengambil kebijakan, pelaku bisnis, dan masyarakat secara umum. Sticky Price Consumer Price Index (CPI) adalah metrik inflasi khusus yang mengukur perubahan harga pada barang dan jasa yang cenderung lambat berubah atau "lengket" (sticky). Prediksi yang akurat terhadap inflasi ini sangat penting karena memiliki dampak signifikan terhadap kebijakan moneter, perencanaan keuangan, dan stabilitas ekonomi. Inflasi jenis sticky price memiliki karakteristik khusus dimana harga barang dan jasa tidak langsung berubah meskipun terjadi perubahan kondisi ekonomi. <br>  
Penelitian oleh Hermansah et al. (2024) menggunakan model RNN-LSTM untuk meramalkan inflasi di Indonesia. Studi ini mengevaluasi berbagai fungsi aktivasi dan metode pembaruan bobot, menemukan bahwa kombinasi fungsi aktivasi logistik dan optimisasi Stochastic Gradient Descent (SGD) menghasilkan akurasi tertinggi, melampaui model tradisional seperti ARIMA dan ETS [[1]](https://journal.uii.ac.id/ENTHUSIASTIC/article/view/36361). Penelitian ini menunjukkan bahwa model RNN seperti LSTM telah digunakan untuk memprediksi inflasi. <br>  
Selain itu ada juga penelitian yang dilakukan oleh Yang dan Guo (2021) menerapkan model GRU-RNN untuk memprediksi inflasi di Tiongkok. Hasilnya menunjukkan bahwa model ini memiliki kinerja yang baik dalam memproses data deret waktu yang kompleks dan nonlinier, serta mengungguli model tradisional dalam akurasi prediksi [[2]](https://pmc.ncbi.nlm.nih.gov/articles/PMC8390133/). Penelitian ini menunjukkan bahwa model GRU cocok untuk memprediksi kasus inflasi. <br>
Project ini penting karena: 

1. Mengantisipasi Perubahan Ekonomi <br> Membantu memprediksi arah inflasi yang sangat penting untuk kebijakan fiskal dan moneter
2. Mendukung Pengambilan Keputusan Keuangan <br> Keputusan yang dapat diambil seperti: anggaran dan kebijakan subsidi, harga produk dan upah, dan juga investasi dan strategi pasar.
3. Mengelola Risiko <br> Dapat meminimalisir resiko yang dapat ditimbulkan oleh inflasi parah.
4. Perencanaan Jangka Panjang <br> Membantu membuat rencana dan keputusan jangka panjang berdasarkan data (data-driven decision) 

## Business Understanding
### Problem Statements
1. Bagaimana cara memahami pola dan tren historis dari Sticky Price CPI agar dapat digunakan sebagai dasar dalam membangun model prediktif?
2. Bagaimana cara melakukan preprocessing pada data time series Sticky Price CPI agar dapat dikonversi menjadi format yang sesuai untuk pemodelan menggunakan deep learning?
3. Bagaimana merancang dan mengembangkan model prediktif berbasis arsitektur LSTM dan GRU untuk memprediksi inflasi dari data Sticky Price CPI?
4. Bagaimana cara mengevaluasi performa dari model LSTM dan GRU dalam memprediksi CPI berdasarkan metrik evaluasi yang sesuai untuk data time series?
5. Model manakah yang memiliki performa terbaik berdasarkan hasil evaluasi, dan bagaimana kriteria pemilihannya ditentukan?
6. Seberapa baik performa model terbaik saat digunakan untuk melakukan prediksi inflasi pada data Sticky Price CPI yang belum pernah dilihat sebelumnya?

### Goals
1. Memahami trend dari Sticky Price CPI
2. Preprocess dan mengubah data time series ke dalam deep learning model
3. Mengembangkan dan membandingkan model LSTM, dan Gru
4. Evaluasi kinerja model
5. Menentukan model yang terbaik berdasarkan hasil evaluasi
6. Test model terbaik untuk prediksi

### Solution Statement
Membangun model prediksi inflasi menggunakan model deep learning LSTM dan GRU berdasarkan dataset nyata dari [Federal Reserve Bank of St. Louis](https://fred.stlouisfed.org/series/CORESTICKM159SFRBATL) lalu diproses (scaling, transform, etc.) agar bisa memprediksi dengan akurat untuk beberapa bulan ke depan. Model dengan loss terkecil yang akan dijadikan sebagai model untuk inference (prediksi).

## Data Understanding
Dataset yang digunakan merupakan data real yang diambil dari Federal Reserve Bank of St. Louis mengenai index harga dari barang-barang dengan harga yang relatif jarang berubah-ubah (sticky price), contohnya adalah: biaya pendidikan, layanan medis, biaya perumahan, dll. Data ini penting untuk menentukan harga dari produk dan jasa dengan harga yang jarang berubah-ubah untuk mempertimbangkan inflasi di masa depan. Sticky Price CPI sendiri memiliki implikasi penting untuk kebijakan moneter. <br>
Dataset dapat diunduh dari website [FRED](https://fred.stlouisfed.org/series/CORESTICKM159SFRBATL) <br>  
### Struktur Dataset:
- Jumlah Baris: 830
- Jumlah Kolom: 2
- Jumlah Nilai Hilang: 0
- Jumlah Data Duplikat: 0
- Jumlah Nilai Unik:
  - observation_date: 830
  - cpi: 705
### Variabel atau Fitur pada Dataset:
- observation_date : Merupakan variabel waktu dilakukannya observasi index consumer price dengan sticky price. Formatnya YYYY-MM-DD. Dimana observasi dilakukan setiap tanggal 1 pada tiap bulannya dimulai dari tahun 1955 sampai 2025. Tipe datanya adalah datetime secara default.
- CPALTT01USM657N : Pada notebook dilakukan rename menjadi cpi. Yakni merupakan variabel Sticky Price Consumer Index. Nilai yang diambil dikalkulasi berdasarkan subset dari barang-barang dan jasa-jasa yang termasuk ke dalam CPI tetapi dengan harga yang relatif jarang berubah. Tipe datanya adalah float64 secara default. <br>
Berikut adalah visualisasi dari dataset dari wakatu ke waktu: 
<br>![stickyPriceCpi](https://github.com/user-attachments/assets/f2418a59-b29c-48d1-8195-84751071074d)
### Kualitas dan Karakteristik Data
- Kualitas Data Baik: Tidak terdapat nilai yang hilang maupun duplikat, yang berarti data siap untuk diproses.
- Distribusi Waktu Konsisten: Semua observasi berada dalam interval waktu bulanan tanpa missing timestamp.
- Potensi Outlier: Nilai minimum yang jauh lebih rendah dari rata-rata (sekitar -1.915) dapat mengindikasikan anomali atau kondisi ekonomi ekstrim pada periode tertentu. Akan tetapi dalam konteks ini, hal ini dapat menunjukkan index inflasi yang sangat ekstrim sehingga tidak perlu dihilangkan.

## Data Preparation
Berikut adalah langkah-langkah yang dilakukan saat Data Preparation:
1. Scaling/Normalization: <br> Variabel CPI diskalakan menggunakan:
   ```python
   MinMaxScaler()
   ```
   seperti pada code snippet berikut:
   ```python
   from sklearn.preprocessing import MinMaxScaler
   scaler = MinMaxScaler()
   scaled = scaler.fit_transform(df[['cpi']])
   ```
   Tahap ini penting karena model seperti LSTM dan GRU sensitif terhadap skala pada data. Sehingga digunakan `MinMaxScaler` untuk mengubah nilai menjadi rentang 0-1, sehingga model dapat lebih stabil dan menghindari vanishing gradient.
   
2. Function Sequence: <br> Selanjutnya adalah membuat function `create_sequences()`:
   ```python
   def create_sequences(data, seq_len):
    X, y = [], []
    for i in range(len(data) - seq_len):
        X.append(data[i:i+seq_len])
        y.append(data[i+seq_len])
    return np.array(X), np.array(y)
   ```
   Tujuannya adalah membuat urutan data time series sebagai input (X) dan label/prediksi berikutnya (y) untuk model seperti LSTM dan GRU, yang dirancang untuk memproses data sekuensial.

3. Mendeklarasikan `SEQ_LEN` dan menjalankan function `create_sequences()`
   ```python
   SEQ_LEN = 12
   X, y = create_sequences(scaled, SEQ_LEN)
   ```
   `SEQ_LEN` adalah variabel yang digunakan untuk menentukan seberapa panjang waktu yang digunakan untuk memprediksi inflasi. Pada kali ini akan menggunakan 12, yang artinya model akan belajar berdasarkan 12 bulan terakhir untuk memprediksi bulan berikutnya. Penentuan `SEQ_LEN` mencerminkan berapa banyak informasi masa lalu yang dianggap relevan. Lalu X dan y untuk menjalankan `create_sequences()` dengan parameter `scaled` dan `SEQ_LEN`.

4. Data Splitting: <br> Membagi dataset sebanyak 80:20 seperti pada code snippet:
    ```python
    split = int(0.8 * len(X))
    X_train, y_train = X[:split], y[:split]
    X_test, y_test = X[split:], y[split:]
    ```
    Dimana dataset dibagi menjadi 80% train dan 20% test. Proses ini membantu mengatasi proses overfitting agar model dapat memprediksi data baru yang tidak ada pada training.

## Modeling
Pada tahap ini dilakukan modeling seperti:
1. `Callback`: <br> Callback yang digunakan adalah:
   ```python
   callbacks = [
      ...
   ]
   ```
   - `EarlyStopping`: Untuk menghentikan training lebih awal jika model tidak berkembang dalam beberapa epoch dan mencegah overfitting. Parameter yang digunakan adalah:
        - `patience = 5` : menunggu epoch 5 jika tanpa peningkatan
        - `restore_best_weights = True` : mengembalikan bobot terbaik sebelum val_loss memburuk
        - `monitor = 'val_loss'` : memantau loss pada data validasi
      ```python
      EarlyStopping(patience=5, restore_best_weights=True, monitor='val_loss')
      ```
   - `ReduceLROnPlateau`: Menurunkan learning rate model jika tidak berkembang. Parameter yang digunakan adalah:
        - `factor = 0.5` : mengurangi learning rate setengahnya
        - `patience = 3` : menunggu 3 epoch untuk mengurangi learning rate
      ```python
      ReduceLROnPlateau(factor=0.5, patience=3, verbose=1)
      ```
   
2. Model LSTM: <br> Membangun model LSTM untuk memprediksi nilai CPI berdasarkan urutan waktu.
   ```python
   lstm_model = Sequential([
       Input(shape=(SEQ_LEN, 1)),
       LSTM(64, activation='tanh', return_sequences=False),
       Dense(1)
   ])
   lstm_model.compile(optimizer='adam', loss='mse')
   ```
   Parameter yang digunakan:
   - `Input(shape=(SEQ_LEN, 1))` : Input pada model yakni 12 bulan dan 1 fitur per bulan
   - `LSTM(64, activation='tanh')` :
        - 64 unit memori untuk menangkap pola sekuensial.
        - `return_sequences=False` : hanya output terakhir yang diambil.
   - `Dense(1)` : Output terakhir berupa satu nilai, yakni CPI yang diprediksi
   - `optimizer='adam'` : Optimisasi Adam yang dinilai adaptif, cepat, dan stabil
   - `loss - 'mse'` : Mean Squared Error untuk regresi dan time series
   <br>
   Kemudian kita melatih model dengan `fit()` : <br>

   ```python
   lstm_history = lstm_model.fit(
       X_train, y_train,
       validation_data=(X_test, y_test),
       epochs=50,
       batch_size=32,
       callbacks=callbacks,
       verbose=2
   )
   ```
   Parameter yang digunakan: <br>  
   - `epochs=50`: maksimum 50 pengulangan pelatihan (dihentikan lebih awal jika perlu)
   - `batch_size=32`: ukuran batch pelatihan
   - `validation_data=(X_test, y_test)`: untuk evaluasi selama training
   - `callbacks`: untuk kontrol pelatihan (early stopping dan LR scheduler)
   <br>
   Kemudian prediksi dan evaluasi model dengan:

   ```python
   lstm_preds = lstm_model.predict(X_test)
   lstm_preds_inv = scaler.inverse_transform(lstm_preds)
   y_test_inv = scaler.inverse_transform(y_test)
   
   mae = mean_absolute_error(y_test_inv, lstm_preds_inv)
   rmse = np.sqrt(mean_squared_error(y_test_inv, lstm_preds_inv))
   ```
   Dimana `inverse_transform()` untuk mengembalikan hasil prediksi ke skala asli dan juga mengukur loss dengan MAE dan RMSE.

4. Model GRU : <br>
   Sama seperti pada LSTM, yang membedakan adalah model yang digunakan yakni dengan GRU. Sehingga hanya perlu mengganti kata LSTM pada kode sebelumnya dengan GRU. Model GRU (Gated Recurrent Unit) digunakan sebagai alternatif LSTM yang lebih ringan dan cepat.
   Contohnya : <br>
   ```python
   gru_model = Sequential([
       Input(shape=(SEQ_LEN, 1)),
       GRU(64, activation='tanh', return_sequences=False),
       Dense(1)
   ])
   ```
   
## Evaluation

Dalam proyek ini, model yang digunakan bertujuan untuk memprediksi nilai inflasi (CPI) berdasarkan data historis time series. Karena tipe permasalahan ini merupakan regresi time series, maka metrik evaluasi yang digunakan adalah **Mean Absolute Error (MAE)** dan **Root Mean Squared Error (RMSE)**. Metrik ini dipilih karena sesuai untuk mengukur kesalahan antara nilai aktual dan nilai prediksi secara langsung dalam satuan yang sama dengan data aslinya.

### Metrik Evaluasi

1. MAE (Mean Absolute Error): <br> Mengukur rata-rata dari selisih absolut antara nilai aktual (`yᵢ`) dan nilai prediksi (`ŷᵢ`). Metrik ini memberikan gambaran seberapa besar rata-rata kesalahan prediksi tanpa memperhitungkan arah kesalahan. Semakin kecil nilai MAE, semakin akurat prediksi model.
  
  $$
  \text{MAE} = \frac{1}{n} \sum_{i=1}^{n} |y_i - \hat{y}_i|
  $$
  
2. RMSE (Root Mean Squared Error): <br> menghitung akar kuadrat dari rata-rata kuadrat selisih antara nilai aktual dan prediksi. Karena menggunakan kuadrat dari error, RMSE lebih sensitif terhadap outlier dibanding MAE. Cocok untuk menyoroti kesalahan besar dalam prediksi. <br>

  $$
  \text{RMSE} = \sqrt{\frac{1}{n} \sum_{i=1}^{n} (y_i - \hat{y}_i)^2}
  $$

### Hasil Evaluasi

- LSTM
  - MAE = 0.2819  
  - RMSE = 0.3440

- GRU
  - MAE = 0.2592  
  - RMSE = 0.3182

Berdasarkan hasil evaluasi di atas, model GRU memiliki performa yang lebih baik dibandingkan LSTM. Hal ini ditunjukkan oleh nilai MAE dan RMSE yang lebih rendah, yang berarti prediksi GRU lebih dekat dengan nilai aktual. Selain itu, GRU cenderung lebih efisien dalam proses pelatihan karena memiliki struktur yang lebih sederhana dibandingkan LSTM, namun tetap mampu menangkap pola temporal pada data CPI dengan baik.
