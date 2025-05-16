# Laporan Proyek Machine Learning - Haiqel Aziizul Hakeem

## Domain Proyek
Inflasi merupakan indikator ekonomi yang sangat penting bagi pengambil kebijakan, pelaku bisnis, dan masyarakat secara umum. Sticky Price Consumer Price Index (CPI) adalah metrik inflasi khusus yang mengukur perubahan harga pada barang dan jasa yang cenderung lambat berubah atau "lengket" (sticky). Prediksi yang akurat terhadap inflasi ini sangat penting karena memiliki dampak signifikan terhadap kebijakan moneter, perencanaan keuangan, dan stabilitas ekonomi. Inflasi jenis sticky price memiliki karakteristik khusus dimana harga barang dan jasa tidak langsung berubah meskipun terjadi perubahan kondisi ekonomi. <br>  
Penelitian oleh Hermansah et al. (2024) menggunakan model RNN-LSTM untuk meramalkan inflasi di Indonesia. Studi ini mengevaluasi berbagai fungsi aktivasi dan metode pembaruan bobot, menemukan bahwa kombinasi fungsi aktivasi logistik dan optimisasi Stochastic Gradient Descent (SGD) menghasilkan akurasi tertinggi, melampaui model tradisional seperti ARIMA dan ETS [[1]](https://journal.uii.ac.id/ENTHUSIASTIC/article/view/36361). Penelitian ini menunjukkan bahwa model RNN seperti LSTM telah digunakan untuk memprediksi inflasi. <br>  
Selain itu ada juga penelitian yang dilakukan oleh Yang dan Guo (2021) menerapkan model GRU-RNN untuk memprediksi inflasi di Tiongkok. Hasilnya menunjukkan bahwa model ini memiliki kinerja yang baik dalam memproses data deret waktu yang kompleks dan nonlinier, serta mengungguli model tradisional dalam akurasi prediksi [[2]](https://pmc.ncbi.nlm.nih.gov/articles/PMC8390133/). Penelitian ini menunjukkan bahwa model GRU cocok untuk memprediksi kasus inflasi. <br>
Project ini penting karena: <br>
1. Mengantisipasi Perubahan Ekonomi <br> Membantu memprediksi arah inflasi yang sangat penting untuk kebijakan fiskal dan moneter
2. Mendukung Pengambilan Keputusan Keuangan <br> Keputusan yang dapat diambil seperti: anggaran dan kebijakan subsidi, harga produk dan upah, dan juga investasi dan strategi pasar.
3. Mengelola Risiko <br> Dapat meminimalisir resiko yang dapat ditimbulkan oleh inflasi parah.
4. Perencanaan Jangka Panjang <br> Membantu membuat rencana dan keputusan jangka panjang berdasarkan data (data-driven decision) 

## Business Understanding
### Problem Statements
1. Bagaiman cara membaca dan mengerti data yang tersedia pada dataset Sticky Price Consumer Price Index (CPI) Less Food and Energy?
2. Bagaimana cara membangun model deep learning yang akurat menggunakan LSTM dan GRU untuk memprediksi inflasi pada Sticky Price Consumer Price Index (CPI) Less Food and Energy?
3. Apakah model tersebut cocok untuk kasus prediksi inflasi dan kenapa?

### Goals
1. Memahami tren dan data pada Sticky Price CPI Less Food and Energy melalui EDA (Exploratory Data Analysis).
2. Melakukan preprocessing dan konversi data time series ke format input deep learning, membangun dan membandingkan performa model LSTM dan GRU, serta mengevaluasi kinerja model.
5. Menentukan model terbaik berdasarkan hasil evaluasi.

### Solution Statement
Membangun model prediksi inflasi menggunakan model deep learning LSTM dan GRU berdasarkan dataset nyata dari [Federal Reserve Bank of St. Louis](https://fred.stlouisfed.org/series/CORESTICKM159SFRBATL) lalu diproses (scaling, transform, etc.) agar bisa memprediksi dengan akurat untuk beberapa bulan ke depan. Model dengan loss terkecil yang akan dijadikan sebagai model untuk inference (prediksi).

## Data Understanding
Dataset yang digunakan merupakan data real yang diambil dari Federal Reserve Bank of St. Louis mengenai index harga dari barang-barang dengan harga yang relatif jarang berubah-ubah (sticky price), contohnya adalah: biaya pendidikan, layanan medis, biaya perumahan, dll. Data ini penting untuk menentukan harga dari produk dan jasa dengan harga yang jarang berubah-ubah untuk mempertimbangkan inflasi di masa depan. Sticky Price CPI sendiri memiliki implikasi penting untuk kebijakan moneter. <br>
Dataset dapat diunduh dari website [FRED](https://fred.stlouisfed.org/series/CORESTICKM159SFRBATL) <br>  
### Variabel atau Fitur pada dataset:
- observation_date : Merupakan variabel waktu dilakukannya observasi index consumer price dengan sticky price. Formatnya YYYY-MM-DD. Dimana observasi dilakukan setiap tanggal 1 pada tiap bulannya dimulai dari tahun 1955 sampai 2025. Tipe datanya adalah datetime secara default.
- CPALTT01USM657N : Pada notebook dilakukan rename menjadi cpi. Yakni merupakan variabel Sticky Price Consumer Index. Nilai yang diambil dikalkulasi berdasarkan subset dari barang-barang dan jasa-jasa yang termasuk ke dalam CPI tetapi dengan harga yang relatif jarang berubah. Tipe datanya adalah float64 secara default. <br>
Berikut adalah visualisasi dari dataset dari wakatu ke waktu: <br>
![Visualisasi Data](/assets/stickyPriceCpi.png)
