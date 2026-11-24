# Retail Demand Forecasting & Inventory Simulator

Simulasi stok ritel harian untuk 10 toko dan 50 produk menggunakan prediksi **LightGBM** dan perhitungan *dynamic safety stock*. Proyek ini membandingkan total biaya operasional—gabungan biaya simpan gudang (*overstock*) dan kerugian akibat kehabisan barang (*lost sales*)—antara metode tebakan konvensional (*baseline*) melawan model *machine learning*.

![Preview Dashboard](assets/dashboard_preview.png)

## Ringkasan Teknis

* **Dataset:** Store Item Demand Forecasting (data historis penjualan harian 2013–2017).
* **Validasi:** *Time-based split* menggunakan periode Q4 2017 (Oktober–Desember) sebagai data uji untuk mencegah *data leakage*.
* **Fitur Utama:** *Lag features* (7, 14, 28 hari), *rolling mean & standard deviation*, serta fitur kalender (hari dalam seminggu, bulan, akhir pekan).
* **Logika Inventaris:** Stok harian dihitung dari hasil prediksi ditambah *safety stock* dinamis berdasarkan standar deviasi error historis tiap item ($Z \times \sigma_{\text{error}}$).

## Hasil Simulasi (Q4 2017)

*(Sesuaikan angka di bawah ini dengan output tabel akhir di `04_business_simulation.ipynb`)*

| Metrik | Metode Lama (Lag-7) | Model LightGBM |
| :--- | :---: | :---: |
| **Total Biaya Simpan** | Rp ... | Rp ... |
| **Potensi Lost Sales** | Rp ... | Rp ... |
| **Total Kerugian** | Rp ... | Rp ... |
| **Service Level Aktual** | ...% | ...% |

## Struktur Folder

```text
├── .streamlit/
│   └── config.toml                 # Konfigurasi tema tampilan Streamlit
├── assets/
│   └── dashboard_preview.png       # Screenshot aplikasi
├── data/
│   └── processed/
│       └── val_predictions.parquet # Hasil prediksi validasi Q4 2017
├── notebooks/
│   ├── 01_eda.ipynb                # Analisis pola tren dan musiman
│   ├── 02_feature_engineering.ipynb# Pembuatan fitur lag, rolling, dan split data
│   ├── 03_modeling_evaluation.ipynb# Training LightGBM dan evaluasi MAE/RMSE
│   └── 04_business_simulation.ipynb# Simulasi biaya inventaris dan safety stock
├── app.py                          # Dashboard interaktif Streamlit
├── requirements.txt
└── README.md