import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# 1. Konfigurasi Halaman
st.set_page_config(
    page_title="Retail Demand & Inventory Simulator",
    page_icon="📦",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 2. Custom CSS untuk KPI Cards & Proporsi Layar
st.markdown("""
<style>
    /* Styling Kartu KPI */
    div[data-testid="stMetric"] {
        background: linear-gradient(135deg, rgba(30, 41, 59, 0.95), rgba(15, 23, 42, 0.98));
        border: 1px solid rgba(59, 130, 246, 0.3);
        border-left: 4px solid #3B82F6;
        padding: 10px 14px;
        border-radius: 10px;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.25);
    }
    div[data-testid="stMetricValue"] {
        font-size: 1.4rem;
        font-weight: 700;
        color: #F8FAFC;
    }
    /* Ruang atas 3.6rem agar badge tidak tertimpa header bar bawaan Streamlit */
    .block-container {
        padding-top: 3.6rem !important;
        padding-bottom: 0.5rem !important;
    }
    /* Merapatkan jarak judul h1 */
    h1 {
        padding-top: 0.2rem !important;
        padding-bottom: 0.2rem !important;
        font-size: 2rem !important;
    }
    .subtitle-badge {
        display: inline-block;
        background-color: rgba(59, 130, 246, 0.15);
        color: #60A5FA;
        padding: 3px 10px;
        border-radius: 20px;
        font-size: 0.75rem;
        font-weight: 600;
        border: 1px solid rgba(59, 130, 246, 0.3);
    }
</style>
""", unsafe_allow_html=True)

# 3. Load Data
@st.cache_data
def load_data():
    df = pd.read_parquet('data/processed/val_predictions.parquet')
    df['date'] = pd.to_datetime(df['date'])
    return df

df = load_data()

# 4. Sidebar untuk Kontrol Parameter Simulasi
with st.sidebar:
    st.markdown("## 🎛️ Control Panel")
    st.caption("Atur parameter operasional untuk melihat simulasi biaya secara real-time.")
    
    st.markdown("### 🏪 Filter Entitas")
    col_s1, col_s2 = st.columns(2)
    with col_s1:
        selected_store = st.selectbox("Cabang Toko", sorted(df['store'].unique()))
    with col_s2:
        selected_item = st.selectbox("ID Produk", sorted(df['item'].unique()))
        
    st.markdown("---")
    st.markdown("### 💰 Parameter Finansial")
    lost_sales_cost = st.number_input("Rugi Lost Sales / Unit (Rp)", value=15000, step=1000)
    holding_cost = st.number_input("Biaya Simpan / Unit / Hari (Rp)", value=3000, step=500)
    
    st.markdown("---")
    st.markdown("### 🛡️ Kebijakan Stok")
    service_level_option = st.selectbox(
        "Target Service Level",
        options=[("90% (Z = 1.28)", 1.28), ("95% (Z = 1.65)", 1.65), ("99% (Z = 2.33)", 2.33)],
        index=1,
        format_func=lambda x: x[0]
    )
    z_score = service_level_option[1]

# 5. Filter & Hitung Simulasi untuk Toko & Produk Terpilih
df_sub = df[(df['store'] == selected_store) & (df['item'] == selected_item)].sort_values('date').copy()

def simulate_item(data, pred_col):
    error = data['sales'] - data[pred_col]
    std_err = error.std()
    safety_stock = np.ceil(z_score * std_err)
    stok_siap = np.maximum(0, np.round(data[pred_col]) + safety_stock)
    
    selisih = stok_siap - data['sales']
    overstock = np.where(selisih > 0, selisih, 0)
    stockout = np.where(selisih < 0, np.abs(selisih), 0)
    
    total_cost = (overstock * holding_cost).sum() + (stockout * lost_sales_cost).sum()
    actual_sl = (stockout == 0).mean() * 100
    return stok_siap, safety_stock, total_cost, actual_sl, overstock.sum(), stockout.sum()

stok_base, ss_base, cost_base, sl_base, ov_base, st_base = simulate_item(df_sub, 'sales_lag_7')
stok_lgb, ss_lgb, cost_lgb, sl_lgb, ov_lgb, st_lgb = simulate_item(df_sub, 'pred_lgb')

df_sub['stok_disiapkan_lgb'] = stok_lgb
penghematan = cost_base - cost_lgb
persen_hemat = (penghematan / cost_base * 100) if cost_base > 0 else 0

# 6. Header Utama
st.markdown('<span class="subtitle-badge">MACHINE LEARNING INVENTORY OPTIMIZATION</span>', unsafe_allow_html=True)
st.title("Retail Demand Forecasting & Inventory Simulator")
st.caption(f"Analisis perbandingan performa **LightGBM** vs **Metode Konvensional** untuk **Store {selected_store} • Item {selected_item}** (Periode Q4 2017).")

# 7. KPI Cards (4 Kolom Sejajar)
col1, col2, col3, col4 = st.columns(4)
col1.metric(
    label="Biaya Metode Lama",
    value=f"Rp {cost_base:,.0f}",
    delta=f"{int(st_base)} unit stockout",
    delta_color="off"
)
col2.metric(
    label="Biaya Model LightGBM",
    value=f"Rp {cost_lgb:,.0f}",
    delta=f"-Rp {penghematan:,.0f} ({persen_hemat:.1f}%)",
    delta_color="inverse"
)
col3.metric(
    label="Safety Stock Harian",
    value=f"{int(ss_lgb)} unit",
    delta=f"{int(ss_lgb - ss_base)} unit vs Lama",
    delta_color="inverse"
)
col4.metric(
    label="Service Level Aktual",
    value=f"{sl_lgb:.1f}%",
    delta=f"Target: {service_level_option[0][:3]}",
    delta_color="normal"
)

# 8. Konten Utama dengan Tabs
tab1, tab2 = st.tabs(["📈 Visualisasi Prediksi & Stok", "📋 Tabel Rekomendasi Harian"])

with tab1:
    plt.style.use('dark_background')
    fig, ax = plt.subplots(figsize=(12, 2.7))
    fig.patch.set_facecolor('#0F172A')
    ax.set_facecolor('#0F172A')

    ax.plot(df_sub['date'], df_sub['sales'], label='Penjualan Aktual', color='#38BDF8', linewidth=2.2)
    ax.plot(df_sub['date'], df_sub['pred_lgb'], label='Prediksi Permintaan (LightGBM)', color='#F59E0B', linestyle='--', linewidth=1.8)
    ax.fill_between(
        df_sub['date'], df_sub['pred_lgb'], df_sub['stok_disiapkan_lgb'],
        color='#10B981', alpha=0.18, label='Area Buffer Safety Stock'
    )
    ax.plot(df_sub['date'], df_sub['stok_disiapkan_lgb'], label='Total Stok Disiapkan', color='#10B981', linewidth=1.5)

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_color('#334155')
    ax.spines['bottom'].set_color('#334155')
    ax.tick_params(colors='#94A3B8', labelsize=8.5)
    ax.set_ylabel("Jumlah Unit", color='#CBD5E1', fontweight='bold', fontsize=9)
    ax.grid(True, linestyle='--', alpha=0.15, color='#94A3B8')
    ax.legend(loc="upper left", frameon=True, facecolor='#1E293B', edgecolor='#334155', fontsize=8.5, ncol=2)

    plt.tight_layout()
    st.pyplot(fig)

with tab2:
    tabel_tampil = df_sub[['date', 'sales', 'pred_lgb', 'stok_disiapkan_lgb']].copy()
    tabel_tampil['date'] = tabel_tampil['date'].dt.strftime('%Y-%m-%d')
    tabel_tampil['pred_lgb'] = tabel_tampil['pred_lgb'].round(1)
    tabel_tampil['Selisih Stok'] = tabel_tampil['stok_disiapkan_lgb'] - tabel_tampil['sales']
    
    tabel_tampil.columns = ['Tanggal', 'Penjualan Aktual', 'Prediksi LightGBM', 'Rekomendasi Stok', 'Sisa / Kurang Stok']
    
    st.dataframe(
        tabel_tampil,
        width="stretch",
        hide_index=True,
        height=280
    )