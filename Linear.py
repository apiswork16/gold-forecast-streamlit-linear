import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import requests
from datetime import datetime, timedelta

# ============================
# PAGE CONFIG
# ============================
st.set_page_config(
    page_title="Gold Price Forecasting",
    page_icon="🟡",
    layout="centered"
)

# ============================
# HEADER
# ============================
st.title("🟡 Gold Price Forecasting")
st.caption("Forecast Tren Harga Emas Dunia menggunakan Regresi Linear")
st.divider()

# ============================
# SIDEBAR
# ============================
st.sidebar.header("⚙️ Pengaturan Analisis")

hist_period = st.sidebar.selectbox(
    "Periode Data Historis",
    ["6 Bulan", "1 Tahun", "2 Tahun", "3 Tahun"]
)

pred_days = st.sidebar.slider(
    "Periode Prediksi (hari)",
    7, 180, 30
)

st.sidebar.subheader("💱 Kurs USD → IDR")
kurs_mode = st.sidebar.radio(
    "Sumber Kurs",
    ["API (Real-time)", "Manual"]
)

@st.cache_data
def get_kurs_api():
    url = "https://open.er-api.com/v6/latest/USD"
    return requests.get(url).json()["rates"]["IDR"]

if kurs_mode == "API (Real-time)":
    kurs_idr = get_kurs_api()
else:
    kurs_idr = st.sidebar.number_input(
        "Masukkan Kurs USD → IDR",
        min_value=10000,
        max_value=20000,
        value=16000,
        step=50
    )

if st.sidebar.button("🔄 Refresh Data"):
    st.cache_data.clear()
    st.rerun()

st.sidebar.divider()

# ============================
# DATE RANGE
# ============================
today = datetime.today()

if hist_period == "6 Bulan":
    start_date = today - timedelta(days=180)
elif hist_period == "1 Tahun":
    start_date = today - timedelta(days=365)
elif hist_period == "2 Tahun":
    start_date = today - timedelta(days=365 * 2)
else:
    start_date = today - timedelta(days=365 * 3)

# ============================
# LOAD GOLD DATA
# ============================
@st.cache_data
def load_gold(start):
    df = yf.download("GC=F", start=start, progress=False)
    df = df[['Close']].reset_index()
    df.columns = ['ds', 'price_usd']
    df['ds'] = pd.to_datetime(df['ds'])
    return df.dropna()

df = load_gold(start_date)

# ============================
# CONVERT TO IDR / gram
# ============================
df['price_idr'] = df['price_usd'] * kurs_idr / 31.1035

# ============================
# KPI
# ============================
current_usd = df['price_usd'].iloc[-1]
current_idr = df['price_idr'].iloc[-1]

st.info(f"📅 Data historis: **{len(df)} hari** ({hist_period})")

c1, c2, c3 = st.columns(3)
c1.metric("🌍 Harga Emas Dunia", f"USD {current_usd:,.2f}")
c2.metric("💱 Kurs USD → IDR", f"Rp {kurs_idr:,.0f}")
c3.metric("🟡 Harga Spot Dunia", f"Rp {current_idr:,.0f}/gr")

st.divider()

# ============================
# LINEAR REGRESSION (OLS MANUAL)
# ============================
df['t'] = (df['ds'] - df['ds'].min()).dt.days

x = df['t'].values
y = df['price_idr'].values

x_mean = np.mean(x)
y_mean = np.mean(y)

beta_1 = np.sum((x - x_mean) * (y - y_mean)) / np.sum((x - x_mean) ** 2)
beta_0 = y_mean - beta_1 * x_mean

df['trend'] = beta_0 + beta_1 * x
df['residual'] = y - df['trend']
noise_std = df['residual'].std()

# ============================
# FUTURE PREDICTION
# ============================
future_x = np.arange(x[-1] + 1, x[-1] + pred_days + 1)
future_y = beta_0 + beta_1 * future_x

future_dates = pd.date_range(
    start=df['ds'].iloc[-1] + timedelta(days=1),
    periods=pred_days
)

future_df = pd.DataFrame({
    "ds": future_dates,
    "trend": future_y
})

future_price = future_y[-1]
change_pct = (future_price - current_idr) / current_idr * 100

trend_status = "📈 Tren Naik" if change_pct > 0 else "📉 Tren Turun / Stabil"

# ============================
# SUMMARY
# ============================
st.subheader("🔎 Ringkasan Prediksi")

st.success(f"""
**Harga saat ini:** Rp {current_idr:,.0f} / gram  
**Estimasi {pred_days} hari ke depan:** Rp {future_price:,.0f} / gram  
**Perubahan estimasi:** {change_pct:.2f}%  
**Status tren:** **{trend_status}**
""")

st.divider()

# ============================
# CHART – HISTORIS + PREDIKSI
# ============================
st.subheader("📊 Grafik Harga Historis & Prediksi")

fig, ax = plt.subplots(figsize=(10, 5))

# historis
ax.plot(df['ds'], df['price_idr'], label="Harga Historis", linewidth=2)

# trend historis
ax.plot(df['ds'], df['trend'], linestyle="--", label="Garis Tren")

# prediksi
ax.plot(future_df['ds'], future_df['trend'], linestyle="--", label="Prediksi")

# noise band
ax.fill_between(
    df['ds'],
    df['trend'] - noise_std,
    df['trend'] + noise_std,
    alpha=0.2,
    label="Noise (ketidakpastian)"
)

ax.set_xlabel("Tanggal")
ax.set_ylabel("Rp / gram")
ax.legend()
ax.grid(alpha=0.3)

st.pyplot(fig)

# ============================
# CARA PAKAI
# ============================
st.subheader("📘 Cara Menggunakan Aplikasi")

st.markdown("""
1. Pilih **periode data historis** untuk melatih model.
2. Tentukan **jumlah hari prediksi** ke depan.
3. Pilih sumber **kurs USD → IDR**.
4. Grafik menunjukkan:
   - Harga emas historis
   - Garis tren regresi linear
   - Prediksi harga masa depan
5. Fokus pada **arah tren**, bukan angka harian.
""")

# ============================
# DISCLAIMER
# ============================
st.subheader("⚠️ Disclaimer")

st.caption("""
Aplikasi ini dibuat untuk **tujuan edukasi dan akademik**.  
Regresi linear digunakan untuk menangkap **tren umum** harga emas dunia.  
Model ini tidak mempertimbangkan faktor eksternal seperti geopolitik dan kebijakan moneter.
""")
