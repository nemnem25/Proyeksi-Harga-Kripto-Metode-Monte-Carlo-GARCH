import streamlit as st
import numpy as np
import pandas as pd
from datetime import datetime
import requests
import pytz
import matplotlib.pyplot as plt


# Konfigurasi halaman Streamlit
st.set_page_config(page_title="Simulasi Monte Carlo Proyeksi Harga Kripto", layout="wide")

# Menampilkan waktu
wib = pytz.timezone("Asia/Jakarta")
st.markdown(
    f"<div style='text-align: right; font-size: 14px;'><strong>🕒 {datetime.now(wib).strftime('%A, %d %B %Y')}</strong></div>",
    unsafe_allow_html=True,
)

st.title("🪙 Simulasi Monte Carlo Proyeksi Harga Kripto")
st.markdown(
    "_Simulasi berbasis data historis untuk memproyeksikan harga kripto selama beberapa hari ke depan menggunakan metode Monte Carlo._"
)

# Mapping simbol ke ID CoinGecko
coingecko_map = {"BTC-USD": "bitcoin", "ETH-USD": "ethereum", "BNB-USD": "binancecoin"}  # Tambahkan sesuai kebutuhan

# Input pengguna
ticker_input = st.selectbox("Pilih simbol kripto:", list(coingecko_map.keys()))
coin_id = coingecko_map[ticker_input]

# Ambil data historis (gunakan cache untuk efisiensi)
@st.cache_data
def get_historical_data(coin_id):
    url = f"https://api.coingecko.com/api/v3/coins/{coin_id}/market_chart"
    params = {"vs_currency": "usd", "days": "365"}
    resp = requests.get(url, params=params)
    resp.raise_for_status()
    data = resp.json()["prices"]
    dates = [datetime.fromtimestamp(p[0] / 1000).date() for p in data]
    prices = [p[1] for p in data]
    return pd.DataFrame({"Date": dates, "Close": prices}).set_index("Date")

try:
    # Ambil data historis
    df = get_historical_data(coin_id)

    # Validasi data historis
    if len(df) < 2:
        st.warning("Data historis tidak mencukupi untuk simulasi.")
        st.stop()

    # Hitung log-return
    log_ret = np.log(df["Close"] / df["Close"].shift(1)).dropna()
    mu, sigma = log_ret.mean(), log_ret.std()

    # Harga penutupan terakhir
    current_price = df["Close"].iloc[-1]

    # Tampilkan informasi harga terakhir
    st.markdown(f"### **Harga Penutupan Terakhir ({ticker_input}): US${current_price:,.2f}**")

    # Tetapkan random seed
    if "random_seed" not in st.session_state:
        st.session_state.random_seed = 42
    np.random.seed(st.session_state.random_seed)

    # Simulasi Monte Carlo
    if "simulation_results" not in st.session_state:
        simulation_results = {}
        for days in [3, 7, 30, 90, 365]:  # Periode simulasi
            sims = np.zeros((days, 100000))
            for i in range(100000):  # Simulasi 100,000 iterasi
                rw = np.random.normal(mu, sigma, days)  # Random walk
                sims[:, i] = current_price * np.exp(np.cumsum(rw))  # Hitung harga simulasi
            simulation_results[days] = sims[-1, :]  # Simpan hasil akhir saja
        st.session_state.simulation_results = simulation_results

    # Tampilkan hasil simulasi
    for days, results in st.session_state.simulation_results.items():
        st.header(f"📅 Proyeksi Harga {ticker_input} untuk {days} Hari ke Depan")

        # Statistik hasil simulasi
        mean_price = np.mean(results)
        median_price = np.median(results)
        std_dev = np.std(results)
        skewness = pd.Series(results).skew()
        lower_bound = np.percentile(results, 5)
        upper_bound = np.percentile(results, 95)
        prob_above_mean = (results > mean_price).mean() * 100

        # Distribusi kumulatif untuk tiga rentang harga tertinggi
        sorted_results = np.sort(results)  # Urutkan hasil simulasi
        top_3_ranges = sorted_results[-3:]  # Ambil tiga rentang harga tertinggi
        prob_top_3 = (results >= top_3_ranges[0]).mean() * 100  # Probabilitas kumulatif

        # Layout menggunakan kolom
        col1, col2 = st.columns([2, 1])

        with col1:
            # Visualisasi histogram
            fig, ax = plt.subplots()
            ax.hist(results, bins=50, color="blue", alpha=0.7)
            ax.set_title(f"Distribusi Harga Simulasi ({days} Hari)")
            ax.set_xlabel("Harga")
            ax.set_ylabel("Frekuensi")
            st.pyplot(fig)

        with col2:
            # Menampilkan statistik
            st.markdown("### 📊 Statistik Simulasi")
            st.write(f"**Rata-rata harga:** US${mean_price:,.2f}")
            st.write(f"**Median harga:** US${median_price:,.2f}")
            st.write(f"**Standar deviasi:** US${std_dev:,.2f}")
            st.write(f"**Rentang (5%-95%):** US${lower_bound:,.2f} - US${upper_bound:,.2f}")
            st.write(f"**Skewness:** {skewness:.2f}")
            st.write(f"**Peluang di atas rata-rata:** {prob_above_mean:.2f}%")
            st.write(f"**Peluang di tiga rentang tertinggi:** {prob_top_3:.2f}%")

        # Kesimpulan pendek untuk media sosial
        social_summary = (
            f"Berdasarkan simulasi Monte Carlo, ada peluang sebesar {prob_top_3:.1f}% "
            f"{ticker_input.split('-')[0]} bergerak di kisaran US${lower_bound:,.0f} - US${upper_bound:,.0f} "
            f"dalam {days} hari ke depan, dengan peluang {prob_above_mean:.1f}% berada di atas rata-rata logaritmik "
            f"US${mean_price:,.0f}."
        )

        st.markdown("### 📢 Kesimpulan untuk Media Sosial")
        st.text_area("Salin Kesimpulan", value=social_summary, height=100, key=f"summary_{days}")

except Exception as e:
    st.error(f"Terjadi kesalahan: {e}")
