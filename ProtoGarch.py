import streamlit as st
import numpy as np
import pandas as pd
from datetime import datetime
import pytz
import time
import requests
from arch import arch_model

# ————————————————————
# Utility formatting angka Indonesia 
# ————————————————————

def format_angka_indonesia(val) -> str:
    try:
        val = float(val)
    except (TypeError, ValueError):
        return str(val)
    if abs(val) < 1:
        s = f"{val:,.8f}"
    else:
        s = f"{val:,.0f}"
    return s.replace(",", "X").replace(".", ",").replace("X", ".")

def format_persen_indonesia(val) -> str:
    try:
        val = float(val)
    except (TypeError, ValueError):
        return str(val)
    s = f"{val:.1f}"
    return s.replace(".", ",") + "%"

# ————————————————————
# Konfigurasi halaman Streamlit
# ————————————————————

st.set_page_config(page_title="Proyeksi Harga Kripto Metode Monte Carlo (GARCH)", layout="centered")

# Tampilkan waktu realtime di atas
wib = pytz.timezone("Asia/Jakarta")
waktu_sekarang = datetime.now(wib).strftime("%A, %d %B %Y")
st.markdown(f"""
<div style='background-color: #5B5B5B; padding: 8px; border-radius: 8px; text-align: center; font-weight: bold; font-size: 16px;'>
⏰ {waktu_sekarang}
</div>
""", unsafe_allow_html=True)

st.title("Proyeksi Harga Kripto Metode Monte Carlo (GARCH)")
st.markdown(
    "_Simulasi berbasis data historis untuk memproyeksikan harga kripto selama beberapa hari ke depan, menggunakan metode Monte Carlo dengan volatilitas dinamis dari model GARCH._",
    unsafe_allow_html=True
)

# ————————————————————
# CSS global untuk styling hasil
# ————————————————————

st.markdown("""
    <style>
    table {
        width: 100%;
        border-collapse: collapse;
    }
    th {
        background-color: #5B5B5B;
        font-weight: bold;
        color: white;
        padding: 6px;
        text-align: left;
        border: 1px solid white;
    }
    td {
        border: 1px solid white;
        padding: 6px;
        text-align: left;
    }
    </style>
""", unsafe_allow_html=True)

# —————————————————————————
# Daftar ticker dan mapping ke CoinGecko
# —————————————————————————

coingecko_map = {
    "BTC-USD":"bitcoin", "ETH-USD":"ethereum", "BNB-USD":"binancecoin", "USDT-USD":"tether", "SOL-USD":"solana"
}

ticker_input = st.selectbox("Pilih simbol kripto:", list(coingecko_map.keys()))
if not ticker_input:
    st.stop()

# ————————————————————
# Logika simulasi dengan GARCH
# ————————————————————

try:
    coin_id = coingecko_map[ticker_input]

    resp = requests.get(
        f"https://api.coingecko.com/api/v3/coins/{coin_id}/market_chart",
        params={"vs_currency":"usd","days":"365"}
    )
    resp.raise_for_status()
    prices = resp.json()["prices"]
    dates = [datetime.fromtimestamp(p[0]/1000).date() for p in prices]
    closes = [p[1] for p in prices]

    df = pd.DataFrame({"Date":dates, "Close":closes}).set_index("Date")
    if len(df) < 2:
        st.warning("Data historis tidak mencukupi untuk simulasi.")
        st.stop()

    log_ret = np.log(df["Close"]/df["Close"].shift(1)).dropna()

    # Model GARCH untuk menghitung volatilitas
    st.spinner("Menghitung volatilitas dinamis dengan model GARCH...")
    garch_model = arch_model(log_ret, vol="Garch", p=1, q=1)
    garch_fit = garch_model.fit(disp="off")
    forecast = garch_fit.forecast(horizon=1)
    garch_volatility = forecast.variance.iloc[-1, 0] ** 0.5

    # Harga penutupan terakhir (dari hari sebelumnya, sesuai historis)
    current_price = df["Close"].iloc[-2]

    harga_penutupan = format_angka_indonesia(current_price)
    st.write(f"**Harga penutupan {ticker_input} sehari sebelumnya: US${harga_penutupan}**")

    for days in [3, 7, 30, 90, 365]:
        st.subheader(f"Proyeksi Harga Kripto {ticker_input} untuk {days} Hari ke Depan")
        sims = np.zeros((days, 100000))
        for i in range(100000):
            rw = np.random.normal(0, garch_volatility, days)
            sims[:, i] = current_price * np.exp(np.cumsum(rw))
        finals = sims[-1, :]

        bins = np.linspace(finals.min(), finals.max(), 10)
        counts, _ = np.histogram(finals, bins=bins)
        probs = counts / len(finals) * 100
        idx_sorted = np.argsort(probs)[::-1]

        table_html = "<table><thead><tr><th>Peluang</th><th>Rentang Harga (US$)</th></tr></thead><tbody>"

        total_peluang = 0
        rentang_bawah = float('inf')
        rentang_atas = 0

        for idx, id_sort in enumerate(idx_sorted):
            if probs[id_sort] == 0:
                continue
            low = bins[id_sort]
            high = bins[id_sort+1] if id_sort+1 < len(bins) else bins[-1]
            low_fmt = format_angka_indonesia(low)
            high_fmt = format_angka_indonesia(high)
            pct = format_persen_indonesia(probs[id_sort])
            table_html += f"<tr><td>{pct}</td><td>{low_fmt} - {high_fmt}</td></tr>"

            if idx < 3:
                total_peluang += probs[id_sort]
                rentang_bawah = min(rentang_bawah, low)
                rentang_atas = max(rentang_atas, high)

        total_peluang_fmt = format_persen_indonesia(total_peluang)
        rentang_bawah_fmt = format_angka_indonesia(rentang_bawah)
        rentang_atas_fmt = format_angka_indonesia(rentang_atas)

        table_html += f"""
        <tr class='highlight-green'><td colspan='2'>
        Peluang kumulatif dari tiga rentang harga tertinggi mencapai {total_peluang_fmt}, dengan kisaran harga US${rentang_bawah_fmt} hingga US${rentang_atas_fmt}.
        </td></tr>
        """
        table_html += "</tbody></table>"
        st.markdown(table_html, unsafe_allow_html=True)

        # Statistik tambahan untuk kesimpulan
        mean_log = np.mean(np.log(finals))
        harga_mean = np.exp(mean_log)
        chance_above_mean = np.mean(finals > harga_mean) * 100

        # Format angka untuk kesimpulan
        harga_mean_fmt = format_angka_indonesia(harga_mean)
        chance_above_mean_fmt = format_persen_indonesia(chance_above_mean)

        # Kesimpulan untuk media sosial
        kesimpulan = (
            f"Berdasarkan simulasi Monte Carlo, ada peluang sebesar {total_peluang_fmt} "
            f"{ticker_input} bergerak di kisaran US${rentang_bawah_fmt}-US${rentang_atas_fmt} "
            f"dalam {days} hari ke depan, dengan peluang {chance_above_mean_fmt} berada di atas rata-rata logaritmik US${harga_mean_fmt}."
        )
        st.text_area(f"Kesimpulan untuk {days} hari (salin untuk media sosial):", kesimpulan, height=100)

except Exception as e:
    st.error(f"Terjadi kesalahan: {e}")
