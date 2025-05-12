import streamlit as st
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import pytz

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

st.set_page_config(page_title="Proyeksi Harga Kripto Metode Monte Carlo", layout="centered")

# Tampilkan waktu realtime di atas
wib = pytz.timezone("Asia/Jakarta")
waktu_sekarang = datetime.now(wib).strftime("%A, %d %B %Y")
st.markdown(f"""
<div style='background-color: #5B5B5B; padding: 8px; border-radius: 8px; text-align: center; font-weight: bold; font-size: 16px;'>
⏰ {waktu_sekarang}
</div>
""", unsafe_allow_html=True)

st.title("Proyeksi Harga Kripto Metode Monte Carlo")
st.markdown(
    "_Simulasi berbasis data historis untuk memproyeksikan harga kripto selama beberapa hari ke depan, menggunakan metode Monte Carlo dengan Geometric Brownian Motion._",
    unsafe_allow_html=True
)

# ————————————————————
# Upload file CSV oleh pengguna
# ————————————————————

st.subheader("Unggah Data Historis Harga Kripto")
uploaded_file = st.file_uploader("Unggah file CSV Anda di sini:", type=["csv"])

if uploaded_file is not None:
    try:
        # Membaca file CSV
        df = pd.read_csv(uploaded_file)

        # Validasi kolom
        if "snapped_at" not in df.columns or "price" not in df.columns:
            st.error("File CSV harus memiliki kolom 'snapped_at' dan 'price'.")
            st.stop()

        # Konversi kolom 'snapped_at' menjadi format datetime dan ambil hanya tanggalnya
        df["Date"] = pd.to_datetime(df["snapped_at"]).dt.date

        # Gunakan kolom 'price' sebagai 'Close'
        df["Close"] = df["price"]

        # Pilih hanya kolom 'Date' dan 'Close'
        processed_df = df[["Date", "Close"]]

        # Filter data untuk memastikan hanya data 5 tahun terakhir
        today = datetime.now().date()
        five_years_ago = today - timedelta(days=5 * 365)
        processed_df = processed_df[processed_df["Date"] >= five_years_ago]

        if processed_df.empty:
            st.error("Data historis harus mencakup setidaknya satu tanggal dalam 5 tahun terakhir.")
            st.stop()

        # Menampilkan data yang valid
        st.write("### Data Historis yang Valid:")
        st.dataframe(processed_df)

        # Memproses data untuk simulasi Monte Carlo
        processed_df = processed_df.sort_values("Date")
        processed_df["Log Return"] = np.log(processed_df["Close"] / processed_df["Close"].shift(1)).dropna()
        mu, sigma = processed_df["Log Return"].mean(), processed_df["Log Return"].std()

        current_price = processed_df["Close"].iloc[-1]
        harga_penutupan = format_angka_indonesia(current_price)
        st.write(f"**Harga penutupan terakhir: US${harga_penutupan}**")

        # Simulasi Monte Carlo menggunakan Geometric Brownian Motion (GBM)
        np.random.seed(42)
        for days in [3, 7, 30, 90, 365]:
            st.subheader(f"Proyeksi Harga Kripto untuk {days} Hari ke Depan")
            sims = np.zeros((days, 100000))
            for i in range(100000):
                dt = 1  # Unit waktu dalam hari
                drift = (mu - 0.5 * sigma**2) * dt
                shocks = sigma * np.random.normal(0, 1, days)
                sims[:, i] = current_price * np.exp(np.cumsum(drift + shocks))
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

    # Hitung harga mean dari data historis
harga_mean = processed_df["Close"].mean()
harga_mean_fmt = format_angka_indonesia(harga_mean)

# Hitung persentase peluang harga berada di atas atau di bawah mean
peluang_di_atas_mean = np.sum(finals > harga_mean) / len(finals) * 100
peluang_di_bawah_mean = np.sum(finals < harga_mean) / len(finals) * 100

peluang_di_atas_mean_fmt = format_persen_indonesia(peluang_di_atas_mean)
peluang_di_bawah_mean_fmt = format_persen_indonesia(peluang_di_bawah_mean)

# Tampilkan informasi di Streamlit
st.write(f"**Harga Rata-rata Historis:** US${harga_mean_fmt}")
st.write(f"**Peluang Harga di Atas Mean:** {peluang_di_atas_mean_fmt}")
st.write(f"**Peluang Harga di Bawah Mean:** {peluang_di_bawah_mean_fmt}")

    except Exception as e:
        st.error(f"Terjadi kesalahan: {e}")
else:
    st.info("Silakan unggah file CSV untuk memulai.")
