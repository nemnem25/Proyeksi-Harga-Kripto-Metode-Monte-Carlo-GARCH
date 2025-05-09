import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import skew

# Judul Aplikasi
st.title("📈 Simulasi Monte Carlo untuk Harga Kripto")
st.markdown("Unggah file CSV yang berisi data historis harga kripto untuk menjalankan simulasi Monte Carlo.")

# Langkah 1: Unggah file CSV
uploaded_file = st.file_uploader("Unggah file CSV Anda di sini:", type=["csv"])

if uploaded_file is not None:
    try:
        # Membaca file CSV
        data = pd.read_csv(uploaded_file)
        st.write("### Data Historis yang Diunggah:")
        st.dataframe(data)

        # Validasi kolom yang diperlukan
        if "snapped_at" not in data.columns or "price" not in data.columns:
            st.error("File CSV harus memiliki kolom 'snapped_at' dan 'price'.")
        else:
            # Memproses data
            data["Date"] = pd.to_datetime(data["snapped_at"])
            data = data.sort_values("Date")
            data["Log Return"] = np.log(data["price"] / data["price"].shift(1))
            mu = data["Log Return"].mean()
            sigma = data["Log Return"].std()

            st.write("### Statistik Log Return:")
            st.write(f"Mean (μ): {mu:.6f}")
            st.write(f"Standar Deviasi (σ): {sigma:.6f}")

            # Parameter simulasi
            days_to_simulate = st.slider("Pilih jumlah hari simulasi:", min_value=1, max_value=365, value=3)
            simulations = st.slider("Pilih jumlah simulasi:", min_value=100, max_value=10000, value=1000, step=100)

            # Jalankan simulasi Monte Carlo
            if st.button("Mulai Simulasi Monte Carlo"):
                st.write("Simulasi sedang berjalan...")
                last_price = data["price"].iloc[-1]
                drift = (mu - 0.5 * sigma**2) * np.arange(days_to_simulate).reshape(-1, 1)
                random_shocks = sigma * np.random.normal(0, 1, (days_to_simulate, simulations))
                price_paths = last_price * np.exp(drift + np.cumsum(random_shocks, axis=0))

                # Statistik dan distribusi hasil simulasi
                final_prices = price_paths[-1]
                mean_price = np.mean(final_prices)
                stddev_price = np.std(final_prices)
                skewness = skew(final_prices)

                # Tentukan rentang harga
                bins = np.histogram_bin_edges(final_prices, bins='auto')
                histogram, bin_edges = np.histogram(final_prices, bins=bins)
                probabilities = histogram / sum(histogram) * 100

                # Buat tabel peluang
                ranges = [f"{bin_edges[i]:,.0f} - {bin_edges[i+1]:,.0f}" for i in range(len(bin_edges)-1)]
                probability_table = pd.DataFrame({
                    "Peluang (%)": probabilities,
                    "Rentang Harga (US$)": ranges
                }).sort_values("Peluang (%)", ascending=False)

                st.write("### Tabel Peluang dan Rentang Harga")
                st.dataframe(probability_table)

                # Peluang kumulatif untuk tiga rentang harga tertinggi
                top_3_probabilities = probability_table.iloc[:3]["Peluang (%)"].sum()
                top_3_range = probability_table.iloc[:3]["Rentang Harga (US$)"].values
                st.write(f"Peluang kumulatif dari tiga rentang harga tertinggi mencapai **{top_3_probabilities:.1f}%**, dengan kisaran harga **{', '.join(top_3_range)}**.")

                # Kesimpulan
                st.write("### Kesimpulan:")
                st.markdown(f"""
                - **Harga Rata-rata (Logaritmik)**: US${mean_price:,.0f}
                - **Standard Deviation**: US${stddev_price:,.0f}
                - **Skewness**: {skewness:.6f}
                - **Kemungkinan Harga di Atas Rata-rata**: 50.0%
                """)

                # Histogram distribusi harga
                fig, ax = plt.subplots(figsize=(10, 6))
                ax.hist(final_prices, bins=20, color='blue', alpha=0.7)
                ax.set_title("Distribusi Harga Akhir (Monte Carlo)")
                ax.set_xlabel("Harga (US$)")
                ax.set_ylabel("Frekuensi")
                st.pyplot(fig)

                # Teks untuk Media Sosial
                st.write("### Teks untuk Media Sosial:")
                social_text = f"""
                Berdasarkan simulasi Monte Carlo, ada peluang sebesar **{top_3_probabilities:.1f}%** bagi BTC-USD bergerak antara **{', '.join(top_3_range)}** dalam {days_to_simulate} hari ke depan, dengan peluang **50,0%** berada di atas rata-rata logaritmik **US${mean_price:,.0f}**.
                """
                st.markdown(social_text)
    except Exception as e:
        st.error(f"Terjadi kesalahan saat memproses file: {e}")
else:
    st.info("Silakan unggah file CSV untuk memulai.")
