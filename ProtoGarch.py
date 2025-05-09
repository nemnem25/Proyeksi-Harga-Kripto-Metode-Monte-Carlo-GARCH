import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt

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
            days_to_simulate = st.slider("Pilih jumlah hari simulasi:", min_value=1, max_value=365, value=30)
            simulations = st.slider("Pilih jumlah simulasi:", min_value=100, max_value=10000, value=1000, step=100)

            # Tambahkan tombol untuk memulai simulasi
            if st.button("Mulai Simulasi Monte Carlo"):
                st.write("Simulasi sedang berjalan...")
                # Jalankan logika simulasi
                last_price = data["price"].iloc[-1]
                drift = (mu - 0.5 * sigma**2) * np.arange(days_to_simulate).reshape(-1, 1)
                random_shocks = sigma * np.random.normal(0, 1, (days_to_simulate, simulations))
                price_paths = last_price * np.exp(drift + np.cumsum(random_shocks, axis=0))

                # Plot hasil simulasi
                fig, ax = plt.subplots(figsize=(10, 6))
                for i in range(price_paths.shape[1]):
                    ax.plot(price_paths[:, i], color='blue', alpha=0.1)
                ax.set_title("Hasil Simulasi Monte Carlo")
                ax.set_xlabel("Hari ke-")
                ax.set_ylabel("Harga (USD)")
                st.pyplot(fig)
    except Exception as e:
        st.error(f"Terjadi kesalahan saat memproses file: {e}")
else:
    st.info("Silakan unggah file CSV untuk memulai.")
