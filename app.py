import streamlit as st
import pandas as pd
import joblib
import numpy as np

# Load the trained pipeline
# Pastikan file .joblib berada di direktori yang sama dengan app.py
try:
    pipeline = joblib.load('sleep_disorder_svm_pipeline.joblib')
except FileNotFoundError:
    st.error("File pipeline model tidak ditemukan. Pastikan 'sleep_disorder_svm_pipeline.joblib' ada di direktori saat ini.")
    st.stop()


# --- Antarmuka Aplikasi Streamlit ---

st.set_page_config(page_title="Prediksi Gangguan Tidur", page_icon="😴", layout="wide")

# Judul Aplikasi
st.title("😴 Aplikasi Prediksi Gangguan Tidur")
st.markdown("Aplikasi ini memprediksi kemungkinan seseorang memiliki gangguan tidur berdasarkan faktor kesehatan dan gaya hidup.")
st.markdown("---")

# --- Bagian Input Pengguna ---
st.header("Masukkan Informasi Kesehatan dan Gaya Hidup Anda:")

# Buat kolom untuk tata letak yang lebih rapi
col1, col2 = st.columns(2)

with col1:
    age = st.slider("Usia (Age)", 20, 70, 35)
    gender = st.selectbox("Jenis Kelamin (Gender)", ["Male", "Female"])
    sleep_duration = st.slider("Durasi Tidur (jam)", 4.0, 10.0, 7.5, 0.1)
    heart_rate = st.slider("Detak Jantung (bpm)", 60, 90, 70)

with col2:
    daily_steps = st.slider("Langkah Harian (Daily Steps)", 2000, 12000, 6000)
    bmi_category = st.selectbox("Kategori BMI", ["Normal", "Overweight", "Obese"])
    systolic_bp = st.slider("Tekanan Darah Sistolik (mmHg)", 110, 180, 120)
    diastolic_bp = st.slider("Tekanan Darah Diastolik (mmHg)", 70, 120, 80)


# --- Logika Prediksi ---
if st.button("Prediksi Status Gangguan Tidur", type="primary"):

    # Buat DataFrame dari input pengguna
    # Nama kolom HARUS sesuai dengan yang digunakan saat pelatihan model
    input_data = {
        'Age': [age],
        'Gender': [gender],
        'Sleep Duration': [sleep_duration],
        'BMI Category': [bmi_category],
        'Heart Rate': [heart_rate],
        'Daily Steps': [daily_steps],
        'Systolic BP': [systolic_bp],
        'Diastolic BP': [diastolic_bp]
    }
    input_df = pd.DataFrame(input_data)

    st.markdown("---")
    st.subheader("Hasil Prediksi:")

    try:
        # Lakukan prediksi probabilitas
        prediction_proba = pipeline.predict_proba(input_df)

        # Dapatkan daftar kelas dari pipeline untuk menemukan indeks kategori 'None'
        classes = pipeline.classes_
        # Temukan indeks untuk kelas 'None' (tidak ada gangguan)
        none_class_index = np.where(classes == 'None')[0][0]

        # Dapatkan probabilitas untuk kategori 'None'
        confidence_no_disorder = prediction_proba[0][none_class_index]

        # Terapkan aturan baru
        if confidence_no_disorder >= 0.80:
            # Jika confidence untuk 'Tidak Punya Gangguan' >= 75%
            st.success("🎉 **Kategori: Tidak Punya Gangguan Tidur**")
            st.write(f"Tingkat kepercayaan bahwa Anda **tidak** memiliki gangguan tidur adalah **{confidence_no_disorder*100:.2f}%**.")
        else:
            # Jika confidence untuk 'Tidak Punya Gangguan' < 75%
            # Tingkat kepercayaan 'Punya Gangguan' adalah 100% - confidence 'Tidak Punya Gangguan'
            confidence_has_disorder = 1 - confidence_no_disorder
            st.warning("⚠️ **Kategori: Punya Gangguan Tidur**")
            st.write(f"Tingkat kepercayaan bahwa Anda **memiliki** gangguan tidur adalah **{confidence_has_disorder*100:.2f}%**.")


    except Exception as e:
        st.error(f"Terjadi kesalahan saat prediksi: {e}")

st.markdown("""
---
*Penafian: Prediksi ini didasarkan pada model machine learning dan tidak boleh dianggap sebagai pengganti nasihat medis profesional.*
""")