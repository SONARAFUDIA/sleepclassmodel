import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC

# --- Fungsi Pelatihan Model (Sama seperti sebelumnya) ---
def train_model():
    # 1. Memuat dan memproses data seperti di notebook
    df = pd.read_csv('data/Sleep_health_and_lifestyle_dataset.csv')
    df['Sleep Disorder'] = df['Sleep Disorder'].fillna('None')
    
    # 2. Seleksi fitur awal dan pemrosesan 'Blood Pressure'
    features_to_keep = [
        'Age', 'Gender', 'Sleep Duration', 'BMI Category',
        'Blood Pressure', 'Heart Rate', 'Daily Steps', 'Sleep Disorder'
    ]
    df_selected = df[features_to_keep].copy()

    bp_split = df_selected['Blood Pressure'].str.split('/', expand=True)
    df_selected['Systolic BP'] = pd.to_numeric(bp_split[0])
    df_selected['Diastolic BP'] = pd.to_numeric(bp_split[1])
    df_processed = df_selected.drop(columns=['Blood Pressure'])

    # 3. Mendefinisikan fitur (X) dan target (y)
    X = df_processed.drop('Sleep Disorder', axis=1)
    y = df_processed['Sleep Disorder']

    # 4. Mendefinisikan pipeline preprocessor
    categorical_features = ['Gender', 'BMI Category']
    numerical_features = ['Age', 'Sleep Duration', 'Heart Rate', 'Daily Steps', 'Systolic BP', 'Diastolic BP']

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numerical_features),
            ('cat', OneHotEncoder(drop='first', handle_unknown='ignore'), categorical_features)
        ],
        remainder='passthrough'
    )

    # 5. Membuat pipeline model SVM (performa terbaik di notebook)
    model_pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', SVC(probability=True, random_state=42))
    ])
    
    model_pipeline.fit(X, y)
    
    return model_pipeline, df_processed

# Melatih model saat aplikasi pertama kali dijalankan
pipeline, df_processed = train_model() # Mengganti nama variabel 'model' menjadi 'pipeline' agar sesuai

# --- Antarmuka Streamlit ---
st.set_page_config(page_title="Prediksi Gangguan Tidur", layout="wide")
st.title("🩺 Aplikasi Prediksi Status Gangguan Tidur")
st.write("Aplikasi ini memprediksi apakah seseorang memiliki risiko gangguan tidur berdasarkan data kesehatan dan gaya hidup.")
st.markdown("---")

# Layout dengan dua kolom
col1, col2 = st.columns(2)

with col1:
    st.header("Masukkan Data Diri:")
    
    # Input dari pengguna
    age = st.slider("Usia", 20, 60, 35, help="Masukkan usia Anda dalam tahun.")
    gender = st.radio("Jenis Kelamin", ('Male', 'Female'), help="Pilih jenis kelamin Anda.")
    sleep_duration = st.slider("Durasi Tidur (jam)", 1.0, 10.0, 7.5, 0.1, help="Rata-rata durasi tidur Anda per malam.")
    daily_steps = st.slider("Langkah Harian", 3000, 10000, 6000, help="Rata-rata jumlah langkah harian Anda.")
    
with col2:
    st.header("Masukkan Data Kesehatan:")
    heart_rate = st.slider("Detak Jantung Istirahat (BPM)", 60, 90, 70, help="Detak jantung Anda per menit saat istirahat.")
    systolic_bp = st.slider("Tekanan Darah Sistolik (mmHg)", 110, 150, 125, help="Nilai atas dari pembacaan tekanan darah.")
    diastolic_bp = st.slider("Tekanan Darah Diastolik (mmHg)", 70, 100, 80, help="Nilai bawah dari pembacaan tekanan darah.")
    
    # Mendapatkan kategori BMI dari data yang ada
    bmi_options = df_processed['BMI Category'].unique().tolist()
    bmi_category = st.selectbox("Kategori BMI", options=bmi_options, help="Pilih kategori Indeks Massa Tubuh (BMI) Anda.")

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
            # Jika confidence untuk 'Tidak Punya Gangguan' >= 80%
            st.success("🎉 **Kategori: Tidak Punya Gangguan Tidur**")
            # st.write(f"Tingkat kepercayaan bahwa Anda **tidak** memiliki gangguan tidur adalah **{confidence_no_disorder*100:.2f}%**.")
        else:
            # Jika confidence untuk 'Tidak Punya Gangguan' < 80%
            # Tingkat kepercayaan 'Punya Gangguan' adalah 100% - confidence 'Tidak Punya Gangguan'
            confidence_has_disorder = 1 - confidence_no_disorder
            st.warning("⚠️ **Kategori: Punya Gangguan Tidur**")
            # st.write(f"Tingkat kepercayaan bahwa Anda **memiliki** gangguan tidur adalah **{confidence_has_disorder*100:.2f}%**.")


    except Exception as e:
        st.error(f"Terjadi kesalahan saat prediksi: {e}")

st.markdown("""
---
*Penafian: Prediksi ini didasarkan pada model machine learning dan tidak boleh dianggap sebagai pengganti nasihat medis profesional.*
""")