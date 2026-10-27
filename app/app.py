import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Konfigurasi Halaman Web
st.set_page_config(
    page_title="MindSleep Self-Assessment",
    page_icon="🌙",
    layout="wide"
)

# Load Model (Path file wajib sesuai dengan struktur folder GitHub)
@st.cache_resource
def load_models():
    model = joblib.load('models/sleep_disorder_model.pkl')
    encoders = joblib.load('models/encoders.pkl')
    target_encoder = joblib.load('models/target_encoder.pkl')
    return model, encoders, target_encoder

try:
    model, encoders, target_encoder = load_models()
    model_loaded = True
except Exception as e:
    model_loaded = False

# Header & Informasi Aplikasi
st.title("🌙 MindSleep: Self-Assessment Kesehatan Tidur & Stres")
st.caption("Alat Skrining Mandiri & Mawas Diri Gaya Hidup Harian")

st.info(
    "💡 **Disclaimer Medis & Etika Data:** Aplikasi ini dirancang sebagai sarana edukasi dan mawas diri gaya hidup mandiri. "
    "Hasil analisis berbasis model Machine Learning (Random Forest) dan **bukan diagnosis medis formal**."
)

st.divider()

if not model_loaded:
    st.error("⚠️ File model belum ditemukan di folder 'models/'. Pastikan file .pkl sudah terunggah di GitHub.")
else:
    # Input Data Pengguna
    st.subheader("📋 Masukkan Data Kebiasaan Harian Anda")
    col1, col2 = st.columns(2)
    
    with col1:
        gender = st.selectbox("Jenis Kelamin", options=["Male", "Female"], format_func=lambda x: "Laki-laki" if x == "Male" else "Perempuan")
        age = st.number_input("Usia (Tahun)", min_value=10, max_value=100, value=25)
        sleep_duration = st.slider("Rata-rata Durasi Tidur Harian (Jam)", min_value=3.0, max_value=12.0, value=7.0, step=0.1)
        quality_of_sleep = st.slider("Kualitas Tidur yang Dirasakan (Skala 1 - 10)", min_value=1, max_value=10, value=7)
        stress_level = st.slider("Tingkat Stres Sehari-hari (Skala 1 - 10)", min_value=1, max_value=10, value=5)

    with col2:
        height = st.number_input("Tinggi Badan (cm)", min_value=100, max_value=220, value=165)
        weight = st.number_input("Berat Badan (kg)", min_value=30, max_value=180, value=60)
        
        # Kalkulasi BMI Otomatis
        bmi_val = weight / ((height / 100) ** 2)
        if bmi_val < 18.5:
            bmi_cat = "Underweight"
        elif 18.5 <= bmi_val < 25:
            bmi_cat = "Normal"
        elif 25 <= bmi_val < 30:
            bmi_cat = "Overweight"
        else:
            bmi_cat = "Obese"
            
        st.caption(f"📊 Indeks Massa Tubuh (BMI): **{bmi_val:.1f}** ({bmi_cat})")
        
        sys_bp = st.number_input("Tekanan Darah Sistolik (Atas)", min_value=80, max_value=200, value=120)
        dia_bp = st.number_input("Tekanan Darah Diastolik (Bawah)", min_value=50, max_value=130, value=80)
        heart_rate = st.number_input("Detak Jantung Istirahat (BPM)", min_value=50, max_value=120, value=70)
        physical_activity = st.slider("Aktivitas Fisik / Olahraga (Menit/Hari)", min_value=0, max_value=180, value=45)
        daily_steps = st.number_input("Estimasi Langkah Kaki Harian", min_value=500, max_value=20000, value=7000, step=500)

    st.divider()

    # Tombol Prediksi
    if st.button("🔍 Cek Hasil Analisis Mawas Diri", type="primary", use_container_width=True):
        # Transformasi Data Terenkode
        gender_encoded = encoders['Gender'].transform([gender])[0]
        
        bmi_input = bmi_cat
        if bmi_input not in encoders['BMI Category'].classes_:
            bmi_input = 'Normal'
        bmi_encoded = encoders['BMI Category'].transform([bmi_input])[0]
        
        input_data = pd.DataFrame([{
            'Gender': gender_encoded,
            'Age': age,
            'Sleep Duration': sleep_duration,
            'Quality of Sleep': quality_of_sleep,
            'Physical Activity Level': physical_activity,
            'Stress Level': stress_level,
            'BMI Category': bmi_encoded,
            'Heart Rate': heart_rate,
            'Daily Steps': daily_steps,
            'Systolic_BP': sys_bp,
            'Diastolic_BP': dia_bp
        }])
        
        prediction = model.predict(input_data)[0]
        result_label = target_encoder.inverse_transform([prediction])[0]
        
        # Tampilan Hasil Evaluasi
        st.subheader("📊 Hasil Penilaian Kesehatan Tidur")
        
        if result_label == 'None':
            st.success("✅ **KONDISI OPTIMAL (TIDAK TERINDIKASI GANGGUAN TIDUR)**")
            st.write("Pola tidur, tingkat aktivitas, dan indikator stres Anda dalam batas normal. Pertahankan pola hidup sehat ini!")
        elif result_label == 'Insomnia':
            st.warning("⚠️ **INDIKASI RISIKO INSOMNIA**")
            st.write("Terdeteksi potensi kendala kualitas tidur. Disarankan mengatur jam tidur konsisten dan mengurangi konsumsi kafein.")
        elif result_label == 'Sleep Apnea':
            st.error("🚨 **INDIKASI RISIKO SLEEP APNEA**")
            st.write("Indikator menunjukkan potensi kendala pernapasan saat tidur. Sangat disarankan berkonsultasi dengan profesional kesehatan.")
