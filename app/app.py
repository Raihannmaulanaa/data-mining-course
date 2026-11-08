import sys
from pathlib import Path

import pandas as pd
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent))
from pipeline import BMI_MAP, FEATURES, GENDER_MAP, load_bundle  # noqa: E402

st.set_page_config(page_title="MindSleep Self-Assessment", page_icon="🌙", layout="wide")


@st.cache_resource(show_spinner="Memuat model...")
def get_bundle():
    return load_bundle()


bundle = get_bundle()
model, ranges, metrics = bundle["model"], bundle["ranges"], bundle["metrics"]

LABEL_ID = {"None": "Tidak terindikasi gangguan", "Insomnia": "Insomnia", "Sleep Apnea": "Sleep Apnea"}

st.title("🌙 MindSleep: Self-Assessment Kesehatan Tidur & Stres")
st.caption("Alat skrining mandiri & mawas diri gaya hidup harian")
st.info(
    "💡 **Disclaimer:** Aplikasi ini untuk edukasi dan mawas diri, **bukan diagnosis medis**. "
    f"Model Random Forest dilatih pada dataset sintetis kecil ({metrics['n_unique_rows']} baris unik), "
    "sehingga hasilnya hanya gambaran kecenderungan pola data, bukan kondisi klinis Anda."
)
st.divider()

# ---------------- Input ----------------
st.subheader("📋 Masukkan Data Kebiasaan Harian Anda")
c1, c2 = st.columns(2)
with c1:
    gender = st.selectbox("Jenis Kelamin", ["Male", "Female"],
                          format_func=lambda x: "Laki-laki" if x == "Male" else "Perempuan")
    age = st.number_input("Usia (tahun)", 15, 100, 35)
    sleep_duration = st.slider("Rata-rata durasi tidur (jam)", 3.0, 12.0, 7.0, 0.1)
    quality = st.slider("Kualitas tidur yang dirasakan (1-10)", 1, 10, 7)
    stress = st.slider("Tingkat stres sehari-hari (1-10)", 1, 10, 5)
with c2:
    height = st.number_input("Tinggi badan (cm)", 100, 220, 165)
    weight = st.number_input("Berat badan (kg)", 30, 180, 60)
    bmi = weight / ((height / 100) ** 2)
    if bmi < 18.5:
        bmi_cat = "Underweight"
    elif bmi < 25:
        bmi_cat = "Normal"
    elif bmi < 30:
        bmi_cat = "Overweight"
    else:
        bmi_cat = "Obese"
    st.caption(f"📊 BMI: **{bmi:.1f}** ({bmi_cat})")
    sbp = st.number_input("Tekanan darah sistolik (atas)", 80, 200, 120)
    dbp = st.number_input("Tekanan darah diastolik (bawah)", 50, 130, 80)
    hr = st.number_input("Detak jantung istirahat (BPM)", 40, 130, 70)
    activity = st.slider("Aktivitas fisik (menit/hari)", 0, 180, 45)
    steps = st.number_input("Langkah kaki harian", 500, 25000, 7000, step=500)

st.divider()

if st.button("🔍 Cek Hasil Analisis", type="primary", use_container_width=True):
    # Underweight tidak ada di data latih -> jangan diam-diam diubah jadi Normal
    bmi_for_model = "Normal" if bmi_cat == "Underweight" else bmi_cat
    row = {
        "Gender": GENDER_MAP[gender], "Age": age, "Sleep Duration": sleep_duration,
        "Quality of Sleep": quality, "Physical Activity Level": activity,
        "Stress Level": stress, "BMI Category": BMI_MAP[bmi_for_model],
        "Heart Rate": hr, "Daily Steps": steps, "Systolic_BP": sbp, "Diastolic_BP": dbp,
    }
    X = pd.DataFrame([row])[FEATURES]
    proba = pd.Series(model.predict_proba(X)[0], index=model.classes_)
    top = proba.idxmax()

    # ---- Peringatan keandalan ----
    warnings = []
    if bmi_cat == "Underweight":
        warnings.append("Kategori BMI *Underweight* tidak ada di data latih; dihitung sebagai *Normal*.")
    names = {"Age": "Usia", "Sleep Duration": "Durasi tidur", "Quality of Sleep": "Kualitas tidur",
             "Physical Activity Level": "Aktivitas fisik", "Stress Level": "Stres",
             "Heart Rate": "Detak jantung", "Daily Steps": "Langkah harian",
             "Systolic_BP": "Sistolik", "Diastolic_BP": "Diastolik"}
    for col, (lo, hi) in ranges.items():
        if not lo <= row[col] <= hi:
            warnings.append(f"**{names[col]}** ({row[col]:g}) di luar rentang data latih ({lo:g}-{hi:g}).")
    if warnings:
        st.warning("⚠️ **Keandalan hasil berkurang:**\n\n" + "\n".join(f"- {w}" for w in warnings))

    # ---- Hasil ----
    st.subheader("📊 Hasil Penilaian")
    conf = proba[top]
    if conf < 0.5:
        st.info(f"🤔 **Hasil belum tegas.** Kemungkinan tertinggi: {LABEL_ID[top]} ({conf:.0%}).")
    elif top == "None":
        st.success(f"✅ **Tidak terindikasi gangguan tidur** ({conf:.0%})")
    elif top == "Insomnia":
        st.warning(f"⚠️ **Pola data mirip risiko Insomnia** ({conf:.0%})")
    else:
        st.error(f"🚨 **Pola data mirip risiko Sleep Apnea** ({conf:.0%})")

    st.write("**Sebaran probabilitas model:**")
    st.bar_chart(proba.rename(index=LABEL_ID), horizontal=True)

    # ---- Catatan gaya hidup (aturan sederhana, BUKAN hasil ML) ----
    tips = []
    if sleep_duration < 7:
        tips.append("Durasi tidur di bawah 7 jam; rekomendasi umum dewasa adalah 7-9 jam.")
    if quality <= 5:
        tips.append("Kualitas tidur yang Anda rasakan rendah; coba jam tidur konsisten dan kurangi layar/kafein malam hari.")
    if stress >= 7:
        tips.append("Tingkat stres tinggi; teknik relaksasi atau berbicara dengan orang terpercaya/profesional bisa membantu.")
    if activity < 30 or steps < 5000:
        tips.append("Aktivitas fisik tergolong rendah; tambah aktivitas ringan secara bertahap.")
    if sbp >= 140 or dbp >= 90:
        tips.append("Tekanan darah tergolong tinggi; sebaiknya diperiksakan ke tenaga kesehatan.")
    if bmi_cat in ("Overweight", "Obese"):
        tips.append("BMI di atas normal; kelebihan berat badan berkaitan dengan risiko gangguan tidur.")
    st.write("**Catatan gaya hidup** *(berbasis aturan umum, bukan prediksi model)*")
    if tips:
        for t in tips:
            st.write(f"- {t}")
    else:
        st.write("- Indikator harian Anda berada dalam kisaran yang umumnya dianggap sehat.")

    if top != "None" and conf >= 0.5:
        st.caption("Jika keluhan tidur mengganggu aktivitas, konsultasikan dengan dokter.")

with st.expander("ℹ️ Tentang model & keterbatasannya"):
    st.write(
        f"- Algoritma: Random Forest (300 pohon), data latih {metrics['n_unique_rows']} baris unik "
        "(374 baris asli, 242 di antaranya duplikat dan dibuang).\n"
        f"- Akurasi validasi silang 5-fold: **{metrics['accuracy_cv']:.0%}**, "
        f"F1 macro: **{metrics['f1_macro_cv']:.2f}**.\n"
        "- Sleep Apnea paling sulit dikenali model (sering tertukar dengan Insomnia/Tidak ada).\n"
        "- Dataset bersifat sintetis dan rentang usianya 27-59 tahun; hasil di luar itu kurang andal."
    )
    cm = pd.DataFrame(metrics["confusion"], index=[f"Asli: {l}" for l in metrics["labels"]],
                      columns=[f"Prediksi: {l}" for l in metrics["labels"]])
    st.dataframe(cm)
    imp = pd.Series(bundle["importances"]).sort_values(ascending=False)
    st.write("Fitur terpenting menurut model:")
    st.bar_chart(imp, horizontal=True)
