# MindSleep: Self-Assessment Kesehatan Tidur & Stres

Prototipe web self-assessment kesehatan tidur dan stres menggunakan algoritma Random Forest Classifier. Proyek data mining ini membantu pengguna memetakan kecenderungan gangguan tidur secara mandiri berdasarkan indikator pola hidup dan data fisiologis harian.

> **Disclaimer:** aplikasi ini untuk edukasi dan mawas diri, bukan diagnosis medis.

Link app pakai streamlit: https://data-mining-course-zj3ok9ak4k7hiwg78nzz5x.streamlit.app/

## Dataset yang dipilih
Sleep Health and Lifestyle Dataset (Kaggle), data sintetis: 374 baris, 132 baris unik setelah duplikat dibuang.

## Metodologi yang digunakan
Random Forest Classifier, evaluasi 5-fold stratified cross-validation pada data unik.
Akurasi sekitar 71%, F1 macro sekitar 0,63.

## Keterbatasan
Data sintetis dan kecil, rentang usia 27-59 tahun, kelas Sleep Apnea paling sulit dikenali.
Aplikasi ini untuk edukasi dan bukan diagnosis medis.

## Struktur repository
```
app/        aplikasi Streamlit (app.py) dan logika data/model (pipeline.py)
data/       dataset CSV
models/     model Random Forest yang sudah dilatih
train_model.py   skrip untuk melatih ulang model
requirements.txt daftar library
```

## Menjalankan
```
pip install -r requirements.txt
streamlit run app/app.py
```

Untuk melatih ulang model: `python train_model.py`
