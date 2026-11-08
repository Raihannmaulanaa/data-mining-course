"""Jalankan dari root repo:  python train_model.py"""
import sys
from pathlib import Path

import joblib

sys.path.insert(0, str(Path(__file__).parent / "app"))
from pipeline import MODEL_PATH, train  # noqa: E402

bundle = train()
MODEL_PATH.parent.mkdir(exist_ok=True)
joblib.dump(bundle, MODEL_PATH)
m = bundle["metrics"]
print(f"Data unik        : {m['n_unique_rows']} baris")
print(f"Akurasi 5-fold CV: {m['accuracy_cv']:.3f}")
print(f"F1 macro 5-fold  : {m['f1_macro_cv']:.3f}")
print("Confusion matrix (urutan", m["labels"], "):")
for row in m["confusion"]:
    print("  ", row)
print("Disimpan ke", MODEL_PATH)
