"""Logika data + model yang dipakai bersama oleh train_model.py dan app.py."""
from pathlib import Path

import joblib
import pandas as pd
import sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict

ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = ROOT / "data" / "Sleep_health_and_lifestyle_dataset.csv"
MODEL_PATH = ROOT / "models" / "sleep_disorder_model.pkl"

FEATURES = [
    "Gender", "Age", "Sleep Duration", "Quality of Sleep",
    "Physical Activity Level", "Stress Level", "BMI Category",
    "Heart Rate", "Daily Steps", "Systolic_BP", "Diastolic_BP",
]
NUMERIC = [f for f in FEATURES if f not in ("Gender", "BMI Category")]
GENDER_MAP = {"Female": 0, "Male": 1}
BMI_MAP = {"Normal": 0, "Overweight": 1, "Obese": 2}  # ordinal
LABELS = ["None", "Insomnia", "Sleep Apnea"]


def load_clean_data(path=DATA_PATH) -> pd.DataFrame:
    # keep_default_na=False WAJIB: kalau tidak, string "None" di kolom
    # Sleep Disorder dibaca pandas sebagai NaN (219 baris hilang/rusak).
    df = pd.read_csv(path, keep_default_na=False)
    df["BMI Category"] = df["BMI Category"].replace({"Normal Weight": "Normal"})
    bp = df["Blood Pressure"].str.split("/", expand=True).astype(int)
    df["Systolic_BP"], df["Diastolic_BP"] = bp[0], bp[1]
    # 242 dari 374 baris adalah duplikat persis -> buang agar evaluasi tidak bocor
    key = [c for c in df.columns if c != "Person ID"]
    return df.drop_duplicates(subset=key).reset_index(drop=True)


def encode(df: pd.DataFrame) -> pd.DataFrame:
    X = df[FEATURES].copy()
    X["Gender"] = X["Gender"].map(GENDER_MAP)
    X["BMI Category"] = X["BMI Category"].map(BMI_MAP)
    return X


def new_model() -> RandomForestClassifier:
    return RandomForestClassifier(
        n_estimators=300, min_samples_leaf=2, random_state=42, n_jobs=-1
    )


def train(path=DATA_PATH) -> dict:
    df = load_clean_data(path)
    X, y = encode(df), df["Sleep Disorder"]

    # Evaluasi jujur: 5-fold stratified pada data unik
    pred = cross_val_predict(
        new_model(), X, y, cv=StratifiedKFold(5, shuffle=True, random_state=42)
    )
    metrics = {
        "n_unique_rows": int(len(df)),
        "accuracy_cv": float(accuracy_score(y, pred)),
        "f1_macro_cv": float(f1_score(y, pred, average="macro")),
        "confusion": confusion_matrix(y, pred, labels=LABELS).tolist(),
        "labels": LABELS,
    }
    model = new_model().fit(X, y)
    ranges = {c: (float(X[c].min()), float(X[c].max())) for c in NUMERIC}
    return {
        "model": model,
        "ranges": ranges,
        "metrics": metrics,
        "sklearn_version": sklearn.__version__,
        "importances": dict(zip(FEATURES, model.feature_importances_.round(4).tolist())),
    }


def load_bundle() -> dict:
    """Load .pkl; jika versi scikit-learn beda/rusak, latih ulang dari CSV."""
    try:
        bundle = joblib.load(MODEL_PATH)
        if bundle.get("sklearn_version") == sklearn.__version__:
            return bundle
    except Exception:
        pass
    return train()
