"""Do the *streaming* (monitor/API) models flag the labelled insiders?
  python stream_model_eval.py
"""
import joblib
import pandas as pd

from config import FEATURE_FILE, MODELS_DIR, TRUTH_FILE
from src.features import FEATURE_COLS

df = pd.read_csv(FEATURE_FILE).merge(pd.read_csv(TRUTH_FILE)[["user", "is_anomalous"]], on="user", how="left")
y = df["is_anomalous"].fillna(0).astype(int)
X = joblib.load(MODELS_DIR / "scaler.pkl").transform(df[FEATURE_COLS])
flags = {n: (joblib.load(MODELS_DIR / f).predict(X) == -1).astype(int)
         for n, f in [("IsolationForest", "isolation_forest.pkl"), ("OneClassSVM", "oneclass_svm.pkl"), ("LOF", "lof.pkl")]}
flags["Alert (>= 2 votes)"] = (sum(flags.values()) >= 2).astype(int)
print(f"{len(df)} users, {y.sum()} labelled insiders")
for n, f in flags.items():
    tp = int(((f == 1) & (y == 1)).sum()); fl = int(f.sum())
    print(f"{n:20s} flagged={fl:4d}  insiders_hit={tp:3d}  precision={tp / max(fl, 1):.3f}  recall={tp / y.sum():.3f}")
