"""Held-out evaluation: features/models are fitted on one half of the users, results are reported on the other half.

  python holdout_eval.py            # 30 random stratified 50/50 splits
Writes Results/holdout_results.csv
"""
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import StandardScaler
from sklearn.svm import OneClassSVM

from config import CONTAMINATION, FEATURE_FILE, RESULTS_DIR, TRUTH_FILE
from src.features import FEATURE_COLS

N_SPLITS, TOP_K_FEATURES = 30, 10

df = pd.read_csv(FEATURE_FILE).merge(pd.read_csv(TRUTH_FILE)[["user", "is_anomalous"]], on="user", how="left")
y = df["is_anomalous"].fillna(0).astype(int).to_numpy()
informative = [c for c in FEATURE_COLS if df[c].nunique() > 1]
print(f"{len(df)} users, {y.sum()} insiders, {len(informative)} informative features, {N_SPLITS} splits")


def scores(model_name, Xtr, Xte):
    if model_name == "IsolationForest":
        m = IsolationForest(contamination=CONTAMINATION, random_state=42).fit(Xtr)
    elif model_name == "OneClassSVM":
        m = OneClassSVM(nu=CONTAMINATION, kernel="rbf", gamma="scale").fit(Xtr)
    else:
        m = LocalOutlierFactor(n_neighbors=20, contamination=CONTAMINATION, novelty=True).fit(Xtr)
    return -m.decision_function(Xte)               # higher = more anomalous


rows = []
splitter = StratifiedShuffleSplit(n_splits=N_SPLITS, test_size=0.5, random_state=42)
for i, (tr, te) in enumerate(splitter.split(df, y)):
    for setup in ["all informative features", f"top {TOP_K_FEATURES} chosen on train half only"]:
        cols = informative
        if setup.startswith("top"):
            auc_tr = {c: roc_auc_score(y[tr], df[c].to_numpy()[tr]) for c in informative}
            cols = sorted(informative, key=lambda c: -abs(auc_tr[c] - 0.5))[:TOP_K_FEATURES]
        sc = StandardScaler().fit(df.loc[tr, cols])
        Xtr, Xte = sc.transform(df.loc[tr, cols]), sc.transform(df.loc[te, cols])
        k = int(len(te) * 0.05)
        for name in ["IsolationForest", "OneClassSVM", "LOF"]:
            s = scores(name, Xtr, Xte)
            top = np.argsort(-s)[:k]
            rows.append(dict(split=i, setup=setup, model=name, auc=roc_auc_score(y[te], s),
                             ap=average_precision_score(y[te], s), precision_at_5pct=y[te][top].mean()))

res = pd.DataFrame(rows)
RESULTS_DIR.mkdir(exist_ok=True)
res.to_csv(RESULTS_DIR / "holdout_results.csv", index=False)
summary = res.groupby(["setup", "model"])[["auc", "ap", "precision_at_5pct"]].agg(["mean", "std"]).round(3)
print("\nTEST-HALF results (mean and std over splits). Chance: AUC 0.5, AP/precision ~", round(y.mean(), 3))
print(summary.to_string())
