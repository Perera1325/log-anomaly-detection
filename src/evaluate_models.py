from itertools import combinations

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.metrics import (cohen_kappa_score, confusion_matrix, f1_score,
                             precision_score, recall_score)
from sklearn.preprocessing import StandardScaler

from config import (COMPARISON_FILE, CONTAMINATION, RESULTS_DIR, TRUTH_FILE)
from src.features import FEATURE_COLS
from src.plots import save_agreement_heatmap, save_confusion_matrices

MODELS = ["IsolationForest", "OneClassSVM", "LOF"]


def seed_stability(feature_file, seeds=range(5)):
    """Mean pairwise Jaccard overlap of Isolation Forest's flagged users across random seeds."""
    df = pd.read_csv(feature_file)
    X = StandardScaler().fit_transform(df[FEATURE_COLS])
    sets = [set(df.loc[IsolationForest(contamination=CONTAMINATION, random_state=s)
                       .fit_predict(X) == -1, "user"]) for s in seeds]
    return round(float(np.mean([len(a & b) / len(a | b) for a, b in combinations(sets, 2)])), 3)


def evaluate_models(results_file, feature_file=None):
    """Counts + consistency always; precision/recall/F1/confusion matrix only if labels exist."""
    print("\nEvaluating models...")
    df = pd.read_csv(results_file)
    flags = {m: (df[m] == -1).astype(int) for m in MODELS}
    flags["Ensemble (>=2 votes)"] = (sum(flags[m] for m in MODELS) >= 2).astype(int)
    RESULTS_DIR.mkdir(exist_ok=True)

    truth = None
    if TRUTH_FILE.exists():
        t = pd.read_csv(TRUTH_FILE)[["user", "is_anomalous"]]
        truth = df[["user"]].merge(t, on="user", how="left")["is_anomalous"].fillna(0).astype(int)
    else:
        print("No ground-truth labels found: precision/recall/F1 skipped.")

    rows = []
    for name, y_pred in flags.items():
        row = {"Model": name, "Anomalies": int(y_pred.sum()), "Normal": int((y_pred == 0).sum()),
               "Anomaly Rate (%)": round(100 * y_pred.mean(), 2)}
        if truth is not None:
            tn, fp, fn, tp = confusion_matrix(truth, y_pred, labels=[0, 1]).ravel()
            row.update({"Precision": round(precision_score(truth, y_pred, zero_division=0), 3),
                        "Recall": round(recall_score(truth, y_pred, zero_division=0), 3),
                        "F1": round(f1_score(truth, y_pred, zero_division=0), 3),
                        "TP": tp, "FP": fp, "FN": fn, "TN": tn})
        rows.append(row)
    summary = pd.DataFrame(rows)

    # anomaly detection consistency
    kappa = pd.DataFrame([[cohen_kappa_score(flags[a], flags[b]) for b in MODELS] for a in MODELS],
                         index=MODELS, columns=MODELS).round(3)
    votes = sum(flags[m] for m in MODELS)
    consistency = {
        "mean_pairwise_kappa": round(float(np.mean([kappa.loc[a, b] for a, b in combinations(MODELS, 2)])), 3),
        "flagged_by_all_3": int((votes == 3).sum()),
        "flagged_by_exactly_2": int((votes == 2).sum()),
        "flagged_by_exactly_1": int((votes == 1).sum()),
    }
    if feature_file:
        consistency["isolation_forest_seed_stability_jaccard"] = seed_stability(feature_file)

    summary.to_csv(COMPARISON_FILE, index=False)
    kappa.to_csv(RESULTS_DIR / "model_consistency.csv")
    pd.Series(consistency).to_csv(RESULTS_DIR / "consistency_summary.csv", header=["value"])
    save_agreement_heatmap(kappa, RESULTS_DIR / "agreement_heatmap.png")
    if truth is not None:
        save_confusion_matrices(summary, RESULTS_DIR / "confusion_matrices.png")

    print(summary.to_string(index=False))
    print("\nConsistency:", consistency)
    return summary
