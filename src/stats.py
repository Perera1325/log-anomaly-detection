"""Uncertainty for the evaluation: percentile-bootstrap confidence intervals over accounts."""
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

MODELS = {"IsolationForest": "IF_score", "OneClassSVM": "SVM_score", "LOF": "LOF_score"}
ENSEMBLE = "Ensemble (>=2 votes)"


def _prf(y, flag):
    tp = int((flag & (y == 1)).sum())
    fp = int((flag & (y == 0)).sum())
    fn = int((~flag & (y == 1)).sum())
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    return p, r, (2 * p * r / (p + r) if p + r else 0.0)


def bootstrap_table(results: pd.DataFrame, truth, n_boot: int = 500, seed: int = 42) -> pd.DataFrame:
    """Point estimate and 95% percentile-bootstrap interval for precision, recall, F1, ROC AUC and AP."""
    y = np.asarray(truth).astype(int)
    n = len(y)
    flags = {m: (results[m].to_numpy() == -1) for m in MODELS}
    flags[ENSEMBLE] = sum(flags.values()) >= 2
    scores = {m: -results[c].to_numpy() for m, c in MODELS.items()}
    rng = np.random.default_rng(seed)
    idx = [rng.integers(0, n, n) for _ in range(n_boot)]
    rows = []

    def add(model, metric, est, vals):
        vals = np.asarray(vals, dtype=float)
        lo, hi = np.percentile(vals, [2.5, 97.5]) if len(vals) else (np.nan, np.nan)
        rows.append({"Model": model, "Metric": metric, "Estimate": est, "Lower": lo, "Upper": hi})

    for name, flag in flags.items():
        est = _prf(y, flag)
        boots = np.array([_prf(y[i], flag[i]) for i in idx])
        for k, metric in enumerate(["Precision", "Recall", "F1"]):
            add(name, metric, est[k], boots[:, k])
        if name in scores:
            s = scores[name]
            ok = [i for i in idx if 0 < y[i].sum() < len(i)]
            add(name, "ROC AUC", roc_auc_score(y, s), [roc_auc_score(y[i], s[i]) for i in ok])
            add(name, "Avg. precision", average_precision_score(y, s),
                [average_precision_score(y[i], s[i]) for i in ok])
    return pd.DataFrame(rows)


def operating_curve(score, truth, rates):
    """Precision / recall / F1 when the top `rate` share of accounts (by anomaly score) is flagged."""
    y = np.asarray(truth).astype(int)
    order = np.argsort(-np.asarray(score), kind="stable")
    out = []
    for r in rates:
        k = max(int(round(r * len(y))), 1)
        flag = np.zeros(len(y), dtype=bool)
        flag[order[:k]] = True
        p, rc, f = _prf(y, flag)
        out.append({"rate": r, "flagged": k, "tp": int((flag & (y == 1)).sum()), "Precision": p, "Recall": rc,
                    "F1": f})
    return pd.DataFrame(out)
