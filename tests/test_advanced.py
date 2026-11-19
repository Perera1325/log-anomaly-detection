"""Tests for the ATT&CK mapping, bootstrap intervals and operating-point curve."""
import numpy as np
import pandas as pd

from src import mitre, stats


def test_mitre_mapping():
    t = mitre.techniques(["Brute force", "Password spraying", "Lateral movement", "Automated burst"])
    assert set(t["Technique"]) == {"T1110", "T1110.003", "T1021"}          # unmapped pattern is skipped
    assert mitre.url("T1110.003") == "https://attack.mitre.org/techniques/T1110/003/"
    assert mitre.ids([]) == "-"


def _toy(n=400, seed=0):
    rng = np.random.default_rng(seed)
    y = (rng.random(n) < 0.1).astype(int)
    s = rng.normal(size=n) + 1.5 * y
    flag = s > np.quantile(s, 0.9)
    res = pd.DataFrame({"IsolationForest": np.where(flag, -1, 1), "OneClassSVM": np.where(flag, -1, 1),
                        "LOF": np.where(rng.random(n) < 0.1, -1, 1),
                        "IF_score": -s, "SVM_score": -s, "LOF_score": -rng.normal(size=n)})
    return res, y


def test_bootstrap_intervals_are_ordered():
    res, y = _toy()
    ci = stats.bootstrap_table(res, y, n_boot=100)
    assert set(ci["Metric"]) == {"Precision", "Recall", "F1", "ROC AUC", "Avg. precision"}
    assert (ci["Lower"] <= ci["Upper"]).all()
    auc = ci[(ci.Model == "IsolationForest") & (ci.Metric == "ROC AUC")].iloc[0]
    assert 0.6 < auc.Estimate < 1.0


def test_operating_curve_is_monotone_in_flags():
    res, y = _toy()
    c = stats.operating_curve(res["IF_score"].to_numpy() * -1, y, [0.01, 0.05, 0.1, 0.2])
    assert list(c["flagged"]) == sorted(c["flagged"])
    assert c["Recall"].is_monotonic_increasing
