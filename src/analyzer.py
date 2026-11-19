"""Analyse an uploaded log file: models + explainable attack indicators. Used by the dashboard's Attack analyzer."""
import io
from functools import lru_cache

import joblib
import numpy as np
import pandas as pd

from config import MODELS_DIR
from src.features import FEATURE_COLS, compute_user_features
from src.model_agreement import LEVELS
from src.preprocess import clean_frame
from src.sources import UNIFIED, load_windows_csv, parse_auth_line

MODEL_FILES = {"IsolationForest": "isolation_forest.pkl", "OneClassSVM": "oneclass_svm.pkl", "LOF": "lof.pkl"}


@lru_cache(maxsize=1)
def _load():
    return joblib.load(MODELS_DIR / "scaler.pkl"), {n: joblib.load(MODELS_DIR / f) for n, f in MODEL_FILES.items()}


def read_log(name: str, data: bytes) -> pd.DataFrame:
    """CSV (our schema, Get-WinEvent export or CERT logon.csv) or Linux sshd auth.log."""
    text = data.decode("utf-8", errors="ignore")
    head = text.splitlines()[0].lower() if text.strip() else ""
    if "," in head and any(k in head for k in ("user", "timestamp", "timecreated", "activity")):
        return load_windows_csv(io.StringIO(text))
    rows = [ev for ev in (parse_auth_line(line) for line in text.splitlines()) if ev]
    if not rows:
        raise ValueError("Unrecognised format: expected a logon CSV or a Linux sshd auth.log")
    return pd.DataFrame(rows, columns=UNIFIED)


def _indicators(f: pd.Series, ev: pd.DataFrame) -> list:
    """Plain-language attack indicators for one user (heuristic rules, not a verdict)."""
    out = []
    n = len(ev)
    if f["max_fail_streak"] >= 5 or (f["fail_ratio"] >= 0.4 and n >= 10):
        out.append("Brute force (many failed logons in a row)")
    if f["fail_then_success"] >= 1 and f["max_fail_streak"] >= 3:
        out.append("Success after repeated failures (possible password guessed)")
    if f["off_hours_ratio"] >= 0.3 and n >= 10:
        out.append("Off-hours access (large share outside 07:00-19:00)")
    if f["wk_max_z_off"] >= 2.5:
        out.append("Sudden change from own normal working hours")
    if f["unique_pcs"] >= 10 or (f["pc_switch_ratio"] >= 0.8 and f["unique_pcs"] >= 5):
        out.append("Lateral movement (many different machines)")
    if f["short_gap_ratio"] >= 0.8 and n >= 30:
        out.append("Automated burst (events seconds apart)")
    if f["weekend_ratio"] >= 0.5 and n >= 10:
        out.append("Weekend-heavy activity")
    return out


def _spraying(ev: pd.DataFrame):
    """One source machine failing against many different users = password spraying."""
    fails = ev[ev["event_id"] == 4625]
    if fails.empty:
        return []
    g = fails.groupby("pc")["user"].nunique()
    return [(pc, int(k)) for pc, k in g[g >= 5].items()]


def analyze(df: pd.DataFrame) -> dict:
    scaler, models = _load()
    ev = clean_frame(df)
    if ev.empty:
        raise ValueError("No usable logon events (Event IDs 4624/4625) found in this file")
    feats = compute_user_features(ev)
    Z = pd.DataFrame(scaler.transform(feats[FEATURE_COLS]), columns=FEATURE_COLS, index=feats["user"])
    X = Z.to_numpy()
    flags = pd.DataFrame({n: (m.predict(X) == -1) for n, m in models.items()}, index=feats["user"])
    votes = flags.sum(axis=1)
    rows = []
    for u, f in feats.set_index("user").iterrows():
        uev = ev[ev["user"] == u]
        top = Z.loc[u].abs().sort_values(ascending=False).head(3)
        rows.append({"user": u, "events": len(uev), "risk": LEVELS[int(votes[u])], "votes": int(votes[u]),
                     "models": ", ".join(flags.columns[flags.loc[u]]) or "-",
                     "indicators": "; ".join(_indicators(f, uev)) or "-",
                     "most unusual": ", ".join(f"{k} ({Z.loc[u, k]:+.1f}σ)" for k in top.index)})
    table = pd.DataFrame(rows).sort_values(["votes", "events"], ascending=[False, False]).reset_index(drop=True)
    spray = _spraying(ev)
    if spray:
        sprayed = {pc: set(ev[(ev["event_id"] == 4625) & (ev["pc"] == pc)]["user"]) for pc, _ in spray}
        for i, u in enumerate(table["user"]):
            for pc, users in sprayed.items():
                if u in users:
                    tag = f"Targeted by password spraying from {pc}"
                    cur = table.at[i, "indicators"]
                    table.at[i, "indicators"] = tag if cur == "-" else f"{cur}; {tag}"
    patterns = sorted({p.split(" (")[0] for s in table["indicators"] for p in s.split("; ")
                       if p != "-" and not p.startswith("Targeted")})
    if spray:
        patterns.append("Password spraying")
    worst = table.iloc[0]
    return {"events": ev, "table": table, "z": Z, "spray": spray, "patterns": patterns,
            "n_events": len(ev), "n_users": int(feats.shape[0]),
            "n_flagged": int((table["votes"] >= 2).sum()), "worst": worst,
            "verdict": "Critical" if (table["votes"] >= 3).any() else "High" if (table["votes"] >= 2).any()
            else "Medium" if (table["votes"] >= 1).any() else "Low"}
