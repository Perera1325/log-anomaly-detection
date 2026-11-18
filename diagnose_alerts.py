"""Explain why the live monitor raised so many alerts:  python diagnose_alerts.py"""
from collections import Counter

import pandas as pd

from config import ALERT_FILE, CLEAN_FILE, FEATURE_FILE, MIN_EVENTS_FOR_SCORING, RISK_FILE
from src.charts import zscores
from src.detector import Detector
from src.features import FEATURE_COLS

feats = pd.read_csv(FEATURE_FILE)
risk = pd.read_csv(RISK_FILE)
alerts = pd.read_csv(ALERT_FILE)
clean = pd.read_csv(CLEAN_FILE, parse_dates=["timestamp"])
per_user = clean.groupby("user").size()

print(f"Users in training: {len(feats)} | flagged by >=2 models in training: {(risk.anomaly_votes >= 2).sum()}")
print(f"Events per user in history: min {per_user.min()}, median {int(per_user.median())}, max {per_user.max()} "
      f"(monitor scores a user only from {MIN_EVENTS_FOR_SCORING} events)")
print(f"Alerts: {len(alerts)} for {alerts.user.nunique()} users | alerted users already flagged in training: "
      f"{alerts.user.isin(risk[risk.anomaly_votes >= 2].user).sum()}")

det = Detector()
det.seed(clean)
hist = Counter((det.assess(u) or {"votes": "not scored"})["votes"] for u in det.buffers)
print("Votes using HISTORY ONLY (no live events):", dict(hist))

zero = [c for c in FEATURE_COLS if feats[c].std() == 0]
print("Constant features (zero variance across users):", zero or "none")

z = zscores(feats, FEATURE_COLS)
top = Counter(z.loc[u].abs().idxmax() for u in alerts.user if u in z.index)
print("Most common biggest-deviation feature among alerted users:", top.most_common(4))
print("Alert time range:", alerts.time.min(), "->", alerts.time.max())
