"""Streaming detector shared by monitor.py and api.py."""
import threading
from collections import defaultdict

import joblib
import pandas as pd

from config import (ALERT_MIN_VOTES, MIN_EVENTS_FOR_SCORING, MODELS_DIR, WINDOW_DAYS)
from src.features import FEATURE_COLS, compute_user_features
from src.model_agreement import LEVELS

COLS = ["timestamp", "user", "pc", "event_id", "logon_type"]


class Detector:
    def __init__(self, models_dir=MODELS_DIR):
        self.scaler = joblib.load(models_dir / "scaler.pkl")
        self.models = {"IsolationForest": joblib.load(models_dir / "isolation_forest.pkl"),
                       "OneClassSVM": joblib.load(models_dir / "oneclass_svm.pkl"),
                       "LOF": joblib.load(models_dir / "lof.pkl")}
        self.buffers = defaultdict(list)     # user -> events inside the rolling window
        self.alerted = {}                    # user -> vote count already alerted
        self.lock = threading.Lock()

    def seed(self, history: pd.DataFrame):
        for user, g in history.groupby("user"):
            self.buffers[user] = g[COLS].to_dict("records")

    def assess(self, user):
        """Score one user from their buffered events. Returns None if too little data."""
        buf = self.buffers.get(user, [])
        if len(buf) < MIN_EVENTS_FOR_SCORING:
            return None
        feats = compute_user_features(pd.DataFrame(buf))
        X = self.scaler.transform(feats[FEATURE_COLS])
        flagged = [n for n, m in self.models.items() if m.predict(X)[0] == -1]
        return {"user": user, "votes": len(flagged), "risk_level": LEVELS[len(flagged)],
                "models": flagged, "features": feats.iloc[0].drop("user").to_dict()}

    def process(self, ev):
        """Add one event; return an alert dict when a user newly crosses the alert threshold."""
        with self.lock:
            user, ts = ev["user"], pd.Timestamp(ev["timestamp"])
            buf = self.buffers[user]
            buf.append({c: ev[c] for c in COLS})
            cutoff = ts - pd.Timedelta(days=WINDOW_DAYS)
            self.buffers[user] = [e for e in buf if pd.Timestamp(e["timestamp"]) >= cutoff]
            res = self.assess(user)
            if res is None:
                return None
            if res["votes"] < ALERT_MIN_VOTES:
                self.alerted.pop(user, None)          # recovered: allow a future alert
                return None
            if res["votes"] <= self.alerted.get(user, 0):
                return None
            self.alerted[user] = res["votes"]
            f = res["features"]
            return {"time": str(ts), "user": user, "risk_level": res["risk_level"],
                    "votes": res["votes"], "models": "+".join(res["models"]),
                    "fail_ratio": round(f["fail_ratio"], 3),
                    "off_hours_ratio": round(f["off_hours_ratio"], 3),
                    "unique_pcs": int(f["unique_pcs"]),
                    "max_fail_streak": int(f["max_fail_streak"])}
