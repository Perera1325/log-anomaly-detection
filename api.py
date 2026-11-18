"""REST API for real-time ingestion and risk lookup.

  uvicorn api:app --host 0.0.0.0 --port 8000        # docs at /docs
Set API_KEY to require an `X-API-Key` header on every request except /health.
"""
import os
from datetime import datetime
from typing import List

import pandas as pd
from fastapi import Depends, FastAPI, Header, HTTPException
from pydantic import BaseModel

from bootstrap import ensure_ready
from config import ALERT_FILE, CLEAN_FILE, RISK_FILE, VALID_EVENT_IDS
from src.alerting import Alerter
from src.detector import Detector

ensure_ready()
detector = Detector()
detector.seed(pd.read_csv(CLEAN_FILE, parse_dates=["timestamp"]))
alerter = Alerter(quiet=True)

app = FastAPI(title="Log Anomaly Detection API", version="1.0")


def require_key(x_api_key: str = Header(default=None)):
    key = os.getenv("API_KEY")
    if key and x_api_key != key:
        raise HTTPException(status_code=401, detail="invalid or missing X-API-Key")


class Event(BaseModel):
    timestamp: datetime
    user: str
    pc: str
    event_id: int            # 4624 success, 4625 failure
    logon_type: int = -1


@app.get("/health")
def health():
    return {"status": "ok", "users_tracked": len(detector.buffers)}


@app.post("/events", dependencies=[Depends(require_key)])
def ingest(events: List[Event]):
    """Send logon events in chronological order; returns any alerts they triggered."""
    alerts, used = [], 0
    for e in events:
        if e.event_id not in VALID_EVENT_IDS:
            continue
        used += 1
        ev = e.model_dump()
        ev["user"] = ev["user"].strip().upper()
        alert = detector.process(ev)
        if alert:
            alerter.send(alert)
            alerts.append(alert)
    return {"received": len(events), "processed": used, "alerts": alerts}


@app.get("/users/{user}", dependencies=[Depends(require_key)])
def user_risk(user: str):
    res = detector.assess(user.upper())
    if res is None:
        raise HTTPException(404, "unknown user or not enough events to score")
    return res


@app.get("/alerts", dependencies=[Depends(require_key)])
def alerts(limit: int = 50):
    if not os.path.exists(ALERT_FILE):
        return []
    return pd.read_csv(ALERT_FILE).tail(limit).to_dict("records")


@app.get("/summary", dependencies=[Depends(require_key)])
def summary():
    risk = pd.read_csv(RISK_FILE)
    return {"users": len(risk), "risk_levels": risk["risk_level"].value_counts().to_dict()}
