import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

from src.alerting import Alerter

ALERT = {"time": "2026-01-01 00:00:00", "user": "X", "risk_level": "Critical", "votes": 3,
         "models": "IsolationForest+OneClassSVM+LOF", "fail_ratio": 0.9, "off_hours_ratio": 1.0,
         "unique_pcs": 3, "max_fail_streak": 9}


def test_webhook_and_csv(tmp_path, monkeypatch):
    got = []

    class H(BaseHTTPRequestHandler):
        def do_POST(self):
            got.append(json.loads(self.rfile.read(int(self.headers["Content-Length"]))))
            self.send_response(200); self.end_headers()
        def log_message(self, *a): pass

    srv = HTTPServer(("127.0.0.1", 0), H)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    monkeypatch.setenv("ALERT_WEBHOOK_URL", f"http://127.0.0.1:{srv.server_port}/")
    csv = tmp_path / "alerts.csv"
    Alerter(csv_path=csv, quiet=True).send(ALERT)
    srv.shutdown()
    assert got and got[0]["user"] == "X" and "CRITICAL" in got[0]["text"]
    assert csv.exists()


def test_webhook_failure_does_not_raise(tmp_path, monkeypatch):
    monkeypatch.setenv("ALERT_WEBHOOK_URL", "http://127.0.0.1:9/")
    Alerter(csv_path=tmp_path / "a.csv", quiet=True).send(ALERT)
