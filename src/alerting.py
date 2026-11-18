"""Alert delivery: console, CSV, log file, optional webhook (Slack/Discord/Teams/any) and email.

Configure with environment variables (see .env.example):
  ALERT_WEBHOOK_URL                      POST JSON here
  SMTP_HOST SMTP_PORT SMTP_USER SMTP_PASS ALERT_EMAIL_TO ALERT_EMAIL_FROM   (email)
  ALERT_EMAIL_MIN_LEVEL                  High | Critical (default Critical)
"""
import json
import logging
import os
import smtplib
import urllib.request
from email.message import EmailMessage

import pandas as pd

from config import ALERT_FILE, RESULTS_DIR

log = logging.getLogger("alerts")


def format_alert(a: dict) -> str:
    return (f"[{a['risk_level'].upper()}] user={a['user']} at {a['time']} | models={a['models']} | "
            f"fail_ratio={a['fail_ratio']} off_hours={a['off_hours_ratio']} "
            f"pcs={a['unique_pcs']} max_fail_streak={a.get('max_fail_streak', '-')}")


class Alerter:
    def __init__(self, csv_path=ALERT_FILE, quiet=False):
        self.csv_path = csv_path
        self.quiet = quiet
        RESULTS_DIR.mkdir(exist_ok=True)
        fh = logging.FileHandler(RESULTS_DIR / "alerts.log")
        fh.setFormatter(logging.Formatter("%(asctime)s %(message)s"))
        log.setLevel(logging.INFO)
        if not any(isinstance(h, logging.FileHandler) for h in log.handlers):
            log.addHandler(fh)

    def send(self, alert: dict):
        text = format_alert(alert)
        if not self.quiet:
            print("[ALERT]", text)
        log.info(text)
        pd.DataFrame([alert]).to_csv(self.csv_path, mode="a", index=False,
                                     header=not os.path.exists(self.csv_path))
        self._webhook(alert, text)
        self._email(alert, text)

    def _webhook(self, alert, text):
        url = os.getenv("ALERT_WEBHOOK_URL")
        if not url:
            return
        body = json.dumps({"text": text, "content": text, **alert}).encode()
        req = urllib.request.Request(url, body, {"Content-Type": "application/json"})
        try:
            urllib.request.urlopen(req, timeout=5).read()
        except Exception as e:                      # alerting must never crash the monitor
            log.error("webhook failed: %s", e)

    def _email(self, alert, text):
        host, to = os.getenv("SMTP_HOST"), os.getenv("ALERT_EMAIL_TO")
        if not (host and to):
            return
        order = ["Medium", "High", "Critical"]
        if order.index(alert["risk_level"]) < order.index(os.getenv("ALERT_EMAIL_MIN_LEVEL", "Critical")):
            return
        msg = EmailMessage()
        msg["Subject"] = f"[{alert['risk_level']}] Anomalous activity: {alert['user']}"
        msg["From"] = os.getenv("ALERT_EMAIL_FROM", os.getenv("SMTP_USER", "alerts@localhost"))
        msg["To"] = to
        msg.set_content(text)
        try:
            with smtplib.SMTP(host, int(os.getenv("SMTP_PORT", "587")), timeout=10) as s:
                s.starttls()
                if os.getenv("SMTP_USER"):
                    s.login(os.getenv("SMTP_USER"), os.getenv("SMTP_PASS", ""))
                s.send_message(msg)
        except Exception as e:
            log.error("email failed: %s", e)
