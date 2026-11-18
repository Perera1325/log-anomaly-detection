"""Real-time monitoring + alerting.

  python monitor.py                              # replay Data/raw/live_events.csv
  python monitor.py --delay 0.05                 # slow replay for a demo
  python monitor.py --source linux --path /var/log/auth.log     # follow a Linux auth log
  python monitor.py --source windows             # live Windows Security log (admin + pywin32)

Each new logon event updates that user's rolling 30-day features, which are scored by the
three saved models. An alert fires when >= 2 models flag the user (console, CSV, log file,
and webhook/email if configured through environment variables).
"""
import argparse

import pandas as pd

from config import ALERT_FILE, CLEAN_FILE, LIVE_FILE, LINUX_LOG
from src.alerting import Alerter
from src.detector import Detector
from src.streams import stream_linux, stream_replay, stream_windows

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", choices=["replay", "linux", "windows"], default="replay")
    ap.add_argument("--path", help="file for replay/linux sources")
    ap.add_argument("--delay", type=float, default=0.0, help="seconds between replayed events")
    args = ap.parse_args()

    det = Detector()
    det.seed(pd.read_csv(CLEAN_FILE, parse_dates=["timestamp"]))
    if ALERT_FILE.exists():
        ALERT_FILE.unlink()
    alerter = Alerter()

    if args.source == "replay":
        stream = stream_replay(args.path or LIVE_FILE, args.delay)
    elif args.source == "linux":
        stream = stream_linux(args.path or LINUX_LOG)
    else:
        stream = stream_windows()

    n = n_alerts = 0
    try:
        for ev in stream:
            n += 1
            alert = det.process(ev)
            if alert:
                n_alerts += 1
                alerter.send(alert)
    except KeyboardInterrupt:
        pass
    print(f"\nProcessed {n:,} events, raised {n_alerts} alert(s). Log: {ALERT_FILE}")
