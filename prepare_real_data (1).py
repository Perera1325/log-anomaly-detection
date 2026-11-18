"""Turn a CERT Insider Threat logon.csv into this project's inputs.

  python prepare_real_data.py path/to/logon.csv [--insiders path/to/insiders.csv] [--live-days 5]

Writes into Data/raw/:
  logon_events.csv   history used for training            (everything before the live window)
  live_events.csv    the last N days, replayed by monitor.py as a "live" stream
  ground_truth.csv   only when --insiders is given (users listed as insiders = anomalous)
The old synthetic auth.log is moved aside so it does not mix into the real data.
"""
import argparse
from pathlib import Path

import pandas as pd

from config import RAW_DIR

ap = argparse.ArgumentParser()
ap.add_argument("logon")
ap.add_argument("--insiders", help="CERT answers file (insiders.csv) for labels")
ap.add_argument("--live-days", type=int, default=5)
ap.add_argument("--out-dir", default=str(RAW_DIR))
a = ap.parse_args()
out = Path(a.out_dir)
out.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(a.logon).rename(columns={"user_id": "user", "date": "timestamp"})
missing = {"timestamp", "user", "pc", "activity"} - set(df.columns)
if missing:
    raise SystemExit(f"Missing columns {sorted(missing)}. Found: {list(df.columns)}. "
                     "Expected CERT logon.csv (id,date,user,pc,activity) or user_id,pc,activity,timestamp.")
df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
df = df.dropna(subset=["timestamp", "user", "pc"])
df["user"] = df["user"].astype(str).str.split("/").str[-1].str.upper()       # DTAA/ACM2278 -> ACM2278
df["event_id"] = df["activity"].map({"Logon": 4624, "Logoff": 4634})
df = df.dropna(subset=["event_id"]).assign(logon_type=-1)
df["event_id"] = df["event_id"].astype(int)
cols = ["timestamp", "user", "pc", "event_id", "logon_type"]

cut = df["timestamp"].max().normalize() - pd.Timedelta(days=a.live_days - 1)
hist, live = df[df["timestamp"] < cut][cols], df[df["timestamp"] >= cut][cols]
hist.to_csv(out / "logon_events.csv", index=False)
live.to_csv(out / "live_events.csv", index=False)
print(f"History: {len(hist):,} rows ({hist['timestamp'].min():%Y-%m-%d} to {hist['timestamp'].max():%Y-%m-%d})")
print(f"Live:    {len(live):,} rows ({cut:%Y-%m-%d} onwards)")

for stale in ["auth.log", "ground_truth.csv"]:
    f = out / stale
    if f.exists():
        f.rename(out / f"{stale}.synthetic.bak")
        print(f"Moved synthetic {stale} aside")

if a.insiders:
    ins = pd.read_csv(a.insiders)
    bad = set(ins["user"].astype(str).str.upper())
    users = sorted(set(hist["user"]) | set(live["user"]))
    pd.DataFrame({"user": users, "is_anomalous": [int(u in bad) for u in users]}).to_csv(
        out / "ground_truth.csv", index=False)
    print(f"Labels:  {len(bad & set(users))} insider users among {len(users)}")
else:
    print("No --insiders file: precision/recall/F1 will be skipped (counts + consistency only).")
