"""Collect log data from several sources into one unified schema:
timestamp, user, pc, event_id (4624 success / 4625 failure), logon_type, source.

Supported inputs
  * Windows Event Log CSV exports (Get-WinEvent / Event Viewer) or our own schema
  * CERT Insider Threat style logon.csv  (id,date,user,pc,activity)
  * Linux sshd authentication logs (/var/log/auth.log, /var/log/secure)
"""
import re
from datetime import datetime
from pathlib import Path

import pandas as pd

from config import LINUX_LOG, LINUX_LOG_YEAR, RAW_FILE

UNIFIED = ["timestamp", "user", "pc", "event_id", "logon_type", "source"]

_WIN_RENAME = {"TimeCreated": "timestamp", "date": "timestamp", "Id": "event_id",
               "EventID": "event_id", "TargetUserName": "user", "WorkstationName": "pc",
               "LogonType": "logon_type"}

_SSH_RE = re.compile(
    r"^(?P<mon>[A-Z][a-z]{2})\s+(?P<day>\d{1,2})\s+(?P<time>\d{2}:\d{2}:\d{2})\s+\S+\s+"
    r"sshd\[\d+\]:\s+(?P<res>Accepted|Failed)\s+\S+\s+for\s+(?:invalid user\s+)?"
    r"(?P<user>\S+)\s+from\s+(?P<ip>\S+)\s+port")


def parse_auth_line(line: str, year: int = LINUX_LOG_YEAR):
    """Parse one sshd line. Returns a dict, or None for lines that are not logon events."""
    m = _SSH_RE.match(line.strip())
    if not m:
        return None
    ts = datetime.strptime(f"{year} {m['mon']} {m['day']} {m['time']}", "%Y %b %d %H:%M:%S")
    return {"timestamp": ts, "user": m["user"], "pc": m["ip"],
            "event_id": 4624 if m["res"] == "Accepted" else 4625,
            "logon_type": 10, "source": "linux"}


def load_linux_auth(path, year: int = LINUX_LOG_YEAR) -> pd.DataFrame:
    rows = []
    with open(path, errors="ignore") as fh:
        for line in fh:
            ev = parse_auth_line(line, year)
            if ev:
                rows.append(ev)
    return pd.DataFrame(rows, columns=UNIFIED)


def load_windows_csv(path) -> pd.DataFrame:
    df = pd.read_csv(path).rename(columns=_WIN_RENAME)
    if "event_id" not in df and "activity" in df:           # CERT logon.csv
        df["event_id"] = df["activity"].map({"Logon": 4624, "Logoff": 4634})
    if "logon_type" not in df:
        df["logon_type"] = -1
    df["source"] = "windows"
    return df[UNIFIED]


def collect_sources(windows_csv=RAW_FILE, linux_log=LINUX_LOG) -> pd.DataFrame:
    frames = []
    if Path(windows_csv).exists():
        frames.append(load_windows_csv(windows_csv))
        print(f"  Windows source: {len(frames[-1]):,} records ({windows_csv})")
    if Path(linux_log).exists():
        frames.append(load_linux_auth(linux_log))
        print(f"  Linux source:   {len(frames[-1]):,} records ({linux_log})")
    if not frames:
        raise FileNotFoundError("No log sources found in Data/raw/")
    return pd.concat(frames, ignore_index=True)
