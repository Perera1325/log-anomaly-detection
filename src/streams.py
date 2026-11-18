"""Event streams for the real-time monitor. Each yields dicts with the unified schema."""
import time
from datetime import datetime

import pandas as pd

from src.preprocess import clean_frame
from src.sources import parse_auth_line


def stream_replay(path, delay=0.0):
    live = clean_frame(pd.read_csv(path))
    for ev in live.to_dict("records"):
        yield ev
        if delay:
            time.sleep(delay)


def stream_linux(path, from_start=False, poll=1.0):
    """Follow a Linux auth.log like `tail -f` and parse sshd logon events as they appear."""
    with open(path, errors="ignore") as fh:
        if not from_start:
            fh.seek(0, 2)
        while True:
            line = fh.readline()
            if not line:
                time.sleep(poll)
                continue
            ev = parse_auth_line(line, datetime.now().year)
            if ev:
                ev["user"] = ev["user"].upper()
                yield ev


def stream_windows(poll=2):
    """Live Windows Security log (Windows + admin rights + `pip install pywin32`). Untested here."""
    import win32evtlog                                     # type: ignore
    h = win32evtlog.OpenEventLog(None, "Security")
    flags = win32evtlog.EVENTLOG_FORWARDS_READ | win32evtlog.EVENTLOG_SEQUENTIAL_READ
    while win32evtlog.ReadEventLog(h, flags, 0):           # skip existing history
        pass
    while True:
        for e in win32evtlog.ReadEventLog(h, flags, 0) or []:
            eid, s = e.EventID & 0xFFFF, (e.StringInserts or [])
            if eid not in (4624, 4625) or len(s) < 14:
                continue
            user = s[5]
            if user.endswith("$") or user in ("-", "SYSTEM"):
                continue
            yield {"timestamp": datetime.fromtimestamp(int(e.TimeGenerated)), "user": user.upper(),
                   "pc": s[11] if eid == 4624 else s[13], "event_id": eid,
                   "logon_type": int(s[8] if eid == 4624 else s[10])}
        time.sleep(poll)
