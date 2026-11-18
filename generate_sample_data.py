"""Generate a synthetic Windows logon dataset (4624/4625 events) with injected anomalies.

Replace Data/raw/logon_events.csv with your real export to use real data
(columns: timestamp, user, pc, event_id, logon_type).
"""
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

from config import RAW_DIR, RAW_FILE, LIVE_FILE, TRUTH_FILE, LINUX_LOG

rng = np.random.default_rng(42)
N_USERS, N_PCS = 200, 250
START = datetime(2026, 10, 5)   # a Monday, so the live window falls on weekdays
HISTORY_DAYS, LIVE_DAYS = 30, 2

users = [f"U{i:03d}" for i in range(1, N_USERS + 1)]
pcs = [f"PC{i:03d}" for i in range(1, N_PCS + 1)]

# anomalous users in the history (5% of users)
shuffled = list(rng.permutation(users))
brute = shuffled[0:4]        # many failed logons
night = shuffled[4:7]        # off-hours / weekend activity
lateral = shuffled[7:10]     # logons on many different machines
truth = {u: "normal" for u in users}
truth.update({u: "brute_force" for u in brute})
truth.update({u: "night_owl" for u in night})
truth.update({u: "lateral_movement" for u in lateral})
# a normally-behaving user who is compromised only during the live window
LIVE_VICTIM = next(u for u in shuffled[10:] if truth[u] == "normal")

home = {u: list(rng.choice(pcs, size=rng.integers(1, 3), replace=False)) for u in users}
rate = {u: rng.uniform(3, 8) for u in users}


def event(ts, user, pc, fail, ltype):
    return dict(timestamp=ts, user=user, pc=pc,
                event_id=4625 if fail else 4624, logon_type=ltype)


def day_events(user, day, kind="normal", compromised=False):
    rows = []
    weekend = day.weekday() >= 5
    if kind == "night_owl" or (compromised and kind == "night_owl"):
        n = rng.poisson(rate[user] * 1.2)
        hours = rng.integers(0, 6, size=n)
    elif weekend:
        n = rng.poisson(0.3) if rng.random() < 0.05 else 0
        hours = rng.integers(9, 17, size=n)
    else:
        n = rng.poisson(rate[user])
        hours = np.clip(rng.normal(13, 2.5, size=n), 8, 18).astype(int)
    for h in hours:
        ts = day + timedelta(hours=int(h), minutes=int(rng.integers(0, 60)),
                             seconds=int(rng.integers(0, 60)))
        pc = rng.choice(home[user])
        ltype = 2 if rng.random() < 0.9 else 10
        fail = rng.random() < 0.03
        if kind == "brute_force":
            fail = rng.random() < 0.45
        if kind == "lateral_movement":
            pc = rng.choice(pcs[:60]); ltype = int(rng.choice([3, 10]))
        rows.append(event(ts, user, pc, fail, ltype))
    if kind == "brute_force" and not weekend:
        for _ in range(int(rng.integers(10, 30))):      # burst of failures
            ts = day + timedelta(hours=int(rng.integers(0, 24)), seconds=int(rng.integers(0, 3600)))
            rows.append(event(ts, user, rng.choice(home[user]), True, 3))
        rows.append(event(day + timedelta(hours=23, minutes=59), user, rng.choice(home[user]), False, 3))
    return rows


# ---------------------------------------------------------------- Linux auth.log
LINUX_USERS = [f"l{i:02d}" for i in range(1, 41)]
LINUX_BAD = {"l01": "brute_force", "l02": "brute_force", "l03": "night_owl", "l04": "lateral_movement"}
l_ip = {u: [f"10.0.{rng.integers(1, 5)}.{rng.integers(2, 250)}" for _ in range(rng.integers(1, 3))]
        for u in LINUX_USERS}


def linux_lines(days):
    lines = []

    def add(ts, user, ip, ok):
        what = "Accepted" if ok else "Failed"
        inv = "" if ok or rng.random() < 0.7 else "invalid user "
        lines.append((ts, f"{ts.strftime('%b')} {ts.day:2d} {ts:%H:%M:%S} srv01 sshd[{rng.integers(1000, 30000)}]: "
                          f"{what} password for {inv}{user} from {ip} port {rng.integers(20000, 60000)} ssh2"))

    for d in range(days):
        day = START + timedelta(days=d)
        if day.weekday() >= 5:
            continue
        for u in LINUX_USERS:
            kind = LINUX_BAD.get(u, "normal")
            n = rng.poisson(3)
            for _ in range(n):
                h = int(rng.integers(0, 6)) if kind == "night_owl" else int(np.clip(rng.normal(13, 2.5), 8, 18))
                ts = day + timedelta(hours=h, minutes=int(rng.integers(0, 60)), seconds=int(rng.integers(0, 60)))
                ip = rng.choice(l_ip[u])
                if kind == "lateral_movement":
                    ip = f"10.0.9.{rng.integers(2, 60)}"
                add(ts, u, ip, rng.random() > (0.45 if kind == "brute_force" else 0.03))
            if kind == "brute_force":
                for _ in range(int(rng.integers(10, 25))):
                    ts = day + timedelta(hours=int(rng.integers(0, 24)), seconds=int(rng.integers(0, 3600)))
                    add(ts, u, f"203.0.113.{rng.integers(2, 250)}", False)
        lines.append((day + timedelta(hours=3), f"{day:%b} {day.day:2d} 03:00:01 srv01 CRON[812]: session opened for user root"))
    lines.sort(key=lambda x: x[0])
    return [l for _, l in lines]


def build(days, offset, compromised_user=None):
    rows = []
    for d in range(days):
        day = START + timedelta(days=offset + d)
        for u in users:
            rows += day_events(u, day, truth[u] if u != compromised_user else "normal")
        if compromised_user:  # attack: night logons, failures, many machines
            for _ in range(60):
                ts = day + timedelta(hours=int(rng.integers(0, 5)), minutes=int(rng.integers(0, 60)),
                                     seconds=int(rng.integers(0, 60)))
                rows.append(event(ts, compromised_user, rng.choice(pcs), rng.random() < 0.4,
                                  int(rng.choice([3, 10]))))
    return pd.DataFrame(rows)


def add_noise(df):
    """Duplicates, missing values and irrelevant events so preprocessing has work to do."""
    dup = df.sample(frac=0.01, random_state=1)
    df = pd.concat([df, dup], ignore_index=True)
    miss = df.sample(frac=0.005, random_state=2).index
    df.loc[miss, "user"] = None
    logoff = df.sample(frac=0.02, random_state=3).copy()
    logoff["event_id"] = 4634
    df = pd.concat([df, logoff], ignore_index=True)
    return df.sort_values("timestamp").reset_index(drop=True)


if __name__ == "__main__":
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    hist = add_noise(build(HISTORY_DAYS, 0))
    live = build(LIVE_DAYS, HISTORY_DAYS, compromised_user=LIVE_VICTIM)
    live = live.sort_values("timestamp").reset_index(drop=True)
    hist.to_csv(RAW_FILE, index=False)
    live.to_csv(LIVE_FILE, index=False)
    LINUX_LOG.write_text("\n".join(linux_lines(HISTORY_DAYS)) + "\n")
    all_truth = dict(truth)
    all_truth.update({u: LINUX_BAD.get(u, "normal") for u in LINUX_USERS})
    pd.DataFrame({"user": [u.upper() for u in all_truth],
                  "is_anomalous": [int(t != "normal") for t in all_truth.values()],
                  "type": list(all_truth.values())}).to_csv(TRUTH_FILE, index=False)
    print(f"Linux:   {LINUX_LOG}")
    print(f"History: {len(hist):,} rows -> {RAW_FILE}")
    print(f"Live:    {len(live):,} rows -> {LIVE_FILE}  (compromised user: {LIVE_VICTIM})")
    print(f"Labels:  {TRUTH_FILE}")
