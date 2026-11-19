"""Create 5 sample attack log files (+1 normal control) in sample_attacks/ for the dashboard's Attack analyzer.

  python generate_attack_samples.py
Every file holds ~8 ordinary office users (45 weekdays of normal behaviour) plus one attack scenario.
All data are synthetic.
"""
from pathlib import Path

import numpy as np
import pandas as pd

OUT = Path(__file__).resolve().parent / "sample_attacks"
START = pd.Timestamp("2026-09-01")
DAYS = [d for d in pd.date_range(START, periods=56, freq="D") if d.dayofweek < 5][:45]   # 45 weekdays


def ev(ts, user, pc, fail=False, ltype=2):
    return dict(timestamp=pd.Timestamp(ts), user=user, pc=pc, event_id=4625 if fail else 4624, logon_type=ltype)


def normal_user(rng, user, days=DAYS):
    """Ordinary office worker, modelled on CERT normals: one morning logon per weekday on one PC."""
    pc = f"PC-{rng.integers(1000, 9999)}"
    rows = []
    for d in days:
        rows.append(ev(d + pd.Timedelta(hours=8, minutes=int(rng.integers(0, 30)), seconds=int(rng.integers(0, 60))),
                       user, pc))
        if rng.random() < 0.3:                                     # sometimes a second logon after lunch
            rows.append(ev(d + pd.Timedelta(hours=13, minutes=int(rng.integers(0, 60))), user, pc))
        if rng.random() < 0.03:                                    # the odd typo in a password
            rows.append(ev(d + pd.Timedelta(hours=8, minutes=31), user, pc, fail=True))
    return rows


def base(seed, n_users=8):
    rng = np.random.default_rng(seed)
    return rng, [r for i in range(n_users) for r in normal_user(rng, f"EMP{seed}{i:02d}")]


def brute_force():
    rng, rows = base(1)
    rows += normal_user(rng, "VICTIM_BF")
    t = DAYS[-1] + pd.Timedelta(hours=14)                          # 60 wrong passwords in 8 minutes, then success
    rows += [ev(t + pd.Timedelta(seconds=8 * i), "VICTIM_BF", "PC-ATTACKER", fail=True, ltype=3) for i in range(60)]
    rows.append(ev(t + pd.Timedelta(seconds=500), "VICTIM_BF", "PC-ATTACKER", ltype=3))
    return rows


def off_hours_access():
    rng, rows = base(2)
    u = "INSIDER_NIGHT"
    rows += normal_user(rng, u, DAYS[:-8])
    pc = "PC-4411"
    for d in DAYS[-8:]:                                            # suddenly works 01:00-04:00 every night
        for _ in range(int(rng.integers(3, 7))):
            rows.append(ev(d + pd.Timedelta(hours=int(rng.integers(1, 4)), minutes=int(rng.integers(0, 60))), u, pc))
    return rows


def lateral_movement():
    rng, rows = base(3)
    u = "SVC_LATERAL"
    rows += normal_user(rng, u, DAYS[:-2])
    for d in DAYS[-2:]:                                            # hops across 25 machines in a few hours
        for k in range(25):
            rows.append(ev(d + pd.Timedelta(hours=11, minutes=7 * k), u, f"SRV-{100 + k}", ltype=3))
    return rows


def password_spraying():
    rng, rows = base(4, n_users=14)
    t = DAYS[-3] + pd.Timedelta(hours=3)                           # one machine tries 1-2 passwords per user
    users = sorted({r["user"] for r in rows})
    for i, u in enumerate(users):
        for j in range(int(rng.integers(1, 3))):
            rows.append(ev(t + pd.Timedelta(seconds=45 * i + 5 * j), u, "WS-ROGUE", fail=True, ltype=3))
    return rows


def account_takeover():
    rng, rows = base(5)
    u = "COMPROMISED_AC"
    rows += normal_user(rng, u, DAYS[:-1])
    t = DAYS[-1] + pd.Timedelta(days=1, hours=3)                   # Saturday 03:00, new machines, burst of activity
    for k in range(90):
        rows.append(ev(t + pd.Timedelta(seconds=40 * k), u, f"NEW-HOST-{k % 12}", fail=(k % 9 == 0), ltype=10))
    return rows


def normal_control():
    rng, rows = base(6, n_users=10)
    return rows


FILES = {"1_brute_force.csv": brute_force, "2_off_hours_insider.csv": off_hours_access,
         "3_lateral_movement.csv": lateral_movement, "4_password_spraying.csv": password_spraying,
         "5_account_takeover.csv": account_takeover, "0_normal_control.csv": normal_control}

if __name__ == "__main__":
    OUT.mkdir(exist_ok=True)
    for name, fn in FILES.items():
        df = pd.DataFrame(fn()).sort_values("timestamp")
        df["timestamp"] = df["timestamp"].dt.strftime("%Y-%m-%d %H:%M:%S")
        df.to_csv(OUT / name, index=False)
        print(f"{name:28s} {len(df):5d} events, {df['user'].nunique()} users")
