from collections import Counter

import numpy as np
import pandas as pd

from config import FEATURE_FILE, MIN_SNAPSHOT_EVENTS, OFF_HOURS, SNAPSHOT_FRACS, TRAINING_FILE

# total_events / active_days are deliberately NOT model features: they measure how long a user was
# observed, so on CERT they mostly detect "left the organisation" rather than suspicious behaviour.
FREQUENCY = ["fail_ratio", "events_per_day", "max_events_per_hour", "off_hours_ratio", "weekend_ratio",
             "hour_entropy", "early_ratio", "late_ratio"]
INTERVAL = ["median_gap_sec", "p90_gap_sec", "min_gap_sec", "short_gap_ratio"]   # robust to one new event
DIVERSITY = ["unique_pcs", "unique_logon_types", "rare_pc_ratio"]
SEQUENCE = ["max_fail_streak", "fail_then_success", "pc_switch_ratio", "transition_entropy"]
# Personal baseline: how far a user's *worst* week/day is from that same user's normal behaviour.
PERSONAL = ["wk_max_z_off", "wk_max_z_early", "wk_max_z_late", "wk_max_z_n", "recent_off_shift",
            "hour_surp_daymax", "hour_surp_p95", "rare_hour_ratio", "newpc_ratio_late"]
FEATURE_COLS = FREQUENCY + INTERVAL + DIVERSITY + SEQUENCE + PERSONAL


def _sequence_features(df: pd.DataFrame) -> pd.DataFrame:
    """Sequential patterns of each user's ordered event stream."""
    out = {}
    for user, g in df.groupby("user", sort=False):
        fail = g["event_id"].to_numpy() == 4625
        pcs = g["pc"].to_numpy()
        streak = max_streak = after_fail_runs = 0
        for f in fail:
            if f:
                streak += 1
                max_streak = max(max_streak, streak)
            else:
                if streak >= 3:            # success right after >=3 failures
                    after_fail_runs += 1
                streak = 0
        switch = float((pcs[1:] != pcs[:-1]).mean()) if len(pcs) > 1 else 0.0
        tokens = list(zip(g["event_id"], g["logon_type"]))
        bigrams = Counter(zip(tokens[:-1], tokens[1:]))
        total = sum(bigrams.values())
        ent = -sum(c / total * np.log2(c / total) for c in bigrams.values()) if total else 0.0
        out[user] = (max_streak, after_fail_runs, switch, ent)
    return pd.DataFrame.from_dict(out, orient="index", columns=SEQUENCE)


def _robust_z(values: pd.Series, users: pd.Series) -> pd.Series:
    """z-score of each value against its own user's median, using a MAD/std floor."""
    g = values.groupby(users)
    med = g.transform("median")
    mad = (values - med).abs().groupby(users).transform("median") * 1.4826
    sd = np.maximum(np.maximum(mad, g.transform("std").fillna(0)), 0.5)
    return (values - med) / sd


def _personal_features(df: pd.DataFrame) -> pd.DataFrame:
    """Per-user deviation features: each user is compared with their own history, not with the crowd."""
    d = df[["user", "timestamp", "pc"]].copy()
    d["hour"] = d["timestamp"].dt.hour
    d["off"] = ((d["hour"] < OFF_HOURS[0]) | (d["hour"] >= OFF_HOURS[1])).astype(int)
    d["early"] = (d["hour"] < 6).astype(int)
    d["late"] = (d["hour"] >= 21).astype(int)
    d["week"] = (d["timestamp"] - d["timestamp"].min()).dt.days // 7
    wk = d.groupby(["user", "week"]).agg(n=("pc", "size"), off=("off", "sum"),
                                         early=("early", "sum"), late=("late", "sum")).reset_index()
    out = pd.DataFrame(index=sorted(d["user"].unique()))
    for col, name in [("off", "wk_max_z_off"), ("early", "wk_max_z_early"),
                      ("late", "wk_max_z_late"), ("n", "wk_max_z_n")]:
        out[name] = _robust_z(wk[col].astype(float), wk["user"]).groupby(wk["user"]).max()

    rows = {}
    for user, g in d.sort_values("timestamp").groupby("user", sort=False):
        n = len(g)
        if n < 30:                                   # too little history to build a personal baseline
            rows[user] = (0.0,) * 5
            continue
        k = int(n * 0.6)
        base, test = g.iloc[:k], g.iloc[k:]
        hp = np.bincount(base["hour"], minlength=24) + 0.5     # smoothed hour-of-day habit
        hp = hp / hp.sum()
        surprise = -np.log2(hp[test["hour"].to_numpy()])       # how unusual is each later logon hour?
        day_mean = pd.Series(surprise).groupby(test["timestamp"].dt.date.to_numpy()).mean()
        newpc = (~test["pc"].isin(set(base["pc"]))).mean()
        cut = g["timestamp"].max() - pd.Timedelta(days=56)
        recent, older = g[g["timestamp"] > cut], g[g["timestamp"] <= cut]
        shift = (recent["off"].mean() - older["off"].mean()) if len(older) >= 10 and len(recent) >= 5 else 0.0
        rows[user] = (float(shift), float(day_mean.max()), float(np.quantile(surprise, 0.95)),
                      float((surprise > np.log2(100)).mean()), float(newpc))
    cols = ["recent_off_shift", "hour_surp_daymax", "hour_surp_p95", "rare_hour_ratio", "newpc_ratio_late"]
    return out.join(pd.DataFrame.from_dict(rows, orient="index", columns=cols)).fillna(0.0)


def compute_user_features(df: pd.DataFrame) -> pd.DataFrame:
    """Behavioural features per user: event frequency, time intervals, sequential patterns."""
    df = df.sort_values("timestamp").copy()
    df["hour"] = df["timestamp"].dt.hour
    df["date"] = df["timestamp"].dt.date
    df["is_fail"] = (df["event_id"] == 4625).astype(int)
    df["off_hours"] = ((df["hour"] < OFF_HOURS[0]) | (df["hour"] >= OFF_HOURS[1])).astype(int)
    df["weekend"] = (df["timestamp"].dt.dayofweek >= 5).astype(int)
    df["gap"] = df.groupby("user")["timestamp"].diff().dt.total_seconds()
    df["hour_bucket"] = df["timestamp"].dt.floor("h")
    df["short_gap"] = (df["gap"] < 300).astype(float).where(df["gap"].notna())
    df["early"] = (df["hour"] < 6).astype(int)
    df["late"] = (df["hour"] >= 21).astype(int)
    pc_count = df.groupby(["user", "pc"])["pc"].transform("size")
    df["rare_pc"] = (pc_count <= 2).astype(int)               # logons on PCs the user hardly ever uses

    g = df.groupby("user")
    feats = pd.DataFrame({
        "total_events": g.size(),
        "fail_count": g["is_fail"].sum(),
        "fail_ratio": g["is_fail"].mean(),
        "unique_pcs": g["pc"].nunique(),
        "unique_logon_types": g["logon_type"].nunique(),
        "off_hours_ratio": g["off_hours"].mean(),
        "weekend_ratio": g["weekend"].mean(),
        "active_days": g["date"].nunique(),                   # only used to derive events_per_day
        "early_ratio": g["early"].mean(),
        "late_ratio": g["late"].mean(),
        "rare_pc_ratio": g["rare_pc"].mean(),
        "median_gap_sec": g["gap"].median().fillna(0),
        "p90_gap_sec": g["gap"].quantile(0.9).fillna(0),
        "min_gap_sec": g["gap"].min().fillna(0),
        "short_gap_ratio": df["short_gap"].groupby(df["user"]).mean().fillna(0),   # bursts: gaps < 5 min
        "max_events_per_hour": df.groupby(["user", "hour_bucket"]).size().groupby("user").max(),
    })
    feats["events_per_day"] = feats["total_events"] / feats["active_days"]

    hour_counts = pd.crosstab(df["user"], df["hour"])
    p = hour_counts.div(hour_counts.sum(axis=1), axis=0).to_numpy()
    with np.errstate(divide="ignore", invalid="ignore"):
        ent = -np.nansum(np.where(p > 0, p * np.log2(p), 0), axis=1)
    feats["hour_entropy"] = pd.Series(ent, index=hour_counts.index)

    feats = feats.join(_sequence_features(df)).join(_personal_features(df))
    # total_events is kept for reporting / snapshot filtering only; it is not in FEATURE_COLS
    return feats[["total_events"] + FEATURE_COLS].reset_index().rename(columns={"index": "user"})


def training_snapshots(clean_df: pd.DataFrame, final: pd.DataFrame) -> pd.DataFrame:
    """Each user's features as they looked at several earlier points in time, plus the final state.

    Training on snapshots teaches the models how features normally drift as new events arrive, so one
    ordinary new event in live monitoring is not mistaken for an anomaly.
    """
    t0, t1 = clean_df["timestamp"].min(), clean_df["timestamp"].max()
    frames = []
    for frac in SNAPSHOT_FRACS:
        sub = clean_df[clean_df["timestamp"] <= t0 + (t1 - t0) * frac]
        snap = compute_user_features(sub)
        frames.append(snap[snap["total_events"] >= MIN_SNAPSHOT_EVENTS])
    return pd.concat(frames + [final], ignore_index=True)


def extract_features(clean_df: pd.DataFrame):
    print("\nExtracting behavioural features...")
    feats = compute_user_features(clean_df)
    feats.to_csv(FEATURE_FILE, index=False)
    snaps = training_snapshots(clean_df, feats)
    snaps.to_csv(TRAINING_FILE, index=False)
    print(f"Training set: {len(snaps)} user snapshots ({len(SNAPSHOT_FRACS)} past cut-offs + final state)")
    print(f"Feature extraction completed: {len(feats)} users x {len(FEATURE_COLS)} features "
          f"({len(FREQUENCY)} frequency, {len(INTERVAL)} interval, {len(DIVERSITY)} diversity, "
          f"{len(SEQUENCE)} sequence, {len(PERSONAL)} personal-baseline)")
    print(f"Saved -> {FEATURE_FILE}")
    return feats
