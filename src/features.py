from collections import Counter

import numpy as np
import pandas as pd

from config import FEATURE_FILE, MIN_SNAPSHOT_EVENTS, OFF_HOURS, SNAPSHOT_FRACS, TRAINING_FILE

FREQUENCY = ["total_events", "fail_count", "fail_ratio", "active_days", "events_per_day",
             "max_events_per_hour", "off_hours_ratio", "weekend_ratio", "hour_entropy"]
INTERVAL = ["median_gap_sec", "p90_gap_sec", "min_gap_sec", "short_gap_ratio"]   # robust to one new event
DIVERSITY = ["unique_pcs", "unique_logon_types"]
SEQUENCE = ["max_fail_streak", "fail_then_success", "pc_switch_ratio", "transition_entropy"]
FEATURE_COLS = FREQUENCY + INTERVAL + DIVERSITY + SEQUENCE


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

    g = df.groupby("user")
    feats = pd.DataFrame({
        "total_events": g.size(),
        "fail_count": g["is_fail"].sum(),
        "fail_ratio": g["is_fail"].mean(),
        "unique_pcs": g["pc"].nunique(),
        "unique_logon_types": g["logon_type"].nunique(),
        "off_hours_ratio": g["off_hours"].mean(),
        "weekend_ratio": g["weekend"].mean(),
        "active_days": g["date"].nunique(),
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

    feats = feats.join(_sequence_features(df))
    return feats[FEATURE_COLS].reset_index().rename(columns={"index": "user"})


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
          f"({len(FREQUENCY)} frequency, {len(INTERVAL)} interval, {len(SEQUENCE)} sequence)")
    print(f"Saved -> {FEATURE_FILE}")
    return feats
