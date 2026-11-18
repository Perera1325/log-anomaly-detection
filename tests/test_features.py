import pandas as pd
from src.features import FEATURE_COLS, compute_user_features


def _events(user, hours, fails=()):
    return [dict(timestamp=pd.Timestamp(f"2026-01-05 {h:02d}:00:00"), user=user, pc="PC1",
                 event_id=4625 if i in fails else 4624, logon_type=2) for i, h in enumerate(hours)]


def test_feature_columns_and_sequences():
    df = pd.DataFrame(_events("A", [9, 10, 11, 12, 13, 14], fails={1, 2, 3}) + _events("B", [9, 10, 11]))
    f = compute_user_features(df).set_index("user")
    assert list(f.columns) == ["total_events"] + FEATURE_COLS      # total_events is reporting-only
    assert "total_events" not in FEATURE_COLS and "active_days" not in FEATURE_COLS
    assert f.loc["A", "max_fail_streak"] == 3 and f.loc["A", "fail_then_success"] == 1
    assert f.loc["B", "max_fail_streak"] == 0
    assert f.loc["B", "median_gap_sec"] == 3600 and f.loc["B", "short_gap_ratio"] == 0


def test_off_hours_ratio():
    df = pd.DataFrame(_events("N", [1, 2, 3, 12]))
    assert compute_user_features(df).set_index("user").loc["N", "off_hours_ratio"] == 0.75


def test_personal_baseline_flags_a_new_after_hours_burst():
    """A user who suddenly logs on at night scores far above their own normal week."""
    days = pd.date_range("2026-01-05", periods=60, freq="D")
    calm = [dict(timestamp=d + pd.Timedelta(hours=9), user="U", pc="PC1", event_id=4624, logon_type=2) for d in days]
    burst = [dict(timestamp=days[-3] + pd.Timedelta(hours=2, minutes=i), user="U", pc="PC1", event_id=4624,
                  logon_type=2) for i in range(12)]
    steady = [dict(timestamp=d + pd.Timedelta(hours=9), user="S", pc="PC1", event_id=4624, logon_type=2) for d in days]
    f = compute_user_features(pd.DataFrame(calm + burst + steady)).set_index("user")
    assert f.loc["U", "wk_max_z_off"] > 2.5 > f.loc["S", "wk_max_z_off"]
    assert f.loc["U", "hour_surp_daymax"] > f.loc["S", "hour_surp_daymax"]
