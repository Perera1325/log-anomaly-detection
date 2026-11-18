import pandas as pd
from src.features import FEATURE_COLS, compute_user_features


def _events(user, hours, fails=()):
    return [dict(timestamp=pd.Timestamp(f"2026-01-05 {h:02d}:00:00"), user=user, pc="PC1",
                 event_id=4625 if i in fails else 4624, logon_type=2) for i, h in enumerate(hours)]


def test_feature_columns_and_sequences():
    df = pd.DataFrame(_events("A", [9, 10, 11, 12, 13, 14], fails={1, 2, 3}) + _events("B", [9, 10, 11]))
    f = compute_user_features(df).set_index("user")
    assert list(f.columns) == FEATURE_COLS
    assert f.loc["A", "max_fail_streak"] == 3 and f.loc["A", "fail_then_success"] == 1
    assert f.loc["B", "max_fail_streak"] == 0
    assert f.loc["B", "mean_gap_sec"] == 3600


def test_off_hours_ratio():
    df = pd.DataFrame(_events("N", [1, 2, 3, 12]))
    assert compute_user_features(df).set_index("user").loc["N", "off_hours_ratio"] == 0.75
