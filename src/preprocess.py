import pandas as pd

from config import CLEAN_FILE, PROC_DIR, VALID_EVENT_IDS
from src.sources import collect_sources


def clean_frame(df: pd.DataFrame) -> pd.DataFrame:
    """Clean a raw log DataFrame (also used by the real-time monitor and the API)."""
    df = df.drop_duplicates()
    df = df.dropna(subset=["timestamp", "user", "pc", "event_id"])
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    df = df.dropna(subset=["timestamp"])
    df["user"] = df["user"].astype(str).str.strip().str.upper()
    df["event_id"] = pd.to_numeric(df["event_id"], errors="coerce")
    df = df[df["event_id"].isin(VALID_EVENT_IDS)]
    df["event_id"] = df["event_id"].astype(int)
    if "logon_type" not in df:
        df["logon_type"] = -1
    df["logon_type"] = pd.to_numeric(df["logon_type"], errors="coerce").fillna(-1).astype(int)
    return df.sort_values("timestamp").reset_index(drop=True)


def preprocess():
    """Collect every available source, clean it and save logon_cleaned.csv."""
    print("\nCollecting log sources...")
    raw = collect_sources()
    print(f"Dataset loaded: {len(raw):,} records")
    df = clean_frame(raw)
    PROC_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(CLEAN_FILE, index=False)
    print(f"Cleaning completed: {len(df):,} records kept ({len(raw) - len(df):,} removed)")
    print(f"Saved -> {CLEAN_FILE}")
    return df
