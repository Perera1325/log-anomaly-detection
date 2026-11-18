import pandas as pd

from config import RESULTS_DIR, RISK_FILE

LEVELS = {0: "Low", 1: "Medium", 2: "High", 3: "Critical"}


def model_agreement(results_file):
    print("\nCalculating model agreement...")
    df = pd.read_csv(results_file)
    df["anomaly_votes"] = ((df["IsolationForest"] == -1).astype(int)
                           + (df["OneClassSVM"] == -1).astype(int)
                           + (df["LOF"] == -1).astype(int))
    df["risk_level"] = df["anomaly_votes"].map(LEVELS)
    RESULTS_DIR.mkdir(exist_ok=True)
    df.sort_values("anomaly_votes", ascending=False).to_csv(RISK_FILE, index=False)
    print(df["risk_level"].value_counts().to_string())
    print(f"Saved -> {RISK_FILE}")
    return df
