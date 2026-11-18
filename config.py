"""Central settings for the log anomaly detection project."""
from pathlib import Path

ROOT = Path(__file__).resolve().parent
RAW_DIR = ROOT / "Data" / "raw"
PROC_DIR = ROOT / "Data" / "processed"
MODELS_DIR = ROOT / "Models"
RESULTS_DIR = ROOT / "Results"

RAW_FILE = RAW_DIR / "logon_events.csv"          # history used for training
LIVE_FILE = RAW_DIR / "live_events.csv"          # events replayed by the monitor
TRUTH_FILE = RAW_DIR / "ground_truth.csv"        # optional labels (synthetic data only)

CLEAN_FILE = PROC_DIR / "logon_cleaned.csv"
FEATURE_FILE = PROC_DIR / "user_features.csv"
MODEL_RESULTS_FILE = PROC_DIR / "model_results.csv"

COMPARISON_FILE = RESULTS_DIR / "model_comparison.csv"
RISK_FILE = RESULTS_DIR / "high_risk_users.csv"
ALERT_FILE = RESULTS_DIR / "alerts.csv"

LINUX_LOG = RAW_DIR / "auth.log"                 # Linux sshd authentication log (optional)
LINUX_LOG_YEAR = 2026                            # syslog lines carry no year

VALID_EVENT_IDS = [4624, 4625]    # 4624 = logon success, 4625 = logon failure
OFF_HOURS = (7, 19)               # outside 07:00-19:00 counts as off-hours
CONTAMINATION = 0.05
RANDOM_STATE = 42
WINDOW_DAYS = 30                  # rolling window used by the real-time monitor
MIN_EVENTS_FOR_SCORING = 20
ALERT_MIN_VOTES = 2               # alert when >= 2 of 3 models flag a user
