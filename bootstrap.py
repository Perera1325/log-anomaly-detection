"""Make sure data + trained models exist (idempotent). Used by Docker, the API and the dashboard."""
import subprocess
import sys

from config import CLEAN_FILE, LIVE_FILE, MODELS_DIR, RAW_FILE, RISK_FILE, ROOT


def ready() -> bool:
    return all(p.exists() for p in [CLEAN_FILE, RISK_FILE, MODELS_DIR / "scaler.pkl",
                                    MODELS_DIR / "lof.pkl", LIVE_FILE])


def ensure_ready():
    if ready():
        return
    py = sys.executable
    if not RAW_FILE.exists():
        print("No raw data found: generating synthetic sample data...")
        subprocess.run([py, "generate_sample_data.py"], cwd=ROOT, check=True)
    elif not LIVE_FILE.exists():
        LIVE_FILE.write_text("timestamp,user,pc,event_id,logon_type\n")   # empty live stream
    subprocess.run([py, "main.py"], cwd=ROOT, check=True)


if __name__ == "__main__":
    ensure_ready()
    print("Ready.")
