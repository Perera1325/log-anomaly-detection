"""Run the whole offline pipeline:  python main.py"""
from config import (CLEAN_FILE, FEATURE_FILE, MODEL_RESULTS_FILE, RAW_FILE)
from src.preprocess import preprocess
from src.features import extract_features
from src.train_models import train_models
from src.evaluate_models import evaluate_models
from src.model_agreement import model_agreement

if __name__ == "__main__":
    if not RAW_FILE.exists():
        raise SystemExit("Raw data missing. Run:  python generate_sample_data.py  (or put your CSV at Data/raw/logon_events.csv)")
    clean = preprocess()
    extract_features(clean)
    train_models(FEATURE_FILE)
    evaluate_models(MODEL_RESULTS_FILE, FEATURE_FILE)
    model_agreement(MODEL_RESULTS_FILE)
    print("\nPipeline finished. Next: python monitor.py   |   streamlit run dashboard.py")
