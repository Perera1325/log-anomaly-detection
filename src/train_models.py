import joblib
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import StandardScaler
from sklearn.svm import OneClassSVM

from config import CONTAMINATION, MODELS_DIR, MODEL_RESULTS_FILE, RANDOM_STATE
from src.features import FEATURE_COLS


def train_models(feature_file):
    print("\nLoading features...")
    df = pd.read_csv(feature_file)
    users = df["user"]
    scaler = StandardScaler().fit(df[FEATURE_COLS])
    X_scaled = scaler.transform(df[FEATURE_COLS])

    print("Training Isolation Forest...")
    iso = IsolationForest(contamination=CONTAMINATION, random_state=RANDOM_STATE).fit(X_scaled)

    print("Training One-Class SVM...")
    svm = OneClassSVM(nu=CONTAMINATION, kernel="rbf", gamma="scale").fit(X_scaled)

    print("Training Local Outlier Factor (novelty=True so it can score new data)...")
    lof = LocalOutlierFactor(contamination=CONTAMINATION, novelty=True).fit(X_scaled)

    # Training-set labels. For LOF, predict() on its own training data is optimistic
    # (each point is its own neighbour), so use the stored training outlier factors.
    lof_scores = lof.negative_outlier_factor_
    lof_pred = (lof_scores < lof.offset_).astype(int) * -2 + 1   # -1 anomaly / 1 normal

    results = pd.DataFrame({
        "user": users,
        "IsolationForest": iso.predict(X_scaled),
        "OneClassSVM": svm.predict(X_scaled),
        "LOF": lof_pred,
        "IF_score": iso.decision_function(X_scaled),
        "SVM_score": svm.decision_function(X_scaled),
        "LOF_score": lof_scores - lof.offset_,
    })
    MODEL_RESULTS_FILE.parent.mkdir(parents=True, exist_ok=True)
    results.to_csv(MODEL_RESULTS_FILE, index=False)

    MODELS_DIR.mkdir(exist_ok=True)
    joblib.dump(iso, MODELS_DIR / "isolation_forest.pkl")
    joblib.dump(svm, MODELS_DIR / "oneclass_svm.pkl")
    joblib.dump(lof, MODELS_DIR / "lof.pkl")
    joblib.dump(scaler, MODELS_DIR / "scaler.pkl")
    print("All models trained and saved.")
    return results
