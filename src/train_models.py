import joblib
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import StandardScaler
from sklearn.svm import OneClassSVM

from config import CONTAMINATION, MODELS_DIR, MODEL_RESULTS_FILE, RANDOM_STATE, TRAINING_FILE
from src.features import FEATURE_COLS


def train_models(feature_file):
    print("\nLoading features...")
    df = pd.read_csv(feature_file)                     # one row per user: what we report on
    users = df["user"]
    train = pd.read_csv(TRAINING_FILE) if TRAINING_FILE.exists() else df   # snapshots over time
    scaler = StandardScaler().fit(train[FEATURE_COLS])
    X_train = scaler.transform(train[FEATURE_COLS])
    X_scaled = scaler.transform(df[FEATURE_COLS])

    print("Training Isolation Forest...")
    iso = IsolationForest(contamination=CONTAMINATION, random_state=RANDOM_STATE).fit(X_train)

    print("Training One-Class SVM...")
    svm = OneClassSVM(nu=CONTAMINATION, kernel="rbf", gamma="scale").fit(X_train)

    # snapshots of one user are near-duplicates, so look beyond them: neighbours > snapshots per user
    LOF_NEIGHBORS = max(20, 3 * (len(train) // max(len(df), 1)) + 10)
    print("Training Local Outlier Factor (novelty=True so it can score new data)...")
    lof = LocalOutlierFactor(n_neighbors=LOF_NEIGHBORS, contamination=CONTAMINATION, novelty=True).fit(X_train)

    lof_scores = lof.score_samples(X_scaled)             # final per-user state, scored against all snapshots
    lof_pred = lof.predict(X_scaled)

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
