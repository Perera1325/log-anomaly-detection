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

    # --- Population models: each user's final behaviour vs. all other users (batch report, evaluation) ---
    # Fitted on one row per user. Training on the time snapshots here would hide real outliers, because an
    # insider's own earlier snapshots sit right next to the final row and look like a dense, normal cluster.
    scaler_pop = StandardScaler().fit(df[FEATURE_COLS])
    X_pop = scaler_pop.transform(df[FEATURE_COLS])
    print("Training Isolation Forest...")
    iso_pop = IsolationForest(contamination=CONTAMINATION, random_state=RANDOM_STATE).fit(X_pop)
    print("Training One-Class SVM...")
    svm_pop = OneClassSVM(nu=CONTAMINATION, kernel="rbf", gamma="scale").fit(X_pop)
    print("Training Local Outlier Factor...")
    lof_pop = LocalOutlierFactor(n_neighbors=20, contamination=CONTAMINATION, novelty=False)
    lof_pred = lof_pop.fit_predict(X_pop)

    results = pd.DataFrame({
        "user": users,
        "IsolationForest": iso_pop.predict(X_pop),
        "OneClassSVM": svm_pop.predict(X_pop),
        "LOF": lof_pred,
        "IF_score": iso_pop.decision_function(X_pop),
        "SVM_score": svm_pop.decision_function(X_pop),
        "LOF_score": lof_pop.negative_outlier_factor_ - lof_pop.offset_,
    })

    # --- Streaming models: used by monitor.py / the Detector on live events ---
    # Trained on snapshots over time so they know how features normally drift as new events arrive.
    scaler = StandardScaler().fit(train[FEATURE_COLS])
    X_train = scaler.transform(train[FEATURE_COLS])
    iso = IsolationForest(contamination=CONTAMINATION, random_state=RANDOM_STATE).fit(X_train)
    svm = OneClassSVM(nu=CONTAMINATION, kernel="rbf", gamma="scale").fit(X_train)
    # snapshots of one user are near-duplicates, so look beyond them: neighbours > snapshots per user
    lof_neighbors = max(20, 3 * (len(train) // max(len(df), 1)) + 10)
    lof = LocalOutlierFactor(n_neighbors=lof_neighbors, contamination=CONTAMINATION, novelty=True).fit(X_train)
    print("Streaming models trained on user snapshots (novelty=True so LOF can score new data).")
    MODEL_RESULTS_FILE.parent.mkdir(parents=True, exist_ok=True)
    results.to_csv(MODEL_RESULTS_FILE, index=False)

    MODELS_DIR.mkdir(exist_ok=True)
    joblib.dump(iso, MODELS_DIR / "isolation_forest.pkl")
    joblib.dump(svm, MODELS_DIR / "oneclass_svm.pkl")
    joblib.dump(lof, MODELS_DIR / "lof.pkl")
    joblib.dump(scaler, MODELS_DIR / "scaler.pkl")
    print("All models trained and saved.")
    return results
