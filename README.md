# Log Anomaly Detection (Windows + Linux authentication logs)

Unsupervised detection of anomalous users from logon logs using Isolation Forest, One-Class SVM and
Local Outlier Factor, with model-agreement risk levels, real-time monitoring and alerting, a REST API,
a Streamlit dashboard and Docker deployment.

## Quick start (local)
```
python -m venv .venv && source .venv/Scripts/activate     # Git Bash  (PowerShell: .venv\Scripts\activate)
pip install -r requirements.txt
python generate_sample_data.py     # synthetic Windows CSV + Linux auth.log + labels (skip with real data)
python main.py                     # full offline pipeline
python monitor.py                  # real-time replay + alerts
uvicorn api:app --port 8000        # REST API, docs at http://localhost:8000/docs
streamlit run dashboard.py         # dashboard at http://localhost:8501
python -m pytest -q                # tests (pip install -r requirements-dev.txt)
```

## Objectives -> where they are met

| # | Objective | Implementation | Evidence |
|---|---|---|---|
| 1 | Collect and preprocess logs (Windows Event Logs, Linux auth logs, public datasets) | `src/sources.py` (Windows CSV / Event Viewer export, CERT `logon.csv`, Linux sshd `auth.log`), `src/preprocess.py` (dedupe, nulls, timestamps, event filter) | `Data/processed/logon_cleaned.csv`, console counts |
| 2 | Behavioural features: event frequency, time intervals, sequential patterns | `src/features.py`: 9 frequency, 4 interval, 2 diversity, 4 sequence features (fail streaks, fail-then-success, PC-switch ratio, event-transition entropy) | `Data/processed/user_features.csv` |
| 3 | Unsupervised anomaly detection (Isolation Forest, One-Class SVM) | `src/train_models.py` (+ LOF as a third model), saved in `Models/` | `Data/processed/model_results.csv` |
| 4 | Compare models: precision, recall, F1, confusion matrix, consistency | `src/evaluate_models.py`, `src/plots.py` | `Results/model_comparison.csv`, `confusion_matrices.png`, `agreement_heatmap.png`, `consistency_summary.csv` |
| 5 | Real-time monitoring and alerting | `src/detector.py` (rolling 30-day per-user scoring), `monitor.py` (replay / Linux tail / Windows live), `api.py` (`POST /events`), `src/alerting.py` (console, CSV, log, webhook, email) | `Results/alerts.csv`, `alerts.log` |

Notes on honest evaluation
* Precision/recall/F1/confusion matrix need ground-truth labels. The synthetic data ships with
  `Data/raw/ground_truth.csv` (`user,is_anomalous`). With an unlabelled real dataset these are skipped and only
  anomaly counts and consistency (Cohen's kappa, seed stability, vote overlap) are reported.
* `contamination=0.05` is a setting (about 5% of users get flagged), not an accuracy.
* A flagged user is an anomaly, not a confirmed incident.

## Using real data
Put files in `Data/raw/`:
* `logon_events.csv` with `timestamp,user,pc,event_id,logon_type` (also accepts Get-WinEvent column names, and the
  CERT insider-threat `logon.csv` format `id,date,user,pc,activity`)
* `auth.log` (Linux sshd lines)
* optional `ground_truth.csv`, and `live_events.csv` for the replay monitor

## API
`POST /events` (JSON list of `{timestamp,user,pc,event_id,logon_type}`) returns triggered alerts;
`GET /users/{user}`, `/alerts`, `/summary`, `/health`. Set `API_KEY` to require an `X-API-Key` header.
```
curl -X POST localhost:8000/events -H "Content-Type: application/json" \
  -d '[{"timestamp":"2026-12-01T02:00:00","user":"bob","pc":"PC1","event_id":4625,"logon_type":3}]'
```

## Alerts
Set environment variables (see `.env.example`): `ALERT_WEBHOOK_URL` (Slack/Discord/Teams/any JSON endpoint),
SMTP settings for email. Alert fires when >= 2 of 3 models flag a user, once per escalation.

## Deployment
**Docker (local or any VPS):**
```
cp .env.example .env     # optional
docker compose up --build
```
API on :8000, dashboard on :8501. `Data/`, `Models/`, `Results/` are mounted from the host, so real data placed in
`Data/raw/` is picked up (delete `Models/` to retrain).

**Render (API):** push to GitHub, create a Blueprint from `render.yaml`. **Streamlit Community Cloud (dashboard):**
point it at `dashboard.py` in your repo; it prepares data on first run.
**CI:** `.github/workflows/ci.yml` runs tests and builds the Docker image on every push.

Live sources: `python monitor.py --source linux --path /var/log/auth.log` follows a Linux log;
`--source windows` reads the Security log (Windows, admin rights, `pip install pywin32`).
