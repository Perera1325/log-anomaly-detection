import os

import pandas as pd
import streamlit as st
from sklearn.metrics import average_precision_score, roc_auc_score

from bootstrap import ensure_ready
from config import (ALERT_FILE, ALERT_MIN_VOTES, CLEAN_FILE, COMPARISON_FILE, CONTAMINATION, FEATURE_FILE,
                    MIN_EVENTS_FOR_SCORING, MODEL_RESULTS_FILE, MODELS_DIR, OFF_HOURS, RESULTS_DIR, RISK_FILE, TRUTH_FILE,
                    WINDOW_DAYS)
from src import attack_tab, charts, ui
from src.features import FEATURE_COLS

st.set_page_config(page_title="Log Anomaly Detection", layout="wide")
ui.inject_css()
ui.reset_numbers()
animate = st.session_state.get("animate", True)
if not animate:
    st.markdown("<style>*, *::before, *::after {animation: none !important;}</style>", unsafe_allow_html=True)

with st.spinner("Preparing data and models (first run only)..."):
    ensure_ready()


plot = ui.plot


@st.cache_data(show_spinner=False)
def load_csv(path, mtime, **kw):
    return pd.read_csv(path, **kw)


def mtime(p):
    return os.path.getmtime(p) if os.path.exists(p) else 0


MODELS = {"IsolationForest": "IF_score", "OneClassSVM": "SVM_score", "LOF": "LOF_score"}

risk = load_csv(RISK_FILE, mtime(RISK_FILE))
feats = load_csv(FEATURE_FILE, mtime(FEATURE_FILE))
results = load_csv(MODEL_RESULTS_FILE, mtime(MODEL_RESULTS_FILE))
clean = load_csv(CLEAN_FILE, mtime(CLEAN_FILE), usecols=["timestamp", "user", "pc", "event_id", "logon_type"],
                 parse_dates=["timestamp"])
summary = load_csv(COMPARISON_FILE, mtime(COMPARISON_FILE))
has_truth = TRUTH_FILE.exists() and "Precision" in summary
z_all = charts.zscores(feats, FEATURE_COLS)
n2, n3 = int((risk["anomaly_votes"] >= 2).sum()), int((risk["anomaly_votes"] >= 3).sum())
d0, d1 = clean["timestamp"].min(), clean["timestamp"].max()

truth = None
if has_truth:
    truth = results[["user"]].merge(pd.read_csv(TRUTH_FILE)[["user", "is_anomalous"]], on="user",
                                    how="left")["is_anomalous"].fillna(0).astype(int)

# ----------------------------------------------------------------- sidebar
with st.sidebar:
    st.markdown("### Log Anomaly Detection")
    st.caption("Unsupervised detection of anomalous logon behaviour in Windows and Linux authentication logs.")
    st.toggle("Animations", value=True, key="animate")
    st.divider()
    ui.status_block(MODELS_DIR, ALERT_FILE)
    st.divider()
    st.markdown(f"<div class='meta'><b>Records</b><br>{len(clean):,} cleaned logon events<br><br>"
                f"<b>Period</b><br>{d0:%d %b %Y} to {d1:%d %b %Y}<br><br>"
                f"<b>Accounts</b><br>{len(risk):,} analysed<br><br>"
                f"<b>Detectors</b><br>Isolation Forest, One-Class SVM, Local Outlier Factor<br><br>"
                f"<b>Decision rule</b><br>Risk level = number of detectors that flag the account</div>",
                unsafe_allow_html=True)
    st.divider()
    st.caption("A flag marks unusual behaviour. It is not evidence of an incident.")

# ------------------------------------------------------------------ header
summary_sentence = ""
if has_truth:
    best = summary.loc[summary["F1"].idxmax()]
    summary_sentence = (f" Against the reference labels, the best single detector ({best['Model']}) reaches "
                        f"F1 = {best['F1']:.2f} (precision {best['Precision']:.2f}, recall {best['Recall']:.2f}).")
hours = clean["timestamp"].dt.hour
fail = (clean["event_id"] == 4625)
bad_h = [int(((hours == h) & fail).sum()) for h in range(24)]
ok_h = [int(((hours == h) & ~fail).sum()) for h in range(24)]
ui.hero("Unsupervised Detection of Anomalous Logon Behaviour",
        "A three-detector ensemble with model-agreement risk scoring, real-time alerting and per-account explanations",
        ok_h, bad_h)
st.markdown(
    f"<p class='abstract'><span class='h'>Summary.</span>This page reports an unsupervised analysis of "
    f"{len(clean):,} cleaned logon events (Windows Event IDs 4624 and 4625) from {len(risk):,} accounts. Each "
    f"account is described by {len(FEATURE_COLS)} behavioural features and scored independently by an Isolation "
    f"Forest, a One-Class SVM and a Local Outlier Factor model. {n2} accounts are flagged by at least two "
    f"detectors; {'none is' if n3 == 0 else f'{n3} are'} flagged by all three.{summary_sentence}</p>",
    unsafe_allow_html=True)

tab_over, tab_eval, tab_users, tab_alerts, tab_attack, tab_method = st.tabs(
    ["Overview", "Evaluation", "Account analysis", "Alerts", "Attack scenarios", "Methodology"])

# ---------------------------------------------------------------- overview
with tab_over:
    n_alerts = len(load_csv(ALERT_FILE, mtime(ALERT_FILE))) if ALERT_FILE.exists() else 0
    ui.key_figures([
        ("Log records", f"{len(clean):,}", "after cleaning"),
        ("Accounts", f"{len(risk):,}", f"{len(FEATURE_COLS)} features each"),
        ("Flagged, 1 or more", int((risk['anomaly_votes'] >= 1).sum()), "at least one detector"),
        ("Flagged, 2 or more", n2, "alert threshold"),
        ("Alerts raised", n_alerts, "real-time monitor")])

    ui.section("1.1  Risk distribution and behavioural structure")
    a, b = st.columns([2, 3])
    with a:
        plot(charts.risk_donut(risk))
        ui.figure_caption("risk", "Accounts by risk level. Low, Medium, High and Critical correspond to 0, 1, 2 and 3 "
                                  "detectors flagging the account.")
    with b:
        plot(charts.user_map(feats, risk, FEATURE_COLS))
        ui.figure_caption("map", "Two-dimensional principal component projection of the standardised behavioural "
                                 "features. Each marker is an account; marker shape and colour give the risk level. "
                                 "Distance reflects dissimilarity of behaviour.")

    ui.section("1.2  Temporal structure of logon activity")
    a, b = st.columns([3, 2])
    with a:
        plot(charts.activity_heatmap(clean))
        ui.figure_caption("heat", "Number of logon events by weekday and hour of day, all accounts pooled.")
    with b:
        plot(charts.hourly_profile(clean))
        ui.figure_caption("hourly", "Successful and failed logons by hour of day. Off-hours are defined as "
                                    f"before {OFF_HOURS[0]:02d}:00 or from {OFF_HOURS[1]:02d}:00.")

# -------------------------------------------------------------- evaluation
with tab_eval:
    cons_path = RESULTS_DIR / "consistency_summary.csv"
    cons = load_csv(cons_path, mtime(cons_path), index_col=0)["value"] if cons_path.exists() else pd.Series(dtype=float)
    if not has_truth:
        ui.callout("No reference labels were found, so precision, recall and F1 are not reported. Detector "
                   "agreement is shown below.", title="Note.")
    else:
        best = summary.loc[summary["F1"].idxmax()]
        ui.key_figures([
            ("Best F1", f"{best['F1']:.3f}", best["Model"]),
            ("Best precision", f"{summary['Precision'].max():.3f}", summary.loc[summary['Precision'].idxmax(), "Model"]),
            ("Best recall", f"{summary['Recall'].max():.3f}", summary.loc[summary['Recall'].idxmax(), "Model"]),
            ("Mean pairwise kappa", f"{cons['mean_pairwise_kappa']:.3f}" if len(cons) else "n/a",
             "agreement between detectors")])
        ui.section("2.1  Detection performance",
                   "Reference labels (Data/raw/ground_truth.csv) are used only to score the detectors, never to "
                   "train them. Threshold-based metrics use the "
                   f"{CONTAMINATION:.0%} flagging rate; AUC and average precision are threshold-free.")
        table = summary[["Model", "Precision", "Recall", "F1", "TP", "FP", "FN", "TN"]].copy()
        table["ROC AUC"] = [f"{roc_auc_score(truth, -results[MODELS[m]]):.3f}" if m in MODELS else "n/a"
                            for m in table["Model"]]
        table["Avg. precision"] = [f"{average_precision_score(truth, -results[MODELS[m]]):.3f}" if m in MODELS
                                   else "n/a" for m in table["Model"]]
        ui.table_caption("perf", "Detection performance of each detector and of the two-vote ensemble. "
                                 "TP, FP, FN and TN are counts of accounts.")
        st.dataframe(table, hide_index=True, width="stretch", column_config={
            c: st.column_config.NumberColumn(format="%.3f") for c in ["Precision", "Recall", "F1"]})
        a, b = st.columns([2, 3])
        with a:
            plot(charts.metrics_bars(summary))
            ui.figure_caption("bars", "Precision, recall and F1 at the flagging threshold.")
        with b:
            plot(charts.confusion_grid(summary))
            ui.figure_caption("cm", "Confusion matrices (counts of accounts); shading shows the share of each "
                                    "actual class.")
        ui.section("2.2  Ranking quality")
        plot(charts.roc_pr(results, truth))
        ui.figure_caption("rocpr", "Ranking quality of the raw anomaly scores, independent of the flagging "
                                   "threshold. Dash-dot lines mark chance performance.")

    ui.section("2.3  Agreement between detectors")
    a, b = st.columns(2)
    kappa = load_csv(RESULTS_DIR / "model_consistency.csv", mtime(RESULTS_DIR / "model_consistency.csv"),
                     index_col=0)
    with a:
        plot(charts.kappa_heatmap(kappa))
        ui.figure_caption("kappa", "Cohen's kappa between detector decisions (1 = identical, 0 = chance level).")
    with b:
        plot(charts.vote_combinations(results))
        ui.figure_caption("votes", "Accounts flagged by each exact combination of detectors "
                                   "(IF, Isolation Forest; SVM, One-Class SVM).")
    if len(cons):
        stab = cons.get("isolation_forest_seed_stability_jaccard")
        if stab is not None:
            st.markdown(f"<p class='meta'>Isolation Forest seed stability (Jaccard overlap of flagged accounts "
                        f"across five random seeds): <b>{float(stab):.3f}</b>.</p>", unsafe_allow_html=True)

# ------------------------------------------------------------------- users
with tab_users:
    ui.section("3.1  Accounts under review")
    f1, f2 = st.columns([2, 1])
    levels = f1.multiselect("Risk level", ["Medium", "High", "Critical", "Low"], default=["High", "Critical"])
    query = f2.text_input("Search account")
    table = risk[risk["risk_level"].isin(levels)]
    if query:
        table = table[table["user"].str.contains(query, case=False)]
    table = table.sort_values(["anomaly_votes", "user"], ascending=[False, True])
    show = table[["user", "risk_level", "anomaly_votes", "IsolationForest", "OneClassSVM", "LOF"]].copy()
    for m in ["IsolationForest", "OneClassSVM", "LOF"]:
        show[m] = show[m].map({-1: "Flagged", 1: "-"})
    ui.table_caption("accounts", "Accounts matching the selected risk levels and the detectors that flagged them.")
    st.dataframe(show, hide_index=True, width="stretch", height=260, column_config={
        "user": "Account", "risk_level": "Risk", "anomaly_votes": st.column_config.ProgressColumn(
            "Votes", min_value=0, max_value=3, format="%d / 3")})

    flagged_users = table["user"].head(25).tolist()
    if flagged_users:
        plot(charts.feature_heatmap(z_all.loc[flagged_users]))
        ui.figure_caption("fheat", "Feature z-scores of the selected accounts relative to all accounts "
                                   "(red = above the population mean, blue = below; clipped at 4).")
        ui.section("3.2  Account profile")
        u = st.selectbox("Account", table["user"].tolist())
        row = risk.set_index("user").loc[u]
        ev = clean[clean["user"] == u]
        off = ((ev["timestamp"].dt.hour < OFF_HOURS[0]) | (ev["timestamp"].dt.hour >= OFF_HOURS[1])).mean()
        ui.key_figures([
            ("Risk level", row["risk_level"], f"{int(row['anomaly_votes'])} of 3 detectors"),
            ("Events", f"{len(ev):,}", f"{(ev['event_id'] == 4625).mean():.0%} failed"),
            ("Distinct sources", ev["pc"].nunique(), "machines or addresses"),
            ("Off-hours share", f"{off:.0%}", f"outside {OFF_HOURS[0]:02d}:00 to {OFF_HOURS[1]:02d}:00")])
        a, b = st.columns(2)
        with a:
            plot(charts.zscore_bar(z_all.loc[u]))
            ui.figure_caption("zbar", f"Ten largest behavioural deviations for account {ui.esc(u)}.")
        with b:
            plot(charts.user_timeline(ev))
            ui.figure_caption("timeline", f"Logon events of account {ui.esc(u)} by time of day. Shaded bands mark "
                                          "off-hours.")
    else:
        ui.callout("No accounts match the selected filter.")

# ------------------------------------------------------------------ alerts
with tab_alerts:
    ui.section("4.1  Alert stream", f"An alert is raised when at least {ALERT_MIN_VOTES} of 3 detectors flag an "
                                     "account, and is escalated when more detectors agree.")
    auto = st.toggle("Refresh every 5 seconds", value=False)

    @st.fragment(run_every=5 if auto else None)
    def live_alerts():
        if not ALERT_FILE.exists():
            ui.callout("No alerts yet. Run <code>python monitor.py</code> or send events to the API "
                       "(<code>POST /events</code>).")
            return
        alerts = pd.read_csv(ALERT_FILE)
        ui.key_figures([("Alerts", len(alerts), "since monitor start"),
                        ("Critical", int((alerts["risk_level"] == "Critical").sum()), "all detectors agree"),
                        ("Accounts alerted", alerts["user"].nunique(), "distinct")])
        plot(charts.alerts_timeline(alerts))
        ui.figure_caption("alerts", "Time at which each account crossed the alert threshold.")
        ui.table_caption("alerttable", "Alert log, most recent first.")
        st.dataframe(alerts.iloc[::-1], hide_index=True, width="stretch", height=300)

    live_alerts()

# --------------------------------------------------------- attack analyzer
with tab_attack:
    attack_tab.render(plot)

# ------------------------------------------------------------- methodology
with tab_method:
    ui.section("6.1  Data")
    st.markdown(
        "Only interactive logon events are used: Windows Event ID 4624 (successful logon) and 4625 (failed "
        "logon). Logs are accepted as CSV, as a Windows Get-WinEvent export, in the CERT r4.2 `logon.csv` format "
        "and as Linux sshd `auth.log`. Records are cleaned and validated before feature extraction. Device, file "
        "and e-mail activity are not used.")
    ui.section("6.2  Features")
    st.markdown(
        f"Each account is summarised by {len(FEATURE_COLS)} behavioural features computed over a rolling window of "
        f"{WINDOW_DAYS} days, including failure ratio and longest failure streak, failure followed by success, "
        "share of off-hours and weekend activity, number of distinct machines and the rate of machine switching, "
        "burstiness of event timing, and deviation from the account's own working-hours baseline. Features are "
        "standardised (z-scores) before modelling. Accounts with fewer than "
        f"{MIN_EVENTS_FOR_SCORING} events are not scored.")
    ui.section("6.3  Detectors and decision rule")
    ui.table_caption("detectors", "Detectors and their principal settings.")
    st.dataframe(pd.DataFrame({
        "Detector": ["Isolation Forest", "One-Class SVM", "Local Outlier Factor"],
        "Principle": ["Random partitioning isolates anomalies in few splits",
                      "Learns a boundary around normal behaviour",
                      "Compares local density with that of neighbours"],
        "Setting": [f"contamination = {CONTAMINATION}", f"RBF kernel, nu = {CONTAMINATION}",
                    f"contamination = {CONTAMINATION}"]}), hide_index=True, width="stretch")
    ui.table_caption("risk", "Risk level as a function of detector agreement.")
    st.dataframe(pd.DataFrame({
        "Detectors flagging": [0, 1, 2, 3], "Risk level": ["Low", "Medium", "High", "Critical"],
        "Alert": ["No", "No", "Yes", "Yes, escalated"]}), hide_index=True, width="stretch")
    ui.section("6.4  Evaluation protocol")
    st.markdown(
        "Reference labels are used only for scoring. Reported measures are precision, recall and F1 at the "
        "flagging threshold, ROC AUC and average precision from the raw scores, Cohen's kappa between detectors, "
        "and the Jaccard overlap of flagged accounts across random seeds. A stratified 50/50 held-out split is "
        "provided in `holdout_eval.py`, and `stream_model_eval.py` evaluates the models used by the real-time "
        "monitor.")
    ui.section("6.5  Limitations")
    for t in ["Recall is low at the alert threshold: many true insiders are not flagged. Alerts are leads for an "
              "analyst, not verdicts.",
              "The One-Class SVM and LOF detectors rank accounts close to chance on the benchmark; the Isolation "
              "Forest carries most of the signal.",
              "Models are trained on the CERT r4.2 benchmark. Logs from another environment can appear unusual "
              "only because they differ from it, so retraining on local data is required before operational use.",
              "Feature design was informed by the labelled scenarios, so scores on the same data are optimistic; "
              "the held-out split reduces but does not remove this.",
              "Account tenure partly separates insiders from normal users in CERT, which is an artefact of the "
              "synthetic data generator.",
              "Only logon events are used; file, device and e-mail behaviour are future work."]:
        st.markdown(f"- {t}")
    ui.section("6.6  Reference")
    st.markdown("<p class='refs'>Glasser, J., &amp; Lindauer, B. (2013). Bridging the gap: A pragmatic approach to "
                "generating insider threat data. <i>2013 IEEE Security and Privacy Workshops</i>, 98-104. "
                "<a href='https://doi.org/10.1109/SPW.2013.37'>https://doi.org/10.1109/SPW.2013.37</a></p>",
                unsafe_allow_html=True)