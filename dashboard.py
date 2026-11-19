"""Streamlit dashboard:  streamlit run dashboard.py"""
import os

import pandas as pd
import streamlit as st

from bootstrap import ensure_ready
from config import (ALERT_FILE, CLEAN_FILE, COMPARISON_FILE, FEATURE_FILE, MODEL_RESULTS_FILE,
                    RESULTS_DIR, RISK_FILE, TRUTH_FILE)
from src import attack_tab, charts
from src.features import FEATURE_COLS

st.set_page_config(page_title="Log Anomaly Detection", page_icon="🛡️", layout="wide")

st.markdown("""
<style>
.block-container {padding-top: 1.6rem; max-width: 1400px;}
#MainMenu, footer {visibility: hidden;}
h1 {font-weight: 700; letter-spacing: -0.5px;}
.hero {background: linear-gradient(120deg,#0F172A 0%,#1E3A8A 100%); color: white; padding: 22px 28px;
       border-radius: 14px; margin-bottom: 18px;}
.hero h1 {color: white; margin: 0; font-size: 1.9rem;}
.hero p {color: #CBD5E1; margin: 4px 0 0 0; font-size: 0.95rem;}
.kpi {background: white; border: 1px solid #E5E7EB; border-radius: 12px; padding: 14px 18px;
      box-shadow: 0 1px 2px rgba(15,23,42,.05); border-left: 5px solid var(--c,#2563EB);}
.kpi .l {color: #6B7280; font-size: .78rem; text-transform: uppercase; letter-spacing: .6px;}
.kpi .v {color: #111827; font-size: 1.85rem; font-weight: 700; line-height: 1.25;}
.kpi .s {color: #6B7280; font-size: .78rem;}
.badge {display:inline-block; padding: 2px 10px; border-radius: 999px; color: white; font-size: .78rem;}
</style>
""", unsafe_allow_html=True)

with st.spinner("Preparing data and models (first run only)..."):
    ensure_ready()


def kpi(col, label, value, sub="", color="#2563EB"):
    col.markdown(f"<div class='kpi' style='--c:{color}'><div class='l'>{label}</div>"
                 f"<div class='v'>{value}</div><div class='s'>{sub}&nbsp;</div></div>", unsafe_allow_html=True)


def plot(fig):
    st.plotly_chart(fig, width="stretch", theme=None, config={"displaylogo": False, "toImageButtonOptions": {"scale": 3}})


@st.cache_data(show_spinner=False)
def load_csv(path, mtime, **kw):
    return pd.read_csv(path, **kw)


def mtime(p):
    return os.path.getmtime(p) if os.path.exists(p) else 0


risk = load_csv(RISK_FILE, mtime(RISK_FILE))
feats = load_csv(FEATURE_FILE, mtime(FEATURE_FILE))
results = load_csv(MODEL_RESULTS_FILE, mtime(MODEL_RESULTS_FILE))
clean = load_csv(CLEAN_FILE, mtime(CLEAN_FILE), usecols=["timestamp", "user", "pc", "event_id", "logon_type"],
                 parse_dates=["timestamp"])
summary = load_csv(COMPARISON_FILE, mtime(COMPARISON_FILE))
has_truth = TRUTH_FILE.exists() and "Precision" in summary
z_all = charts.zscores(feats, FEATURE_COLS)

# ----------------------------------------------------------------- sidebar
with st.sidebar:
    st.markdown("### 🛡️ Log Anomaly Detection")
    st.caption("Unsupervised detection of suspicious logon behaviour in Windows and Linux logs.")
    st.divider()
    st.markdown("**Data**")
    st.write(f"{len(clean):,} cleaned log records")
    st.write(f"{clean['timestamp'].min():%d %b %Y} – {clean['timestamp'].max():%d %b %Y}")
    st.divider()
    st.markdown("**Objectives**")
    for o in ["Collect & preprocess logs", "Behavioural feature extraction", "Unsupervised detection",
              "Model comparison", "Real-time monitoring & alerts"]:
        st.markdown(f"✅ {o}")
    st.divider()
    st.caption("A flag marks unusual behaviour, not a confirmed incident.")

st.markdown("<div class='hero'><h1>Log Anomaly Detection</h1>"
            "<p>Isolation Forest · One-Class SVM · Local Outlier Factor with model-agreement risk scoring and "
            "real-time alerting</p></div>", unsafe_allow_html=True)

tab_over, tab_eval, tab_users, tab_alerts, tab_attack = st.tabs(
    ["📊 Overview", "🧪 Model evaluation", "🔎 Users & investigation", "🚨 Live alerts", "🧨 Attack analyzer"])

# ---------------------------------------------------------------- overview
with tab_over:
    n_alerts = len(load_csv(ALERT_FILE, mtime(ALERT_FILE))) if ALERT_FILE.exists() else 0
    c = st.columns(5)
    kpi(c[0], "Log records", f"{len(clean):,}", "after cleaning", "#2563EB")
    kpi(c[1], "Users analysed", f"{len(risk):,}", f"{len(FEATURE_COLS)} behaviour features", "#0D9488")
    kpi(c[2], "Flagged users", int((risk['anomaly_votes'] >= 1).sum()), "≥ 1 model", "#F2B01E")
    kpi(c[3], "High / Critical", int((risk['anomaly_votes'] >= 2).sum()), "≥ 2 models agree", "#E4572E")
    kpi(c[4], "Alerts raised", n_alerts, "real-time monitor", "#9B1C31")
    st.write("")
    a, b = st.columns([2, 3])
    with a:
        plot(charts.risk_donut(risk))
    with b:
        plot(charts.user_map(feats, risk, FEATURE_COLS))
    a, b = st.columns([3, 2])
    with a:
        plot(charts.activity_heatmap(clean))
    with b:
        plot(charts.hourly_profile(clean))

# -------------------------------------------------------------- evaluation
with tab_eval:
    cons_path = RESULTS_DIR / "consistency_summary.csv"
    cons = load_csv(cons_path, mtime(cons_path), index_col=0)["value"] if cons_path.exists() else pd.Series(dtype=float)
    if not has_truth:
        st.info("No ground-truth labels found, so precision / recall / F1 are not shown. Anomaly counts and "
                "consistency are below.")
    else:
        st.caption("Metrics use the labels in `Data/raw/ground_truth.csv` (synthetic anomalies unless you supplied "
                   "your own).")
        c = st.columns(4)
        best = summary.loc[summary["F1"].idxmax()]
        kpi(c[0], "Best F1", f"{best['F1']:.2f}", best["Model"], "#7C3AED")
        kpi(c[1], "Best precision", f"{summary['Precision'].max():.2f}",
            summary.loc[summary['Precision'].idxmax(), "Model"], "#2563EB")
        kpi(c[2], "Best recall", f"{summary['Recall'].max():.2f}",
            summary.loc[summary['Recall'].idxmax(), "Model"], "#0D9488")
        kpi(c[3], "Mean model kappa", f"{cons['mean_pairwise_kappa']:.2f}", "agreement between detectors",
            "#F2B01E")
        st.write("")
        a, b = st.columns([2, 3])
        with a:
            plot(charts.metrics_bars(summary))
        with b:
            plot(charts.confusion_grid(summary))
        truth = results[["user"]].merge(pd.read_csv(TRUTH_FILE)[["user", "is_anomalous"]], on="user",
                                        how="left")["is_anomalous"].fillna(0).astype(int)
        plot(charts.roc_pr(results, truth))

    a, b = st.columns(2)
    kappa = load_csv(RESULTS_DIR / "model_consistency.csv", mtime(RESULTS_DIR / "model_consistency.csv"),
                     index_col=0)
    with a:
        plot(charts.kappa_heatmap(kappa))
    with b:
        plot(charts.vote_combinations(results))
    if len(cons):
        stab = cons.get("isolation_forest_seed_stability_jaccard")
        if stab is not None:
            st.caption(f"Isolation Forest seed stability (Jaccard overlap of flagged users across 5 random seeds): "
                       f"**{float(stab):.2f}**")
    with st.expander("Full comparison table"):
        st.dataframe(summary, hide_index=True, width="stretch")

# ------------------------------------------------------------------- users
with tab_users:
    f1, f2 = st.columns([2, 1])
    levels = f1.multiselect("Risk level", ["Medium", "High", "Critical", "Low"], default=["High", "Critical"])
    query = f2.text_input("Search user")
    table = risk[risk["risk_level"].isin(levels)]
    if query:
        table = table[table["user"].str.contains(query, case=False)]
    table = table.sort_values(["anomaly_votes", "user"], ascending=[False, True])
    show = table[["user", "risk_level", "anomaly_votes", "IsolationForest", "OneClassSVM", "LOF"]].copy()
    for m in ["IsolationForest", "OneClassSVM", "LOF"]:
        show[m] = show[m].map({-1: "⚠ flagged", 1: "–"})
    st.dataframe(show, hide_index=True, width="stretch", height=260, column_config={
        "user": "User", "risk_level": "Risk", "anomaly_votes": st.column_config.ProgressColumn(
            "Votes", min_value=0, max_value=3, format="%d / 3")})

    flagged_users = table["user"].head(25).tolist()
    if flagged_users:
        plot(charts.feature_heatmap(z_all.loc[flagged_users]))
        st.markdown("#### Investigate a user")
        u = st.selectbox("User", table["user"].tolist())
        row = risk.set_index("user").loc[u]
        ev = clean[clean["user"] == u]
        c = st.columns(4)
        kpi(c[0], "Risk level", row["risk_level"], f"{int(row['anomaly_votes'])} of 3 models",
            charts.RISK_COLORS[row["risk_level"]])
        kpi(c[1], "Events", f"{len(ev):,}", f"{(ev['event_id'] == 4625).mean():.0%} failed", "#2563EB")
        kpi(c[2], "Machines / IPs", ev["pc"].nunique(), "distinct sources", "#0D9488")
        off = ((ev["timestamp"].dt.hour < 7) | (ev["timestamp"].dt.hour >= 19)).mean()
        kpi(c[3], "Off-hours share", f"{off:.0%}", "before 07:00 / after 19:00", "#F2B01E")
        st.write("")
        a, b = st.columns(2)
        with a:
            plot(charts.zscore_bar(z_all.loc[u]))
        with b:
            plot(charts.user_timeline(ev))
    else:
        st.info("No users match the filter.")

# ------------------------------------------------------------------ alerts
with tab_alerts:
    auto = st.toggle("Auto-refresh every 5 s", value=False)

    @st.fragment(run_every=5 if auto else None)
    def live_alerts():
        if not ALERT_FILE.exists():
            st.info("No alerts yet. Run `python monitor.py` or send events to the API (`POST /events`).")
            return
        alerts = pd.read_csv(ALERT_FILE)
        c = st.columns(3)
        kpi(c[0], "Alerts", len(alerts), "since monitor start", "#9B1C31")
        kpi(c[1], "Critical", int((alerts["risk_level"] == "Critical").sum()), "", "#9B1C31")
        kpi(c[2], "Users alerted", alerts["user"].nunique(), "", "#E4572E")
        st.write("")
        plot(charts.alerts_timeline(alerts))
        st.dataframe(alerts.iloc[::-1], hide_index=True, width="stretch", height=300)

    live_alerts()

# --------------------------------------------------------- attack analyzer
with tab_attack:
    attack_tab.render(kpi, plot)
