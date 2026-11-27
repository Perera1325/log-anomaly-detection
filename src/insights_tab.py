"""Research insights tab: ablation, sensitivity, error analysis and latency (from Results/insights.json)."""
import json

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from config import RESULTS_DIR
from src import ui

INSIGHTS_FILE = RESULTS_DIR / "insights.json"


def _load():
    if not INSIGHTS_FILE.exists():
        return None
    return json.loads(INSIGHTS_FILE.read_text(encoding="utf-8"))


def render():
    d = _load()
    if d is None:
        st.info("Results/insights.json not found. Run the analysis scripts to generate it.")
        return
    lat = d["lat"]
    ui.key_figures([
        ("Events scored", f"{lat['n']}", "latency benchmark"),
        ("Mean latency", f"{lat['mean']} ms", "per event"),
        ("95th percentile", f"{lat['p95']} ms", "per event"),
        ("Throughput", f"{lat['per_sec']}/s", "single process"),
        ("Correlated pairs", d["n_high_corr"], "|rho| >= 0.90")])

    ui.section("Feature-family ablation",
               "Isolation Forest re-trained with each feature family removed or used alone (5 seeds). "
               "The personal-baseline family carries most of the signal.")
    ab = pd.DataFrame([{"Feature set": n, "ROC AUC": m["auc"], "Avg. precision": m["ap"], "Precision": m["prec"],
                        "Recall": m["rec"], "F1": m["f1"]} for n, _, m in d["ablation"]])
    fig = go.Figure(go.Bar(x=ab["ROC AUC"], y=ab["Feature set"], orientation="h", text=ab["ROC AUC"],
                           marker_color=["#4CC9F0" if "Personal" in s else "#8FA3C8" for s in ab["Feature set"]]))
    fig.update_layout(height=420, xaxis_range=[0.4, 0.85], yaxis_autorange="reversed",
                      margin=dict(l=10, r=10, t=10, b=30), xaxis_title="ROC AUC")
    ui.plot(fig)
    ui.table_caption("ablation", "Metrics for each feature set (labels used for analysis only).")
    st.dataframe(ab, hide_index=True, width="stretch")

    ui.section("Sensitivity to the contamination setting",
               "Flagging fewer accounts raises precision and lowers recall. The right setting follows analyst capacity.")
    names = {"IF": "Isolation Forest", "SVM": "One-Class SVM", "LOF": "LOF"}
    fig = go.Figure()
    for k, rows in d["cont"].items():
        fig.add_scatter(x=[r[0] for r in rows], y=[r[4] for r in rows], mode="lines+markers", name=names[k])
    fig.update_layout(height=340, xaxis_title="Contamination / nu", yaxis_title="F1",
                      margin=dict(l=10, r=10, t=10, b=30))
    ui.plot(fig)
    hp = []
    for k, rows in d["hp"].items():
        for v, auc in rows:
            hp.append({"Parameter": k, "Value": str(v), "ROC AUC": auc})
    ui.table_caption("hyper", "ROC AUC when one hyperparameter is varied. LOF improves with a larger "
                     "neighbourhood, but the value was chosen with labels, so it is optimistic.")
    st.dataframe(pd.DataFrame(hp), hide_index=True, width="stretch", height=35 * 16 + 3)

    ui.section("Error analysis",
               "Missed insiders look like ordinary users; false positives look like administrators.")
    err = pd.DataFrame(d["err"]).T.reset_index().rename(columns={"index": "Group", "n": "Users"})
    st.dataframe(err, hide_index=True, width="stretch")
    sc = pd.DataFrame([{"Scenario": k, "Insiders": v["n"], "Detected at threshold": v["tp"],
                        "AUC vs all normal users": d["scen_auc"][k]} for k, v in d["err_scen"].items()])
    ui.table_caption("scen", "Detection by insider scenario. Scenario 3 reflects an administrator-like role.")
    st.dataframe(sc, hide_index=True, width="stretch")
    st.caption(d.get("note", ""))

    ui.section("About this project")
    st.markdown("**Author:** G.M. Methum Dulneth (HCL/IT/CYBR/2095), BSc (Hons) Cyber Security, "
                "Birmingham City University. **Supervisor:** Anjana Piyasena.")
