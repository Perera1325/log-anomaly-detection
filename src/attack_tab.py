"""Dashboard tab: upload log files (or use the bundled sample attacks) and see an analysis for each one."""
import hashlib
from pathlib import Path

import pandas as pd
import streamlit as st

from src import charts
from src.analyzer import analyze, read_log

SAMPLES = Path(__file__).resolve().parent.parent / "sample_attacks"
ICON = {"Critical": "🔴", "High": "🟠", "Medium": "🟡", "Low": "🟢"}
WHAT = {
    "0_normal_control.csv": "Normal control: ordinary office users, no attack (should stay quiet)",
    "1_brute_force.csv": "Brute force: 60 wrong passwords in minutes, then a successful logon",
    "2_off_hours_insider.csv": "Insider working at night: sudden 01:00-04:00 logons for 8 days",
    "3_lateral_movement.csv": "Lateral movement: one account hops across 25 machines",
    "4_password_spraying.csv": "Password spraying: one machine tries a few passwords on 14 users",
    "5_account_takeover.csv": "Account takeover: weekend 03:00 burst from new machines",
}


@st.cache_data(show_spinner=False)
def _run(name: str, data: bytes, digest: str):
    return analyze(read_log(name, data))


def _banner(r):
    w = r["worst"]
    what = "; ".join(r["patterns"]) if r["patterns"] else "no specific attack pattern matched"
    msg = (f"**{ICON[r['verdict']]} {r['verdict']} risk.** Most suspicious account: **{w['user']}** "
           f"({w['votes']} of 3 models agree). Indicators: {what}.")
    if r["verdict"] == "Critical":
        st.error(msg)
    elif r["verdict"] == "High":
        st.warning(msg)
    elif r["verdict"] == "Medium":
        st.info(msg + " One model only, so this is a weak signal and raises no alert.")
    else:
        st.success("🟢 Low risk. No account looks unusual.")


def render(kpi, plot):
    st.markdown("#### Analyse log files")
    st.caption("Upload up to several logon files (CSV with timestamp, user, pc, event_id, logon_type; a Windows "
               "Get-WinEvent export; a CERT logon.csv; or a Linux sshd auth.log). Each file is scored by the three "
               "models and checked for explainable attack indicators.")
    uploads = st.file_uploader("Log files", type=["csv", "log", "txt"], accept_multiple_files=True,
                               key="attack_uploads")
    samples = sorted(SAMPLES.glob("*.csv")) if SAMPLES.exists() else []
    c1, c2 = st.columns([1, 3])
    if samples and c1.button("▶ Use the sample attack files", key="use_samples"):
        st.session_state["samples_on"] = True
    if st.session_state.get("samples_on") and c2.button("Clear samples", key="clear_samples"):
        st.session_state["samples_on"] = False
        st.rerun()
    if samples:
        with st.expander("Sample files (download to try the uploader)"):
            for p in samples:
                st.download_button(f"⬇ {p.name}", p.read_bytes(), file_name=p.name, mime="text/csv",
                                   key=f"dl_{p.name}", help=WHAT.get(p.name, ""))
                st.caption(WHAT.get(p.name, ""))

    files = [(u.name, u.getvalue()) for u in uploads]
    if st.session_state.get("samples_on"):
        files += [(p.name, p.read_bytes()) for p in samples]
    if not files:
        st.info("Upload files or press **Use the sample attack files** to see the analysis.")
        return

    results, errors = {}, {}
    for name, data in files:
        try:
            results[name] = _run(name, data, hashlib.md5(data).hexdigest())
        except Exception as e:                                  # unreadable file: report it, keep the rest
            errors[name] = str(e)
    for name, msg in errors.items():
        st.error(f"**{name}**: {msg}")
    if not results:
        return

    st.markdown("##### Summary of all files")
    summary = pd.DataFrame([{
        "File": n, "What it contains": WHAT.get(n, ""), "Events": r["n_events"], "Users": r["n_users"],
        "Flagged users (≥2 models)": r["n_flagged"], "Risk": f"{ICON[r['verdict']]} {r['verdict']}",
        "Attack patterns found": ", ".join(r["patterns"]) or "-"} for n, r in results.items()])
    st.dataframe(summary, hide_index=True, width="stretch")

    for name, r in results.items():
        with st.expander(f"{ICON[r['verdict']]} {name}", expanded=len(results) <= 2):
            if WHAT.get(name):
                st.caption(WHAT[name])
            _banner(r)
            c = st.columns(4)
            kpi(c[0], "Events", f"{r['n_events']:,}", "Event IDs 4624 / 4625", "#2563EB")
            kpi(c[1], "Users", r["n_users"], "accounts in file", "#0D9488")
            kpi(c[2], "Flagged users", r["n_flagged"], "≥ 2 models agree", "#E4572E")
            kpi(c[3], "Highest risk", r["verdict"], r["worst"]["user"], charts.RISK_COLORS[r["verdict"]])
            st.write("")
            for pc, k in r["spray"]:
                st.warning(f"**Password spraying:** machine **{pc}** failed against **{k}** different accounts.")
            t = r["table"]
            show = t[t["votes"] >= 1].head(10)
            if show.empty:
                st.caption("No user was flagged by any model.")
            else:
                st.dataframe(show[["user", "risk", "votes", "models", "indicators", "most unusual", "events"]],
                             hide_index=True, width="stretch", column_config={
                    "user": "User", "risk": "Risk", "models": "Models that flagged", "events": "Events",
                    "votes": st.column_config.ProgressColumn("Votes", min_value=0, max_value=3, format="%d / 3"),
                    "indicators": "Attack indicators", "most unusual": "Most unusual features (vs normal)"})
                u = show.iloc[0]["user"]
                a, b = st.columns(2)
                with a:
                    plot(charts.zscore_bar(r["z"].loc[u]))
                with b:
                    plot(charts.user_timeline(r["events"][r["events"]["user"] == u]))
                st.caption(f"Charts show **{u}**, the most suspicious account in this file.")
    st.caption("Indicators are simple explainable rules; risk levels come from models trained on the CERT r4.2 "
               "benchmark. A flag marks unusual behaviour, not a confirmed incident.")
