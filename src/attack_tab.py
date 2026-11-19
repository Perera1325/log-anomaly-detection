"""Dashboard tab: upload log files (or use the bundled sample attacks) and read an analysis of each one."""
import hashlib
from pathlib import Path

import pandas as pd
import streamlit as st

from src import charts, ui
from src.analyzer import analyze, read_log

SAMPLES = Path(__file__).resolve().parent.parent / "sample_attacks"
WHAT = {
    "0_normal_control.csv": "Control: ordinary office accounts, no attack (expected to stay quiet).",
    "1_brute_force.csv": "Brute force: 60 failed logons within minutes, followed by a success.",
    "2_off_hours_insider.csv": "Off-hours insider: sudden logons between 01:00 and 04:00 on 8 consecutive days.",
    "3_lateral_movement.csv": "Lateral movement: one account logs on to 25 different machines.",
    "4_password_spraying.csv": "Password spraying: one machine attempts a few passwords against 14 accounts.",
    "5_account_takeover.csv": "Account takeover: weekend 03:00 burst of activity from previously unseen machines.",
}


@st.cache_data(show_spinner=False)
def _run(name: str, data: bytes, digest: str):
    return analyze(read_log(name, data))


def _banner(r):
    w = r["worst"]
    patterns = ui.esc("; ".join(r["patterns"])) if r["patterns"] else "no specific attack pattern matched"
    if r["verdict"] == "Low":
        ui.callout("No account looks unusual in this file.", "Low", "Low risk.")
        return
    text = (f"Most suspicious account: <b>{ui.esc(w['user'])}</b> ({w['votes']} of 3 detectors agree). "
            f"Indicators: {patterns}.")
    if r["verdict"] == "Medium":
        text += " Only one detector flags this account, so the signal is weak and no alert is raised."
    ui.callout(text, r["verdict"], f"{r['verdict']} risk.")


def render(plot):
    ui.section("5.1  Analysis of uploaded log files",
               "Upload one or more logon files: CSV with timestamp, user, pc, event_id and logon_type; a Windows "
               "Get-WinEvent export; a CERT logon.csv; or a Linux sshd auth.log. Each file is scored by the three "
               "detectors and checked against explainable attack indicators.")
    uploads = st.file_uploader("Log files", type=["csv", "log", "txt"], accept_multiple_files=True,
                               key="attack_uploads")
    samples = sorted(SAMPLES.glob("*.csv")) if SAMPLES.exists() else []
    c1, c2, _ = st.columns([1, 1, 3])
    if samples and c1.button("Load sample files", key="use_samples"):
        st.session_state["samples_on"] = True
    if st.session_state.get("samples_on") and c2.button("Clear samples", key="clear_samples"):
        st.session_state["samples_on"] = False
        st.rerun()
    if samples:
        with st.expander("Sample files (synthetic) for trying the uploader"):
            for p in samples:
                st.download_button(f"Download {p.name}", p.read_bytes(), file_name=p.name, mime="text/csv",
                                   key=f"dl_{p.name}")
                st.caption(WHAT.get(p.name, ""))

    files = [(u.name, u.getvalue()) for u in uploads]
    if st.session_state.get("samples_on"):
        files += [(p.name, p.read_bytes()) for p in samples]
    if not files:
        ui.callout("Upload files or choose <b>Load sample files</b> to see the analysis.")
        return

    results, errors = {}, {}
    for name, data in files:
        try:
            results[name] = _run(name, data, hashlib.md5(data).hexdigest())
        except Exception as e:                                  # unreadable file: report it, keep the rest
            errors[name] = str(e)
    for name, msg in errors.items():
        ui.callout(f"<b>{ui.esc(name)}</b>: {ui.esc(msg)}", "High", "Could not read file.")
    if not results:
        return

    ui.table_caption("atk-summary", "Summary of the analysed files. Flagged accounts are those on which at least "
                                    "two detectors agree.")
    st.dataframe(pd.DataFrame([{
        "File": n, "Events": r["n_events"], "Accounts": r["n_users"],
        "Flagged": r["n_flagged"], "Risk": r["verdict"],
        "Patterns found": ", ".join(r["patterns"]) or "-"} for n, r in results.items()]),
        hide_index=True, width="stretch")

    for name, r in results.items():
        with st.expander(f"{name}  |  {r['verdict']} risk", expanded=len(results) <= 2):
            if WHAT.get(name):
                st.caption(WHAT[name])
            _banner(r)
            ui.key_figures([("Events", f"{r['n_events']:,}", "Event IDs 4624 and 4625"),
                            ("Accounts", r["n_users"], "in file"),
                            ("Flagged", r["n_flagged"], "2 or more detectors"),
                            ("Highest risk", r["verdict"], r["worst"]["user"])])
            for pc, k in r["spray"]:
                ui.callout(f"Machine <b>{ui.esc(pc)}</b> failed against <b>{k}</b> different accounts.",
                           "High", "Password spraying.")
            t = r["table"]
            show = t[t["votes"] >= 1].head(10)
            if show.empty:
                st.caption("No account was flagged by any detector.")
                continue
            ui.table_caption(f"atk-t-{name}", f"Accounts flagged in {ui.esc(name)}, with the detectors that flagged "
                                              "them, rule-based attack indicators and their most unusual features.")
            st.dataframe(show[["user", "risk", "votes", "models", "indicators", "most unusual", "events"]],
                         hide_index=True, width="stretch", column_config={
                "user": "Account", "risk": "Risk", "models": "Detectors", "events": "Events",
                "votes": st.column_config.ProgressColumn("Votes", min_value=0, max_value=3, format="%d / 3"),
                "indicators": "Attack indicators", "most unusual": "Most unusual features (z-score)"})
            u = show.iloc[0]["user"]
            a, b = st.columns(2)
            with a:
                plot(charts.zscore_bar(r["z"].loc[u]))
                ui.figure_caption(f"atk-z-{name}", f"Largest behavioural deviations of {ui.esc(u)}, the most "
                                                   f"suspicious account in {ui.esc(name)}.")
            with b:
                plot(charts.user_timeline(r["events"][r["events"]["user"] == u]))
                ui.figure_caption(f"atk-tl-{name}", f"Logon events of {ui.esc(u)} by time of day; shaded bands "
                                                    "mark off-hours.")
    st.markdown("<p class='meta'>Indicators are simple, explainable rules. Risk levels come from models trained on "
                "the CERT r4.2 benchmark. A flag marks unusual behaviour, not a confirmed incident.</p>",
                unsafe_allow_html=True)
