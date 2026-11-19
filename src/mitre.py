"""Indicative mapping from detected behaviour patterns to MITRE ATT&CK techniques (https://attack.mitre.org)."""
import pandas as pd

# (pattern prefix produced by src.analyzer, technique id, technique name, tactic)
MAP = [
    ("Brute force", "T1110", "Brute Force", "Credential Access"),
    ("Success after repeated failures", "T1110", "Brute Force", "Credential Access"),
    ("Password spraying", "T1110.003", "Brute Force: Password Spraying", "Credential Access"),
    ("Lateral movement", "T1021", "Remote Services", "Lateral Movement"),
    ("Off-hours access", "T1078", "Valid Accounts",
     "Initial Access, Persistence, Privilege Escalation, Defense Evasion"),
    ("Sudden change", "T1078", "Valid Accounts",
     "Initial Access, Persistence, Privilege Escalation, Defense Evasion"),
]


def url(technique_id: str) -> str:
    return "https://attack.mitre.org/techniques/" + technique_id.replace(".", "/") + "/"


def techniques(patterns) -> pd.DataFrame:
    """One row per detected pattern that has an ATT&CK counterpart."""
    rows = []
    for p in patterns:
        for prefix, tid, name, tactic in MAP:
            if p.startswith(prefix):
                rows.append({"Detected pattern": p, "Technique": tid, "Name": name, "Tactic": tactic,
                             "Reference": url(tid)})
                break
    return pd.DataFrame(rows, columns=["Detected pattern", "Technique", "Name", "Tactic", "Reference"])


def ids(patterns) -> str:
    t = techniques(patterns)
    return ", ".join(sorted(set(t["Technique"]))) if len(t) else "-"
