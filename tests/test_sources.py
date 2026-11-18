import pandas as pd
from src.sources import parse_auth_line, load_windows_csv
from src.preprocess import clean_frame

OK = "Oct  5 09:12:01 srv01 sshd[1234]: Accepted password for alice from 10.0.0.5 port 22 ssh2"
BAD = "Oct 15 23:59:59 srv01 sshd[99]: Failed password for invalid user admin from 203.0.113.9 port 4444 ssh2"


def test_parse_success_and_failure():
    a, b = parse_auth_line(OK, 2026), parse_auth_line(BAD, 2026)
    assert (a["user"], a["event_id"], a["pc"]) == ("alice", 4624, "10.0.0.5")
    assert (b["user"], b["event_id"]) == ("admin", 4625)
    assert a["timestamp"].year == 2026


def test_non_logon_line_ignored():
    assert parse_auth_line("Oct  5 03:00:01 srv01 CRON[812]: session opened for user root") is None


def test_cert_style_csv(tmp_path):
    p = tmp_path / "logon.csv"
    p.write_text("id,date,user,pc,activity\n1,01/04/2010 07:00:00,DTAA/ABC1,PC-1,Logon\n2,01/04/2010 17:00:00,DTAA/ABC1,PC-1,Logoff\n")
    df = clean_frame(load_windows_csv(p))
    assert len(df) == 1 and df.loc[0, "event_id"] == 4624    # logoff filtered out


def test_clean_removes_dupes_nulls():
    row = dict(timestamp="2026-01-01 10:00:00", user="a", pc="p", event_id=4624, logon_type=2)
    df = pd.DataFrame([row, row, {**row, "user": None}, {**row, "event_id": 4634}])
    assert len(clean_frame(df)) == 1
