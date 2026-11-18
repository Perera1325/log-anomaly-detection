from fastapi.testclient import TestClient
import api

client = TestClient(api.app)


def test_health():
    assert client.get("/health").json()["status"] == "ok"


def test_attack_triggers_alert():
    events = [{"timestamp": f"2026-12-01T02:{i % 60:02d}:{i % 50:02d}", "user": "victim9",
               "pc": f"PC{i % 70}", "event_id": 4625 if i % 3 else 4624, "logon_type": 3}
              for i in range(200)]
    r = client.post("/events", json=events)
    assert r.status_code == 200 and r.json()["processed"] == 200
    assert any(a["user"] == "VICTIM9" for a in r.json()["alerts"]) or \
        client.get("/users/victim9").json()["votes"] >= 2


def test_logoff_ignored_and_summary():
    r = client.post("/events", json=[{"timestamp": "2026-12-01T10:00:00", "user": "z", "pc": "p", "event_id": 4634}])
    assert r.json()["processed"] == 0
    assert client.get("/summary").json()["users"] > 0


def test_api_key_enforced(monkeypatch):
    monkeypatch.setenv("API_KEY", "secret")
    assert client.get("/summary").status_code == 401
    assert client.get("/summary", headers={"X-API-Key": "secret"}).status_code == 200
