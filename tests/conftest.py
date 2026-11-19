import shutil

import pytest

from bootstrap import ensure_ready
from config import ALERT_FILE


@pytest.fixture(scope="session", autouse=True)
def pipeline_ready():
    ensure_ready()


@pytest.fixture(scope="session", autouse=True)
def keep_real_alerts_clean():
    """API tests raise test alerts; put the real alert files back afterwards so the dashboard stays clean."""
    files = [ALERT_FILE, ALERT_FILE.with_suffix(".log")]
    backups = {}
    for f in files:
        if f.exists():
            backups[f] = f.with_name(f.name + ".testbak")
            shutil.copy2(f, backups[f])
    yield
    for f in files:
        if f in backups:
            shutil.move(backups[f], f)
        elif f.exists():
            f.unlink()
