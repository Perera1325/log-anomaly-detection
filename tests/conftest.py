import pytest
from bootstrap import ensure_ready


@pytest.fixture(scope="session", autouse=True)
def pipeline_ready():
    ensure_ready()
