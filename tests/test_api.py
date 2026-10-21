import pytest

pytest.importorskip("fastapi")
pytest.importorskip("httpx")

from fastapi.testclient import TestClient  # noqa: E402

from demandcast import api  # noqa: E402


@pytest.fixture()
def client(trained, monkeypatch):
    monkeypatch.setattr(api, "ARTIFACTS_DIR", str(trained[0]))
    with TestClient(api.app) as c:
        yield c


def test_health_and_series(client):
    assert client.get("/health").json()["status"] == "ok"
    assert "store_00" in client.get("/series").json()["series_ids"]


def test_forecast(client):
    r = client.get("/forecast/store_00?horizon=5")
    assert r.status_code == 200 and len(r.json()["forecast"]) == 5


def test_errors(client):
    assert client.get("/forecast/nope").status_code == 404
    assert client.get("/forecast/store_00?horizon=0").status_code == 422
