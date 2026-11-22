"""Tests for src/api.py: /health, /predict, the strict request contract and start-up errors."""

import copy
import json

import pandas as pd
import pytest
from fastapi.testclient import TestClient

from src import api
from src.config import SETTINGS
from src.features import get_feature_preprocessor
from src.infer import run_inference
from src.train import train_model
from src.utils import save_model

FEATURES = SETTINGS["features"]
FEATURE_COLUMNS = FEATURES["categorical"] + FEATURES["numeric"]
SAMPLE = api.SAMPLE_SHIPMENT


def _make_shipments(n_rows: int = 30) -> pd.DataFrame:
    """Build shipments that pass every validation rule, with a simple known cost."""
    df = pd.DataFrame({
        "shipment_id": range(1, n_rows + 1),
        "transport_mode": [["Road", "Sea", "Air"][i % 3] for i in range(n_rows)],
        "cargo_type": [["General", "Refrigerated", "Hazardous"][i % 3] for i in range(n_rows)],
        "distance_km": [100.0 + i * 50 for i in range(n_rows)],
        "weight_kg": [50.0 + i * 20 for i in range(n_rows)],
        "volume_m3": [1.0 + i * 0.5 for i in range(n_rows)],
        "urgent": [i % 2 for i in range(n_rows)],
        "month": [i % 12 + 1 for i in range(n_rows)],
    })
    df["cost_usd"] = 0.5 * df["distance_km"] + 0.2 * df["weight_kg"] + 50
    return df


@pytest.fixture
def client(tmp_path, monkeypatch):
    """Serve a small model trained in tmp_path, with logs in tmp_path; the real models/ is never read."""
    df = _make_shipments()
    model = train_model(
        df[FEATURE_COLUMNS], df["cost_usd"],
        get_feature_preprocessor(FEATURES["categorical"], FEATURES["numeric"]),
        "regression", model_params={"n_estimators": 10},
    )
    settings = copy.deepcopy(SETTINGS)
    settings["paths"]["model"] = tmp_path / "model.joblib"
    settings["paths"]["model_info"] = tmp_path / "model_info.json"
    settings["paths"]["log_file"] = tmp_path / "logs" / "pipeline.log"
    save_model(model, settings["paths"]["model"])
    monkeypatch.setattr(api, "SETTINGS", settings)
    with TestClient(api.app) as test_client:
        yield test_client, model


def test_unit_health_reports_ok_and_model(client):
    """/health answers 200 with the release version, the served model file and no lineage when none was fetched."""
    test_client, _ = client

    response = test_client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok", "version": "dev", "model": "model.joblib", "model_info": {}}


def test_integration_health_shows_release_and_lineage(client, monkeypatch):
    """With the lineage file of the prod model, /health traces the service to its MLflow run, data and commit."""
    _, model = client
    settings = api.SETTINGS
    info = {"model": "shipping_cost_model@prod", "version": "3", "run_id": "abc123", "git_commit": "def456",
            "data_md5": "4db8"}
    settings["paths"]["model_info"].write_text(json.dumps(info), encoding="utf-8")
    monkeypatch.setattr(api, "APP_VERSION", "v1.2.0")

    with TestClient(api.app) as test_client:
        body = test_client.get("/health").json()

    assert body["version"] == "v1.2.0"
    assert body["model_info"] == info


def test_integration_predict_matches_batch_inference(client):
    """/predict returns one estimate per shipment, in order, equal to run_inference on the same rows."""
    test_client, model = client
    shipments = _make_shipments(3).drop(columns="cost_usd")

    response = test_client.post("/predict", json={"shipments": shipments.to_dict(orient="records")})

    assert response.status_code == 200
    body = response.json()["predictions"]
    assert [row["shipment_id"] for row in body] == [1, 2, 3]
    expected = run_inference(model, shipments[FEATURE_COLUMNS])["prediction"].tolist()
    assert [row["prediction"] for row in body] == pytest.approx(expected)


def test_unit_predict_trims_text_like_the_pipeline(client):
    """Spaces around a category are trimmed by clean_dataframe, as in batch, so the shipment is accepted."""
    test_client, _ = client

    response = test_client.post("/predict", json={"shipments": [{**SAMPLE, "transport_mode": " Sea "}]})

    assert response.status_code == 200


@pytest.mark.parametrize("body", [
    {"shipments": [{**SAMPLE, "discount": 5}]},                # unknown field (extra="forbid")
    {"shipments": [{k: v for k, v in SAMPLE.items() if k != "month"}]},  # missing field
    {"shipments": [{**SAMPLE, "distance_km": "8928"}]},         # number sent as text (strict)
    {"shipments": [{**SAMPLE, "urgent": True}]},                # bool where 0 or 1 is expected (strict)
    {"shipments": []},                                          # nothing to price
    {"shipment": [SAMPLE]},                                     # wrong top-level key
])
def test_unit_predict_rejects_invalid_body_with_422(client, body):
    """Bodies that break the contract are rejected by Pydantic with 422 before any prediction."""
    test_client, _ = client

    assert test_client.post("/predict", json=body).status_code == 422


@pytest.mark.parametrize("change, message", [
    ({"transport_mode": "Rocket"}, "transport_mode: unknown values"),
    ({"month": 13}, "month: 1 rows outside the range"),
    ({"distance_km": 20000.0}, "distance_km: 1 rows outside the range"),
])
def test_unit_predict_rejects_rule_breaks_with_config_message(client, change, message):
    """Values that break a config.yaml rule get 422 with the same message as the batch pipeline."""
    test_client, _ = client

    response = test_client.post("/predict", json={"shipments": [{**SAMPLE, **change}]})

    assert response.status_code == 422
    assert message in response.json()["detail"]


def test_unit_docs_is_the_only_ui(client):
    """/docs serves the contract with the sample body; /redoc is switched off."""
    test_client, _ = client

    assert test_client.get("/docs").status_code == 200
    assert test_client.get("/redoc").status_code == 404
    schema = test_client.get("/openapi.json").json()["components"]["schemas"]["PredictionRequest"]
    assert schema["examples"] == [{"shipments": [SAMPLE]}]


def test_unit_shipment_fields_match_config():
    """The request fields are exactly the ID column and the feature columns of config.yaml, in order."""
    assert list(api.Shipment.model_fields) == [SETTINGS["problem"]["id_column"]] + FEATURE_COLUMNS


def test_unit_startup_without_model_fails_clearly(tmp_path, monkeypatch):
    """Without a released model the service does not start, and the error says how to create one."""
    settings = copy.deepcopy(SETTINGS)
    settings["paths"]["model"] = tmp_path / "missing.joblib"
    settings["paths"]["model_info"] = tmp_path / "model_info.json"
    settings["paths"]["log_file"] = tmp_path / "pipeline.log"
    monkeypatch.setattr(api, "SETTINGS", settings)

    with pytest.raises(FileNotFoundError, match="missing.joblib"):
        with TestClient(api.app):
            pass
