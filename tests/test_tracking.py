"""Tests for src/tracking.py: MLflow runs, the prod model and lineage on a local MLflow store (no DagsHub)."""

import hashlib
import json
import subprocess

import mlflow
import pytest
from mlflow import MlflowClient
from mlflow.exceptions import MlflowException

from src import tracking


@pytest.fixture
def local_mlflow(tmp_path, monkeypatch):
    """Tracking settings that point MLflow at a store in tmp_path, with a stand-in token so tracking is online."""
    monkeypatch.setenv("MLFLOW_ALLOW_FILE_STORE", "true")  # the local folder store is only for these tests
    monkeypatch.setenv("MLFLOW_TRACKING_PASSWORD", "local-test")
    settings = {
        "mode": "online", "tracking_uri": (tmp_path / "mlruns").as_uri(), "experiment": "shipping-cost-test",
        "model_name": "shipping_cost_model", "prod_alias": "prod",
    }
    yield settings
    mlflow.end_run()


def _model_file(tmp_path, content: bytes = b"released model"):
    """Write a stand-in model file named like the real one."""
    path = tmp_path / "models" / "model.joblib"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(content)
    return path


def test_unit_file_md5_matches_hashlib(tmp_path):
    """file_md5 returns the plain md5 of the file, the value DVC writes in the .dvc pointer."""
    data_file = tmp_path / "shipments.csv"
    data_file.write_bytes(b"shipment_id,cost_usd\n1,100\n")

    assert tracking.file_md5(data_file) == hashlib.md5(b"shipment_id,cost_usd\n1,100\n").hexdigest()


def test_unit_file_md5_missing_file_raises(tmp_path):
    """A missing data file stops the run with a clear message that says how to get it."""
    with pytest.raises(FileNotFoundError, match="dvc pull"):
        tracking.file_md5(tmp_path / "missing.csv")


def test_unit_git_commit_prefers_github_sha(monkeypatch):
    """In CI the commit comes from GITHUB_SHA, without calling git."""
    monkeypatch.setenv("GITHUB_SHA", "abc123")

    assert tracking.git_commit() == "abc123"


def test_unit_git_commit_reads_head(monkeypatch):
    """Outside CI the commit comes from git rev-parse HEAD, without the trailing newline."""
    monkeypatch.delenv("GITHUB_SHA", raising=False)
    monkeypatch.setattr(
        tracking.subprocess, "run",
        lambda *args, **kwargs: subprocess.CompletedProcess(args, 0, stdout="def456\n"),
    )

    assert tracking.git_commit() == "def456"


def test_unit_git_commit_without_git_returns_unknown(monkeypatch):
    """Outside a Git checkout (or without git installed) the commit is recorded as 'unknown'."""
    monkeypatch.delenv("GITHUB_SHA", raising=False)

    def fail(*args, **kwargs):
        raise subprocess.CalledProcessError(128, "git")

    monkeypatch.setattr(tracking.subprocess, "run", fail)

    assert tracking.git_commit() == "unknown"


def test_unit_start_run_unknown_mode_raises():
    """A typo in tracking.mode stops the run with a clear error that lists the allowed modes."""
    with pytest.raises(ValueError, match="Unknown tracking mode: 'onlin'"):
        tracking.start_run({"mode": "onlin"}, {})


def test_unit_start_run_without_token_is_disabled(monkeypatch, caplog):
    """Online without MLFLOW_TRACKING_PASSWORD does not stop the pipeline: tracking is off, with a warning."""
    monkeypatch.delenv("MLFLOW_TRACKING_PASSWORD", raising=False)

    assert tracking.start_run({"mode": "online"}, {}) is None
    assert "MLFLOW_TRACKING_PASSWORD is not set" in caplog.text


def test_unit_start_run_disabled_mode():
    """mode: disabled never contacts the tracking server."""
    assert tracking.start_run({"mode": "disabled"}, {}) is None


def test_integration_start_run_records_settings_and_lineage(local_mlflow):
    """The run gets the data hash and commit as tags, the settings as flat parameters and as run_config.json."""
    run_config = {"split": {"random_state": 42}, "quality_gate": {"mape": 0.18}, "paths": {"model": "m.joblib"},
                  "data_md5": "4db8", "git_commit": "abc123"}

    run = tracking.start_run(local_mlflow, run_config)
    tracking.end_run(run)

    stored = MlflowClient().get_run(run.info.run_id)
    assert stored.data.tags["data_md5"] == "4db8" and stored.data.tags["git_commit"] == "abc123"
    assert stored.data.params == {"split.random_state": "42", "quality_gate.mape": "0.18"}
    assert mlflow.artifacts.load_dict(f"{run.info.artifact_uri}/run_config.json") == run_config


def test_integration_log_metrics_and_end_run(local_mlflow):
    """Metrics are logged with the split as prefix; a finished run ends FINISHED, a failed one FAILED with its error."""
    ok_run = tracking.start_run(local_mlflow, {})
    tracking.log_metrics(ok_run, {"mae": 274.5, "mape": 0.165}, "val")
    tracking.end_run(ok_run)
    failed_run = tracking.start_run(local_mlflow, {})
    tracking.end_run(failed_run, ValueError("Model not released, quality gate failed: mape = 0.2 > 0.18"))

    client = MlflowClient()
    ok, failed = client.get_run(ok_run.info.run_id), client.get_run(failed_run.info.run_id)
    assert ok.data.metrics == {"val_mae": 274.5, "val_mape": 0.165} and ok.info.status == "FINISHED"
    assert failed.info.status == "FAILED" and failed.data.tags["error"].startswith("Model not released")


def test_unit_disabled_run_logs_and_registers_nothing(tmp_path, caplog):
    """With tracking off (run None) metrics and end are skipped, and the log says the model is not registered."""
    tracking.log_metrics(None, {"mape": 0.1}, "val")
    tracking.log_model(None, tmp_path / "model.joblib", {}, {})
    tracking.end_run(None)

    assert "model is not registered" in caplog.text


def _promote(settings: dict, version: str) -> None:
    """Move prod to a version, as a person does after comparing the runs (README section 4)."""
    MlflowClient().set_registered_model_alias(settings["model_name"], settings["prod_alias"], version)


def test_integration_log_model_registers_candidates_and_never_moves_prod(local_mlflow, tmp_path):
    """Each model that passed the gate becomes a new version with its lineage tags; prod stays where a person put it."""
    model_path = _model_file(tmp_path)
    run = tracking.start_run(local_mlflow, {})
    tracking.log_model(run, model_path, local_mlflow, {"data_md5": "4db8", "git_commit": "abc123"})
    tracking.end_run(run)
    client = MlflowClient()
    assert client.get_model_version("shipping_cost_model", "1").aliases == []

    _promote(local_mlflow, "1")
    later = tracking.start_run(local_mlflow, {})
    tracking.log_model(later, model_path, local_mlflow, {"data_md5": "9f01", "git_commit": "def456"})
    tracking.end_run(later)

    candidate = client.get_model_version("shipping_cost_model", "2")
    assert candidate.run_id == later.info.run_id and candidate.aliases == []
    assert candidate.tags == {"data_md5": "9f01", "git_commit": "def456"}
    prod = client.get_model_version_by_alias("shipping_cost_model", "prod")
    assert str(prod.version) == "1" and prod.run_id == run.info.run_id


def test_unit_log_model_reraises_unexpected_registry_errors(local_mlflow, tmp_path, monkeypatch):
    """A registry error other than 'model does not exist' (e.g. no permission) stops the release instead of hiding."""
    class DeniedClient:
        def get_registered_model(self, name):
            raise MlflowException("permission denied", error_code="PERMISSION_DENIED")

    monkeypatch.setattr(tracking, "MlflowClient", DeniedClient)
    run = tracking.start_run(local_mlflow, {})

    with pytest.raises(MlflowException, match="permission denied"):
        tracking.log_model(run, _model_file(tmp_path), local_mlflow, {})


def test_integration_fetch_prod_model_replaces_model_and_writes_lineage(local_mlflow, tmp_path):
    """The prod model replaces an older local file, and its lineage (version, run, commit, data, metrics) is saved."""
    released = _model_file(tmp_path / "release", b"prod model")
    run = tracking.start_run(local_mlflow, {"data_md5": "4db8", "git_commit": "abc123"})
    tracking.log_metrics(run, {"mape": 0.165}, "val")
    tracking.log_model(run, released, local_mlflow, {"data_md5": "4db8", "git_commit": "abc123"})
    tracking.end_run(run)
    _promote(local_mlflow, "1")
    model_path = _model_file(tmp_path / "serve", b"older local model")
    info_path = model_path.parent / "model_info.json"

    info = tracking.fetch_prod_model(local_mlflow, model_path, info_path)

    assert model_path.read_bytes() == b"prod model"
    assert info == {"model": "shipping_cost_model@prod", "version": "1", "run_id": run.info.run_id,
                    "git_commit": "abc123", "data_md5": "4db8", "metrics": {"val_mape": 0.165}}
    assert json.loads(info_path.read_text(encoding="utf-8")) == info


def test_unit_fetch_prod_model_without_token_raises(monkeypatch, tmp_path):
    """Without MLFLOW_TRACKING_PASSWORD the deploy stops with a clear message instead of shipping an unknown model."""
    monkeypatch.delenv("MLFLOW_TRACKING_PASSWORD", raising=False)

    with pytest.raises(RuntimeError, match="MLFLOW_TRACKING_PASSWORD is not set"):
        tracking.fetch_prod_model({}, tmp_path / "model.joblib", tmp_path / "info.json")


def test_integration_fetch_prod_model_wrong_file_raises(local_mlflow, tmp_path):
    """A prod version without the expected file name stops the deploy and names the missing file."""
    run = tracking.start_run(local_mlflow, {})
    tracking.log_model(run, _model_file(tmp_path), local_mlflow, {})
    tracking.end_run(run)
    _promote(local_mlflow, "1")

    with pytest.raises(FileNotFoundError, match="has no file other.joblib"):
        tracking.fetch_prod_model(local_mlflow, tmp_path / "other.joblib", tmp_path / "info.json")


def test_integration_fetch_prod_model_without_prod_alias_raises(local_mlflow, tmp_path):
    """A candidate that nobody promoted is never shipped: with no prod version the deploy stops at the registry."""
    run = tracking.start_run(local_mlflow, {})
    tracking.log_model(run, _model_file(tmp_path), local_mlflow, {})
    tracking.end_run(run)

    with pytest.raises(MlflowException):
        tracking.fetch_prod_model(local_mlflow, tmp_path / "model.joblib", tmp_path / "info.json")
