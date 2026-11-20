"""Tests for src/tracking.py: W&B runs, the prod model and lineage, without an account or internet."""

import hashlib
import json
import subprocess
from pathlib import Path

import pytest

from src import tracking

TRACKING = {
    "mode": "online", "project": "shipping-cost", "entity": None,
    "model_name": "shipping_cost_model", "prod_alias": "prod",
}


class FakeRun:
    """Stand-in for a W&B run that records what it receives."""

    def __init__(self, **init_kwargs):
        self.init_kwargs, self.logged, self.artifacts, self.summary = init_kwargs, [], [], {}
        self.id, self.url, self.exit_code = "run123", None, None
        self.disabled = init_kwargs.get("mode") == "disabled"

    def log(self, values):
        self.logged.append(values)

    def log_artifact(self, artifact, aliases):
        self.artifacts.append((artifact, aliases))

    def finish(self, exit_code):
        self.exit_code = exit_code


class FakeArtifact:
    """Stand-in for wandb.Artifact that records its files."""

    def __init__(self, name, type, metadata):  # pylint: disable=redefined-builtin
        self.name, self.type, self.metadata, self.files = name, type, metadata, []

    def add_file(self, path):
        self.files.append(path)


@pytest.fixture
def fake_wandb(monkeypatch):
    """Replace wandb.init and wandb.Artifact inside src.tracking with the fakes above."""
    monkeypatch.setattr(tracking.wandb, "init", lambda **kwargs: FakeRun(**kwargs))
    monkeypatch.setattr(tracking.wandb, "Artifact", FakeArtifact)


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


def test_unit_start_run_online_with_key(fake_wandb, monkeypatch):
    """With WANDB_API_KEY set, the run starts online in the configured project with the run config."""
    monkeypatch.setenv("WANDB_API_KEY", "not-a-real-key")

    run = tracking.start_run(TRACKING, {"git_commit": "abc123"})

    assert run.init_kwargs == {
        "project": "shipping-cost", "entity": None, "job_type": "train",
        "config": {"git_commit": "abc123"}, "mode": "online",
    }


def test_unit_start_run_without_key_is_disabled(fake_wandb, monkeypatch, caplog):
    """Online without WANDB_API_KEY does not stop the pipeline: tracking is disabled, with a warning."""
    monkeypatch.delenv("WANDB_API_KEY", raising=False)

    run = tracking.start_run(TRACKING, {})

    assert run.init_kwargs["mode"] == "disabled"
    assert "WANDB_API_KEY is not set" in caplog.text


def test_unit_start_run_unknown_mode_raises(fake_wandb):
    """A typo in tracking.mode stops the run with a clear error that lists the allowed modes."""
    with pytest.raises(ValueError, match="Unknown tracking mode: 'onlin'"):
        tracking.start_run({**TRACKING, "mode": "onlin"}, {})


def test_unit_log_metrics_prefixes_split():
    """Metrics are logged with the split as prefix, so validation and test values stay apart."""
    run = FakeRun()

    tracking.log_metrics(run, {"mae": 274.5, "mape": 0.165}, "val")

    assert run.logged == [{"val_mae": 274.5, "val_mape": 0.165}]


def test_unit_log_model_uploads_file_with_prod_alias(fake_wandb, tmp_path):
    """The model file becomes a 'model' artifact with its metadata, aliased latest and prod."""
    model_path = tmp_path / "model.joblib"
    model_path.write_bytes(b"model")
    run = FakeRun()

    tracking.log_model(run, model_path, TRACKING, {"data_md5": "d41d8", "git_commit": "abc123"})

    artifact, aliases = run.artifacts[0]
    assert (artifact.name, artifact.type, artifact.files) == ("shipping_cost_model", "model", [str(model_path)])
    assert artifact.metadata == {"data_md5": "d41d8", "git_commit": "abc123"}
    assert aliases == ["latest", "prod"]


def test_unit_log_model_disabled_run_registers_nothing(fake_wandb, tmp_path, caplog):
    """With tracking disabled, nothing is uploaded and the log says so instead of claiming a prod model."""
    run = FakeRun(mode="disabled")

    tracking.log_model(run, tmp_path / "model.joblib", TRACKING, {})

    assert run.artifacts == []
    assert "model is not registered" in caplog.text


def test_unit_end_run_success_and_failure():
    """A finished run exits with 0; a failed one exits with 1 and keeps the error message in its summary."""
    ok_run, failed_run = FakeRun(), FakeRun()

    tracking.end_run(ok_run)
    tracking.end_run(failed_run, ValueError("Model not released, quality gate failed: mape = 0.2 > 0.18"))

    assert (ok_run.exit_code, ok_run.summary) == (0, {})
    assert failed_run.exit_code == 1
    assert failed_run.summary["error"].startswith("Model not released")


def test_integration_real_wandb_disabled_mode(tmp_path, monkeypatch):
    """With the real wandb library in disabled mode, a full run (metrics, model, finish) works offline."""
    monkeypatch.chdir(tmp_path)
    model_path = tmp_path / "model.joblib"
    model_path.write_bytes(b"model")

    run = tracking.start_run({**TRACKING, "mode": "disabled"}, {"git_commit": "abc123"})
    tracking.log_metrics(run, {"mape": 0.165}, "val")
    tracking.log_model(run, model_path, TRACKING, {"git_commit": "abc123"})
    tracking.end_run(run)

    assert not (tmp_path / "wandb").exists()


class FakeLoggedRun:
    """Stand-in for the W&B run that logged an artifact."""

    path = ["team", "shipping-cost", "run123"]


class FakeRegistryArtifact:
    """Stand-in for a downloaded registry artifact that holds one file."""

    qualified_name, version, aliases = "team/shipping-cost/shipping_cost_model:v3", "v3", ["latest", "prod"]
    metadata = {"git_commit": "abc123", "data_md5": "4db8", "val_metrics": {"mape": 0.165}}

    def __init__(self, file_name, run):
        self.file_name, self.run = file_name, run

    def download(self, root):
        (Path(root) / self.file_name).write_bytes(b"prod model")
        return root

    def logged_by(self):
        return self.run


@pytest.fixture
def fake_registry(monkeypatch):
    """Replace wandb.Api so artifact(name) returns a fake artifact and records the requested name."""
    requested = {}

    def install(file_name="model.joblib", run=FakeLoggedRun()):
        class FakeApi:
            def artifact(self, name, type):  # pylint: disable=redefined-builtin
                requested.update(name=name, type=type)
                return FakeRegistryArtifact(file_name, run)

        monkeypatch.setattr(tracking.wandb, "Api", FakeApi)
        return requested

    return install


def test_unit_fetch_prod_model_replaces_model_and_writes_lineage(fake_registry, tmp_path, monkeypatch):
    """The prod model replaces any older file, and its lineage (version, run, commit, data) is written next to it."""
    monkeypatch.setenv("WANDB_API_KEY", "not-a-real-key")
    requested = fake_registry()
    model_path, info_path = tmp_path / "models" / "model.joblib", tmp_path / "models" / "model_info.json"
    model_path.parent.mkdir()
    model_path.write_bytes(b"older local model")

    info = tracking.fetch_prod_model(TRACKING, model_path, info_path)

    assert requested == {"name": "shipping-cost/shipping_cost_model:prod", "type": "model"}
    assert model_path.read_bytes() == b"prod model"
    assert info["version"] == "v3" and info["wandb_run"] == "team/shipping-cost/run123"
    assert info["git_commit"] == "abc123" and info["data_md5"] == "4db8"
    assert json.loads(info_path.read_text(encoding="utf-8")) == info


def test_unit_fetch_prod_model_uses_entity_and_handles_unknown_run(fake_registry, tmp_path, monkeypatch):
    """With an entity in config.yaml the name starts with it; an artifact without a logging run keeps wandb_run None."""
    monkeypatch.setenv("WANDB_API_KEY", "not-a-real-key")
    requested = fake_registry(run=None)

    info = tracking.fetch_prod_model({**TRACKING, "entity": "team"}, tmp_path / "model.joblib", tmp_path / "info.json")

    assert requested["name"] == "team/shipping-cost/shipping_cost_model:prod"
    assert info["wandb_run"] is None


def test_unit_fetch_prod_model_without_key_raises(monkeypatch, tmp_path):
    """Without WANDB_API_KEY the deploy stops with a clear message instead of shipping an unknown model."""
    monkeypatch.delenv("WANDB_API_KEY", raising=False)

    with pytest.raises(RuntimeError, match="WANDB_API_KEY is not set"):
        tracking.fetch_prod_model(TRACKING, tmp_path / "model.joblib", tmp_path / "info.json")


def test_unit_fetch_prod_model_wrong_file_raises(fake_registry, tmp_path, monkeypatch):
    """An artifact without model.joblib stops the deploy and names the missing file."""
    monkeypatch.setenv("WANDB_API_KEY", "not-a-real-key")
    fake_registry(file_name="other.pkl")

    with pytest.raises(FileNotFoundError, match="has no file model.joblib"):
        tracking.fetch_prod_model(TRACKING, tmp_path / "model.joblib", tmp_path / "info.json")
