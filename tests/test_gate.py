import json
from types import SimpleNamespace

import pytest

from fraud.gate import (
    load_champion_metrics,
    load_metrics,
    model_size_mb,
    passes_gate,
    promote_to_champion,
    register_challenger,
    rollback,
)


def good_candidate(**changes):
    metrics = {
        "recall_at_p80": 0.80,
        "pr_auc": 0.90,
        "p95_ms": 80,
        "model_mb": 20,
    }
    metrics.update(changes)
    return metrics


def test_candidate_passes_all_thresholds():
    passed, reasons = passes_gate(
        candidate=good_candidate(),
        champion=None,
    )

    assert passed is True
    assert reasons == []


def test_candidate_fails_recall():
    passed, reasons = passes_gate(
        candidate=good_candidate(recall_at_p80=0.60),
        champion=None,
    )

    assert passed is False
    assert any("recall_at_p80" in reason for reason in reasons)


def test_candidate_fails_against_champion():
    passed, reasons = passes_gate(
        candidate=good_candidate(pr_auc=0.80),
        champion={"pr_auc": 0.90},
    )

    assert passed is False
    assert any("pr_auc" in reason for reason in reasons)


def test_missing_latency_does_not_pass():
    candidate = good_candidate()
    del candidate["p95_ms"]

    passed, reasons = passes_gate(
        candidate=candidate,
        champion=None,
    )

    assert passed is False
    assert any("p95_ms" in reason for reason in reasons)


def test_register_challenger_sets_alias_and_tags(monkeypatch):
    calls = {"tags": []}

    def fake_register_model(**kwargs):
        calls["register"] = kwargs
        return SimpleNamespace(version=7)

    class FakeClient:
        def set_registered_model_alias(self, name, alias, version):
            calls["alias"] = (name, alias, version)

        def set_model_version_tag(self, name, version, key, value):
            calls["tags"].append((name, version, key, value))

    monkeypatch.setattr(
        "fraud.gate.mlflow.register_model",
        fake_register_model,
    )
    monkeypatch.setattr(
        "fraud.gate.MlflowClient",
        lambda: FakeClient(),
    )

    version = register_challenger(
        run_id="run-123",
        metrics={"recall_at_p80": 0.80, "pr_auc": 0.90},
    )

    assert version == "7"
    assert calls["register"] == {
        "model_uri": "runs:/run-123/model",
        "name": "fraud-detector",
    }
    assert calls["alias"] == (
        "fraud-detector",
        "challenger",
        "7",
    )
    assert (
        "fraud-detector",
        "7",
        "run_id",
        "run-123",
    ) in calls["tags"]
    assert (
        "fraud-detector",
        "7",
        "metric.recall_at_p80",
        "0.8",
    ) in calls["tags"]


def test_promote_to_champion_moves_alias(monkeypatch):
    calls = {}

    class FakeClient:
        def set_registered_model_alias(self, name, alias, version):
            calls["alias"] = (name, alias, version)

    monkeypatch.setattr(
        "fraud.gate.MlflowClient",
        lambda: FakeClient(),
    )

    promote_to_champion("7")

    assert calls["alias"] == (
        "fraud-detector",
        "champion",
        "7",
    )


def test_rollback_moves_champion_to_requested_version(monkeypatch):
    calls = {}

    class FakeClient:
        def set_registered_model_alias(self, name, alias, version):
            calls["alias"] = (name, alias, version)

    monkeypatch.setattr(
        "fraud.gate.MlflowClient",
        lambda: FakeClient(),
    )

    rollback("4")

    assert calls["alias"] == (
        "fraud-detector",
        "champion",
        "4",
    )


def test_model_size_mb_sums_all_artifact_files(monkeypatch, tmp_path):
    model_dir = tmp_path / "model"
    model_dir.mkdir()
    (model_dir / "model.pkl").write_bytes(b"a" * 1024)

    metadata_dir = model_dir / "metadata"
    metadata_dir.mkdir()
    (metadata_dir / "MLmodel").write_bytes(b"b" * 2048)

    class FakeClient:
        def download_artifacts(self, run_id, path):
            assert run_id == "run-123"
            assert path == "model"
            return str(model_dir)

    monkeypatch.setattr(
        "fraud.gate.MlflowClient",
        lambda: FakeClient(),
    )

    size = model_size_mb("run-123")

    assert size == pytest.approx(3072 / (1024 * 1024))


def test_load_champion_metrics_returns_none_without_alias(monkeypatch):
    class FakeClient:
        def get_registered_model(self, name):
            assert name == "fraud-detector"
            return SimpleNamespace(aliases={"challenger": "7"})

    monkeypatch.setattr(
        "fraud.gate.MlflowClient",
        lambda: FakeClient(),
    )

    assert load_champion_metrics() is None


def test_load_champion_metrics_reads_test_pr_auc(monkeypatch):
    class FakeClient:
        def get_registered_model(self, name):
            assert name == "fraud-detector"
            return SimpleNamespace(aliases={"champion": "4"})

        def get_model_version(self, name, version):
            assert name == "fraud-detector"
            assert version == "4"
            return SimpleNamespace(run_id="champion-run")

        def get_run(self, run_id):
            assert run_id == "champion-run"
            return SimpleNamespace(
                data=SimpleNamespace(
                    metrics={"test_pr_auc": 0.82},
                ),
            )

    monkeypatch.setattr(
        "fraud.gate.MlflowClient",
        lambda: FakeClient(),
    )

    assert load_champion_metrics() == {"pr_auc": 0.82}


def test_load_metrics_reads_best_model_report(tmp_path):
    report = {
        "name": "lightgbm_none",
        "metrics": {
            "test_recall": 0.733,
            "test_recall_at_p80": 0.773,
            "test_pr_auc": 0.796,
            "latency_p95_ms": 17.7,
            "model_mb": 0.82,
        },
    }
    path = tmp_path / "best_model.json"
    path.write_text(json.dumps(report), encoding="utf-8")

    candidate = load_metrics(path)
    passed, reasons = passes_gate(candidate, champion=None)

    assert candidate == {
        "recall_at_p80": 0.773,
        "pr_auc": 0.796,
        "p95_ms": 17.7,
        "model_mb": 0.82,
    }
    assert passed is True
    assert reasons == []


def test_old_recall_cannot_replace_recall_at_p80():
    candidate = good_candidate()
    del candidate["recall_at_p80"]
    candidate["recall"] = 0.99

    passed, reasons = passes_gate(candidate, champion=None)

    assert passed is False
    assert "missing metric: recall_at_p80" in reasons


def test_load_metrics_accepts_flat_metrics(tmp_path):
    candidate = good_candidate()
    path = tmp_path / "candidate.json"
    path.write_text(json.dumps(candidate), encoding="utf-8")

    assert load_metrics(path) == candidate