import pytest

from pipelines.flow import (
    add_model_size,
    check,
    promote_if_approved,
    register_candidate,
    retrain_pipeline,
    train,
)


def good_metrics(**changes):
    metrics = {
        "recall_at_p80": 0.80,
        "pr_auc": 0.90,
        "p95_ms": 80,
        "model_mb": 20,
    }
    metrics.update(changes)
    return metrics


def test_register_candidate_uses_run_and_metrics(monkeypatch):
    calls = {}

    def fake_register(run_id, metrics):
        calls["run_id"] = run_id
        calls["metrics"] = metrics
        return "7"

    monkeypatch.setattr(
        "pipelines.flow.register_challenger",
        fake_register,
    )

    metrics = good_metrics()
    version = register_candidate.fn("run-123", metrics)

    assert version == "7"
    assert calls == {
        "run_id": "run-123",
        "metrics": metrics,
    }


def test_approved_candidate_becomes_champion(monkeypatch):
    promoted = []

    monkeypatch.setattr(
        "pipelines.flow.load_champion_metrics",
        lambda: None,
    )
    monkeypatch.setattr(
        "pipelines.flow.promote_to_champion",
        promoted.append,
    )

    version = promote_if_approved.fn(
        version="7",
        candidate_metrics=good_metrics(),
        champion_metrics=None,
    )

    assert version == "7"
    assert promoted == ["7"]


def test_failed_candidate_keeps_current_champion(monkeypatch):
    promoted = []

    monkeypatch.setattr(
        "pipelines.flow.load_champion_metrics",
        lambda: None,
    )
    monkeypatch.setattr(
        "pipelines.flow.promote_to_champion",
        promoted.append,
    )

    with pytest.raises(RuntimeError, match="recall_at_p80"):
        promote_if_approved.fn(
            version="7",
            candidate_metrics=good_metrics(recall_at_p80=0.60),
            champion_metrics=None,
        )

    assert promoted == []


def test_add_model_size_preserves_input_metrics(monkeypatch):
    monkeypatch.setattr(
        "pipelines.flow.model_size_mb",
        lambda run_id: 12.5,
    )

    original = {
        "recall_at_p80": 0.80,
        "pr_auc": 0.90,
        "p95_ms": 80,
    }

    result = add_model_size.fn("run-123", original)

    assert result == {
        "recall_at_p80": 0.80,
        "pr_auc": 0.90,
        "p95_ms": 80,
        "model_mb": 12.5,
    }
    assert "model_mb" not in original


def test_check_uses_raw_validation(monkeypatch):
    expected = object()

    monkeypatch.setattr(
        "pipelines.flow.validate_raw",
        lambda df: expected,
    )

    assert check.fn(object()) is expected


def test_train_selects_best_model_and_maps_metrics(monkeypatch):
    calls = {}

    best = {
        "name": "lightgbm_none",
        "run_id": "run-123",
        "metrics": {
            "test_recall_at_p80": 0.773,
            "test_pr_auc": 0.796,
            "latency_p95_ms": 17.7,
            "model_mb": 0.82,
        },
    }
    results = [best]

    def fake_run_all(**kwargs):
        calls["run_all"] = kwargs
        return results

    def fake_select_best(received_results):
        calls["select_best"] = received_results
        return best

    def fake_write_reports(received_results, received_best):
        calls["write_reports"] = (
            received_results,
            received_best,
        )

    monkeypatch.setattr(
        "pipelines.flow.run_all",
        fake_run_all,
    )
    monkeypatch.setattr(
        "pipelines.flow.select_best",
        fake_select_best,
    )
    monkeypatch.setattr(
        "pipelines.flow.write_reports",
        fake_write_reports,
    )

    run_id, metrics = train.fn(
        train_df="train-data",
        val_df="val-data",
        test_df="test-data",
        data_version="hash-123",
    )

    assert calls["run_all"]["train_df"] == "train-data"
    assert calls["run_all"]["val_df"] == "val-data"
    assert calls["run_all"]["test_df"] == "test-data"
    assert calls["run_all"]["data_version"] == "hash-123"
    assert calls["run_all"]["n_boot"] == 200

    assert calls["select_best"] is results
    assert calls["write_reports"] == (results, best)

    assert run_id == "run-123"
    assert metrics == {
        "recall_at_p80": 0.773,
        "pr_auc": 0.796,
        "p95_ms": 17.7,
        "model_mb": 0.82,
    }


def test_add_model_size_keeps_logged_size(monkeypatch):
    def unexpected_download(run_id):
        raise AssertionError("model artifact should not be downloaded")

    monkeypatch.setattr(
        "pipelines.flow.model_size_mb",
        unexpected_download,
    )

    result = add_model_size.fn(
        "run-123",
        {"model_mb": 1.5},
    )

    assert result == {"model_mb": 1.5}


def test_candidate_is_compared_with_loaded_champion(monkeypatch):
    promoted = []

    monkeypatch.setattr(
        "pipelines.flow.load_champion_metrics",
        lambda: {"pr_auc": 0.90},
    )
    monkeypatch.setattr(
        "pipelines.flow.promote_to_champion",
        promoted.append,
    )

    with pytest.raises(RuntimeError, match="pr_auc"):
        promote_if_approved.fn(
            version="7",
            candidate_metrics=good_metrics(pr_auc=0.80),
            champion_metrics=None,
        )

    assert promoted == []


def test_retrain_signal_starts_training_pipeline(monkeypatch):
    calls = []

    monkeypatch.setattr(
        "pipelines.flow.training_pipeline",
        lambda: calls.append("started") or "7",
    )

    result = retrain_pipeline.fn("RETRAIN")

    assert result == "7"
    assert calls == ["started"]


def test_other_signal_does_not_start_training_pipeline(monkeypatch):
    calls = []

    monkeypatch.setattr(
        "pipelines.flow.training_pipeline",
        lambda: calls.append("started"),
    )

    result = retrain_pipeline.fn("NORMAL")

    assert result == "ignored"
    assert calls == []