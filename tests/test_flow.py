import pytest

from pipelines.flow import (
    add_model_size,
    check,
    promote_if_approved,
    register_candidate,
)


def good_metrics(**changes):
    metrics = {
        "recall": 0.80,
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
        "pipelines.flow.promote_to_champion",
        promoted.append,
    )

    with pytest.raises(RuntimeError, match="recall"):
        promote_if_approved.fn(
            version="7",
            candidate_metrics=good_metrics(recall=0.60),
            champion_metrics=None,
        )

    assert promoted == []

def test_add_model_size_preserves_input_metrics(monkeypatch):
    monkeypatch.setattr(
        "pipelines.flow.model_size_mb",
        lambda run_id: 12.5,
    )
    original = {
        "recall": 0.80,
        "pr_auc": 0.90,
        "p95_ms": 80,
    }

    result = add_model_size.fn("run-123", original)

    assert result == {
        "recall": 0.80,
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