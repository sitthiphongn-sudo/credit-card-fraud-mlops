from fraud.gate import passes_gate


def test_gate_rejects_missing_measurements():
    passed, reasons = passes_gate({"recall": 0.9, "pr_auc": 0.8}, None)
    assert not passed
    assert "missing or invalid metric: p95_ms" in reasons
    assert "missing or invalid metric: model_mb" in reasons


def test_gate_rejects_low_recall():
    candidate = {"recall": 0.1, "pr_auc": 0.8, "p95_ms": 1.0, "model_mb": 0.1}
    passed, reasons = passes_gate(candidate, None)
    assert not passed
    assert any("recall" in reason for reason in reasons)
