from fraud.gate import passes_gate


def good_candidate(**changes):
    metrics = {
        "recall": 0.80,
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
        candidate=good_candidate(recall=0.60),
        champion=None,
    )

    assert passed is False
    assert any("recall" in reason for reason in reasons)


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