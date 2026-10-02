"""Combine drift and performance evidence into OK, WATCH or RETRAIN."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from common import (
    DEFAULT_REPORT_DIR,
    contract_test_recall,
    load_model_contract,
    load_params,
    monitoring_params,
    read_json,
    write_json,
)

EXIT_CODES = {"OK": 0, "WATCH": 1, "RETRAIN": 2}
PERIOD_ORDER = ["period_1", "period_2", "period_3", "period_4"]


def baseline_recall(evidently_summary: dict[str, Any]) -> float | None:
    """Prefer approved test recall, then fall back to healthy monitoring windows."""
    try:
        contract = load_model_contract()
        value = contract_test_recall(contract)
        if value is not None:
            return value
    except (FileNotFoundError, KeyError, TypeError, ValueError):
        pass

    periods = evidently_summary.get("periods", {})
    values = [
        float(periods[name]["actual_recall"])
        for name in ("period_1", "period_2")
        if isinstance(periods.get(name), dict)
        and periods[name].get("actual_recall") is not None
    ]
    return sum(values) / len(values) if values else None


def decide_period(
    period: str,
    evidently: dict[str, Any],
    *,
    drift_threshold: float,
    max_recall_drop: float,
    concept_gap_threshold: float,
    normal_recall: float | None,
    estimated_recall: float | None,
) -> dict[str, Any]:
    """Apply monitoring thresholds without treating unavailable metrics as zero."""
    drift_share = float(evidently["drift_share"])
    actual_value = evidently.get("actual_recall")
    actual_recall = float(actual_value) if actual_value is not None else None

    concept_gap = (
        estimated_recall - actual_recall
        if estimated_recall is not None and actual_recall is not None
        else None
    )
    recall_drop = (
        normal_recall - actual_recall
        if normal_recall is not None and actual_recall is not None
        else None
    )

    retrain_reasons: list[str] = []
    watch_reasons: list[str] = []

    if concept_gap is not None and concept_gap >= concept_gap_threshold:
        retrain_reasons.append("estimated-vs-actual recall gap reached concept_gap")
    if recall_drop is not None and recall_drop >= max_recall_drop:
        retrain_reasons.append("actual recall drop reached max_recall_drop")

    if retrain_reasons:
        action = "RETRAIN"
        reasons = retrain_reasons
    elif drift_share >= drift_threshold:
        action = "WATCH"
        watch_reasons.append("feature drift share reached drift_share")
        reasons = watch_reasons
    else:
        action = "OK"
        reasons = ["drift and available model-quality evidence are within thresholds"]

    if estimated_recall is None:
        reasons.append("CBPE estimate unavailable; decision used realized recall when labels exist")
    if actual_recall is None:
        reasons.append("actual recall unavailable because this window has no positive fraud labels")
    if normal_recall is None:
        reasons.append("normal recall baseline unavailable")

    return {
        "period": period,
        "action": action,
        "exit_code": EXIT_CODES[action],
        "drift_share": drift_share,
        "normal_recall_baseline": normal_recall,
        "actual_recall": actual_recall,
        "estimated_recall": estimated_recall,
        "recall_drop": recall_drop,
        "concept_gap": concept_gap,
        "reasons": reasons,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Return OK, WATCH or RETRAIN for one period.")
    parser.add_argument("--period", choices=PERIOD_ORDER, default="period_4")
    parser.add_argument(
        "--evidently-summary",
        type=Path,
        default=DEFAULT_REPORT_DIR / "evidently_summary.json",
    )
    parser.add_argument(
        "--nannyml-summary",
        type=Path,
        default=DEFAULT_REPORT_DIR / "nannyml_summary.json",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_REPORT_DIR / "monitoring_summary.json",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    thresholds = monitoring_params(load_params())
    evidently_summary = read_json(args.evidently_summary)
    normal_recall = baseline_recall(evidently_summary)

    estimated_recall = None
    if args.nannyml_summary.exists():
        nanny_summary = read_json(args.nannyml_summary)
        period_data = nanny_summary.get("periods", {}).get(args.period, {})
        value = period_data.get("estimated_recall") if isinstance(period_data, dict) else None
        if value is not None:
            estimated_recall = float(value)

    decision = decide_period(
        args.period,
        evidently_summary["periods"][args.period],
        drift_threshold=float(thresholds["drift_share"]),
        max_recall_drop=float(thresholds["max_recall_drop"]),
        concept_gap_threshold=float(thresholds["concept_gap"]),
        normal_recall=normal_recall,
        estimated_recall=estimated_recall,
    )
    decision["last_run_timestamp"] = datetime.now(timezone.utc).timestamp()
    decision["thresholds"] = {
        "drift_share": float(thresholds["drift_share"]),
        "max_recall_drop": float(thresholds["max_recall_drop"]),
        "concept_gap": float(thresholds["concept_gap"]),
    }

    write_json(args.output, decision)
    print(json.dumps(decision, indent=2, ensure_ascii=False))
    raise SystemExit(int(decision["exit_code"]))


if __name__ == "__main__":
    main()
