"""Run the complete monitoring check for one production-drift scenario.

Examples:
    python monitoring/drift_check.py --scenario normal1
    python monitoring/drift_check.py --scenario normal2
    python monitoring/drift_check.py --scenario data
    python monitoring/drift_check.py --scenario concept

Exit codes:
    0 = OK
    1 = WATCH
    2 = RETRAIN
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

from common import (
    DEFAULT_PERIOD_DIR,
    DEFAULT_REPORT_DIR,
    DEFAULT_TEST_PATH,
    DEFAULT_TRAIN_PATH,
    ROOT,
    load_params,
    monitoring_params,
    read_json,
    write_json,
)
from decide_action import baseline_recall, decide_period

SCENARIO_TO_PERIOD = {
    "normal1": "period_1",
    "normal2": "period_2",
    "data": "period_3",
    "concept": "period_4",
}


def run_required(command: list[str]) -> None:
    """Run a required monitoring stage and stop immediately if it fails."""
    print("+", " ".join(command))
    completed = subprocess.run(command, cwd=ROOT, check=False)
    if completed.returncode != 0:
        raise SystemExit(completed.returncode)


def run_optional_nannyml(period_dir: Path) -> bool:
    """Run CBPE when NannyML is available; otherwise continue with labeled evidence."""
    command = [
        sys.executable,
        str(ROOT / "monitoring" / "run_nannyml.py"),
        "--period-dir",
        str(period_dir),
    ]
    print("+", " ".join(command))
    completed = subprocess.run(command, cwd=ROOT, check=False)
    if completed.returncode == 0:
        return True

    print(
        "WARNING: NannyML CBPE was not produced. "
        "Continuing with Evidently and realized recall only."
    )
    return False


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run drift monitoring for one scenario.")
    parser.add_argument(
        "--scenario",
        choices=SCENARIO_TO_PERIOD,
        required=True,
        help="normal1/normal2 should stay quiet; data/concept inject different drift types.",
    )
    parser.add_argument("--source", type=Path, default=DEFAULT_TEST_PATH)
    parser.add_argument("--reference", type=Path, default=DEFAULT_TRAIN_PATH)
    parser.add_argument("--period-dir", type=Path, default=DEFAULT_PERIOD_DIR)
    parser.add_argument(
        "--skip-simulate",
        action="store_true",
        help="Reuse existing generated period CSV files instead of recreating them.",
    )
    parser.add_argument(
        "--skip-nannyml",
        action="store_true",
        help="Skip optional CBPE while the team dependency is unresolved.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if not args.skip_simulate:
        run_required(
            [
                sys.executable,
                str(ROOT / "monitoring" / "simulate_drift.py"),
                "--input",
                str(args.source),
                "--output-dir",
                str(args.period_dir),
            ]
        )

    run_required(
        [
            sys.executable,
            str(ROOT / "monitoring" / "run_evidently.py"),
            "--period-dir",
            str(args.period_dir),
            "--reference",
            str(args.reference),
        ]
    )

    nanny_available = False
    if not args.skip_nannyml:
        nanny_available = run_optional_nannyml(args.period_dir)

    thresholds = monitoring_params(load_params())
    evidently_summary = read_json(DEFAULT_REPORT_DIR / "evidently_summary.json")
    period = SCENARIO_TO_PERIOD[args.scenario]
    normal_recall = baseline_recall(evidently_summary)

    estimated_recall = None
    nanny_summary_path = DEFAULT_REPORT_DIR / "nannyml_summary.json"
    if nanny_available and nanny_summary_path.exists():
        nanny_summary = read_json(nanny_summary_path)
        nanny_period = nanny_summary.get("periods", {}).get(period, {})
        value = nanny_period.get("estimated_recall") if isinstance(nanny_period, dict) else None
        if value is not None:
            estimated_recall = float(value)

    evidently_period = evidently_summary["periods"][period]
    decision = decide_period(
        period,
        evidently_period,
        drift_threshold=float(thresholds["drift_share"]),
        max_recall_drop=float(thresholds["max_recall_drop"]),
        concept_gap_threshold=float(thresholds["concept_gap"]),
        normal_recall=normal_recall,
        estimated_recall=estimated_recall,
    )
    decision.update(
        {
            "scenario": args.scenario,
            "actual_precision": evidently_period.get("actual_precision"),
            "drifted_features": evidently_period.get("drifted_features", []),
            "prediction_ratios": evidently_period.get("prediction_ratios", {}),
            "nannyml_available": nanny_available,
            "last_run_timestamp": datetime.now(timezone.utc).timestamp(),
            "thresholds": {
                "drift_share": float(thresholds["drift_share"]),
                "max_recall_drop": float(thresholds["max_recall_drop"]),
                "concept_gap": float(thresholds["concept_gap"]),
            },
        }
    )

    selected_path = DEFAULT_REPORT_DIR / f"drift_check_{args.scenario}.json"
    write_json(selected_path, decision)
    write_json(DEFAULT_REPORT_DIR / "monitoring_summary.json", decision)

    print(json.dumps(decision, indent=2, ensure_ascii=False))
    print(f"STATUS={decision['action']}")
    if decision["action"] == "RETRAIN":
        print("RETRAIN")

    raise SystemExit(int(decision["exit_code"]))


if __name__ == "__main__":
    main()
