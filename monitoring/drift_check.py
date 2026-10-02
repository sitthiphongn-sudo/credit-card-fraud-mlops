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
from typing import Any

from common import (
    DEFAULT_PERIOD_DIR,
    DEFAULT_REPORT_DIR,
    DEFAULT_TEST_PATH,
    DEFAULT_TRAIN_PATH,
    ROOT,
    contract_test_recall,
    load_model_contract,
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


def run_command(command: list[str]) -> None:
    """Run one monitoring stage and fail immediately if it fails."""
    print()
    print("+", " ".join(command))

    completed = subprocess.run(
        command,
        cwd=ROOT,
        check=False,
    )

    if completed.returncode != 0:
        raise SystemExit(completed.returncode)


def default_nannyml_python() -> Path:
    """Return the Python executable from the isolated NannyML environment."""
    windows_python = (
        ROOT
        / ".venv-nannyml"
        / "Scripts"
        / "python.exe"
    )

    posix_python = (
        ROOT
        / ".venv-nannyml"
        / "bin"
        / "python"
    )

    if windows_python.exists():
        return windows_python

    if posix_python.exists():
        return posix_python

    raise FileNotFoundError(
        "NannyML environment was not found. "
        "Expected .venv-nannyml in the project root."
    )


def fallback_nanny_summary(
    evidently_summary: dict[str, Any],
) -> dict[str, Any]:
    """Build realized-only evidence when NannyML is intentionally skipped."""
    contract = load_model_contract()

    periods: dict[str, Any] = {}

    for period, evidence in (
        evidently_summary["periods"].items()
    ):
        periods[period] = {
            "estimated_recall": None,
            "actual_recall": evidence.get(
                "actual_recall"
            ),
            "concept_gap": None,
        }

    return {
        "metric": "recall",
        "contract_test_recall": contract_test_recall(
            contract
        ),
        "reference_actual_recall": None,
        "periods": periods,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run complete drift monitoring "
            "for one simulated production scenario."
        )
    )

    parser.add_argument(
        "--scenario",
        choices=SCENARIO_TO_PERIOD,
        required=True,
        help=(
            "normal1/normal2 should remain OK; "
            "data should WATCH; concept should RETRAIN."
        ),
    )

    parser.add_argument(
        "--source",
        type=Path,
        default=DEFAULT_TEST_PATH,
    )

    parser.add_argument(
        "--reference",
        type=Path,
        default=DEFAULT_TRAIN_PATH,
    )

    parser.add_argument(
        "--period-dir",
        type=Path,
        default=DEFAULT_PERIOD_DIR,
    )

    parser.add_argument(
        "--skip-simulate",
        action="store_true",
        help=(
            "Reuse generated production periods "
            "instead of regenerating them."
        ),
    )

    parser.add_argument(
        "--skip-nannyml",
        action="store_true",
        help=(
            "Skip CBPE and use realized recall only. "
            "Useful when NannyML is unavailable."
        ),
    )

    parser.add_argument(
        "--nannyml-python",
        type=Path,
        default=None,
        help=(
            "Python executable for isolated NannyML env. "
            "Defaults to .venv-nannyml."
        ),
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    # ---------------------------------------------------------
    # 1. Production simulation
    # ---------------------------------------------------------
    if not args.skip_simulate:
        run_command(
            [
                sys.executable,
                str(
                    ROOT
                    / "monitoring"
                    / "simulate_drift.py"
                ),
                "--input",
                str(args.source),
                "--output-dir",
                str(args.period_dir),
            ]
        )

    # ---------------------------------------------------------
    # 2. Evidently
    #
    # Must run in the main project environment because
    # it loads the MLflow champion.
    # ---------------------------------------------------------
    run_command(
        [
            sys.executable,
            str(
                ROOT
                / "monitoring"
                / "run_evidently.py"
            ),
            "--period-dir",
            str(args.period_dir),
            "--reference",
            str(args.reference),
        ]
    )

    evidently_summary = read_json(
        DEFAULT_REPORT_DIR
        / "evidently_summary.json"
    )

    # ---------------------------------------------------------
    # 3. NannyML
    #
    # Main env creates champion predictions first.
    # Isolated NannyML env then performs CBPE using CSV only.
    # ---------------------------------------------------------
    if args.skip_nannyml:
        nanny_summary = fallback_nanny_summary(
            evidently_summary
        )
        nannyml_available = False
    else:
        run_command(
            [
                sys.executable,
                str(
                    ROOT
                    / "monitoring"
                    / "prepare_nannyml_inputs.py"
                ),
                "--period-dir",
                str(args.period_dir),
            ]
        )

        nanny_python = (
            args.nannyml_python
            if args.nannyml_python
            is not None
            else default_nannyml_python()
        )

        if not nanny_python.exists():
            raise FileNotFoundError(
                "NannyML Python executable "
                f"not found: {nanny_python}"
            )

        run_command(
            [
                str(nanny_python),
                str(
                    ROOT
                    / "monitoring"
                    / "run_nannyml.py"
                ),
            ]
        )

        nanny_summary = read_json(
            DEFAULT_REPORT_DIR
            / "nannyml_summary.json"
        )

        nannyml_available = True

    # ---------------------------------------------------------
    # 4. Decision
    # ---------------------------------------------------------
    thresholds = monitoring_params(
    load_params()
    )

    period = SCENARIO_TO_PERIOD[
        args.scenario
    ]

    normal_recall = baseline_recall(
        evidently_summary
    )

    nanny_period = nanny_summary.get(
        "periods",
        {},
    ).get(
        period,
        {},
    )

    estimated_recall = None

    if isinstance(
        nanny_period,
        dict,
    ):
        value = nanny_period.get(
            "estimated_recall"
        )

        if value is not None:
            estimated_recall = float(
                value
            )

    decision = decide_period(
        period,
        evidently_summary[
            "periods"
        ][period],
        drift_threshold=float(
            thresholds["drift_share"]
        ),
        max_recall_drop=float(
            thresholds[
                "max_recall_drop"
            ]
        ),
        concept_gap_threshold=float(
            thresholds[
                "concept_gap"
            ]
        ),
        normal_recall=normal_recall,
        estimated_recall=estimated_recall,
    )

    evidently_period = (
        evidently_summary[
            "periods"
        ][period]
    )

    decision.update(
        {
            "scenario": args.scenario,
            "actual_precision": evidently_period.get(
                "actual_precision"
            ),
            "drifted_features": evidently_period.get(
                "drifted_features",
                [],
            ),
            "prediction_ratios": evidently_period.get(
                "prediction_ratios",
                {},
            ),
            "nannyml_available": (
                nannyml_available
            ),
            "last_run_timestamp": (
                datetime.now(
                    timezone.utc
                ).timestamp()
            ),
            "thresholds": {
                "drift_share": float(
                    thresholds[
                        "drift_share"
                    ]
                ),
                "max_recall_drop": float(
                    thresholds[
                        "max_recall_drop"
                    ]
                ),
                "concept_gap": float(
                    thresholds[
                        "concept_gap"
                    ]
                ),
            },
        }
    )

    selected_path = (
        DEFAULT_REPORT_DIR
        / (
            "drift_check_"
            f"{args.scenario}.json"
        )
    )

    write_json(
        selected_path,
        decision,
    )

    write_json(
        DEFAULT_REPORT_DIR
        / "monitoring_summary.json",
        decision,
    )

    # ---------------------------------------------------------
    # 5. Human/pipeline output
    # ---------------------------------------------------------
    print()
    print(
        json.dumps(
            decision,
            indent=2,
            ensure_ascii=False,
        )
    )

    print()
    print(
        f"STATUS={decision['action']}"
    )

    if (
        decision["action"]
        == "RETRAIN"
    ):
        print("RETRAIN")

    raise SystemExit(
        int(
            decision[
                "exit_code"
            ]
        )
    )


if __name__ == "__main__":
    main()