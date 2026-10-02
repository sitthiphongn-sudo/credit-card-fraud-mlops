"""Exercise the existing validation/gate CLIs and retain CI evidence."""

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

EVIDENCE = Path("reports/ci")
CANDIDATE = Path("reports/experiments/best_model.json")
BAD_SAMPLES = (
    "bad_missing_column.csv",
    "bad_missing_value.csv",
    "bad_negative_amount.csv",
    "bad_out_of_range.csv",
    "bad_wrong_type.csv",
)


def save(name: str, value: dict) -> None:
    EVIDENCE.mkdir(parents=True, exist_ok=True)
    (EVIDENCE / name).write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding="utf-8")


def run_cli(module: str, arguments: list[str], log_name: str) -> int:
    result = subprocess.run(
        [sys.executable, "-m", module, *arguments],
        capture_output=True, text=True, encoding="utf-8", check=False,
    )
    output = result.stdout + result.stderr
    print(output, end="", flush=True)
    EVIDENCE.mkdir(parents=True, exist_ok=True)
    (EVIDENCE / log_name).write_text(output, encoding="utf-8")
    return result.returncode


def validate_samples() -> int:
    results = []
    for name, expected in [("valid.csv", 0), *((name, 1) for name in BAD_SAMPLES)]:
        actual = run_cli("fraud.validate", [f"data/sample/{name}"], f"{name}.log")
        results.append({"sample": name, "expected_exit": expected, "actual_exit": actual})
        print(f"{name}: expected={expected}, actual={actual}", flush=True)
    passed = all(row["expected_exit"] == row["actual_exit"] for row in results)
    save("data-validation.json", {"passed": passed, "results": results})
    return 0 if passed else 1


def check_requirements() -> int:
    from packaging.requirements import InvalidRequirement, Requirement
    from packaging.version import InvalidVersion, Version

    errors = []
    checked = 0
    for number, line in enumerate(Path("requirements.txt").read_text(encoding="utf-8").splitlines(), 1):
        line = line.split(" #", 1)[0].strip()
        if not line or line.startswith("#"):
            continue
        checked += 1
        try:
            requirement = Requirement(line)
            specifiers = list(requirement.specifier)
            if requirement.url or len(specifiers) != 1 or specifiers[0].operator != "==":
                raise ValueError("requires one exact == version")
            Version(specifiers[0].version)  # Reject wildcards such as ==1.*.
        except (InvalidRequirement, InvalidVersion, ValueError) as error:
            errors.append(f"line {number}: {line}: {error}")
    passed = checked > 0 and not errors
    save("requirements.json", {"passed": passed, "checked": checked, "errors": errors})
    print(f"Exact dependency pins: {checked}, errors: {errors}")
    return 0 if passed else 1


def model_gate() -> int:
    # All required metrics are present; rejection must be caused by low recall.
    save("low-recall.json", {"recall_at_p80": 0.60, "pr_auc": 0.9, "p95_ms": 10, "model_mb": 1})
    control_exit = run_cli(
        "fraud.gate", ["check", "--candidate", str(EVIDENCE / "low-recall.json")], "control-gate.log"
    )
    candidate_exit = run_cli("fraud.gate", ["check", "--candidate", str(CANDIDATE)], "candidate-gate.log")
    passed = control_exit == 1 and candidate_exit == 0
    save("model-gate.json", {
        "passed": passed,
        "candidate_exit": candidate_exit,
        "control_exit": control_exit,
        "expected_candidate_exit": 0,
        "expected_control_exit": 1,
    })
    return 0 if passed else 1


def summary(job_status: str) -> int:
    evidence = {"job_status": job_status, "job": os.environ.get("GITHUB_JOB", "local")}
    for name in ("requirements", "data-validation", "data-generation", "model-gate"):
        path = EVIDENCE / f"{name}.json"
        if path.exists():
            evidence[name] = json.loads(path.read_text(encoding="utf-8"))
    if CANDIDATE.exists() and ("model-gate" in evidence or evidence["job"] == "model-gate"):
        evidence["data_kind"] = "synthetic CI fixture; not real-world model performance"
        evidence["candidate"] = json.loads(CANDIDATE.read_text(encoding="utf-8"))
    save("summary.json", evidence)
    rendered = "## CI evidence\n\n```json\n" + json.dumps(evidence, indent=2, ensure_ascii=False) + "\n```\n"
    (EVIDENCE / "summary.md").write_text(rendered, encoding="utf-8")
    if os.environ.get("GITHUB_STEP_SUMMARY"):
        with Path(os.environ["GITHUB_STEP_SUMMARY"]).open("a", encoding="utf-8") as output:
            output.write(rendered)
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "command", choices=("check-requirements", "validate-samples", "model-gate", "summary")
    )
    parser.add_argument("--job-status", default="local")
    args = parser.parse_args()
    if args.command == "check-requirements":
        return check_requirements()
    if args.command == "validate-samples":
        return validate_samples()
    if args.command == "model-gate":
        return model_gate()
    return summary(args.job_status)


if __name__ == "__main__":
    raise SystemExit(main())
