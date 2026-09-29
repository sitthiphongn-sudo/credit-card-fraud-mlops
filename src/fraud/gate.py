"""ด่านตรวจก่อนอนุมัติ และการย้อนเวอร์ชัน — ผู้รับผิดชอบ: ธรรมรักษ์"""

import argparse
import json
from pathlib import Path

import mlflow
from mlflow import MlflowClient

from fraud.config import load_params


def passes_gate(candidate: dict, champion: dict | None) -> tuple[bool, list[str]]:
    g = load_params()["gate"]
    reasons = []
    required_metrics = ("recall", "pr_auc", "p95_ms", "model_mb")

    for metric_name in required_metrics:
        if metric_name not in candidate:
            reasons.append(f"missing metric: {metric_name}")

    if reasons:
        return False, reasons
    if candidate["recall"] < g["min_recall"]:
        reasons.append(f"recall {candidate['recall']:.3f} < {g['min_recall']}")
    if champion and candidate["pr_auc"] < champion["pr_auc"] - g["max_pr_auc_drop"]:
        reasons.append(f"pr_auc {candidate['pr_auc']:.4f} แย่กว่า champion {champion['pr_auc']:.4f}")
    if candidate.get("p95_ms", 0) > g["max_p95_latency_ms"]:
        reasons.append(f"p95 {candidate['p95_ms']} ms เกิน SLO")
    if candidate.get("model_mb", 0) > g["max_model_mb"]:
        reasons.append(f"model {candidate['model_mb']} MB ใหญ่เกิน")
    return (not reasons), reasons


def load_metrics(path: Path) -> dict:
    with path.open(encoding="utf-8") as file:
        metrics = json.load(file)

    if not isinstance(metrics, dict):
        raise ValueError("metrics file must contain a JSON object")

    return metrics

def model_size_mb(run_id: str) -> float:
    """ดาวน์โหลด model artifact และคำนวณขนาดรวมเป็น MB."""
    artifact_path = Path(
        MlflowClient().download_artifacts(
            run_id,
            "model",
        )
    )

    if artifact_path.is_file():
        total_bytes = artifact_path.stat().st_size
    else:
        total_bytes = sum(
            path.stat().st_size
            for path in artifact_path.rglob("*")
            if path.is_file()
        )

    return total_bytes / (1024 * 1024)

def load_champion_metrics() -> dict | None:
    """อ่าน test metrics ของ champion ปัจจุบันจาก MLflow Registry."""
    registry = load_params()["registry"]
    model_name = registry["model_name"]
    champion_alias = registry["champion_alias"]

    client = MlflowClient()
    registered_model = client.get_registered_model(model_name)
    champion_version = registered_model.aliases.get(champion_alias)

    if champion_version is None:
        return None

    model_version = client.get_model_version(
        model_name,
        str(champion_version),
    )
    run = client.get_run(model_version.run_id)
    metrics = run.data.metrics

    if "test_pr_auc" not in metrics:
        raise RuntimeError(
            "champion MLflow run is missing test_pr_auc"
        )

    return {
        "pr_auc": float(metrics["test_pr_auc"]),
    }

def register_challenger(run_id: str, metrics: dict) -> str:
    """ลงทะเบียนโมเดลจาก MLflow run และตั้ง alias เป็น challenger."""
    registry = load_params()["registry"]
    model_name = registry["model_name"]

    result = mlflow.register_model(
        model_uri=f"runs:/{run_id}/model",
        name=model_name,
    )
    version = str(result.version)

    client = MlflowClient()
    client.set_registered_model_alias(
        model_name,
        registry["challenger_alias"],
        version,
    )
    client.set_model_version_tag(
        model_name,
        version,
        "run_id",
        run_id,
    )

    for metric_name, metric_value in sorted(metrics.items()):
        client.set_model_version_tag(
            model_name,
            version,
            f"metric.{metric_name}",
            str(metric_value),
        )

    return version


def promote_to_champion(version: str) -> None:
    """เลื่อน challenger ที่ผ่าน gate ให้เป็น champion."""
    registry = load_params()["registry"]

    MlflowClient().set_registered_model_alias(
        registry["model_name"],
        registry["champion_alias"],
        version,
    )

    print(f"champion -> version {version}")

def rollback(to_version: str) -> None:
    """ย้าย alias champion กลับไปยังเวอร์ชันที่ระบุ"""
    registry = load_params()["registry"]

    MlflowClient().set_registered_model_alias(
        registry["model_name"],
        registry["champion_alias"],
        to_version,
    )

    print(f"champion -> version {to_version}")

def main() -> int:
    parser = argparse.ArgumentParser(
        description="ตรวจ model gate หรือย้อนเวอร์ชัน champion"
    )
    subparsers = parser.add_subparsers(
        dest="command",
        required=True,
    )

    check_parser = subparsers.add_parser(
        "check",
        help="ตรวจ metrics ของ candidate",
    )
    check_parser.add_argument(
        "--candidate",
        type=Path,
        required=True,
    )
    check_parser.add_argument(
        "--champion",
        type=Path,
    )

    rollback_parser = subparsers.add_parser(
        "rollback",
        help="ย้อน champion ไปยังเวอร์ชันที่ระบุ",
    )
    rollback_parser.add_argument("version")

    args = parser.parse_args()

    if args.command == "rollback":
        rollback(args.version)
        return 0

    candidate = load_metrics(args.candidate)
    champion = load_metrics(args.champion) if args.champion else None

    passed, reasons = passes_gate(candidate, champion)

    if passed:
        print("PASS: candidate ผ่าน model gate")
        return 0

    print("FAIL: candidate ไม่ผ่าน model gate")
    for reason in reasons:
        print(f"- {reason}")

    return 1


if __name__ == "__main__":
    raise SystemExit(main())