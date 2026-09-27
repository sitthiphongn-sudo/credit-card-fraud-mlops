"""ด่านตรวจก่อนอนุมัติ และการย้อนเวอร์ชัน — ผู้รับผิดชอบ: ธรรมรักษ์"""
import math
import sys

from mlflow import MlflowClient

from fraud.config import load_params


def passes_gate(candidate: dict, champion: dict | None) -> tuple[bool, list[str]]:
    g = load_params()["gate"]
    reasons = []
    for key in ("recall", "pr_auc", "p95_ms", "model_mb"):
        value = candidate.get(key)
        if not isinstance(value, (int, float)) or not math.isfinite(value):
            reasons.append(f"missing or invalid metric: {key}")
    if reasons:
        return False, reasons
    if candidate["recall"] < g["min_recall"]:
        reasons.append(f"recall {candidate['recall']:.3f} < {g['min_recall']}")
    if champion and candidate["pr_auc"] < champion["pr_auc"] - g["max_pr_auc_drop"]:
        reasons.append(f"pr_auc {candidate['pr_auc']:.4f} แย่กว่า champion {champion['pr_auc']:.4f}")
    if candidate["p95_ms"] > g["max_p95_latency_ms"]:
        reasons.append(f"p95 {candidate['p95_ms']} ms เกิน SLO")
    if candidate["model_mb"] > g["max_model_mb"]:
        reasons.append(f"model {candidate['model_mb']} MB ใหญ่เกิน")
    return (not reasons), reasons


def rollback(to_version: str) -> None:
    """ย้าย alias champion กลับไปเวอร์ชันที่ระบุ — API โหลด models:/<name>@champion จึงไม่ต้องแก้โค้ด"""
    r = load_params()["registry"]
    MlflowClient().set_registered_model_alias(r["model_name"], r["champion_alias"], to_version)
    print(f"champion -> version {to_version}")


if __name__ == "__main__":
    if len(sys.argv) == 3 and sys.argv[1] == "rollback":
        rollback(sys.argv[2])
