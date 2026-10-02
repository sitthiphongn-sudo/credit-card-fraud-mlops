"""Load test สำหรับ /predict ที่ระดับ concurrency 1, 10 และ 100 — ผู้รับผิดชอบ: สหรัฐ

ตัวอย่าง (PowerShell ใช้ backtick ` ต่อบรรทัด ไม่ใช่ backslash):
    python serving/load_test.py `
        --url http://localhost:8000 --requests 1000 --levels 1,10,100 `
        --out reports/serving/load_test_champion.json

SLO (อัปงาน.pdf หัวข้อ 7, configs/params.yaml gate.max_p95_latency_ms):
    - p95 <= 100 ms เมื่อ concurrency <= 10
    - error rate < 1%
Concurrency 100 เป็นผล stress test: รายงาน p95 จริง แต่ไม่ใช้ตัดสิน SLO ด้าน latency

ก่อนยิงจะตรวจ /health ถ้า API ยังไม่ healthy จะหยุด (exit 2) เพื่อไม่ให้ได้หลักฐานจากโมเดลที่ยังไม่พร้อม
ไฟล์ผลลัพธ์บันทึก model_version / run_id / threshold ที่ /health รายงาน เพื่อยืนยันว่าวัดกับ champion ตัวไหน
"""

from __future__ import annotations

import argparse
import json
import platform
import statistics
import sys
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

SLO_P95_MS = 100.0
SLO_MAX_CONCURRENCY = 10
SLO_ERROR_RATE = 0.01

# ไม่ผ่าน proxy ของระบบ (เช่น proxy องค์กรบน Windows) เวลายิง localhost
_NO_PROXY_OPENER = urllib.request.build_opener(urllib.request.ProxyHandler({}))


def build_payload() -> dict:
    """payload 30 features [Time, V1..V28, Amount] ให้ตรงกับ serving.app"""
    return {"features": [0.0] * 29 + [5000.0]}


def one_request(url: str, data: bytes) -> tuple[float, int]:
    req = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"})
    started = time.perf_counter()
    try:
        with _NO_PROXY_OPENER.open(req, timeout=10) as response:
            response.read()
            status = response.status
    except urllib.error.HTTPError as exc:
        status = exc.code
    except Exception:
        status = 0  # connection error / timeout
    return (time.perf_counter() - started) * 1000, status


def percentile(sorted_values: list[float], percentile_value: float) -> float:
    """nearest-rank percentile บนค่าที่เรียงแล้ว"""
    if not sorted_values:
        return 0.0
    index = min(
        len(sorted_values) - 1,
        int(round(percentile_value / 100 * (len(sorted_values) - 1))),
    )
    return sorted_values[index]


def run(url: str, concurrency: int, total: int, data: bytes) -> dict:
    started = time.perf_counter()
    with ThreadPoolExecutor(max_workers=concurrency) as executor:
        results = list(executor.map(lambda _: one_request(url, data), range(total)))
    wall_seconds = time.perf_counter() - started

    latencies = sorted(r[0] for r in results)
    errors = sum(1 for r in results if r[1] != 200)
    status_counts: dict[str, int] = {}
    for _, status in results:
        status_counts[str(status)] = status_counts.get(str(status), 0) + 1

    error_rate = errors / total
    p95 = percentile(latencies, 95)
    slo_applicable = concurrency <= SLO_MAX_CONCURRENCY

    return {
        "concurrency": concurrency,
        "requests": total,
        "wall_seconds": round(wall_seconds, 3),
        "throughput_rps": round(total / wall_seconds, 2),
        "p50_ms": round(percentile(latencies, 50), 2),
        "p95_ms": round(p95, 2),
        "p99_ms": round(percentile(latencies, 99), 2),
        "mean_ms": round(statistics.mean(latencies), 2),
        "max_ms": round(latencies[-1], 2),
        "errors": errors,
        "error_rate": round(error_rate, 4),
        "status_counts": status_counts,
        "slo_applicable": slo_applicable,
        "slo_p95_pass": (p95 <= SLO_P95_MS) if slo_applicable else None,
        "slo_error_pass": error_rate < SLO_ERROR_RATE,
    }


def get_health(base_url: str) -> tuple[int, dict]:
    try:
        with _NO_PROXY_OPENER.open(f"{base_url}/health", timeout=5) as response:
            return response.status, json.loads(response.read())
    except urllib.error.HTTPError as exc:
        try:
            return exc.code, json.loads(exc.read())
        except Exception:
            return exc.code, {}


def main() -> int:
    parser = argparse.ArgumentParser(description="Load test /predict")
    parser.add_argument("--url", default="http://localhost:8000")
    parser.add_argument("--requests", type=int, default=1000)
    parser.add_argument("--levels", default="1,10,100")
    parser.add_argument("--warmup", type=int, default=50)
    parser.add_argument("--out", default="reports/serving/load_test_champion.json")
    parser.add_argument("--note", default="", help="บันทึกสภาพแวดล้อม เช่น 'docker, 1 worker, laptop'")
    args = parser.parse_args()

    base_url = args.url.rstrip("/")
    print("Checking health...", flush=True)
    try:
        health_status, health = get_health(base_url)
    except Exception as exc:
        print(f"ERROR: cannot reach {base_url}/health: {exc}", flush=True)
        return 2
    print(f"health: HTTP {health_status} {health}", flush=True)
    if health_status != 200 or health.get("status") != "healthy":
        print("ERROR: API is not healthy; refusing to produce load test evidence.", flush=True)
        return 2

    data = json.dumps(build_payload()).encode()
    target = f"{base_url}/predict"

    print(f"Warm-up ({args.warmup} requests)...", flush=True)
    run(target, 1, max(1, args.warmup), data)

    rows = []
    for level in (int(v) for v in args.levels.split(",")):
        total = max(args.requests, level)
        print(f"Running concurrency={level}, requests={total}...", flush=True)
        result = run(target, level, total, data)
        rows.append(result)
        print(result, flush=True)

    _, health_after = get_health(base_url)

    report = {
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "target": target,
        "note": args.note,
        "client": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "http": "urllib, new TCP connection per request, ThreadPoolExecutor",
        },
        "model": {
            "model_uri": health.get("model_uri"),
            "model_version": health.get("model_version"),
            "run_id": health.get("run_id"),
            "threshold": health.get("threshold"),
            "model_version_after_test": health_after.get("model_version"),
        },
        "payload": "features = [0.0] * 29 + [5000.0]",
        "slo": {
            "p95_ms_max": SLO_P95_MS,
            "applies_at_concurrency_le": SLO_MAX_CONCURRENCY,
            "error_rate_lt": SLO_ERROR_RATE,
        },
        "results": rows,
    }

    output_path = Path(args.out)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nSaved: {output_path}", flush=True)

    print("\nconcurrency |     RPS |   p50 ms |   p95 ms |   p99 ms |  err% | SLO", flush=True)
    all_slo_pass = True
    for r in rows:
        if r["slo_applicable"]:
            passed = bool(r["slo_p95_pass"]) and r["slo_error_pass"]
            all_slo_pass &= passed
            slo_text = "PASS" if passed else "FAIL"
        else:
            slo_text = "N/A (stress)" + ("" if r["slo_error_pass"] else ", error >= 1%")
        print(
            f'{r["concurrency"]:>11} | {r["throughput_rps"]:>7.2f} | {r["p50_ms"]:>8.2f} | '
            f'{r["p95_ms"]:>8.2f} | {r["p99_ms"]:>8.2f} | {r["error_rate"] * 100:>5.2f} | {slo_text}',
            flush=True,
        )

    return 0 if all_slo_pass else 1


if __name__ == "__main__":
    sys.exit(main())
