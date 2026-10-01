"""Load test สำหรับ /predict -> P50/P95/P99, RPS, error rate เทียบ SLO.

รัน: python serving/load_test.py --url http://localhost:8000 --requests 500
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

SLO_P95_MS = 100.0
SLO_ERROR_RATE = 0.01

def build_payload() -> dict:
    """สร้าง Payload แบบ 30 Features ให้ตรงตาม Schema ของ serving/app.py"""
    # 30 features: [Time, V1..V28, Amount]
    features = [0.0] * 29 + [5000.0]  # บังคับ Amount = 5000.0
    return {"features": features}

_NO_PROXY_OPENER = urllib.request.build_opener(urllib.request.ProxyHandler({}))

def one_request(url: str, data: bytes):
    req = urllib.request.Request(
        url, data=data, headers={"Content-Type": "application/json"}
    )
    t0 = time.perf_counter()
    try:
        with _NO_PROXY_OPENER.open(req, timeout=10) as r:
            r.read()
            ok = r.status == 200
    except Exception:
        ok = False
    return (time.perf_counter() - t0) * 1000, ok

def percentile(sorted_vals, p):
    if not sorted_vals:
        return 0.0
    idx = min(len(sorted_vals) - 1, int(round(p / 100 * (len(sorted_vals) - 1))))
    return sorted_vals[idx]

def run(url: str, concurrency: int, total: int, data: bytes) -> dict:
    start = time.perf_counter()
    with ThreadPoolExecutor(max_workers=concurrency) as ex:
        results = list(ex.map(lambda _: one_request(url, data), range(total)))
    wall = time.perf_counter() - start
    lat = sorted(r[0] for r in results)
    errors = sum(1 for r in results if not r[1])
    return {
        "concurrency": concurrency,
        "requests": total,
        "rps": round(total / wall, 2),
        "p50_ms": round(percentile(lat, 50), 2),
        "p95_ms": round(percentile(lat, 95), 2),
        "p99_ms": round(percentile(lat, 99), 2),
        "mean_ms": round(statistics.mean(lat), 2) if lat else 0.0,
        "error_rate": round(errors / total, 4),
    }

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://localhost:8000")
    ap.add_argument("--requests", type=int, default=500)
    ap.add_argument("--levels", default="1,10,100")
    ap.add_argument("--out", default="serving/load_test_results.json")
    args = ap.parse_args()

    print("checking health...", flush=True)
    try:
        health = json.load(urllib.request.urlopen(f"{args.url}/health", timeout=5))
        print("health:", health, flush=True)
        if health.get("status") not in ["ok", "healthy"]:
            print("WARNING: model not loaded or status degraded -> ผลที่ได้ไม่ใช่ latency ของโมเดลจริง", flush=True)
    except Exception as e:
        print(f"WARNING: Cannot connect to health check endpoint: {e}", flush=True)

    data = json.dumps(build_payload()).encode()
    target = f"{args.url}/predict"

    print("warm-up...", flush=True)
    run(target, 1, 20, data)  # warm-up

    rows = []
    for level in (int(x) for x in args.levels.split(",")):
        print(f"running concurrency={level} ...", flush=True)
        r = run(target, level, max(args.requests, level), data)
        rows.append(r)
        print(r, flush=True)

    Path(args.out).write_text(json.dumps(rows, indent=2))

    print("\nconcurrency |   RPS   | p50 | p95 | p99 | err% | SLO", flush=True)
    passed = True
    for r in rows:
        ok = r["p95_ms"] <= SLO_P95_MS and r["error_rate"] < SLO_ERROR_RATE
        passed &= ok
        print(f"{r['concurrency']:>11} | {r['rps']:>7} | {r['p50_ms']:>3} | "
              f"{r['p95_ms']:>3} | {r['p99_ms']:>3} | {r['error_rate']*100:>4.1f} | "
              f"{'PASS' if ok else 'FAIL'}", flush=True)
    return 0 if passed else 1

if __name__ == "__main__":
    sys.exit(main())