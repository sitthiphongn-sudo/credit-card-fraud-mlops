"""Expose batch monitoring results and alert thresholds as Prometheus gauges."""

from __future__ import annotations

import argparse
import time
from pathlib import Path

from prometheus_client import Gauge, start_http_server

from common import DEFAULT_REPORT_DIR, load_params, monitoring_params, read_json

DRIFT_SHARE = Gauge("fraud_monitor_drift_share", "Share of model features detected as drifted")
ACTUAL_RECALL = Gauge("fraud_monitor_actual_recall", "Realized fraud recall after labels arrive")
ESTIMATED_RECALL = Gauge("fraud_monitor_estimated_recall", "NannyML CBPE estimated fraud recall")
CONCEPT_GAP = Gauge("fraud_monitor_concept_gap", "Estimated recall minus realized recall")
RECALL_DROP = Gauge("fraud_monitor_recall_drop", "Normal-baseline recall minus realized recall")
LAST_RUN = Gauge(
    "fraud_monitor_last_run_timestamp_seconds",
    "Unix timestamp of the latest completed drift monitoring batch",
)
ACTION = Gauge("fraud_monitor_action", "One-hot monitoring decision", ["action"])
DRIFT_THRESHOLD = Gauge("fraud_monitor_drift_share_threshold", "Configured feature drift share threshold")
RECALL_DROP_THRESHOLD = Gauge(
    "fraud_monitor_max_recall_drop_threshold",
    "Configured maximum recall drop before retraining",
)
CONCEPT_GAP_THRESHOLD = Gauge(
    "fraud_monitor_concept_gap_threshold",
    "Configured estimated-vs-actual recall gap threshold",
)

ERROR_RATE_THRESHOLD = Gauge(
    "fraud_monitor_max_error_rate_threshold",
    "Configured maximum serving error rate",
)
FRAUD_RATE_THRESHOLD = Gauge(
    "fraud_monitor_max_fraud_prediction_rate_threshold",
    "Configured maximum online fraud prediction share before a spike alert",
)
STALE_AFTER_SECONDS = Gauge(
    "fraud_monitor_stale_after_seconds",
    "Configured maximum age of monitoring results before they are stale",
)
CONFIG_VALID = Gauge(
    "fraud_monitor_config_valid",
    "1 when monitoring config contains every threshold required by alert rules",
)
P95_SLO_SECONDS = Gauge(
    "fraud_monitor_p95_slo_seconds",
    "Configured serving p95 latency SLO in seconds",
)


def set_optional(gauge: Gauge, value) -> None:
    """Set a gauge to NaN when a batch metric is not statistically available."""
    gauge.set(float("nan") if value is None else float(value))


def update_config_metrics() -> None:
    """Publish thresholds even when no batch summary exists yet."""
    params = load_params()
    monitoring = monitoring_params(params)

    DRIFT_THRESHOLD.set(float(monitoring["drift_share"]))
    RECALL_DROP_THRESHOLD.set(float(monitoring["max_recall_drop"]))
    CONCEPT_GAP_THRESHOLD.set(float(monitoring["concept_gap"]))
    P95_SLO_SECONDS.set(float(params["gate"]["max_p95_latency_ms"]) / 1000.0)

    stale_after = monitoring.get("stale_after_seconds")
    max_error_rate = monitoring.get("max_error_rate")
    max_fraud_rate = monitoring.get("max_fraud_prediction_rate")

    stale_is_valid = (
        isinstance(stale_after, (int, float))
        and not isinstance(stale_after, bool)
        and stale_after > 0
    )
    error_rate_is_valid = (
        isinstance(max_error_rate, (int, float))
        and not isinstance(max_error_rate, bool)
        and 0 <= max_error_rate <= 1
    )
    fraud_rate_is_valid = (
        isinstance(max_fraud_rate, (int, float))
        and not isinstance(max_fraud_rate, bool)
        and 0 <= max_fraud_rate <= 1
    )

    STALE_AFTER_SECONDS.set(float(stale_after) if stale_is_valid else float("nan"))
    ERROR_RATE_THRESHOLD.set(float(max_error_rate) if error_rate_is_valid else float("nan"))
    FRAUD_RATE_THRESHOLD.set(float(max_fraud_rate) if fraud_rate_is_valid else float("nan"))
    CONFIG_VALID.set(float(stale_is_valid and error_rate_is_valid and fraud_rate_is_valid))


def update_batch_metrics(summary_path: Path) -> None:
    """Publish values produced by the latest successful monitoring batch."""
    summary = read_json(summary_path)

    DRIFT_SHARE.set(float(summary["drift_share"]))
    set_optional(ACTUAL_RECALL, summary.get("actual_recall"))
    set_optional(ESTIMATED_RECALL, summary.get("estimated_recall"))
    set_optional(CONCEPT_GAP, summary.get("concept_gap"))
    set_optional(RECALL_DROP, summary.get("recall_drop"))
    LAST_RUN.set(float(summary["last_run_timestamp"]))

    selected = str(summary["action"])
    if selected not in {"OK", "WATCH", "RETRAIN"}:
        raise ValueError(f"Unknown monitoring action: {selected}")

    for action in ("OK", "WATCH", "RETRAIN"):
        ACTION.labels(action=action).set(1.0 if action == selected else 0.0)


def update_metrics(summary_path: Path) -> None:
    """Update config gauges and the latest batch gauges once."""
    update_config_metrics()
    update_batch_metrics(summary_path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Serve batch monitoring results to Prometheus.")
    parser.add_argument(
        "--summary",
        type=Path,
        default=DEFAULT_REPORT_DIR / "monitoring_summary.json",
    )
    parser.add_argument("--port", type=int, default=9108)
    parser.add_argument("--refresh-seconds", type=float, default=5.0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    start_http_server(args.port)
    print(f"Monitoring exporter listening on :{args.port}")

    while True:
        try:
            update_config_metrics()
        except (FileNotFoundError, KeyError, TypeError, ValueError) as exc:
            CONFIG_VALID.set(0.0)
            print(f"Monitoring exporter cannot load config: {exc}")

        try:
            update_batch_metrics(args.summary)
        except (FileNotFoundError, KeyError, TypeError, ValueError) as exc:
            print(f"Monitoring exporter waiting for a valid batch summary: {exc}")

        time.sleep(args.refresh_seconds)


if __name__ == "__main__":
    main()
