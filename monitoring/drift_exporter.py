"""Expose batch monitoring results and alert thresholds as Prometheus gauges."""

from __future__ import annotations

import argparse
import time
from pathlib import Path

from common import DEFAULT_REPORT_DIR, load_params, monitoring_params, read_json
from prometheus_client import Gauge, start_http_server

DRIFT_SHARE = Gauge(
    "fraud_monitor_drift_share",
    "Share of model features detected as drifted",
)
ACTUAL_RECALL = Gauge(
    "fraud_monitor_actual_recall",
    "Realized fraud recall after labels arrive",
)
ESTIMATED_RECALL = Gauge(
    "fraud_monitor_estimated_recall",
    "NannyML CBPE estimated fraud recall",
)
CONCEPT_GAP = Gauge(
    "fraud_monitor_concept_gap",
    "Estimated recall minus realized recall",
)
RECALL_DROP = Gauge(
    "fraud_monitor_recall_drop",
    "Normal-baseline recall minus realized recall",
)
LAST_RUN = Gauge(
    "fraud_monitor_last_run_timestamp_seconds",
    "Unix timestamp of the latest completed drift monitoring batch",
)
ACTION = Gauge(
    "fraud_monitor_action",
    "One-hot monitoring decision",
    ["action"],
)
DRIFT_THRESHOLD = Gauge(
    "fraud_monitor_drift_share_threshold",
    "Configured feature drift share threshold",
)
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
FRAUD_RATE_THRESHOLD_CONFIGURED = Gauge(
    "fraud_monitor_fraud_prediction_rate_threshold_configured",
    "1 when max_fraud_prediction_rate has been calibrated and configured",
)
STALE_AFTER_SECONDS = Gauge(
    "fraud_monitor_stale_after_seconds",
    "Configured maximum age of monitoring results before they are stale",
)
CONFIG_VALID = Gauge(
    "fraud_monitor_config_valid",
    "1 when required monitoring configuration is valid",
)
P95_SLO_SECONDS = Gauge(
    "fraud_monitor_p95_slo_seconds",
    "Configured serving p95 latency SLO in seconds",
)


def set_optional(gauge: Gauge, value) -> None:
    """Set a gauge to NaN when a batch metric is not statistically available."""
    gauge.set(float("nan") if value is None else float(value))


def valid_rate(value) -> bool:
    """Return True for numeric rates in the inclusive range [0, 1]."""
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and 0.0 <= float(value) <= 1.0
    )


def update_config_metrics() -> None:
    """Publish alert thresholds even when no batch summary exists yet."""
    params = load_params()
    monitoring = monitoring_params(params)

    DRIFT_THRESHOLD.set(float(monitoring["drift_share"]))
    RECALL_DROP_THRESHOLD.set(float(monitoring["max_recall_drop"]))
    CONCEPT_GAP_THRESHOLD.set(float(monitoring["concept_gap"]))
    ERROR_RATE_THRESHOLD.set(float(monitoring["max_error_rate"]))
    STALE_AFTER_SECONDS.set(float(monitoring["stale_after_seconds"]))
    P95_SLO_SECONDS.set(float(params["gate"]["max_p95_latency_ms"]) / 1000.0)

    max_fraud_rate = monitoring.get("max_fraud_prediction_rate")
    fraud_rate_is_valid = valid_rate(max_fraud_rate)
    FRAUD_RATE_THRESHOLD_CONFIGURED.set(1.0 if fraud_rate_is_valid else 0.0)
    FRAUD_RATE_THRESHOLD.set(float(max_fraud_rate) if fraud_rate_is_valid else float("nan"))

    required_rates_valid = all(
        valid_rate(monitoring[key])
        for key in ("drift_share", "max_recall_drop", "concept_gap", "max_error_rate")
    )
    stale_value = monitoring["stale_after_seconds"]
    stale_is_valid = (
        isinstance(stale_value, (int, float))
        and not isinstance(stale_value, bool)
        and float(stale_value) > 0
    )
    CONFIG_VALID.set(1.0 if required_rates_valid and stale_is_valid else 0.0)


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
