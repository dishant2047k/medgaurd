"""
backend/utils/logger.py
Structured JSON logging with structlog + Prometheus counters.
"""
import logging
import sys
import structlog
from prometheus_client import Counter

# Prometheus metrics
DETECTION_COUNTER = Counter(
    "medguard_detections_total",
    "Total medical events detected",
    ["event_type", "camera_id"],
)
ALERT_COUNTER = Counter(
    "medguard_alerts_total",
    "Total emergency alerts sent",
    ["channel"],
)
INFERENCE_ERRORS = Counter(
    "medguard_inference_errors_total",
    "Total inference pipeline errors",
)


def setup_logging(log_level: str = "INFO") -> None:
    logging.basicConfig(
        format="%(message)s",
        stream=sys.stdout,
        level=getattr(logging, log_level.upper(), logging.INFO),
    )

    structlog.configure(
        processors=[
            structlog.contextvars.merge_contextvars,
            structlog.stdlib.add_log_level,
            structlog.stdlib.add_logger_name,
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.processors.StackInfoRenderer(),
            structlog.dev.ConsoleRenderer(),
        ],
        wrapper_class=structlog.make_filtering_bound_logger(
            getattr(logging, log_level.upper(), logging.INFO)
        ),
        context_class=dict,
        logger_factory=structlog.stdlib.LoggerFactory(),
    )


def get_logger(name: str):
    return structlog.get_logger(name)
