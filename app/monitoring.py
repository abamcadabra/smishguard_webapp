from collections import Counter

METRICS = Counter()


def record_request():
    METRICS["request_count"] += 1


def record_success():
    METRICS["success_count"] += 1


def record_failure():
    METRICS["failure_count"] += 1


def record_reason(reason: str):
    METRICS[f"reason_{reason}"] += 1


def record_has_url(has_url: bool):
    if has_url:
        METRICS["has_url_count"] += 1
    else:
        METRICS["no_url_count"] += 1


def get_metrics() -> dict:
    return dict(METRICS)