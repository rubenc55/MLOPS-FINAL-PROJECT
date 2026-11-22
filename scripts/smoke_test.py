"""Check that a running API answers like the released service: python scripts/smoke_test.py --url <url>

Inputs: --url, the base URL of the service (Render, or http://localhost:8000 for Docker);
--wait, seconds to wait for /health (Render free services sleep and take up to a minute to wake);
--expect-version, the release tag /health must report (deploy waits until the new release is live).
Outputs: one log line per check; exit code 0 if every check passes, 1 otherwise.
Used by: deploy.yml after every deploy and rollback, and by hand before the deadline.
Why: a deploy is done only when the live URL answers real JSON: /health is ok, /predict returns
one positive cost per shipment in the same order, and an invalid body gets 422.
Only the Python standard library, so it runs anywhere without the project environment.
"""

import argparse
import json
import logging
import sys
import time
import urllib.error
import urllib.request

logger = logging.getLogger("smoke_test")

SHIPMENTS = [
    {"shipment_id": 1, "transport_mode": "Sea", "cargo_type": "General", "distance_km": 8928.0,
     "weight_kg": 1683.2, "volume_m3": 4.719, "urgent": 0, "month": 3},
    {"shipment_id": 2, "transport_mode": "Road", "cargo_type": "General", "distance_km": 1792.5,
     "weight_kg": 2354.9, "volume_m3": 11.754, "urgent": 1, "month": 12},
    {"shipment_id": 3, "transport_mode": "Air", "cargo_type": "Refrigerated", "distance_km": 5200.0,
     "weight_kg": 350.0, "volume_m3": 2.1, "urgent": 1, "month": 7},
]


def call(url: str, body: dict | None = None, timeout: float = 30.0) -> tuple[int, dict]:
    """Send GET (no body) or POST (JSON body) and return the status code and the JSON answer."""
    data = None if body is None else json.dumps(body).encode("utf-8")
    request = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return response.status, json.loads(response.read() or b"{}")
    except urllib.error.HTTPError as error:
        return error.code, json.loads(error.read() or b"{}")


def wait_for_health(base_url: str, wait: float, expect_version: str | None = None) -> tuple[int, dict]:
    """Call /health until it answers 200 (with expect_version, if given) or the wait is over.

    Connection errors are retried: a sleeping Render service, or the old release still answering
    while the new one starts, are normal for the first minutes after a deploy.
    """
    deadline = time.monotonic() + wait
    while True:
        try:
            status, body = call(f"{base_url}/health")
            live = status == 200 and expect_version in (None, body.get("version"))
            if live or time.monotonic() >= deadline:
                return status, body
        except (urllib.error.URLError, OSError) as error:
            if time.monotonic() >= deadline:
                raise ConnectionError(f"{base_url}/health did not answer within {wait:.0f} s: {error}") from error
        logger.info("Waiting for %s/health ...", base_url)
        time.sleep(5)


def run_checks(base_url: str, wait: float, expect_version: str | None = None) -> list[str]:
    """Run the checks and return the problems found (empty list when every check passes)."""
    problems = []
    status, body = wait_for_health(base_url, wait, expect_version)
    logger.info("GET /health -> %s %s", status, body)
    if status != 200 or body.get("status") != "ok":
        problems.append(f"/health answered {status} {body}, expected 200 with status ok")
    elif expect_version and body.get("version") != expect_version:
        problems.append(f"/health reports version {body.get('version')}, expected {expect_version}")

    status, body = call(f"{base_url}/predict", {"shipments": SHIPMENTS})
    predictions = body.get("predictions", [])
    logger.info("POST /predict (%d shipments) -> %s %s", len(SHIPMENTS), status, predictions)
    if status != 200:
        problems.append(f"/predict answered {status} {body}, expected 200")
    elif [row.get("shipment_id") for row in predictions] != [row["shipment_id"] for row in SHIPMENTS]:
        problems.append(f"/predict returned {len(predictions)} predictions for {len(SHIPMENTS)} shipments")
    elif not all(isinstance(row.get("prediction"), (int, float)) and row["prediction"] > 0 for row in predictions):
        problems.append(f"/predict returned a cost that is not a positive number: {predictions}")

    status, body = call(f"{base_url}/predict", {"shipments": [{**SHIPMENTS[0], "discount": 5}]})
    logger.info("POST /predict with an unknown field -> %s", status)
    if status != 422:
        problems.append(f"/predict accepted an unknown field with {status}, expected 422")
    return problems


def main(argv: list[str] | None = None) -> int:
    """Parse the arguments, run the checks and return the exit code (0 pass, 1 fail)."""
    parser = argparse.ArgumentParser(description="Smoke test of the shipping cost API")
    parser.add_argument("--url", required=True, help="Base URL, e.g. https://<service>.onrender.com")
    parser.add_argument("--wait", type=float, default=120.0, help="Seconds to wait for /health (default 120)")
    parser.add_argument("--expect-version", help="Release tag /health must report, e.g. v1.0.0")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(name)s | %(message)s")

    base_url = args.url.rstrip("/")
    try:
        problems = run_checks(base_url, args.wait, args.expect_version)
    except (ConnectionError, ValueError) as error:  # ValueError: the answer is not JSON
        problems = [str(error)]
    for problem in problems:
        logger.error("FAILED: %s", problem)
    if not problems:
        logger.info("Smoke test passed: %s", base_url)
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
