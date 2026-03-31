"""
api/server_smoke_test.py

Tests the FastAPI endpoints without triggering a full evaluation run.
Verifies health check and run listing work correctly.

Usage:
    # Terminal 1: start MLflow
    mlflow ui --port 5000
    # Terminal 2: start API
    uvicorn api.main:app --port 8000 --reload
    # Terminal 3: run this
    python -m api.server_smoke_test
"""

import httpx
from loguru import logger


API_BASE = "http://127.0.0.1:8000"


def run_smoke_test():
    logger.info("=" * 60)
    logger.info("FASTAPI SERVER SMOKE TEST")
    logger.info("=" * 60)

    # Test health endpoint
    logger.info("\nStep 1: GET /health")
    try:
        r = httpx.get(f"{API_BASE}/health", timeout=10)
        r.raise_for_status()
        data = r.json()
        logger.success(f"  Status: {data['status']}")
        logger.info(f"  Papers available: {data['papers_available']}")
        logger.info(f"  Index exists: {data['index_exists']}")
        logger.info(f"  MLflow URI: {data['mlflow_uri']}")
    except Exception as e:
        logger.error(f"  Health check failed: {e}")
        logger.error("  Is the server running? uvicorn api.main:app --port 8000")
        return

    # Test runs listing
    logger.info("\nStep 2: GET /runs")
    try:
        r = httpx.get(f"{API_BASE}/runs?limit=5", timeout=10)
        r.raise_for_status()
        runs = r.json()
        logger.success(f"  Found {len(runs)} runs")
        for run in runs[:3]:
            logger.info(
                f"    {run['run_name']} | "
                f"faithfulness={run.get('ragas_faithfulness', 'N/A')}"
            )
    except Exception as e:
        logger.error(f"  Runs listing failed: {e}")

    # Test OpenAPI docs
    logger.info("\nStep 3: GET /docs (OpenAPI)")
    try:
        r = httpx.get(f"{API_BASE}/docs", timeout=10)
        if r.status_code == 200:
            logger.success("  OpenAPI docs available at http://127.0.0.1:8000/docs")
        else:
            logger.warning(f"  Docs returned {r.status_code}")
    except Exception as e:
        logger.error(f"  Docs check failed: {e}")

    logger.info("\n" + "=" * 60)
    logger.success("API smoke test complete")
    logger.info("=" * 60)


if __name__ == "__main__":
    run_smoke_test()
