# RAG Eval Harness - Makefile

.PHONY: run-local run-prod mlflow-local ingest eval redteam mode lint lint-fix test test-ci quality-gate help

# --- Local demo (zero cost) ---

run-local:
	@echo Starting local demo stack...
	DEMO_MODE=true mlflow ui --port 5000 &
	DEMO_MODE=true uvicorn api.main:app --port 8000 --reload &
	DEMO_MODE=true streamlit run ui/dashboard.py

mlflow-local:
	mlflow ui --port 5000

ingest:
	DEMO_MODE=true python -m ingestion.document_loader

eval:
	DEMO_MODE=true pytest evaluation/ -v --tb=short

redteam:
	DEMO_MODE=true python -m red_team.smoke_test

# --- Production (Azure) ---

run-prod:
	@echo Starting production stack - ensure AZURE_* vars are set in .env
	DEMO_MODE=false uvicorn api.main:app --port 8000 --reload &
	DEMO_MODE=false streamlit run ui/dashboard.py

# --- Developer utilities ---

mode:
	@python -c "import os; from dotenv import load_dotenv; load_dotenv(); m=os.getenv('DEMO_MODE','true'); print('MODE: LOCAL DEMO' if m=='true' else 'MODE: [PRODUCTION] Azure')"

lint:
	ruff check . --select "E,F,W,I" --ignore "E501"

lint-fix:
	ruff check . --select "E,F,W,I" --ignore "E501" --fix

test:
	pytest tests/ -v --tb=short --cov=. --cov-report=term-missing

test-ci:
	pytest tests/ -v --tb=short --cov=. --cov-report=xml

quality-gate:
	python tests/quality_gate.py

help:
	@echo RAG Eval Harness - available targets:
	@echo   LOCAL DEMO: run-local, mlflow-local, ingest, eval, redteam
	@echo   PRODUCTION: run-prod
	@echo   UTILITIES:  mode, lint, lint-fix, test, quality-gate
