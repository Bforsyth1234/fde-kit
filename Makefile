.PHONY: install test lint eval eval-mock trace up down

install:
	pip install -e ".[dev]"

test:
	pytest --cov=fde_kit --cov-report=term-missing

lint:
	ruff check . && ruff format --check .

eval:            ## golden set against the real model; exits 1 below threshold
	python -m evals.run_evals --json eval-results.json

eval-mock:       ## offline harness self-check
	FDE_OTEL_ENABLED=false python -m evals.run_evals --provider mock --threshold 1.0

up:              ## start Phoenix (traces UI at http://localhost:6006)
	docker compose up -d

down:
	docker compose down
