.PHONY: install test lint eval eval-mock trace up down worker crash-demo

worker:          ## Temporal worker for durable runs (needs `make up`)
	fde worker

crash-demo:      ## kill a worker mid-run and watch the run finish (needs `make up`)
	python examples/crash_demo.py

install:
	pip install -e ".[dev]"

test:
	pytest --cov=fde_kit --cov-report=term-missing

lint:
	ruff check .

eval:            ## golden set against the real model; exits 1 below threshold
	python -m evals.run_evals --json eval-results.json

eval-mock:       ## offline harness self-check
	FDE_OTEL_ENABLED=false python -m evals.run_evals --provider mock --threshold 1.0

up:              ## start Phoenix (traces UI at http://localhost:6006)
	docker compose up -d

down:
	docker compose down
