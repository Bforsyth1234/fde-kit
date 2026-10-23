.PHONY: install test lint eval eval-mock trace up down worker crash-demo upgrade-demo replay-check \
	rag-demo rag-eval rag-test rag-agent-eval

worker:          ## Temporal worker for durable runs (needs `make up`)
	fde worker

crash-demo:      ## kill a worker mid-run and watch the run finish (needs `make up`)
	python examples/crash_demo.py

upgrade-demo:    ## deploy v2 while a v1 run is in flight (needs `make up`)
	python examples/upgrade_demo.py

replay-check:    ## replay v1 histories through the live workflow; the pre-deploy gate
	pytest tests/test_versioning.py -q

rag-demo:        ## three tenants, one lease template, same question (needs `make up`)
	python examples/rag_demo.py

rag-eval:        ## per-tenant retrieval eval with bge-small embeddings (needs `make up`)
	python -m evals.rag_eval

rag-agent-eval:  ## the lease-corpus agent eval: scripted model, real tools under RLS (needs `make up`, seeded)
	FDE_OTEL_ENABLED=false FDE_RAG_EMBEDDER=hash python -m evals.run_evals --set rag --provider mock --threshold 1.0

rag-test:        ## the tenant-isolation attack suite (needs `make up`)
	FDE_TEST_PG_URL=postgresql://fde:fde@localhost:5432/fde pytest tests/test_rag_isolation.py -v

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

up:              ## Phoenix (:6006), Temporal (:7233, UI :8233), Postgres (:5432)
	docker compose up -d

down:
	docker compose down
