"""The eval harness: golden-set runner (`runner`) and check implementations (`scorers`).

    fde eval --golden my_cases.jsonl --tools my_project.tools:registry
    python -m fde_kit.evals.runner --golden my_cases.jsonl

The kit's own golden sets (`--set golden|rag|edgar`) live in the repo's `evals/` directory and
are available from a checkout; `--golden` works from anywhere.
"""
