# Current reproducible commands

The only formal experiment command, inherited unchanged by every node:

```text
uv sync --frozen && uv run python repro/run_all.py
```

Winning launch:

```text
orx exp run 05e02249-7a93-40f8-9337-db905b8e4f44 --backend hf --flavor cpu-upgrade --image ghcr.io/astral-sh/uv:python3.12-bookworm --timeout 30m
```

Terminal evidence:

```text
orx exp wait 05e02249-7a93-40f8-9337-db905b8e4f44 --timeout 300
orx logs 863614e1-3ab9-44aa-8c1b-8c2f60fe48ff --bytes 200000
```

Local short validation:

```text
uv run marimo check notebooks/reproduction.py
uv run python repro/checkers/run_evidence_bundle_checker.py
uv run python repro/checkers/audit_candidate_space.py <fresh-candidate-directory>
```

All long or runtime-uncertain CPU work used Hugging Face `cpu-upgrade`. Local
commands above are deterministic single-core checks finishing well within five
minutes. No GPU command was executed.

The exact previous unsupported command page is preserved at
[Historical rejected baseline commands](historical/judged-1e8c465/commands.md).
