# Environment and reproduction entry points

## Pinned environment

- Python: 3.12
- Dependency manager: uv
- Lockfile SHA-256:
  95f8cd0826a479d495738fba954dc4e4a8712ce13614c21e2769667d3714c165
- Final cumulative evidence run: CPU-only Hugging Face cpu-upgrade compute
- Logical CPUs allocated: 64
- Reported cumulative runtime: 604.796 seconds

## Formal commands

    uv sync --frozen
    uv run python repro/run_all.py

## Short checks

    uv run python repro/tests/test_c1_gars.py
    uv run python repro/tests/test_controls.py
    uv run python repro/checkers/run_claim6_checker_suite.py
    uv run python repro/checkers/run_evidence_bundle_checker.py

## Interpretation

The committed evidence bundle records the cumulative run and its controls.
The standardization pass verifies repository structure, metadata, branch
names, attribution, and machine-readable gate files; it does not rerun the
full CPU campaign. Re-running the formal commands is the appropriate next
step after changing scientific code or dependencies.
