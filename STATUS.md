# Status — Nonparametric LLM Evaluation from Preference Data

- Paper: arXiv 2601.21816, accepted at ICML 2026
- OpenReview: rHndxbqWyh
- Repository owner: MachineLearning-Nerd
- Scope: independent claim-level reproduction and evidence audit
- Current state: evidence bundle complete; repository normalization in progress

## Current verdict

- C1 GARS special cases: `VERIFIED`
- C2 EIF theorem certificate: `VERIFIED`
- C3 Table 1 finite numbers: `BLOCKED`
- C4 A-optimal policy: `VERIFIED`
- C5 Table 2 finite numbers: `BLOCKED`
- C6 full-scale Chatbot Arena intervals: `VERIFIED`

The canonical evidence bundle and its independent checker pass. The two
blocked claims are deliberately not promoted to passes because the exact
author-controlled finite-run realization is unavailable.

## Evidence locations

- `.openresearch/artifacts/` — claim contracts, methods, raw outputs, and limitations
- `space/pages/index.md` — evaluator-visible evidence matrix
- `reports/reproduction/report.md` — technical report
- `outputs/verdict.json` — six-claim machine-readable verdict
- `BRANCH_AUDIT.md` — branch purpose and old-to-new branch mapping

## Reproduction entry point

~~~text
uv sync --frozen
uv run python repro/run_all.py
~~~

The final cumulative run used CPU-only Hugging Face `cpu-upgrade` compute,
64 logical CPUs, and 604.796 seconds.

## Remaining limitations

The unresolved finite claims need the authors’ exact v1 seeds, coefficient
draws, metric conventions, and data/code realization. The full-scale Arena
claim uses a pinned mirror because the official LMSYS source is gated; cohort
invariants match, but byte identity is not established.
