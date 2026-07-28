# Reproduction result: four verified claims, two blocked tables

[![Open in molab](https://marimo.io/molab-shield.svg)](https://molab.marimo.io/github/MachineLearning-Nerd/icml26-repro-rHndxbqWyh-nonparametric-llm-eval/blob/master/notebooks/reproduction.py)

We reproduced all six judged claims from
“Nonparametric LLM Evaluation from Preference Data” (arXiv 2601.21816) with
one locked `uv` environment and CPU-only compute. The exact GARS identities,
EIF derivation, A-optimal policy, and full Chatbot Arena interval-width claim
are `VERIFIED`. Tables 1 and 2 remain `BLOCKED` after four routes each because
their exact author configurations are unavailable; they are not counted as
passes.

The strongest empirical result is full scale, not a proxy: 33,000 raw Arena
rows become the paper's 32,980 contexts and 20 models. Median plugin/debiased
interval-width ratios are 0.0402 (Borda), 0.0235 (Bradley–Terry), and 0.0141
(Rank Centrality); disabling EIF makes the ratio 1.0 and the verifier rejects
it. The official dataset is gated, so a pinned mirror with matching cohort
invariants substitutes for it; byte identity cannot be proven. All uncertain
or multi-core work used Hugging Face `cpu-upgrade` (64 logical CPUs); only
short deterministic one-core audits ran locally.

- [Illustrated reproduction report](reports/reproduction/report.md)
- [Self-contained marimo tutorial](notebooks/reproduction.py)
- [Machine-readable claim artifacts](.openresearch/artifacts/)

## Experiment log

| Branch / experiment | Purpose or change | Exact run command | Assessment / outcome | Compute |
|---|---|---|---|---|
| [`orx/baseline-judged-reproduction-plus-locked-uv-envi`](https://github.com/MachineLearning-Nerd/icml26-repro-rHndxbqWyh-nonparametric-llm-eval/tree/orx/baseline-judged-reproduction-plus-locked-uv-envi) | Freeze judged code and locked environment | `uv sync --frozen && uv run python repro/run_all.py` | Baseline completed; historical assertions lacked evaluator-visible evidence | HF cpu-upgrade · 64 CPUs |
| [`orx/exact-theorem-contracts-and-analytic-checkers`](https://github.com/MachineLearning-Nerd/icml26-repro-rHndxbqWyh-nonparametric-llm-eval/tree/orx/exact-theorem-contracts-and-analytic-checkers) | Independent Claims 1, 2, and 4 certificates | `uv sync --frozen && uv run python repro/run_all.py` | VERIFIED / VERIFIED / VERIFIED | HF cpu-upgrade · 64 CPUs |
| [`orx/table-1-dedicated-falsification-audit`](https://github.com/MachineLearning-Nerd/icml26-repro-rHndxbqWyh-nonparametric-llm-eval/tree/orx/table-1-dedicated-falsification-audit) | Claim 3 mandatory fourth route | `uv sync --frozen && uv run python repro/run_all.py` | BLOCKED; no exact counterexample | local · one-core task |
| [`orx/table-2-dedicated-falsification-audit`](https://github.com/MachineLearning-Nerd/icml26-repro-rHndxbqWyh-nonparametric-llm-eval/tree/orx/table-2-dedicated-falsification-audit) | Claim 5 mandatory fourth route | `uv sync --frozen && uv run python repro/run_all.py` | BLOCKED; no exact counterexample | local · one-core task |
| [`orx/independent-claim-6-checker-and-cumulative-evide`](https://github.com/MachineLearning-Nerd/icml26-repro-rHndxbqWyh-nonparametric-llm-eval/tree/orx/independent-claim-6-checker-and-cumulative-evide) | Literal-scale Arena run and independent corruption test | `uv sync --frozen && uv run python repro/run_all.py` | VERIFIED; cumulative regression passed | HF cpu-upgrade · 64 CPUs |
| `master` | Reader-facing publication surface | Not run as an experiment (publication surface) | Awaiting evidence-gated publication | none |

# icml26-repro-rHndxbqWyh — Nonparametric LLM Evaluation from Preference Data (DMLRank)

ICML 2026 Agent Reproduction Challenge. OpenReview: rHndxbqWyh. arXiv: 2601.21816.

Clean-room reproduction of the 6 anchored claims (GARS functional, debiased EIF
estimator, A-optimal labeling policy, Tables 1 & 2, Chatbot Arena application).
CPU-only Monte-Carlo verification against a known data-generating process.
