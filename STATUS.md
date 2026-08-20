# Status — Nonparametric LLM Evaluation from Preference Data

- Paper: arXiv 2601.21816, accepted at ICML 2026
- OpenReview: rHndxbqWyh
- Method: DMLRank
- Repository owner: MachineLearning-Nerd
- Scope: independent clean-room claim-level reproduction and evidence audit
- Current state: standardized and publication-ready as an evidence audit

## Current verdict

- C1 GARS special cases: VERIFIED_SCOPED
- C2 finite EIF theorem certificate: VERIFIED_SCOPED
- C3 Table 1 finite numbers: BLOCKED
- C4 A-optimal policy: VERIFIED_SCOPED
- C5 Table 2 finite numbers: BLOCKED
- C6 full-scale Chatbot Arena intervals: VERIFIED_SCOPED

Four scoped contracts have supporting evidence and two finite-result claims
remain blocked. Complete paper-level claims independently verified: 0 of 6.
The repository does not claim author endorsement or a current external score.

## Evidence locations

- CLAIM_EVIDENCE.md — claim anchors, production paths, controls, and limits
- SOURCE_AUDIT.md — paper identity, available sources, and source gaps
- ENVIRONMENT.md — pinned environment and reproduction commands
- .openresearch/artifacts/ — claim contracts, methods, raw outputs, and limits
- space/pages/index.md — evaluator-visible evidence matrix
- reports/reproduction/report.md — illustrated technical report
- outputs/verdict.json — six-claim machine-readable evidence verdict
- outputs/gate.json — publication/documentation gate
- BRANCH_AUDIT.md — branch purposes and normalization record

## Reproduction entry points

The committed cumulative evidence records a historical formal campaign using
uv sync --frozen and uv run python repro/run_all.py. That runner is not
present in this checkout. The runnable checked-in short checks are listed in
ENVIRONMENT.md and live under repro/tests/ and
space/evidence/code/repro/checkers/.

The cumulative run used CPU-only Hugging Face cpu-upgrade compute, 64 logical
CPUs, and 604.796 seconds. The standardization verifier checks the published
structure and metadata without rerunning that campaign.

## Remaining limitations

C3 and C5 need the exact author-controlled finite-run realization, including
seeds, coefficient draws, metric conventions, and data/code revision. C6
uses a pinned mirror because the official LMSYS source is gated; cohort
invariants match, but byte identity is not established.
