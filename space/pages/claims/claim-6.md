# Claim 6 — full-scale Chatbot Arena intervals

**Verdict: VERIFIED · confidence: MEDIUM**

## Exact claim and source

Section 7.2, Figure 3, and Appendix K describe `n=32,980` prompts, `K=20`
models, released toxicity probability plus TF-IDF prompt embeddings and turn
index, cross-fitted nuisance estimation, and plugin confidence intervals that
collapse toward zero while debiased intervals retain interpretable
uncertainty.

The official `lmsys/chatbot_arena_conversations` dataset was gated (HTTP 403).
The run used public mirror `dim/lmsys_chatbot_arena_conversations` pinned at
revision `2878f2c…`; parquet/LFS SHA-256 `d1b51b…` and normalized core-content
SHA-256 `96ca2f…` are recorded in the
[raw output](../../evidence/claim6/raw/final_with_width_vectors.json).
This is a declared source substitution, not hidden.

## Literal-scale result

The pipeline begins with 33,000 rows, deduplicates `question_id` to exactly
32,980 contexts, retains exactly 20 models, and produces 102 covariates:
toxicity, 100-dimensional TF-IDF/SVD, and turn index. Toxicity is nonzero on
all 32,980 rows with range `[0.006507, 0.893740]`; turns range 1–25.

| GARS | Plugin median width | Debiased EIF median width | Plugin / EIF | Contract `<0.5` |
|---|---:|---:|---:|---|
| Borda | `0.00219651` | `0.0549071` | `0.040182` | pass |
| Bradley–Terry | `0.00661337` | `0.288420` | `0.023531` | pass |
| Rank Centrality | `0.000113518` | `0.00796306` | `0.014134` | pass |

The independent fresh-process checker recomputes medians from all 20 width
vectors: actual evidence exits 0. Corrupting `n_contexts` to 3000 exits 1.
Disabling the EIF gives plugin/EIF ratio exactly 1.0 for all three GARS and is
rejected. Independent output:
[`independent_checker.json`](../../evidence/claim6/raw/independent_checker.json).

Code:
[`arena_full.py`](../../evidence/code/repro/src/arena_full.py),
[`check_claim6.py`](../../evidence/code/repro/checkers/check_claim6.py), and
[`run_claim6_checker_suite.py`](../../evidence/code/repro/checkers/run_claim6_checker_suite.py).
The complete vectors and fold diagnostics are downloadable in
[`final_with_width_vectors.json`](../../evidence/claim6/raw/final_with_width_vectors.json).

Fixed command:

```text
uv sync --frozen && uv run python repro/run_all.py
```

Git `1ff5009f4524a84f1a0743697ebeda122e22a54f`; seeds `260121816`,
`260121817`, `260121900`, `260121901`; estimated 48 simultaneously useful
cores; HF `cpu-upgrade`; actual 64 logical CPUs; Arena 298.715 s, independent
checker 0.120 s, cumulative 527.503 s.

The frozen cumulative winner reran the same contract at Git
`1c7b41b21a4c7ef1fb99821b33bef01316633dcd`: Arena 380.438 s and whole suite
604.796 s, with identical displayed widths. See the
[terminal log](../../evidence/final-cumulative-run.log).

Why confidence is MEDIUM: literal scale, covariates, cross-fitting, all three
GARS, and the qualitative interval claim are direct. The official gated bytes
and undisclosed author tuning realization could not be matched, so this does
not claim numerical replication of Figure 3 bars. Full deviations:
[`limitations_and_deviations.md`](../../evidence/claim6/limitations_and_deviations.md).
