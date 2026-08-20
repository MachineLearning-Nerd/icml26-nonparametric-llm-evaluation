# ICML 2026 — Nonparametric LLM Evaluation from Preference Data

Independent reproduction and evidence audit for **DMLRank**, the paper
“Nonparametric LLM Evaluation from Preference Data.”

**Current evidence status:** Claims 1, 2, 4, and 6 are evidence-backed;
Claims 3 and 5 remain `BLOCKED`. The blocked claims are not counted as passes,
and this repository does not claim that every published number was reproduced.

Paper and source links:

- [arXiv abstract and metadata](https://arxiv.org/abs/2601.21816)
- [arXiv HTML paper](https://arxiv.org/html/2601.21816)
- [ICML 2026 OpenReview record](https://openreview.net/forum?id=rHndxbqWyh)
- [Authors’ linked implementation](https://github.com/DennisFrauen/NonparametricLLMEval)

## What the paper does

[Read the claim-to-evidence ledger](CLAIM_EVIDENCE.md), [source audit](SOURCE_AUDIT.md),
[environment notes](ENVIRONMENT.md), and [publication report](REPORT.md) for the
reproduction boundary and exact production paths. This is an independent audit,
not an author-maintained implementation.

DMLRank estimates generalized average ranking scores (GARS) from selectively
observed pairwise preference data. The framework covers Borda scores,
Bradley–Terry projections, and Rank Centrality/PageRank-style scores. It uses
an efficient-influence-function (EIF) correction with cross-fitting so that
flexible nuisance models can be used while retaining uncertainty estimates,
and it derives an A-optimal policy for collecting preference labels under a
budget.

## Claim ledger

| Claim | Paper statement | Evidence production path | Repository verdict |
|---|---|---|---|
| C1 | GARS contains Borda, Bradley–Terry, and Rank Centrality | `space/evidence/code/repro/src/exact_theory.py` and `repro/tests/test_c1_gars.py` construct a BT model, evaluate all three maps, compare Jacobians with finite differences, and run a wrong-symmetrization control. | `VERIFIED` · high confidence |
| C2 | The cross-fitted EIF estimator is orthogonal, asymptotically normal, and efficient under the theorem assumptions | The finite MAR certificate and independent derivation in `space/evidence/code/repro/src/exact_theory.py` check the EIF identity and reject an omitted-IPW control. | `VERIFIED` · medium confidence; finite certificate is not a proof assistant for every continuous-context regularity condition |
| C3 | Table 1: plugin inference has poor coverage while debiasing repairs it at the reported finite-run scale | `space/evidence/code/repro/src/table1_borda.py`, `space/evidence/code/repro/src/claim3_metric_audit.py`, and `space/evidence/code/repro/src/claim3_falsification.py` run p=2 and p=5 interpretations, metric audits, and a fourth falsification route. | `BLOCKED` · low confidence; the author realization, metric scale, and exact code revision are unavailable |
| C4 | The A-optimal acquisition policy has the clipped square-root form | `space/evidence/code/repro/src/exact_theory.py` derives the KKT solution, solves the budget by bisection, compares with SLSQP, and runs an inverted-information control. | `VERIFIED` · high confidence |
| C5 | Table 2: A-optimal acquisition beats random acquisition for the three GARS objectives | `space/evidence/code/repro/src/table2_oracle.py`, `space/evidence/code/repro/src/table2_learned.py`, `space/evidence/code/repro/src/table2_crossfit.py`, and `space/evidence/code/repro/src/claim5_falsification.py` separate oracle, learned, acquired-data, and falsification routes. | `BLOCKED` · low confidence; the exact v1 author realization is unavailable |
| C6 | On Chatbot Arena, naive plugin intervals collapse relative to debiased EIF intervals | `space/evidence/code/repro/src/arena_full.py` runs the literal 32,980-context/20-model pipeline; `space/evidence/code/repro/checkers/run_claim6_checker_suite.py` independently checks serialized evidence and rejects a corrupted dataset-size control. | `VERIFIED` · medium confidence; the official gated file cannot be proven byte-identical to the pinned mirror |

Canonical machine-readable evidence is under
`.openresearch/artifacts/` and the evaluator-visible snapshot under `space/`.
The older files in `outputs/` are retained as supporting sub-run outputs; the
canonical six-claim verdict is recorded in `outputs/verdict.json`.

## Results at a glance

- Complete paper-level claims independently verified: 0/6. Four scoped
  contracts pass; two finite claims remain blocked.
- C1: BT recovery, Rank Centrality stationarity, Borda/Rank Centrality ranking,
  and Jacobian checks pass at numerical precision.
- C2: the finite pathwise EIF identity agrees to `2.78e-17`; the omitted-IPW
  control fails as expected.
- C3: the qualitative coverage correction appears, but the reported error
  magnitudes are not reproduced, so the claim remains blocked.
- C4: the budget error is `4.44e-16`; independent SLSQP disagreement is
  `5.86e-8`.
- C5: the oracle route improves BT and Rank Centrality, while the Borda
  comparison is effectively tied; the published finite-run claim remains
  blocked because exact author inputs are missing.
- C6: the full-scale run uses 33,000 raw rows, 32,980 contexts, 20 models,
  and 102 features. Median plugin/EIF interval-width ratios are 0.0402
  (Borda), 0.0235 (Bradley–Terry), and 0.0141 (Rank Centrality). The
  independent checker passes the real evidence and rejects a 3,000-context
  corruption.

## Repository map

| Path | Role |
|---|---|
| `space/evidence/code/repro/src/exact_theory.py` | C1, C2, and C4 analytic certificates |
| `space/evidence/code/repro/src/table1_borda.py` | C3 finite Table 1 reconstruction |
| `space/evidence/code/repro/src/claim3_metric_audit.py` | C3 metric-definition audit |
| `space/evidence/code/repro/src/claim3_falsification.py` | C3 assumption-matched falsification route |
| `space/evidence/code/repro/src/table2_oracle.py` | C5 oracle acquisition route |
| `space/evidence/code/repro/src/table2_learned.py` | C5 pilot-learned acquisition route |
| `space/evidence/code/repro/src/table2_crossfit.py` | C5 acquired-data cross-fitting route |
| `space/evidence/code/repro/src/claim5_falsification.py` | C5 paired/sign-flip falsification route |
| `space/evidence/code/repro/src/arena_full.py` | C6 full-scale Arena analysis |
| `space/evidence/code/repro/checkers/` | Independent serialized-evidence and corruption checkers |
| `.openresearch/artifacts/` | Claim contracts, methods, raw outputs, and limitations |
| `space/` | Evaluator-visible evidence snapshot and release manifest |
| `reports/reproduction/report.md` | Illustrated technical report |
| `notebooks/reproduction.py` | Self-contained result walkthrough |
| `docs/CLAIMS_PINNED.md` | Paper-to-claim contract |

## Branch audit

`main` is the reader-facing publication branch. The former internal `orx/*`
branches are retained as clean `audit/*` branches so that every experiment
route remains inspectable.

| Clean branch | Former branch | Purpose |
|---|---|---|
| [`audit/baseline-locked-env`](https://github.com/MachineLearning-Nerd/icml26-nonparametric-llm-evaluation/tree/audit/baseline-locked-env) | `orx/baseline-judged-reproduction-plus-locked-uv-envi` | Freeze the judged baseline and lock the `uv` environment |
| [`audit/theorem-contracts`](https://github.com/MachineLearning-Nerd/icml26-nonparametric-llm-evaluation/tree/audit/theorem-contracts) | `orx/exact-theorem-contracts-and-analytic-checkers` | Independent C1, C2, and C4 certificates |
| [`audit/chatbot-arena-full`](https://github.com/MachineLearning-Nerd/icml26-nonparametric-llm-evaluation/tree/audit/chatbot-arena-full) | `orx/full-chatbot-arena-cross-fitted-inference` | Full Chatbot Arena inference route |
| [`audit/table1-borda`](https://github.com/MachineLearning-Nerd/icml26-nonparametric-llm-evaluation/tree/audit/table1-borda) | `orx/published-dgp-table-1-borda-reconstruction` | First Table 1 Borda reconstruction |
| [`audit/table1-covariate-interpretation`](https://github.com/MachineLearning-Nerd/icml26-nonparametric-llm-evaluation/tree/audit/table1-covariate-interpretation) | `orx/table-1-five-covariate-source-interpretation` | Test the five-covariate interpretation |
| [`audit/table1-metric-audit`](https://github.com/MachineLearning-Nerd/icml26-nonparametric-llm-evaluation/tree/audit/table1-metric-audit) | `orx/table-1-error-definition-audit` | Audit the unresolved Table 1 error definition |
| [`audit/table1-falsification`](https://github.com/MachineLearning-Nerd/icml26-nonparametric-llm-evaluation/tree/audit/table1-falsification) | `orx/table-1-dedicated-falsification-audit` | Complete the fourth C3 falsification route |
| [`audit/table2-oracle`](https://github.com/MachineLearning-Nerd/icml26-nonparametric-llm-evaluation/tree/audit/table2-oracle) | `orx/oracle-published-dgp-table-2-calibration` | Calibrate the C5 oracle policy route |
| [`audit/table2-pilot-learned`](https://github.com/MachineLearning-Nerd/icml26-nonparametric-llm-evaluation/tree/audit/table2-pilot-learned) | `orx/pilot-learned-table-2-reconstruction` | Reconstruct the pilot-learned C5 route |
| [`audit/table2-crossfit`](https://github.com/MachineLearning-Nerd/icml26-nonparametric-llm-evaluation/tree/audit/table2-crossfit) | `orx/cross-fitted-acquired-data-table-2` | Cross-fit acquired data for C5 |
| [`audit/table2-falsification`](https://github.com/MachineLearning-Nerd/icml26-nonparametric-llm-evaluation/tree/audit/table2-falsification) | `orx/table-2-dedicated-falsification-audit` | Complete the fourth C5 falsification route |
| [`audit/cumulative-evidence`](https://github.com/MachineLearning-Nerd/icml26-nonparametric-llm-evaluation/tree/audit/cumulative-evidence) | `orx/cumulative-evaluator-evidence-bundle` | Assemble the final claim contracts, full-scale C6 run, controls, and evidence gate |
| [`audit/arena-integrity`](https://github.com/MachineLearning-Nerd/icml26-nonparametric-llm-evaluation/tree/audit/arena-integrity) | `orx/arena-mirror-integrity-and-full-inference` | Audit mirror integrity and full Arena inference |
| [`audit/claim6-checker`](https://github.com/MachineLearning-Nerd/icml26-nonparametric-llm-evaluation/tree/audit/claim6-checker) | `orx/independent-claim-6-checker-and-cumulative-evide` | Add the independent C6 checker and cumulative evidence |

The branch purpose, old-to-new mapping, and final tip audit are also recorded
in [`BRANCH_AUDIT.md`](BRANCH_AUDIT.md).

## Reproduction commands

The committed evidence records a historical formal campaign using the pinned
environment:

~~~text
uv sync --frozen
uv run python repro/run_all.py
~~~

The current checkout does not contain that historical runner. The runnable
short checks are:

~~~text
uv run python repro/tests/test_c1_gars.py
uv run python repro/tests/test_controls.py
uv run python space/evidence/code/repro/checkers/run_claim6_checker_suite.py
uv run python space/evidence/code/repro/checkers/run_evidence_bundle_checker.py
~~~

The final cumulative evidence run used CPU-only Hugging Face `cpu-upgrade`
compute with 64 logical CPUs and completed in 604.796 seconds. The exact
source anchors, seeds, raw outputs, limitations, and run metadata are linked
from `space/pages/index.md`.

## Scope limitations

- This is an independent clean-room audit, not an author-maintained
  implementation or an author endorsement.
- C3 and C5 are finite numerical claims. The exact author seeds, coefficient
  draws, metric conventions, and v1 code/data realization are not available,
  so nearby clean-room numbers are not promoted to verification or
  falsification.
- C6 uses a pinned mirror because the official LMSYS file is gated in the
  compute environment. Cohort invariants and row counts match, but byte
  identity is not established.
- A `VERIFIED` result means the stated contract passed the repository’s
  evidence checks; it does not imply that every theorem assumption or every
  paper number has been independently proven.
- The current arXiv record links the authors’ implementation. This repository
  documents an independent audit and its evidence routes; it is not an
  endorsement or a replacement for the authors’ code.

## Citation

~~~bibtex
@article{frauen2026nonparametric,
  title         = {Nonparametric LLM Evaluation from Preference Data},
  author        = {Frauen, Dennis and Deviyani, Athiya and van der Schaar, Mihaela and Feuerriegel, Stefan},
  journal       = {arXiv preprint arXiv:2601.21816},
  year          = {2026},
  note          = {Accepted at ICML 2026}
}
~~~

## Thank you

Thank you to Dennis Frauen, Athiya Deviyani, Mihaela van der Schaar, and
Stefan Feuerriegel for developing DMLRank and for making the paper,
claim structure, and implementation trail available for independent study.
Their work gives the reproduction community a useful test case for
evidence-first evaluation of statistical claims in LLM leaderboards.

## Attribution

The repository’s approved publication history is attributed to
`MachineLearning-Nerd <MachineLearning-Nerd@users.noreply.github.com>`.
The final branch set uses `main` plus the clean `audit/*` names above.
