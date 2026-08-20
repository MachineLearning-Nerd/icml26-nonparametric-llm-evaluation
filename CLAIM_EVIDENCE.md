# Claim evidence — Nonparametric LLM Evaluation from Preference Data

Paper: Nonparametric LLM Evaluation from Preference Data (DMLRank)

Paper record: arXiv 2601.21816, OpenReview rHndxbqWyh

Repository scope: independent clean-room claim audit with finite analytic
certificates, finite simulation routes, and a full-scale Chatbot Arena
analysis.

The status VERIFIED in the original evidence bundle means that the named
repository contract passed. It does not mean that every theorem premise or
every author-reported number was independently reproduced. On the stricter
paper-level accounting used here, 0 of 6 complete paper claims are verified.

## C1 — GARS contains the named ranking scores

Paper anchor: the definition of generalized average ranking scores and the
special cases for Borda, Bradley–Terry, and Rank Centrality.

Production path:

- space/evidence/code/repro/src/exact_theory.py constructs a centered
  Bradley–Terry model and
  evaluates the three ranking maps.
- repro/tests/test_c1_gars.py checks score identities, rankings, stationarity,
  and finite-difference Jacobians.
- space/evidence/claim1/raw/exact_theory.json stores the canonical result.

Controls: an intentionally wrong Bradley–Terry symmetrization is rejected.

Observed evidence: algebraic residuals are at numerical precision; the wrong
symmetrization differs by about 0.248955.

Verdict: VERIFIED_SCOPED, high confidence. This certifies the stated finite
ranking identities, not the complete surrounding statistical theory.

## C2 — Cross-fitted EIF inference

Paper anchor: Theorem 5.1 and the efficient-influence-function construction.

Production path:

- space/evidence/code/repro/src/exact_theory.py builds the finite
  missing-at-random observed-data
  model and checks the pathwise EIF identity.
- The same certificate checks mean-zero behavior and the efficient correction.
- space/evidence/claim2/raw/exact_theory.json stores the result.

Controls: an omitted-IPW correction is rejected.

Observed evidence: maximum pathwise identity error is 2.78e-17 and the
omitted-IPW control has error 0.051316.

Verdict: VERIFIED_SCOPED, medium confidence. The finite certificate is not a
proof assistant for all continuous-context regularity and asymptotic
conditions in the theorem.

## C3 — Table 1 coverage and error magnitudes

Paper anchor: the reported finite-simulation comparison between plugin and
debiased inference in Table 1.

Production path:

- space/evidence/code/repro/src/table1_borda.py reconstructs the Borda
  simulation.
- space/evidence/code/repro/src/claim3_metric_audit.py tests the competing
  error definitions.
- space/evidence/code/repro/src/claim3_falsification.py provides an
  assumption-matched
  falsification route.
- audit/table1-borda, audit/table1-covariate-interpretation,
  audit/table1-metric-audit, and audit/table1-falsification preserve the
  separate routes.

Controls: metric recomputation, impossible coverage checks, and corruption
checks are retained in the evidence bundle.

Observed evidence: the qualitative coverage direction appears in clean-room
runs, but the published error magnitudes are not reproduced. The exact author
seed, coefficient draw, metric scale, and code revision are unavailable.

Verdict: BLOCKED, low confidence. This is an evidence boundary, not a
falsification.

## C4 — A-optimal acquisition policy

Paper anchor: Theorem 6.2 and the clipped square-root policy formula.

Production path:

- space/evidence/code/repro/src/exact_theory.py derives the KKT solution and
  solves the budget by
  bisection.
- The result is compared with an independent SLSQP constrained optimizer.
- The canonical finite certificate is under space/evidence/claim4.

Controls: an inverted-information policy is rejected.

Observed evidence: budget error is 4.44e-16 and maximum SLSQP disagreement is
5.86e-8.

Verdict: VERIFIED_SCOPED, high confidence. This audits the closed-form policy
contract and its finite optimizer checks.

## C5 — Table 2 acquisition advantage

Paper anchor: the reported A-optimal-versus-random acquisition comparison in
Table 2 for the three GARS objectives.

Production path:

- space/evidence/code/repro/src/table2_oracle.py tests the oracle policy
  route.
- space/evidence/code/repro/src/table2_learned.py tests the pilot-learned
  route.
- space/evidence/code/repro/src/table2_crossfit.py tests acquired-data
  cross-fitting.
- space/evidence/code/repro/src/claim5_falsification.py tests paired/sign-flip
  evidence.
- audit/table2-oracle, audit/table2-pilot-learned,
  audit/table2-crossfit, and audit/table2-falsification preserve the routes.

Controls: negative MSE corruption and sign-flip checks are retained.

Observed evidence: the direction changes across clean-room interpretations;
the exact author v1 realization is unavailable. The oracle route improves
Bradley–Terry and Rank Centrality but is effectively tied for Borda.

Verdict: BLOCKED, low confidence. Nearby numbers are not promoted to
verification or falsification.

## C6 — Chatbot Arena interval widths

Paper anchor: the full-scale Chatbot Arena comparison between plugin and
debiased EIF intervals.

Production path:

- space/evidence/code/repro/src/arena_full.py runs the 32,980-context,
  20-model pipeline with
  102 features.
- space/evidence/code/repro/checkers/run_claim6_checker_suite.py independently
  checks serialized
  evidence in a fresh process.
- audit/chatbot-arena-full, audit/arena-integrity, and audit/claim6-checker
  preserve the full analysis and integrity checks.

Controls: disabling the EIF correction and changing the context count to
3,000 are rejected by the independent checker.

Observed evidence: median plugin-to-debiased width ratios are 0.0402 for
Borda, 0.0235 for Bradley–Terry, and 0.0141 for Rank Centrality. The pinned
mirror has the expected cohort invariants, but its byte identity to the
gated official source is not established.

Verdict: VERIFIED_SCOPED, medium confidence. This is a clean-room
full-scale evidence contract, not a claim that the gated author file was
byte-identically reproduced.

## Evidence accounting

| Measure | Result |
|---|---:|
| Scoped contracts with evidence | 4 of 6 |
| Blocked finite-result claims | 2 of 6 |
| Supported evidence points | 8 of 12 |
| Complete paper-level claims independently verified | 0 of 6 |
| Current external score claimed | No |

The exact author-controlled realizations required to unblock C3 and C5, and
the gated-source byte identity required to strengthen C6, are not inferred
from nearby clean-room results.
