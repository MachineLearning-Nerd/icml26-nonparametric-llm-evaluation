# Evidence conclusion — DMLRank

This repository provides evidence-backed support for four of the six pinned
claims in “Nonparametric LLM Evaluation from Preference Data” (arXiv
2601.21816). Claims 1, 2, 4, and 6 pass their repository evidence contracts.
Claims 3 and 5 are `BLOCKED`, not falsified, because the exact author-controlled
finite-run realizations are not available.

| Claim | Current conclusion |
|---|---|
| C1 — GARS special cases | Verified by algebraic identities, BT recovery, Rank Centrality stationarity, Jacobian checks, and a rejecting control |
| C2 — efficient cross-fitted EIF | Verified by a finite MAR pathwise certificate and an independent theorem-logic audit; continuous regularity assumptions remain scoped limitations |
| C3 — Table 1 | Blocked after p=2, p=5, metric-definition, and falsification routes could not identify the author’s exact finite-run scale |
| C4 — A-optimal policy | Verified by KKT derivation, exact budget, independent SLSQP comparison, and an inverted-information control |
| C5 — Table 2 | Blocked after oracle, learned, acquired-data, and falsification routes could not recover the exact v1 author realization |
| C6 — Chatbot Arena | Verified at literal 32,980-context/20-model scale with an independent serialized-evidence checker; source byte identity remains unproven |

The full claim contracts and raw evidence are in `.openresearch/artifacts/`.
The evaluator-visible matrix is in `space/pages/index.md`. The final report
and branch history explain how each result was produced.
