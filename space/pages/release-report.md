# Release report

Previous live judged score: `0/12`

Conservative projected score range after the proposed change: **6–8/12**.

Best-supported possible new score: **8/12, forecast only—not a judge result**.

## Claim forecast

| Claim | Current points | Possible points | Confidence | Evidence status | Basis and remaining risk |
|---|---:|---:|---|---|---|
| 1 | 0 | 2 | HIGH | VERIFIED | Exact identities, independent BT/RC checks, and a rejecting control; low residual risk. |
| 2 | 0 | 2 | MEDIUM | VERIFIED | Complete finite EIF pathwise certificate and independent derivation; continuous-context asymptotic premises remain assumptions. |
| 3 | 0 | 0 | LOW | BLOCKED | Three verification routes plus falsification route completed; author realization and error definition unavailable. |
| 4 | 0 | 2 | HIGH | VERIFIED | KKT reconstruction, independent constrained optimizer, exact budget, and rejecting inverted policy. |
| 5 | 0 | 0 | LOW | BLOCKED | Oracle, pilot, acquired-data, and falsification routes completed; v1 author realization unavailable. |
| 6 | 0 | 2 | MEDIUM | VERIFIED | Literal 32,980×20 scale, stated covariates, all GARS, independent checker; mirror bytes substitute for gated official source. |

Current total score remains **0/12** until the live judge evaluates a new
revision. Claims 1, 2, 4, and 6 have changed from INCONCLUSIVE to
evidence-backed candidate verdicts. Claims 3 and 5 remain BLOCKED for the
specific reasons above.

Exact publication action after every gate passes: upload the text allowlist to
the existing Hugging Face Space `DineshAI/rHndxbqWyh` using the Hugging Face
commit API, verify the returned revision by fresh download and SHA-256, then
mirror the published text paths and reader-facing report/notebook to GitHub
`master`. No second Space will be created.

## Compute and provenance

All formal experiments used one pinned `uv` environment and fixed command
`uv sync --frozen && uv run python repro/run_all.py`. Long or uncertain work
used HF `cpu-upgrade` with 64 logical CPUs exposed; no GPU was used. Per-claim
Git SHAs, seeds, useful-core estimates, actual allocations, and runtimes are
inline on each claim page and downloadable in each `run_metadata.json`.
Provider billing cost was not exposed in run logs and is therefore not
invented.

The baseline root was frozen before research changes. The experiment tree
descended through exact-theory checks, two Table 1 interpretations and audits,
three Table 2 interpretations and falsification, a literal-scale Arena run, an
independent Arena checker, and the cumulative evidence bundle.

Winning branch: `orx/cumulative-evaluator-evidence-bundle`; Git SHA
`1c7b41b21a4c7ef1fb99821b33bef01316633dcd`; run
`863614e1-3ab9-44aa-8c1b-8c2f60fe48ff`. The final HF `cpu-upgrade` allocation
was 64 logical CPUs, with 48 cores estimated simultaneously useful before
launch. It ran for 604.796 seconds. The exact terminal output is
[`final-cumulative-run.log`](../evidence/final-cumulative-run.log).

Historical judged files remain reachable under `historical/judged-1e8c465/`.
The current canonical pages label the previous unsupported summary exactly
“Historical rejected baseline.”
The machine-readable [subset check](../evidence/historical-subset.json) found
zero missing old paths and records exact SHA-256 copies of every historical
page that was superseded at its canonical path.
