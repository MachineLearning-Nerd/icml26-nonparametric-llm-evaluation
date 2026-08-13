# Branch audit

This repository originally used `master` as its publication branch and
`orx/*` as internal experiment branches. The clean publication layout is
`main` plus the `audit/*` branches below. The audit branches are retained
because each one records a distinct evidence route or control.

| Former branch | Clean branch | Purpose |
|---|---|---|
| `master` | `main` | Reader-facing publication surface |
| `orx/baseline-judged-reproduction-plus-locked-uv-envi` | `audit/baseline-locked-env` | Freeze the baseline and locked environment |
| `orx/exact-theorem-contracts-and-analytic-checkers` | `audit/theorem-contracts` | Independent C1, C2, and C4 certificates |
| `orx/full-chatbot-arena-cross-fitted-inference` | `audit/chatbot-arena-full` | Full Chatbot Arena inference route |
| `orx/published-dgp-table-1-borda-reconstruction` | `audit/table1-borda` | First Table 1 Borda reconstruction |
| `orx/table-1-five-covariate-source-interpretation` | `audit/table1-covariate-interpretation` | Five-covariate Table 1 interpretation |
| `orx/table-1-error-definition-audit` | `audit/table1-metric-audit` | Table 1 error-definition audit |
| `orx/table-1-dedicated-falsification-audit` | `audit/table1-falsification` | Fourth C3 falsification route |
| `orx/oracle-published-dgp-table-2-calibration` | `audit/table2-oracle` | Oracle Table 2 calibration |
| `orx/pilot-learned-table-2-reconstruction` | `audit/table2-pilot-learned` | Pilot-learned Table 2 reconstruction |
| `orx/cross-fitted-acquired-data-table-2` | `audit/table2-crossfit` | Cross-fitted acquired-data Table 2 route |
| `orx/table-2-dedicated-falsification-audit` | `audit/table2-falsification` | Fourth C5 falsification route |
| `orx/cumulative-evaluator-evidence-bundle` | `audit/cumulative-evidence` | Final contracts, full-scale C6 run, controls, and evidence gate |
| `orx/arena-mirror-integrity-and-full-inference` | `audit/arena-integrity` | Arena mirror-integrity and inference audit |
| `orx/independent-claim-6-checker-and-cumulative-evide` | `audit/claim6-checker` | Independent C6 checker and cumulative evidence |

The final branch tips and remote branch list are verified after publication.
All approved commits use:

~~~text
MachineLearning-Nerd <37579156+MachineLearning-Nerd@users.noreply.github.com>
~~~

No co-author trailers are retained in the approved history.
