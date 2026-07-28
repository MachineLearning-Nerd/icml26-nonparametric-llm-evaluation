# Claim 3 method: four routes

The fixed command is:

```text
uv sync --frozen && uv run python repro/run_all.py
```

Four routes were completed because confidence remained LOW. Route 1 uses
Appendix Table 5's `p=2`; route 2 uses Section 7.1's contradictory `p=5`.
Both compute truth over one million fresh contexts and run 100 deterministic
`n=1000` repetitions with the published NonlinearTie equations, known
propensities, two-fold cross-fitting, and 30-draw/three-fold-CV LightGBM
tuning.

Route 3 recomputes L1, L2, Linf, MAE, MSE, and RMSE directly from route-2 raw
vectors. Route 4 is the mandatory falsification route: it restates the exact
finite-experiment claim and tests whether any divergence is an
assumption-satisfying counterexample.

Routes 1 and 2 use Hugging Face `cpu-upgrade`; the deterministic metric and
falsification audits are single-core, sub-five-minute local tasks.
