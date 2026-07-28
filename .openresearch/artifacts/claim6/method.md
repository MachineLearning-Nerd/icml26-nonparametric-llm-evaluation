# Claim 6 method

Run the fixed project command:

```text
uv sync --frozen && uv run python repro/run_all.py
```

The cumulative runner invokes `repro/src/arena_full.py`. It downloads the
pinned public dataset revision, asserts the paper's full dimensions, builds
the 102 covariates, and uses deterministic two-fold cross-fitting. In each
training fold it tunes the outcome and selection LightGBM models using the
paper's real-data grid and two randomized draws. Selection negatives are drawn
within context at 10:1 and receive inverse-inclusion weights.

Held-out nuisance predictions are evaluated for all 380 ordered model pairs.
For the observed pair, the code evaluates the analytic Jacobian of each GARS
and forms the paper's EIF summand. Bonferroni simultaneous intervals use the
empirical covariance of the cross-fitted summands. The independent numerical
audit records every fitted propensity range and tuning result. The negative
control omits the EIF correction, which must reproduce the plugin width and
must not pass the debiased-width contract.

Estimated requirement before launch: 16 useful CPU cores, uncertain runtime
of 10--35 minutes, and less than 3 GB working memory. The authorized backend is
Hugging Face `cpu-upgrade`; local execution is prohibited for this run.
