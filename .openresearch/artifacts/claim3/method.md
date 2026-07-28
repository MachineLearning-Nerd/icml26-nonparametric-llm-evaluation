# Claim 3 method

The fixed command is:

```text
uv sync --frozen && uv run python repro/run_all.py
```

`repro/src/table1_borda.py` regenerates the published nonlinear-tie DGP,
computes truth over one million fresh contexts, and runs 100 independent
`n=1000` datasets. Each repetition draws all ordered-pair selection indicators
from the known contextual propensity and outcomes from the ternary preference
distribution. A global LightGBM classifier is tuned in each training fold and
predicts all held-out ordered pairs. The paper's analytic weighted-Borda
Jacobian forms the EIF correction.

The primary error is the Euclidean distance between the estimated and true
three-vector. Coverage is joint across all three coordinates using the
Bonferroni simultaneous interval from Appendix H. The omitted-EIF plugin is the
negative control. The result exits nonzero unless its four Monte Carlo intervals
overlap the paper intervals and its directional/coverage checks pass.

Pre-launch estimate: 16--64 useful cores, 30--120 minutes uncertain runtime,
less than 4 GB memory. This requires Hugging Face `cpu-upgrade`; it must not run
locally.
