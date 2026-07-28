# Claim 6 method

The fixed command is:

```text
uv sync --frozen && uv run python repro/run_all.py
```

`repro/src/arena_full.py` downloads the pinned dataset revision, asserts the
paper's full dimensions, builds the 102 covariates, and uses deterministic
two-fold cross-fitting. Each training fold tunes outcome and selection
LightGBM models with the paper's grid and two randomized draws. Selection
negatives are drawn within context at 10:1 and receive inverse-inclusion
weights.

Held-out nuisance predictions cover all 380 ordered model pairs. For each
observed pair, the analytic Jacobian of each GARS forms the EIF summand.
Bonferroni simultaneous intervals use the empirical covariance of cross-fitted
summands.

The mechanism check confirms that omitting EIF reproduces plugin widths. The
negative control reruns the whole contrast with EIF disabled; plugin and
candidate-debiased widths then coincide, ratio one, and the contract must fail.
`repro/checkers/check_claim6.py` independently reads serialized width vectors,
recomputes all ratios, and exits nonzero for an injected `n=3000` corruption.

The final run estimated 48 simultaneously useful cores and used Hugging Face
`cpu-upgrade`; 64 logical CPUs were allocated. Arena inference took `298.715`
seconds and the cumulative run took `527.503` seconds.
