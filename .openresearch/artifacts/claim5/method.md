# Claim 5 method: four routes

The fixed command is:

```text
uv sync --frozen && uv run python repro/run_all.py
```

Route 1 generates 1,500 common contexts from the published BTMisspec
simulator. For each GARS it solves the clipped square-root policy to expected
budget 2,000 and compares with a budget-matched uniform policy. Both policies
use oracle outcome probabilities and common random numbers over 15
repetitions.

Route 2 uses a separate pilot cohort to learn policy-design nuisances, then
acquires a fresh cohort. Route 3 learns two-fold nuisances separately from the
data acquired by each policy. Route 4 is the mandatory falsification route: it
recomputes every raw mean, runs paired t intervals and exhaustive sign-flip
tests for reversals, audits source identity, and injects a negative-MSE
corruption.

Routes 1--3 use Hugging Face `cpu-upgrade`. Route 4 is a deterministic
single-core local audit.
