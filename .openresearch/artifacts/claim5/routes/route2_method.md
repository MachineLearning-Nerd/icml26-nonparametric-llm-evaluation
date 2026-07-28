# Claim 5 route 2 method

Section 7 says that the acquisition comparison labels 1,500 contexts under
both A-optimal and random policies. Appendix Table 5 instead lists
`nctx=3000` for optimal data collection. Route 2 tests the defensible
interpretation that the first 1,500 contexts form an independent random pilot
for nuisance fitting and the second 1,500 are the acquisition sample.

A global binary LightGBM classifier is trained on all labeled ordered pairs
from the pilot. It uses the paper's judge/data-collection hyperparameter grid,
30 randomized configurations, three-fold cross-validation, and pair indices
as categorical features. Its predictions on fresh contexts define each
GARS-specific A-optimal policy and supply the sample-split nuisance values for
the EIF estimator. Separate budget-matched random acquisition uses the same
contexts and outcome uniforms.

The strict v1 numerical contract is unchanged. A budget-matched policy derived
from inverted information weights must have a worse variance objective, while
independent checks enforce the budget, objective ordering, and nonnegative
influence-variance weights. The verifier exits nonzero when any required check
or paper interval fails.

The randomized tuning runs parallel fits and is estimated to use 64 CPU cores
with uncertain runtime beyond five minutes. It must therefore run on Hugging
Face `cpu-upgrade`; actual allocation and runtime are printed in the result.
