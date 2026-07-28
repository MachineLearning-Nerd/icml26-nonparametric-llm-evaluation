# Claim 5 route 3 method

This route reads Appendix Table 5's `nctx=3000` as two independent datasets:
1,500 contexts acquired under the A-optimal policy and 1,500 under the random
policy. Exact synthetic preference probabilities are used only to design the
A-optimal policies. This reflects a common simulation-oracle evaluation of an
acquisition rule and is different from route 2's historical pilot.

For each policy dataset, two-fold cross-fitting trains a global binary
LightGBM classifier on labeled ordered pairs from the training contexts and
predicts every ordered pair in the held-out contexts. Tuning is repeated
inside each training fold using the paper's data-collection grid, 30
randomized draws, and three-fold cross-validation. The EIF summands use only
out-of-fold nuisance predictions and the known acquisition probabilities.

The v1 contract remains unchanged: all three A-optimal means must beat random,
all six Monte Carlo intervals must overlap the judged paper intervals, and
all implementation checks must pass. A budget-matched inverted-information
policy is the negative control.

The route performs 120 randomized hyperparameter searches and is estimated to
use 64 CPU cores for ten or more minutes. It must run on Hugging Face
`cpu-upgrade`; the terminal result records actual allocation and runtime.
