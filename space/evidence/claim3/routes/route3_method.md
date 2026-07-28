# Route 3: error-definition audit

The Table 1 caption says “estimation error” but does not define its norm or
display a multiplier. Route 3 reads the exact 100 per-repetition point estimates
from Route 2 and independently computes L1, L2, RMSE, MAE, L-infinity, and MSE.
For each single standard metric, both the plugin and debiased Monte Carlo
intervals must overlap the corresponding paper intervals.

The checker also reports, but does not accept, post-hoc common scale ranges.
An independent recomputation must recover Route 2's recorded L2 means within
`1e-12`. A corruption control adds `0.1` to one coordinate and must be detected.
The script exits nonzero if no disclosed, unscaled standard metric reconciles
the claim.
