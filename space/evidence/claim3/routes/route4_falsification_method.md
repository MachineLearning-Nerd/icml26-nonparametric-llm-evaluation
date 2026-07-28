# Route 4: dedicated falsification

The exact claim is a Monte Carlo summary of one author-controlled experiment,
not a universal statement over all NonlinearTie parameter draws. A valid
falsification must therefore match the author DGP realization, nuisance
configuration, error definition, folds, and repetitions before a numerical
contradiction counts.

This route independently checks whether the reported errors or coverages violate
the mathematical range of three Borda scores, whether the coverage means are
attainable on a 100-run grid, and whether the current arXiv PDF contradicts the
older Table 1. It then audits every assumption needed to treat Routes 1 or 2 as
a counterexample. An impossible coverage of `1.08 +/- 0.09` is the negative
control and must be rejected.

If the numbers are mathematically feasible and the candidate counterexample
lacks any defining author condition, the only honest result is BLOCKED, not
FALSIFIED. The script exits nonzero in that case.
