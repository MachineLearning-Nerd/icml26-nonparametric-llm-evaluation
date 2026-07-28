# Claim 4 method and derivation

The A-optimal objective separates as the expectation of `q/pi`. Its interior
Lagrangian derivative is `-q/pi^2 + lambda c`; setting it to zero gives
`pi=sqrt(q/(lambda c))`. Convexity of `q/pi` and the box KKT conditions clip
that interior solution to `[alpha,1]`. Monotonicity in `lambda` yields the
unique multiplier that makes a feasible nontrivial budget bind.

`repro/src/exact_theory.py` implements that derivation, solves the multiplier
by bisection, and compares it to an independently parameterized multi-start
SLSQP optimization. It audits the budget and interior KKT ratios. The negative
control inverts informativeness and must have a worse objective.

Fixed command: `uv sync --frozen && uv run python repro/run_all.py`.
