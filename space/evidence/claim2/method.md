# Claim 2 method and independent derivation

Differentiating `theta(P)=E_P[F(mu_P(X))]` gives the direct
`F(mu)-theta` term plus the chain rule `J_F(mu)` applied to the
conditional-mean path. Under MAR, the latter is represented by the
inverse-propensity residual
`A/pi(X) * J_F(mu(X)) * (Y-mu(X))`. Conditional expectation makes the
residual term mean zero.

`repro/src/exact_theory.py` reconstructs these identities on a complete
saturated finite observed-data model and verifies every elementary
distributional path. The maximum pathwise-derivative discrepancy is
`2.78e-17`. Omitting inverse propensity weighting produces discrepancy
`0.051316` and is rejected.

Cross-fitting makes the first-order empirical-process term conditionally
evaluable; Neyman orthogonality leaves the nuisance product remainder in
Theorem 5.1; its rate assumptions make that remainder `o_p(n^-1/2)`. The
i.i.d. finite-moment CLT yields the Gaussian law. The reconstructed score is
the canonical gradient in the nonparametric MAR tangent space, so its
covariance is the efficiency bound.

Fixed command: `uv sync --frozen && uv run python repro/run_all.py`.
