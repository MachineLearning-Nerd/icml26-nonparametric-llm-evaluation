# Claim 3 source audit

Table 1 reports the Borda row at `n=1000` over 100 runs: plugin error
`0.38 ± 0.08` and coverage `0.17 ± 0.09`; debiased error `0.15 ± 0.03`
and coverage `0.94 ± 0.06`. The caption identifies the uncertainty as 95%
Monte Carlo intervals.

Appendix J specifies the NonlinearTie simulator, `K=3`, `p=2`,
`X ~ Uniform([0,1]^p)`, selection base `0.3`, selection mixture `0.1`,
outcome minimum mass `0.05`, propensity bounds `[0.05,0.5]`, and a
one-million-context ground-truth calculation. Appendix I specifies a shared
global LightGBM outcome classifier with pair indices as categorical features
and 30-draw randomized tuning with three-fold CV. The propensity is known in
this synthetic DGP.

Two ambiguities are material. Section 7.1 says five covariates while Appendix
Table 5 says `p=2`. Route 1 followed the appendix and produced much smaller
errors than the table. This second route follows the explicit main-text
five-covariate interpretation. The paper does not state the synthetic
cross-fitting fold count, DGP seed, or distributions of every coefficient
whose scale is given. The cited author archive returned HTTP 410 and no public
mirror was found on 2026-07-28. The later arXiv PDF names a GitHub repository,
but that URL also returns “Repository not found.” This clean-room route uses
two folds (the minimum allowed by Theorem 5.1) and explicitly fixes all
otherwise unavailable draws. It tests statistical agreement, not seed identity.
