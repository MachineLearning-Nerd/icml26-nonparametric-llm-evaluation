# Claim 5 source audit

The judged claim uses the v1 Table 2: at `n=1500`, `beta=2000`, and 15 runs,
MSE times 100 is A-optimal versus random `0.108 +/- 0.041` vs
`0.133 +/- 0.060` (Borda), `2.565 +/- 1.046` vs `2.910 +/- 1.143` (BT),
and `0.010 +/- 0.008` vs `0.017 +/- 0.007` (Rank Centrality).
Appendix J identifies BTMisspec `K=3`, `p=2`, `gamma=1`.

The current arXiv PDF is materially revised: it says 50 runs and reports
`0.130/0.141`, `2.861/2.974`, and `0.017/0.020`. This campaign retains the
judge's v1 values as the exact contract and treats the revision as a source
risk, not permission to move the target.

Route 1 uses the exact DGP equations and an oracle `mu` in both policy design
and EIF estimation. This isolates whether the closed-form allocation itself
reduces MSE under the published setup. It is not a reproduction of an
unavailable learned-nuisance realization, so numerical agreement is required
before it can verify the empirical claim.
