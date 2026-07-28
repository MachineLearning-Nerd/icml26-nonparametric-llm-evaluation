# Claim 3 evaluation

**Verdict: BLOCKED. Confidence: LOW.**

Route 1 (`p=2`) observed plugin error/coverage `0.02288/0.04` and debiased
`0.01504/0.97`. Route 2 (`p=5`) observed `0.02880/0.06` and
`0.01462/0.99`. The qualitative coverage correction is strong, but neither
route reproduces the paper's error magnitudes. Route 3 found that no standard
unscaled vector-error definition reconciles both paper error rows. Route 4
found no valid assumption-satisfying counterexample.

The exact author realization, metric scale, and code revision are unavailable,
so the finite Table 1 report is neither verified nor falsified. Raw files are
`routes/route1_p2_raw.json`, `route2_p5_raw.json`, `route3_raw.json`, and
`route4_raw.json`.
