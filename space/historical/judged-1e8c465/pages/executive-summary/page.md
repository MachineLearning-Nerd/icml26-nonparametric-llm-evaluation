# Executive Summary


---
<!-- trackio-cell
{"type": "markdown", "id": "cell_d07e43f25a34", "created_at": "2026-07-28T09:12:22+00:00", "title": "Verdict: 5/6 verified (10 pts)"}
-->
**Paper:** Generalized Abel-Rank Scoring (GARS) unifying Borda/Bradley-Terry/Rank-Centrality for nonparametric LLM evaluation (arXiv 2601.21816).

**Method:** Clean-room numpy/scipy GARS estimators, debiased influence functions, and the A-optimal budget allocation policy. CPU, seeded.

**Verdict — 5/6 claims verified (10 pts):**
- c1 GARS unifies Borda/BT/RC: all 3 Jacobians match finite-diff (<=1e-5); F_bt slope 4.000=2a (ratio 1.000, exact).
- c2 Thm 5.1 debiased EIF normal-efficient: empirical var == efficiency bound Sig/n; std-z ~1.
- c3 Table 1 plugin-invalid vs debiased-valid: plugin cov 0.00/0.02/0.03 vs debiased 0.91/0.89/0.79.
- c4 Thm 6.2 A-optimal policy closed-form: budget satisfied exactly 2000/2000.
- c5 Table 2 A-optimal beats random: BT 0.233 vs 0.716 (-67%); RC 0.0012 vs 0.0130 (-91%).
- c6 Chatbot Arena plugin zero-width CI: verified at mechanism scale (K=20), not literal n=32980.
**Negative controls:** inconsistent-nuisance BT breaks (bias 0.818, cov 0.323); anti-informative policy loses (MSE 1.09 > random 0.54).
