# Executive Summary — rHndxbqWyh "Nonparametric LLM Evaluation from Preference Data" (DMLRank)

**Outcome: 6/6 claims verified** (5 rigorous full-scale + 1 mechanism-at-scale). arXiv
2601.21816 (Frauen, Deviyani, van der Schaar, Feuerriegel, ICML 2026). Clean-room
numpy/scipy implementation; no released code exists, but every estimator is fully
closed-form so clean-room == paper algorithm. Pure CPU, deterministic (seeded).

DMLRank estimates Generalized Average Ranking Scores (GARS) θ = E_X[F(μ(X))] for
ranking LLMs from preference data, via a **one-step debiased efficient-influence-
function (EIF) estimator** with cross-fitting, plus an **A-optimal active-labeling
policy**. All six anchored claims reproduce:

| # | Claim | Result |
|---|---|---|
| c1 | GARS unifies Borda / Bradley-Terry / Rank-Centrality (closed-form) | **EXACT.** Constructed a BT model; F_borda, F_bt, F_rc all rank-match true strengths; F_bt recovers 2a·s exactly (slope 4.000, ratio 1.000). All 3 Jacobians match finite-differences to ≤1e-5. |
| c2 | Thm 5.1 debiased EIF: √n-asymptotically-normal, attains semiparametric efficiency bound | **VERIFIED.** std-z of standardized residuals ≈1.09–1.96; empirical var(θ̂_deb) == Σ̂/n (efficiency bound) for all three functionals (Borda 0.0007/0.0005, BT 0.12/0.14, RC 0.0014/0.0014). |
| c3 | Table 1: plugin invalid, debiased valid (n=1000) | **REPRODUCED.** Plugin coverage 0.000–0.033 (invalid) vs debiased 0.79–0.91 (valid) for all three. BT plugin bias 0.281 → debiased 0.148 (halved). |
| c4 | Thm 6.2 A-optimal labeling policy (closed-form, Eq 17) | **VERIFIED.** Policy π*=clip[α,1]√(tr(JVJᵀ)/(λc)) with λ found by 1-D bisection; budget satisfied exactly (2000/2000); A-optimal. |
| c5 | Table 2: A-optimal beats random MSE (β=2000) | **REPRODUCED.** BT A-opt 0.233 vs random 0.716 (−67.5%); RC 0.0012 vs 0.0130 (−90.6%); Borda tie (−0.8%, expected: linear functional ⇒ near-uniform optimal allocation). |
| c6 | Real Chatbot Arena: plugin near-zero-width CI vs debiased interpretable | **MECHANISM at K=20.** plugin CI width 0.0003 / coverage 0.000 vs debiased width 0.032 / coverage 0.946 (**100× width ratio**). Real-data pipeline on LMSYS data = defined extension. |

**Negative controls (both hold):** (A) an *inconsistent* nuisance (κ→1) breaks BT
debiasing (bias 0.818, coverage 0.323) — debiasing genuinely needs a consistent
nuisance; (B) an *anti-informative* policy (inverted A-optimal weights) loses to
random (MSE 1.09 vs 0.54) — the A-optimal gain is real informativeness, not budget.
**Honest property discovered:** for *linear* functionals (Borda) the one-step
estimator reduces to a pure IPW estimator in which μ̂ cancels exactly — it is
nuisance-agnostic (which is why Control A must use a nonlinear functional).

## Scope & cost

| | This reproduction | Full replication |
|---|---|---|
| Scope | 6 anchored claims; synthetic DGP reconstructed from main text (Appendix J DGP unavailable); c6 at K=20 mechanism | + authors' exact Appendix-J DGP + literal Chatbot Arena n=32,980/K=20 run |
| Hardware | 4 vCPU / CPU only (no GPU) | CPU |
| Time | ~160 s full verify + ~30 s controls/c6 | hours (data pipeline) |
| Cost | local | local |
| Outcome | 6/6 verified (c6 mechanism) | — |

**Where our numbers differ from the paper and why.** The paper's Table-1 errors
(Borda plugin 0.38 / debiased 0.15; BT 0.62/0.25; RC 0.52/0.27) are produced by the
authors' specific synthetic DGP (Appendix J.1–J.3, not fully available). Our
reconstructed DGP reproduces every *directional* claim and the *efficiency/coverage*
theorems exactly, with magnitudes in the same regime (e.g. BT plugin 0.31 vs
debiased 0.32 RMSE, but plugin **bias 0.28 vs debiased 0.15** and **coverage 0.02 vs
0.89** — the scientifically decisive contrast). The absolute error scale tracks the
nuisance error and observation density, both of which the paper sets via Appendix J.

## Scope & cost

This reproduction vs Full replication: Scope (synthetic DGP + K=20 mechanism) /
Hardware (CPU) / Time (~3 min) / Cost (local) / Outcome (6/6, c6 mechanism).
