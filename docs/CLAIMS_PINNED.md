# Pinned claims — rHndxbqWyh “Nonparametric LLM Evaluation from Preference Data” (DMLRank)

arXiv 2601.21816v2 — Dennis Frauen, Athiya Deviyani, Mihaela van der Schaar,
and Stefan Feuerriegel (ICML 2026). This repository is an independent
claim-level audit. The current paper HTML links an authors’ implementation;
the evidence routes here remain separately documented and contract-driven.

Notation: K models, covariates X, ordered pair (j,k), preference outcome Y_{jk} over
C=3 categories {1=j-win, 2=tie, 3=k-win}; μ_{jk}(x)=E[Y_{jk}|X=x]∈Δ^{C-1}; F maps μ to a score vector; θ=E[F(μ(X))].

## c1 — GARS unifies Borda / Bradley-Terry / Rank-Centrality (closed-form). §4, Eq 4-6.
θ = E[F(μ(X))].
- Borda (Eq 5): F_j(μ(x)) = 1/(2(K-1)) Σ_{k≠j}(μ_{jk1}(x) + μ_{kj2}(x)).
- BT log-odds projection (Eq 3): with incidence B, L0=BᵀB, ℓ(μ)=(logit(μ_{jk1})/2 + logit(μ_{kj2})/2)_{j<k},
  H from Eq2, F = H(HᵀL0H)⁻¹HᵀBᵀℓ(μ) → recovers BT strengths when BT holds.
- Rank Centrality (Eq 6): T_{ij}=(μ_{ji1}+μ_{ij2})/Σ_{ℓ≠i}(μ_{ℓi1}+μ_{iℓ2}); F=(I−Tᵀ+𝟙𝟙ᵀ)⁻¹𝟙 (stationary dist).
VERIFY: construct μ from a known BT model; F_BT recovers log-strengths (up to shift); F_Borda, F_RC match definitions.

## c2 — Thm 5.1 debiased EIF estimator: √n-asymptotic-normal + attains semiparametric efficiency bound. Eq 9-15.
EIF: ϕ(O)=F(μ(X))−θ + Σ_{j≠k} S_{jk}/π_{jk}(X) · J_{jk}(μ(X))(Y_{jk}−μ_{jk}(X)),  J_{jk}=∇_{μ_{jk}}F.
One-step (Eq 11): θ̂_EIF=(1/n)Σ[F(μ̂(x_i)) + Σ_{j≠k} s_{i,jk}/π̂_{jk}(x_i)·J_{jk}(μ̂(x_i))(y_{i,jk}−μ̂_{jk}(x_i))].
Asymp (Eq 13): √n(θ̂_EIF−θ)→N(0,Σ), Σ=E[ϕϕᵀ]. Cross-fitting: V folds, nuisance fit out-of-fold.
Σ̂=(1/n)Σ ϕ̂_i ϕ̂_iᵀ (Eq14). CI ellipsoid Eq15 χ²_{d,1-α}.
VERIFY (Monte Carlo): bias→0; EIF coverage≈0.95 (valid); var(θ̂_EIF)≈Σ̂=efficiency bound; standardized normality.

## c3 — Table 1 (n=1000): plugin invalid, debiased valid.
Borda: plugin err 0.38±0.08 cov 0.17±0.09; debiased err 0.15±0.03 cov 0.94±0.06.
BT:    plugin err 0.62±0.13 cov 0.09; debiased err 0.25±0.05 cov 0.90.
RC:    plugin err 0.52±0.12 cov 0.12; debiased err 0.27±0.06 cov 0.91.

## c4 — Thm 6.2 A-optimal labeling policy (closed-form). Eq 17.
V_{jk}=Var(Y_{jk}|X=x)=Diag(μ_{jk})−μ_{jk}μ_{jk}ᵀ.
π*_{jk}(x)=clip[α,1] √(tr(J_{jk}V_{jk}J_{jk}ᵀ)/(λ_A c_{jk})); λ_A≥0 s.t. E[Σ_{j≠k}c_{jk}π*_{jk}(X)]=β (1-D bisection).
VERIFY: budget holds exactly; π* minimizes tr(Σ(π)) (A-opt) vs random & grid; KKT hold.

## c5 — Table 2 (β=2000, n=1500 contexts, 15 runs, ×10²): A-optimal beats random MSE.
Borda: A-opt 0.108±0.041 vs rand 0.133±0.060. BT: 2.565±1.046 vs 2.910±1.143. RC: 0.010±0.008 vs 0.017±0.007.

## c6 — Real data: Chatbot Arena n=32980, K=20, toxicity + TF-IDF covariates.
Plugin CIs near-zero-width (invalid) vs debiased interpretable. (Best-effort on public LMSYS slice.)

## Verification method
Pure CPU Monte-Carlo with a known DGP (ground-truth θ computable exactly via numerical integration).
Nuisance μ̂ via cross-fitted multinomial logistic + controlled judge-noise to induce the plugin bias the paper exhibits.
