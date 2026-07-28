# Source and live-state audit

Audit date: 2026-07-28 (Asia/Kolkata).

## Paper source

- Paper: arXiv:2601.21816, “Nonparametric LLM Evaluation from Preference Data”.
- Retrieved URL: `https://ar5iv.labs.arxiv.org/html/2601.21816`.
- Retrieval method: HTTPS GET with explicit user agent
  `OpenResearch-Reproduction/1.0 (contact: reproducibility-audit)`.
- Retrieved bytes: 1,018,861.
- SHA-256: `25f14e3a172cfa9427c501e9b1ed973c4a0e1549ee03a7593e5312f9e6474847`.
- Structured cross-check: `orx paper 2601.21816 --full`.
- The paper’s code link,
  `https://anonymous.4open.science/r/NonparametricLLMEval-603E`, returned
  HTTP 410 on 2026-07-28. No author implementation was available from that
  stated source.

## Claim anchors and exact scope

1. Section 4.2, GARS definition and examples: a known differentiable map
   `F:[0,1]^(K×K×C)→R^d` defines `θ=E[F(μ(X))]`. Sections 4.1–4.2 and
   Appendix C instantiate Bradley–Terry projection, Borda, and Rank
   Centrality.
2. Theorem 5.1, equations (9)–(13): under positivity, missing at random,
   i.i.d. sampling, differentiable locally-Lipschitz `F`, finite EIF second
   moment, `V≥2`-fold cross-fitting, the stated nuisance product rates, and a
   lower-bounded fitted propensity, the one-step estimator is asymptotically
   normal and semiparametrically efficient.
3. Table 1, Borda at `n=1000`: over 100 runs, plugin error/coverage is
   `0.38±0.08 / 0.17±0.09`; debiased error/coverage is
   `0.15±0.03 / 0.94±0.06`. The table caption describes mean and 95% CIs.
4. Theorem 6.2, equation (17): if the feasible policy class is nonempty and
   selection indicators are conditionally independent (Assumption B.1), any
   A-optimal policy has the clipped square-root form for almost every `x` and
   every ordered `j≠k`; `λ_A` makes the expected budget bind.
5. Table 2: `n=1500` contexts are labeled under each policy with
   `β=2000`; ranking MSE `(×10^2)` is averaged over 15 runs. A-optimal must be
   lower than random for Borda (`0.108±0.041` vs `0.133±0.060`), BT
   (`2.565±1.046` vs `2.910±1.143`), and Rank Centrality
   (`0.010±0.008` vs `0.017±0.007`).
6. Section 7.2, Figure 3, and Appendix K: Chatbot Arena is deduplicated to one
   comparison per `question_id`, giving exactly 32,980 contexts/comparisons,
   20 models, and four outcome categories. Covariates are a RoBERTa toxicity
   probability, a 100-dimensional TF-IDF/truncated-SVD prompt representation,
   and turn index (`p=102`). The experiment uses two-fold cross-fitting and
   reports plugin and debiased simultaneous 95% intervals for all three GARS.

Theorem claims are universally/asymptotically quantified and cannot be marked
VERIFIED from a finite Monte Carlo run alone. Experimental theorem calibration
will therefore be scoped corroboration unless paired with an independent
symbolic derivation or a valid assumption-satisfying counterexample.

## Baseline repository and evaluator state

- Required and observed Git baseline: `54ccb45533064bd6e515317ccc2ab25e411ff3a0`.
- Experiment tree before setup: empty; no runs; no project command.
- Exact judged Space: `DineshAI/rHndxbqWyh` at
  `1e8c465bf2e2573b8693e61e600f83dad4505ffa`.
- Protected Space manifest location:
  `project/protected-judged-space-1e8c465/manifest.sha256` in the local
  project files directory.
- Live verdict dataset: `ICML-2026-agent-repro/verdicts`, revision
  `4bb5d4f8af26114aac45c9db69730b06de43690e`.
- Downloaded verdict file SHA-256:
  `5d3ebfc3d2fadf8fea051465afefbe0233f247c21e87da620532b54d0ea7f7bc`.
- Filtering strictly by `space_id == "DineshAI/rHndxbqWyh"` produced exactly
  one record: six `inconclusive` claims, score 0/12, judged at
  `2026-07-28T10:03:35+00:00`.

Canonical-entrypoint-only traversal of the judged Space found one current
executive-summary page. None of the six claims exposed executable code, a
pinned environment, raw data, checker output, control output, runtime/CPU
metadata, or faithful limitations. This is the frozen baseline visibility
failure to be corrected additively in a child experiment.
