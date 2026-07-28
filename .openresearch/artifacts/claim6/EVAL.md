# Claim 6 evaluation

**Verdict: VERIFIED. Confidence: MEDIUM.**

The hard scale gate passes at 33,000 raw rows, 32,980 deduplicated contexts,
20 models, and 102 features. All 32,980 toxicity values are nonzero and
nonconstant; turn spans 1--25. Median plugin/debiased width ratios are
`0.040182` (Borda), `0.023531` (BT), and `0.014134` (Rank Centrality).
Disabled-EIF controls have ratio exactly one and are rejected for all three.

The independent checker exits 0 on the raw result and 1 on an injected
downscaled corruption. Confidence remains MEDIUM because official gated-file
byte identity and author preprocessing defaults cannot be established.
