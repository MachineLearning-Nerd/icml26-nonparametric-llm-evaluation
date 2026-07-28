# Claim 6 source audit

Source: arXiv 2601.21816, retrieved from the ar5iv HTML on 2026-07-28.
The retrieved source SHA-256 is
`25f14e3a172cfa9427c501e9b1ed973c4a0e1549ee03a7593e5312f9e6474847`.

Section 7.2 and Appendix K specify one deduplicated comparison per
`question_id`: 32,980 contexts/comparisons, 20 models, four outcomes, and 102
features (released RoBERTa toxicity probability, a 100-dimensional
TF-IDF/truncated-SVD prompt representation, and turn index). Appendix K
specifies two-fold cross-fitting; multiclass LightGBM for outcome probabilities;
binary LightGBM propensities with 10 within-context negatives per positive and
subsampling-correcting weights; and two randomized-search iterations with
three-fold CV. Figure 3 reports 95% simultaneous intervals for weighted Borda,
Bradley--Terry, and Rank Centrality and says the plugin intervals attain
near-zero width.

The paper does not give a numerical definition of “near-zero.” The executable
contract therefore pre-registers a relative, scale-free test: the median plugin
interval width must be less than half the corresponding debiased width for
each of the three GARS. This operationalization tests the asserted collapse but
does not turn the qualitative phrase into a claimed paper constant.

The anonymous source-code URL cited by the paper returned HTTP 410 on
2026-07-28. Consequently, unpublished seeds and text-preprocessing defaults
cannot be audited; all clean-room choices are explicit in the verifier.
