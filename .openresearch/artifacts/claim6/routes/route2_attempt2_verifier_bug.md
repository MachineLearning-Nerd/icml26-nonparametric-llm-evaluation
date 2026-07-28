# Route 2 attempt 2 — scientific contrast passed, verifier control rejected

- Run: `f42a22dc-fabb-4628-a596-24b79118e282`
- Commit: `ef92b7a4f6187be040d6609bb9af365dd9344242`
- Fixed command: `uv sync --frozen && uv run python repro/run_all.py`
- Selected compute: Hugging Face `cpu-upgrade`
- Estimated useful cores: 48
- Actual allocation: 64 logical CPUs
- Arena runtime: 248.635 seconds
- Terminal status: `failed` because the executable verifier returned 1

The literal scale gate passed: 33,000 raw rows, 32,980 deduplicated contexts,
20 models, and 102 features. The pinned parquet SHA-256 equaled its recorded
LFS SHA-256. Median plugin/debiased simultaneous-interval width ratios were
0.040182 (Borda), 0.023531 (Bradley–Terry), and 0.014134 (Rank Centrality), so
all three scientific contrasts passed their predeclared `< 0.5` threshold.

The result remained `BLOCKED` because the original control compared an
omit-EIF width (identical to the plugin width) with the *real* debiased width.
That calculation necessarily repeated the passing scientific contrast instead
of rerunning the full contract under corruption.

The repaired negative control disables the EIF correction for the candidate
debiased estimator too. Plugin and corrupted-debiased widths must then
coincide, giving a ratio of one and causing the `< 0.5` contract to fail. The
repair does not alter the observed data, estimators, folds, model searches,
scientific threshold, or scientific outputs.
