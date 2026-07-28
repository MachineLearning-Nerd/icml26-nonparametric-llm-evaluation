# Route 2 attempt 1 — cancelled runtime overrun

- Run: `6819e36e-bfb9-4151-bb0a-5f4b0040967b`
- Commit: `4aefd603f103cf20dcf111583f7d17b1f340d590`
- Fixed command: `uv sync --frozen && uv run python repro/run_all.py`
- Selected compute: Hugging Face `cpu-upgrade`
- Estimated requirement: 64 useful CPU cores, 10–35 minutes
- Actual allocation reported by the run: 64 logical CPUs
- Recorded wall duration: 1 hour 21 minutes
- Terminal status: `cancelled`

The cumulative exact-theory suite completed. The pinned 41.6 MB Arena mirror
downloaded, but the implementation emitted no later stage checkpoint and no
`ARENA_FULL_RESULTS` record. The run was cancelled after a preannounced grace
period because it exceeded its one-hour launch contract and remained
uninstrumented.

This attempt is **not claim evidence** and cannot pass Claim 6. The repair keeps
the data, folds, two randomized tuning draws, tuning grid, and verification
thresholds fixed. It adds flushed stage timings and avoids nested LightGBM
oversubscription by running three CV fits with 16 threads per fit.
