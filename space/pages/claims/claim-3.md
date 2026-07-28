# Claim 3 — Table 1 Borda simulation

**Verdict: BLOCKED · confidence: LOW · expected points: 0**

## Exact claim and source

Table 1, Section 7.1, and Appendices H–J report, at `n=1000`, plugin
error/coverage `0.38±0.08` / `0.17±0.09` and debiased error/coverage
`0.15±0.03` / `0.94±0.06`. The finite table is a claim about one particular
author-generated NonlinearTie experiment, not a universal statement over all
admissible coefficient draws.

The [source audit](../../evidence/claim3/source_audit.md) records material
ambiguities: the main text says `p=5`, Appendix Table 5 says `p=2`; the error
norm/scale, DGP seed, coefficient distributions, and synthetic fold count are
not disclosed. The anonymous archive returned HTTP 410 and the later named
repository was unavailable.

## Four materially different routes

| Route | Interpretation / method | Plugin error / coverage | Debiased error / coverage | Result |
|---|---|---|---|---|
| 1 | Appendix `p=2`; 100 runs; 1M-context truth; cross-fitted LightGBM | `0.02288±0.00202` / `0.04±0.0386` | `0.01504±0.00147` / `0.97±0.0336` | Direction aligns; exact intervals do not |
| 2 | Main-text `p=5`; otherwise same contract | `0.02880±0.00317` / `0.06±0.0468` | `0.01462±0.00154` / `0.99±0.0196` | Direction aligns; exact errors do not |
| 3 | Independent metric audit over L1/L2/L∞/MAE/MSE/RMSE | no standard unscaled metric matches both paper error intervals | — | No defensible metric reconciliation |
| 4 | Dedicated assumption-satisfying falsification audit | paper intervals are mathematically feasible | impossible `1.08±0.09` coverage rejected | No valid counterexample |

Raw downloads: [route 1](../../evidence/claim3/routes/route1_p2_raw.json),
[route 2](../../evidence/claim3/routes/route2_p5_raw.json),
[route 3](../../evidence/claim3/routes/route3_raw.json), and
[route 4](../../evidence/claim3/routes/route4_raw.json). Code:
[`table1_borda.py`](../../evidence/code/repro/src/table1_borda.py),
[`claim3_metric_audit.py`](../../evidence/code/repro/src/claim3_metric_audit.py),
and [`claim3_falsification.py`](../../evidence/code/repro/src/claim3_falsification.py).

Controls: omitting EIF is rejected by coverage/error direction; corrupting one
coordinate changes the recomputed metric; impossible coverage is rejected.
Every verifier exits nonzero on its corrupted evidence.

Fixed command:

```text
uv sync --frozen && uv run python repro/run_all.py
```

Seeds: DGP `260121816`, repetitions `260221816`–`260221915`. Routes 1/2 used
HF `cpu-upgrade`, 64 logical CPUs, and 506.574/501.392 s; routes 3/4 were
single-core local checks of 0.022 s/0.000016 s. Exact metadata:
[`run_metadata.json`](../../evidence/claim3/run_metadata.json).

Unblocker: the exact public author revision and Table 1 seed/config, or the raw
100-run output plus the error-definition code. A clean-room mismatch cannot
falsify this particular finite realization.
