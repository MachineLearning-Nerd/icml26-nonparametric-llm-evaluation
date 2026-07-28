# Claim 5 — Table 2 A-optimal versus random

**Verdict: BLOCKED · confidence: LOW · expected points: 0**

## Exact claim and source

The judged v1 Table 2 statement reports, for `K=3`, `p=2`, `n=1500`,
`β=2000`, `γ=1`, and 15 runs, A-optimal versus random MSE×100:
Borda `0.108±0.041` vs `0.133±0.060`; BT `2.565±1.046` vs
`2.910±1.143`; RC `0.010±0.008` vs `0.017±0.007`.

The current PDF revises Table 2 to 50 runs and different numbers. The exact
author seed, coefficient draws, v1 revision, and meaning of Appendix
`nctx=3000` are unavailable. See the
[source audit](../../evidence/claim5/source_audit.md).

## Four materially different routes

| Route | Defensible interpretation | Borda A/R | BT A/R | RC A/R | Result |
|---|---|---:|---:|---:|---|
| 1 | Oracle nuisances isolate allocation theorem | `0.00988/0.01067` | `0.3726/0.9005` | `0.00292/0.00344` | Direction all three; only 1/6 paper intervals |
| 2 | Independent 1500-context pilot + 1500 acquisition | `0.03482/0.03817` | `3.6825/3.4198` | `0.01243/0.01370` | BT reverses, not significant |
| 3 | Separate acquired data, two-fold cross-fit estimator | `0.01213/0.01691` | `1.9131/2.4464` | `0.00559/0.00325` | RC reverses, not significant |
| 4 | Exact paired-t and exhaustive sign-flip falsification audit | — | — | — | No reversal meets the valid-falsification rule |

All allocation routes meet the `β=2000` expected budget to
`4.55e-13` and independently verify A-optimal variance objective ≤ random.
The clean-room reversals cannot falsify a single undisclosed finite author
realization, and their paired intervals cross zero.

Raw downloads: [route 1](../../evidence/claim5/routes/route1_raw.json),
[route 2](../../evidence/claim5/routes/route2_raw.json),
[route 3](../../evidence/claim5/routes/route3_raw.json), and
[route 4](../../evidence/claim5/routes/route4_raw.json). Code:
[`table2_oracle.py`](../../evidence/code/repro/src/table2_oracle.py),
[`table2_learned.py`](../../evidence/code/repro/src/table2_learned.py),
[`table2_crossfit.py`](../../evidence/code/repro/src/table2_crossfit.py), and
[`claim5_falsification.py`](../../evidence/code/repro/src/claim5_falsification.py).

Controls: budget-matched uniform and inverted-information policies; route 4
injects a negative MSE and the independent checker rejects it. Verifiers exit
nonzero on corrupted evidence.

Fixed command:

```text
uv sync --frozen && uv run python repro/run_all.py
```

Seeds: DGP `260121816`, repetitions `260321816`–`260321830`. Routes 1–3 used
HF `cpu-upgrade`, actual 64 logical CPUs, scientific runtimes
20.960/75.019/358.379 s. Route 4 was a single-core local 0.707 s check.
[Exact metadata](../../evidence/claim5/run_metadata.json).

Unblocker: the author v1 code revision, exact seed/coefficient draws, and raw
15-run Table 2 configuration. Until then the honest status is BLOCKED.
