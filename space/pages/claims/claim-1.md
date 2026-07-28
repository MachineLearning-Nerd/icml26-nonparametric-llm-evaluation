# Claim 1 — GARS special cases

**Verdict: VERIFIED · confidence: HIGH**

## Exact claim and source

Section 4.1 Eq. (3), Section 4.2 Eqs. (4)–(6), and Appendix C define
\(\theta=E[F(\mu(X))]\) over ordered-pair binary preference probabilities and
state that Borda, the centered Bradley–Terry projection, and Rank Centrality
are special cases. This verification tests all three identities, not only
ranking agreement.

Assumptions audited: valid probabilities in `(0,1)`, both ordered directions,
centered BT strengths, irreducible row-stochastic RC transition matrix, and
the paper's normalization. See the [source audit](../../evidence/claim1/source_audit.md)
and [machine-readable contract](../../evidence/claim1/claim_contract.json).

## Reproducible result

| Check | Observed | Acceptance |
|---|---:|---:|
| BT recovery max absolute residual | `2.498e-16` | `<2e-9` |
| Independent constrained BT residual | `4.441e-16` | `<2e-9` |
| RC stationarity residual | `8.327e-17` | `<2e-9` |
| Borda ranking identity | `true` | `true` |
| RC ranking identity | `true` | `true` |
| Wrong forward-only BT control separation | `0.248955` | `>1e-3` |

The executable implementation is
[`exact_theory.py`](../../evidence/code/repro/src/exact_theory.py). Raw output:
[`exact_theory.json`](../../evidence/claim1/raw/exact_theory.json).
The independent constrained least-squares and stationarity calculations do not
reuse the claimed closed forms.

Fixed command:

```text
uv sync --frozen && uv run python repro/run_all.py
```

Git `1e584bb4319f785fba3790f7119aaa5d6f29fa74`; seeds 0–11; estimated useful
cores 1; HF `cpu-upgrade`; actual allocation 64 logical CPUs because the
inherited cumulative suite was uncertain and exceeded five minutes; checker
1.872 s, cumulative 245.003 s. Full metadata:
[`run_metadata.json`](../../evidence/claim1/run_metadata.json).

Limit: this verifies the exact finite identities and their implementation; it
does not assert that every possible ranking functional is GARS. See
[`limitations_and_deviations.md`](../../evidence/claim1/limitations_and_deviations.md).
