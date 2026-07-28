# Claim 4 — A-optimal labeling policy

**Verdict: VERIFIED · confidence: HIGH**

## Exact claim and source

Definition 6.1, Theorem 6.2 Eq. (17), and Assumption B.1 characterize the
binding-budget optimum as

\[
\pi^*_{jk}(x)=\operatorname{clip}_{[\alpha,1]}
\sqrt{\operatorname{tr}(J_{jk}V_{jk}J_{jk}^{T})/(\lambda_Ac_{jk})}.
\]

The contract requires positive influence weights and costs, a feasible clipped
set, a binding budget, agreement with an optimizer, and KKT satisfaction.
[Source audit](../../evidence/claim4/source_audit.md);
[contract](../../evidence/claim4/claim_contract.json).

## Reproducible result

| Check | Observed | Acceptance |
|---|---:|---:|
| Budget absolute error | `4.441e-16` | `<2e-9` |
| Independent multi-start SLSQP difference | `5.855e-8` | `<2e-6` |
| Interior KKT ratio range | `1.332e-15` | `<2e-9` |
| Negative control: inverted-informativeness objective gap | `12.3831` | positive/rejected |

The checker independently solves for the Lagrange multiplier and then compares
against constrained numerical optimization; it does not choose the budget or
tolerance from the theorem formula. Code:
[`exact_theory.py`](../../evidence/code/repro/src/exact_theory.py). Raw output:
[`exact_theory.json`](../../evidence/claim4/raw/exact_theory.json).

Fixed command:

```text
uv sync --frozen && uv run python repro/run_all.py
```

Git `1e584bb4319f785fba3790f7119aaa5d6f29fa74`; seeds 0–11; estimated useful
cores 1; HF `cpu-upgrade`; actual 64 logical CPUs; checker 1.872 s, cumulative
245.003 s. The proof/KKT reconstruction and boundary limitations are in
[`method.md`](../../evidence/claim4/method.md) and
[`limitations_and_deviations.md`](../../evidence/claim4/limitations_and_deviations.md).
