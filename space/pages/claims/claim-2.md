# Claim 2 — efficient cross-fitted EIF

**Verdict: VERIFIED · confidence: MEDIUM**

## Exact claim and source

Theorem 5.1 Eqs. (9)–(13) and Appendices B.1–B.2 state that, under iid
sampling, missing at random, positivity, smoothness/moment conditions,
cross-fitting, and product-rate nuisance convergence, the debiased estimator
has the displayed efficient influence function, is asymptotically normal at
root-\(n\) rate, and attains the semiparametric efficiency bound.

The [source audit](../../evidence/claim2/source_audit.md) enumerates every
assumption. The [contract](../../evidence/claim2/claim_contract.json) deliberately
does not treat one Gaussian-looking simulation as proof.

## Machine-checkable theorem calibration

An independently reconstructed saturated observed-data model exhausts the
finite state space and checks the EIF mean-zero and pathwise derivative
identities for every nuisance perturbation:

| Check | Observed | Acceptance |
|---|---:|---:|
| EIF mean absolute value | `0.0` | `<2e-9` |
| Maximum pathwise identity residual | `2.776e-17` | `<2e-9` |
| Omitted-IPW control residual | `0.051316` | `>1e-3` |

The checker therefore certifies the algebraic EIF used in Theorem 5.1 and
rejects the intended incorrect score. Code:
[`exact_theory.py`](../../evidence/code/repro/src/exact_theory.py). Raw output:
[`exact_theory.json`](../../evidence/claim2/raw/exact_theory.json). The
independent derivation and cross-fit construction are documented in
[`method.md`](../../evidence/claim2/method.md).

Fixed command:

```text
uv sync --frozen && uv run python repro/run_all.py
```

Git `1e584bb4319f785fba3790f7119aaa5d6f29fa74`; deterministic finite paths
0–11; estimated useful cores 1; HF `cpu-upgrade`; actual 64 logical CPUs;
checker 1.872 s, cumulative 245.003 s.

Why confidence is MEDIUM: the certificate reconstructs the general EIF on a
complete finite MAR model. It cannot mechanically certify that arbitrary
continuous-context nuisance learners satisfy the theorem's rate and
regularity premises; those remain explicit assumptions, not empirical
conclusions. Details:
[`limitations_and_deviations.md`](../../evidence/claim2/limitations_and_deviations.md).
