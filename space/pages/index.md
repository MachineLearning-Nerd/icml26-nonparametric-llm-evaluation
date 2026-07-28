# Current evaluator-visible evidence

Paper source: ar5iv HTML retrieved 2026-07-28 with explicit browser
User-Agent, SHA-256
`25f14e3a172cfa9427c501e9b1ed973c4a0e1549ee03a7593e5312f9e6474847`.
The source audit records exact theorem/section anchors and quantifiers.

Environment: Python 3.12, repository-level `.venv`, `uv.lock` SHA-256
`95f8cd0826a479d495738fba954dc4e4a8712ce13614c21e2769667d3714c165`.
All formal nodes inherited exactly:

```text
uv sync --frozen && uv run python repro/run_all.py
```

## Evidence visibility matrix

| Claim | Canonical page | Code visible | Data inline | Raw link | Checker | Control | Exact claim tested | Reviewer verdict |
|---|---|---|---|---|---|---|---|---|
| 1 | [Claim 1](claims/claim-1.md) | Yes | Yes | Yes | Independent BT/RC identities | Wrong BT symmetrization rejected | Yes | VERIFIED |
| 2 | [Claim 2](claims/claim-2.md) | Yes | Yes | Yes | Finite pathwise certificate | Omitted IPW rejected | Yes, scoped theorem certificate | VERIFIED |
| 3 | [Claim 3](claims/claim-3.md) | Yes | Yes | Four routes | Metric recomputation | Corruption and impossible coverage rejected | Yes | BLOCKED |
| 4 | [Claim 4](claims/claim-4.md) | Yes | Yes | Yes | SLSQP plus KKT | Inverted informativeness rejected | Yes | VERIFIED |
| 5 | [Claim 5](claims/claim-5.md) | Yes | Yes | Four routes | Paired/sign-flip audit | Negative MSE corruption rejected | Yes | BLOCKED |
| 6 | [Claim 6](claims/claim-6.md) | Yes | Yes | Full vectors | Independent fresh-process checker | Disabled EIF and n=3000 corruption rejected | Yes | VERIFIED |

## Navigation

- [Claim 1 — GARS special cases](claims/claim-1.md)
- [Claim 2 — efficient cross-fitted EIF](claims/claim-2.md)
- [Claim 3 — Table 1](claims/claim-3.md)
- [Claim 4 — A-optimal policy](claims/claim-4.md)
- [Claim 5 — Table 2](claims/claim-5.md)
- [Claim 6 — Chatbot Arena](claims/claim-6.md)
- [Release report](release-report.md)
- [Blind evaluator review](red-team.md)
- [Exact commands](../commands.md)
- [Final cumulative raw log](../evidence/final-cumulative-run.log)
- [Text upload manifest](../release-manifest.sha256)
- [Historical rejected baseline](executive-summary/page.md)
