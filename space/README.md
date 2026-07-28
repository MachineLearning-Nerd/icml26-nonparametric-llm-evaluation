---
title: "Nonparametric LLM Evaluation — Reproduction Evidence"
emoji: 🎯
colorFrom: yellow
colorTo: red
sdk: static
pinned: false
tags:
 - trackio
 - trackio-logbook
 - open-experiment
 - icml2026-repro
 - paper-rHndxbqWyh
---

# Nonparametric LLM Evaluation — current reproduction

This is the canonical evaluator entrypoint for the current verification of
arXiv `2601.21816`. The fixed command is:

```text
uv sync --frozen && uv run python repro/run_all.py
```

Current evidence supersedes the judged revision `1e8c465bf2e2573b8693e61e600f83dad4505ffa`.
That revision scored 0/12 because it exposed assertions without code or raw
outputs. Its files are preserved under
[Historical rejected baseline](historical/judged-1e8c465/README.md).

## Current verdicts

| Claim | Current verdict | Confidence | Canonical evidence |
|---|---|---|---|
| 1 · GARS special cases | VERIFIED | HIGH | [Claim 1](pages/claims/claim-1.md) |
| 2 · Cross-fitted EIF theorem | VERIFIED | MEDIUM | [Claim 2](pages/claims/claim-2.md) |
| 3 · Table 1 finite simulation | BLOCKED | LOW | [Claim 3](pages/claims/claim-3.md) |
| 4 · A-optimal policy formula | VERIFIED | HIGH | [Claim 4](pages/claims/claim-4.md) |
| 5 · Table 2 finite simulation | BLOCKED | LOW | [Claim 5](pages/claims/claim-5.md) |
| 6 · Full-scale Arena intervals | VERIFIED | MEDIUM | [Claim 6](pages/claims/claim-6.md) |

`BLOCKED` is not a pass. Claims 3 and 5 each have three independent
verification routes and a fourth falsification route; neither route sequence
could recover the undisclosed author realization or establish a valid
assumption-satisfying counterexample.

Start with the [current evidence index](pages/index.md), which includes the
visibility matrix, exact source anchors, raw downloads, executable code,
checker/control results, limitations, CPU allocation, runtime, seeds, and Git
revisions.
