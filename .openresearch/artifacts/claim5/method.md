# Claim 5 route 1 method

The fixed command is:

```text
uv sync --frozen && uv run python repro/run_all.py
```

For each of 15 repetitions, the verifier generates 1,500 common contexts from
the published BTMisspec simulator and computes all three GARS and Jacobians.
For each GARS separately it solves the clipped square-root policy by bisection
to expected budget 2,000 and compares it to uniform probability
`2000 / (1500 * 3 * 2)`. Common outcome and selection uniforms reduce
paired-comparison noise. Both policies use the exact EIF estimator with oracle
outcome probabilities. Ground truth uses one million fresh contexts.

The independent check evaluates the A-optimal variance objective and exact
budget. The random policy is the negative control. The verifier exits nonzero
unless A-optimal improves all three MSEs and every numerical interval overlaps
the judged v1 Table 2 interval.

Before launch, the workload was estimated to benefit from up to 8 CPU cores
and to have uncertain one-core runtime because it evaluates 15 repetitions,
one million truth contexts, and context-specific Jacobians. It is therefore
routed through Hugging Face `cpu-upgrade`; the verifier records the actual
logical CPU allocation and wall-clock runtime.
