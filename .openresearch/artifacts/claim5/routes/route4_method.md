# Claim 5 route 4 falsification method

The exact target is the judged v1 Table 2 finite simulation result, not a
universal theorem over every possible BTMisspec coefficient draw. A valid
counterexample must therefore use the exact author-side simulation identity
and produce a statistically supported reversal of at least one of the three
reported A-optimal improvements.

The audit loads the raw 15-repetition outputs from routes 1–3, recomputes every
mean, and evaluates every paired A-optimal-minus-random difference with both a
95% paired-t interval and an exhaustive `2^15` sign-flip test. A reversal is
scientifically supported only when the interval is wholly above zero and the
two-sided sign-flip p-value is below 0.05. It is an exact falsification only if
the author v1 seed, coefficient draws, code revision, and raw configuration
also match.

An independent checker must exactly reproduce stored summaries and detect a
`+0.1` corruption. The negative control injects an impossible negative MSE and
must be rejected by the raw-evidence schema.

This deterministic audit uses one CPU core and is expected to finish within
five minutes, so it is routed through the local backend. The current author
archive is unavailable and the paper does not publish the needed source
identity fields; absent a new exact-source artifact, the honest expected
outcome is `BLOCKED`, not `FALSIFIED`.
