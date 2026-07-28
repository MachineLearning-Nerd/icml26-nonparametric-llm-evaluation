# Claim 1 method

`repro/src/exact_theory.py` constructs one ordered-pair tensor from a valid
Bradley--Terry latent-strength model and evaluates all three published maps.
It checks Borda and Rank Centrality ordering, reconstructs centered
Bradley--Terry strengths by the paper's constrained projection, and verifies
the Rank Centrality transition rows and stationarity equation.

An independently formed constrained least-squares projection must agree with
the analytic BT result. Analytic Jacobians are checked separately. The
negative control discards the paper's reverse-direction symmetrization and
must differ materially.

Fixed command: `uv sync --frozen && uv run python repro/run_all.py`.
