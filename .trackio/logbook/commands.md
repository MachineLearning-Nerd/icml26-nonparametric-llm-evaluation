# Captured runs (clean relative commands; run from paper dir with .venv active)

source .venv/bin/activate

# Claim 1 (GARS unifies Borda/BT/RC + Jacobians) — EXACT
trackio logbook run --page "Claim 1" -- python repro/tests/test_c1_gars.py

# Claims 2/3/4/5 (Table 1 & Table 2; debiased EIF + A-optimal policy)
trackio logbook run --page "Claims 2-5" -- python repro/src/verify.py

# Claim 6 mechanism (K=20; plugin near-zero-width CI vs debiased)
trackio logbook run --page "Claim 6" -- python repro/src/c6_demo.py

# Negative controls
trackio logbook run --page "Negative controls" -- python repro/tests/test_controls.py
