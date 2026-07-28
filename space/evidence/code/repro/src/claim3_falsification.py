"""Claim 3 mandatory route 4: dedicated, assumption-aware falsification audit."""

from __future__ import annotations

import json
import math
from pathlib import Path
import sys
import time


ROOT = Path(__file__).resolve().parents[2]
PAPER = {
    "plugin_error": {"mean": 0.38, "half_width": 0.08},
    "plugin_coverage": {"mean": 0.17, "half_width": 0.09},
    "debiased_error": {"mean": 0.15, "half_width": 0.03},
    "debiased_coverage": {"mean": 0.94, "half_width": 0.06},
}


def _interval(entry: dict[str, float]) -> tuple[float, float]:
    return (
        entry["mean"] - entry["half_width"],
        entry["mean"] + entry["half_width"],
    )


def _within(interval: tuple[float, float], bounds: tuple[float, float]) -> bool:
    return bounds[0] <= interval[0] <= interval[1] <= bounds[1]


def main() -> int:
    started = time.monotonic()
    max_borda_l2_error = math.sqrt(3)
    feasibility = {
        "plugin_error_interval_within_borda_l2_bounds": _within(
            _interval(PAPER["plugin_error"]), (0.0, max_borda_l2_error)
        ),
        "debiased_error_interval_within_borda_l2_bounds": _within(
            _interval(PAPER["debiased_error"]), (0.0, max_borda_l2_error)
        ),
        "plugin_coverage_interval_within_probability_bounds": _within(
            _interval(PAPER["plugin_coverage"]), (0.0, 1.0)
        ),
        "debiased_coverage_interval_within_probability_bounds": _within(
            _interval(PAPER["debiased_coverage"]), (0.0, 1.0)
        ),
        "reported_coverage_means_attainable_on_100_run_grid": all(
            abs(PAPER[name]["mean"] * 100 - round(PAPER[name]["mean"] * 100))
            < 1e-12
            for name in ("plugin_coverage", "debiased_coverage")
        ),
        "current_arxiv_pdf_retains_same_table1_values": True,
    }

    # Routes 1 and 2 are candidates for contradiction, but a valid
    # falsification of a reported empirical result must reproduce every
    # condition defining that experiment. These author-controlled fields are
    # absent from both paper versions and their code URLs are inaccessible.
    counterexample_assumption_audit = {
        "published_dgp_equations_matched": True,
        "K_3_and_n_1000_matched": True,
        "100_repetitions_matched": True,
        "author_dgp_seed_matched": False,
        "author_coefficient_distributions_matched": False,
        "author_error_definition_and_scale_matched": False,
        "author_crossfit_fold_count_matched": False,
        "author_code_revision_matched": False,
    }
    valid_counterexample = all(counterexample_assumption_audit.values())

    impossible_control = dict(PAPER)
    impossible_control["debiased_coverage"] = {"mean": 1.08, "half_width": 0.09}
    control_rejected = not _within(
        _interval(impossible_control["debiased_coverage"]), (0.0, 1.0)
    )
    independent_bounds_checker = {
        "borda_coordinate_bounds": [0.0, 1.0],
        "three_coordinate_l2_error_bound": max_borda_l2_error,
        "all_paper_intervals_feasible": all(feasibility.values()),
    }

    falsified = (
        valid_counterexample
        or not independent_bounds_checker["all_paper_intervals_feasible"]
    )
    result = {
        "schema_version": 1,
        "claim_id": 3,
        "route": 4,
        "status": "FALSIFIED" if falsified else "BLOCKED",
        "exact_claim": (
            "For Table 1 Borda at n=1000 over 100 runs, plugin error/coverage "
            "are 0.38+/-0.08 and 0.17+/-0.09, while debiased values are "
            "0.15+/-0.03 and 0.94+/-0.06."
        ),
        "assumptions_domain_quantifiers": {
            "domain": (
                "The paper's single author-generated NonlinearTie experiment, "
                "K=3, ternary outcomes, n=1000, learned cross-fitted LightGBM "
                "nuisance, known synthetic propensity, 100 repetitions."
            ),
            "quantifier": (
                "A Monte Carlo summary of that particular experiment, not a "
                "universal statement over all admissible DGP seeds."
            ),
            "source_ambiguities": [
                "Section 7.1 says p=5 while Appendix Table 5 says p=2.",
                "The error norm and any scale multiplier are undefined.",
                "DGP seed, several coefficient distributions, and synthetic fold count are omitted.",
                "The v1 anonymous archive returns HTTP 410; the later named GitHub URL returns Repository not found.",
            ],
        },
        "paper": PAPER,
        "mathematical_feasibility_checks": feasibility,
        "counterexample_candidate_assumption_audit": counterexample_assumption_audit,
        "valid_assumption_satisfying_counterexample": valid_counterexample,
        "independent_checker": independent_bounds_checker,
        "negative_control": {
            "name": "replace debiased coverage by impossible 1.08+/-0.09",
            "rejected": control_rejected,
        },
        "falsification_conclusion": (
            "No valid falsification was established. Routes 1 and 2 diverge "
            "numerically but cannot contradict a particular reported Monte "
            "Carlo experiment without its omitted realization and metric."
        ),
        "unblocker": (
            "A public author-code revision with the exact Table 1 seed/config, "
            "or the raw 100-run Table 1 outputs and error-definition code."
        ),
        "runtime_seconds": round(time.monotonic() - started, 6),
    }
    output = ROOT / "outputs/claim3_falsification.json"
    output.parent.mkdir(exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print("CLAIM3_FALSIFICATION=" + json.dumps(result, sort_keys=True), flush=True)
    return 0 if falsified and control_rejected else 1


if __name__ == "__main__":
    sys.exit(main())
