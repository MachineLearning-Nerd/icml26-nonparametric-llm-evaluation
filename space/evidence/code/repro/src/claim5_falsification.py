"""Claim 5 mandatory route 4: dedicated falsification audit."""

from __future__ import annotations

import copy
import itertools
import json
import math
import os
from pathlib import Path
import sys
import time

import numpy as np
from scipy.stats import t


ROOT = Path(__file__).resolve().parents[2]
ROUTE_FILES = {
    route: ROOT
    / f".openresearch/artifacts/claim5/routes/route{route}_raw.json"
    for route in (1, 2, 3)
}
GARS = ("borda", "bt", "rank_centrality")
AUTHOR_SOURCE_IDENTITY = {
    "exact_author_code_revision_available": False,
    "exact_author_dgp_seed_available": False,
    "exact_author_coefficient_draws_available": False,
    "exact_v1_raw_repetitions_available": False,
}


def _available_cpus() -> int:
    if hasattr(os, "sched_getaffinity"):
        return len(os.sched_getaffinity(0))
    return os.cpu_count() or 1


def _load_routes() -> dict[int, dict[str, object]]:
    return {
        route: json.loads(path.read_text(encoding="utf-8"))
        for route, path in ROUTE_FILES.items()
    }


def _mse_vector(
    route: dict[str, object], name: str, policy: str
) -> np.ndarray:
    values = [
        row["gars"][name][f"{policy}_mse_x100"]
        for row in route["raw_repetitions"]
    ]
    return np.asarray(values, dtype=float)


def _sign_flip_pvalue(differences: np.ndarray) -> float:
    observed = abs(float(differences.mean()))
    exceed = 0
    total = 2 ** len(differences)
    for signs in itertools.product((-1.0, 1.0), repeat=len(differences)):
        permuted = abs(
            float(np.mean(differences * np.asarray(signs)))
        )
        if permuted >= observed - 1e-15:
            exceed += 1
    return exceed / total


def _difference_audit(
    aopt: np.ndarray, random: np.ndarray
) -> dict[str, object]:
    differences = aopt - random
    mean = float(differences.mean())
    standard_error = float(
        differences.std(ddof=1) / math.sqrt(len(differences))
    )
    half_width = float(
        t.ppf(0.975, len(differences) - 1) * standard_error
    )
    pvalue = _sign_flip_pvalue(differences)
    return {
        "mean_aopt_minus_random": mean,
        "paired_t_95_interval": [
            mean - half_width,
            mean + half_width,
        ],
        "exhaustive_two_sided_sign_flip_pvalue": pvalue,
        "statistically_supported_aopt_worse": bool(
            mean - half_width > 0 and pvalue < 0.05
        ),
    }


def _schema_valid(route: dict[str, object]) -> bool:
    if len(route.get("raw_repetitions", [])) != 15:
        return False
    for name in GARS:
        for policy in ("aopt", "random"):
            values = _mse_vector(route, name, policy)
            if not np.isfinite(values).all() or np.any(values < 0):
                return False
    return True


def main() -> int:
    started = time.monotonic()
    routes = _load_routes()
    summary_recomputation: dict[str, object] = {}
    reversal_audits: dict[str, object] = {}
    maximum_summary_error = 0.0

    for route_number, route in routes.items():
        summary_recomputation[str(route_number)] = {}
        reversal_audits[str(route_number)] = {}
        for name in GARS:
            summary_recomputation[str(route_number)][name] = {}
            for policy in ("aopt", "random"):
                values = _mse_vector(route, name, policy)
                recomputed = float(values.mean())
                stored = float(
                    route["observed_mse_x100"][name][policy]["mean"]
                )
                error = abs(recomputed - stored)
                maximum_summary_error = max(
                    maximum_summary_error, error
                )
                summary_recomputation[str(route_number)][name][policy] = {
                    "recomputed_mean": recomputed,
                    "stored_mean": stored,
                    "absolute_error": error,
                }
            reversal_audits[str(route_number)][name] = (
                _difference_audit(
                    _mse_vector(route, name, "aopt"),
                    _mse_vector(route, name, "random"),
                )
            )

    exact_source_identity = all(AUTHOR_SOURCE_IDENTITY.values())
    statistically_supported_reversals = [
        {
            "route": route,
            "gars": name,
            **audit,
        }
        for route, route_audits in reversal_audits.items()
        for name, audit in route_audits.items()
        if audit["statistically_supported_aopt_worse"]
    ]
    valid_counterexamples = (
        statistically_supported_reversals
        if exact_source_identity
        else []
    )

    corrupted = copy.deepcopy(routes[3])
    original = float(
        corrupted["raw_repetitions"][0]["gars"]["borda"][
            "aopt_mse_x100"
        ]
    )
    corrupted["raw_repetitions"][0]["gars"]["borda"][
        "aopt_mse_x100"
    ] = -abs(original) - 1.0
    corrupted_control_rejected = not _schema_valid(corrupted)

    changed = copy.deepcopy(routes[3])
    changed["raw_repetitions"][0]["gars"]["borda"][
        "aopt_mse_x100"
    ] += 0.1
    changed_mean = float(
        _mse_vector(changed, "borda", "aopt").mean()
    )
    stored_mean = float(
        routes[3]["observed_mse_x100"]["borda"]["aopt"]["mean"]
    )
    independent_checker_detects_corruption = (
        abs(changed_mean - stored_mean) > 1e-6
    )

    falsified = len(valid_counterexamples) > 0
    result = {
        "schema_version": 1,
        "claim_id": 5,
        "route": 4,
        "status": "FALSIFIED" if falsified else "BLOCKED",
        "exact_claim_restatement": (
            "In the v1 Table 2 experiment, at n=1500 contexts, "
            "beta=2000 comparisons, and 15 runs, A-optimal acquisition "
            "has lower mean MSE times 100 than random for Borda, BT, and "
            "Rank Centrality, with the six reported means and 95% "
            "intervals 0.108+/-0.041 vs 0.133+/-0.060, "
            "2.565+/-1.046 vs 2.910+/-1.143, and "
            "0.010+/-0.008 vs 0.017+/-0.007."
        ),
        "assumptions_and_domain": {
            "simulator": "BTMisspec",
            "K": 3,
            "p": 2,
            "gamma": 1,
            "contexts": 1500,
            "budget": 2000,
            "repetitions": 15,
            "estimators": [
                "debiased Borda GARS",
                "debiased BT-projection GARS",
                "debiased Rank Centrality GARS",
            ],
        },
        "falsification_acceptance_rule": (
            "A candidate must reverse A-optimal versus random with a "
            "paired-t 95% interval wholly above zero and exhaustive "
            "two-sided sign-flip p<0.05, while using the exact author "
            "v1 DGP seed, coefficient draws, code revision, and raw-run "
            "configuration."
        ),
        "source_identity": {
            **AUTHOR_SOURCE_IDENTITY,
            "all_required_identity_fields_available": exact_source_identity,
        },
        "revision_audit": {
            "judged_v1": (
                "15 runs; means 0.108/0.133, 2.565/2.910, "
                "0.010/0.017"
            ),
            "current_pdf": (
                "50 runs; revised means 0.130/0.141, 2.861/2.974, "
                "0.017/0.020"
            ),
            "interpretation": (
                "The revision preserves all three directions and is not "
                "itself a falsification of the judged v1 statement."
            ),
        },
        "summary_recomputation": summary_recomputation,
        "maximum_summary_recomputation_abs_error": maximum_summary_error,
        "reversal_audits": reversal_audits,
        "statistically_supported_clean_room_reversals": (
            statistically_supported_reversals
        ),
        "valid_exact_source_counterexamples": valid_counterexamples,
        "independent_checker": {
            "all_route_schemas_valid": all(
                _schema_valid(route) for route in routes.values()
            ),
            "summary_recomputation_exact": (
                maximum_summary_error < 1e-12
            ),
            "detects_plus_0_1_corruption": (
                independent_checker_detects_corruption
            ),
        },
        "negative_control": {
            "name": "negative MSE injected into route 3 raw evidence",
            "rejected": corrupted_control_rejected,
        },
        "limitations": [
            "A clean-room DGP reversal cannot contradict the exact empirical table without the author seed, coefficient draws, and v1 raw configuration.",
            "The Table 2 statement reports one finite simulation study rather than a universal theorem over every BTMisspec coefficient draw.",
            "Failure to match the paper numbers is not a valid falsification.",
        ],
        "unblocker": (
            "A public exact v1 author archive or raw 15-run Table 2 "
            "outputs with DGP seed, coefficient draws, policy-design "
            "nuisance source, fold assignments, and MSE implementation."
        ),
        "estimated_required_cores": 1,
        "logical_cpu_allocation": _available_cpus(),
        "runtime_seconds": round(time.monotonic() - started, 3),
    }
    output = ROOT / "outputs/claim5_falsification_results.json"
    output.parent.mkdir(exist_ok=True)
    output.write_text(
        json.dumps(result, indent=2) + "\n", encoding="utf-8"
    )
    print(
        "CLAIM5_FALSIFICATION_RESULTS="
        + json.dumps(result, sort_keys=True),
        flush=True,
    )
    return 0 if falsified else 1


if __name__ == "__main__":
    sys.exit(main())
