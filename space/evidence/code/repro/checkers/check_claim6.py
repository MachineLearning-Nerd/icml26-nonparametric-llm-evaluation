"""Independent checker for the serialized full-scale Claim 6 result.

This module intentionally does not import the estimator implementation. It
validates only the machine-readable output contract and recomputes the
reported interval-width ratios from their component medians.
"""

from __future__ import annotations

import argparse
import copy
import json
import math
from pathlib import Path
import statistics


EXPECTED_FILE_SHA256 = (
    "d1b51b6052343ea0ef3762f9b1a6607146769d673e35a2b318cc77175daddd4d"
)
EXPECTED_CONTENT_SHA256 = (
    "96ca2fb0caf42adfaa9ba7439d7175232f4e88970795bd6048b524a916a395fe"
)
EXPECTED_GARS = {"borda", "bt", "rank_centrality"}


def validate(result: dict[str, object]) -> list[str]:
    errors: list[str] = []

    exact_fields = {
        "status": "VERIFIED",
        "raw_rows": 33000,
        "n_contexts": 32980,
        "k_models": 20,
        "feature_dimension": 102,
        "cross_fitting_folds": 2,
        "tuning_iterations": 2,
        "negative_samples_per_positive": 10,
        "dataset_file_sha256": EXPECTED_FILE_SHA256,
        "dataset_lfs_sha256": EXPECTED_FILE_SHA256,
        "normalized_core_content_sha256": EXPECTED_CONTENT_SHA256,
    }
    for field, expected in exact_fields.items():
        if result.get(field) != expected:
            errors.append(
                f"{field}: expected {expected!r}, observed {result.get(field)!r}"
            )

    feature_audit = result.get("feature_audit")
    if not isinstance(feature_audit, dict):
        errors.append("feature_audit: missing or not an object")
    else:
        if feature_audit.get("toxicity_nonzero_count") != 32980:
            errors.append("feature_audit: toxicity is not present for all contexts")
        if not (
            float(feature_audit.get("toxicity_max", 0))
            > float(feature_audit.get("toxicity_min", 0))
            >= 0
        ):
            errors.append("feature_audit: toxicity is constant or invalid")
        if int(feature_audit.get("turn_unique_count", 0)) < 2:
            errors.append("feature_audit: turn covariate is constant or missing")

    jacobians = result.get("jacobian_finite_difference_max_abs_errors")
    if not isinstance(jacobians, dict) or set(jacobians) != EXPECTED_GARS:
        errors.append("jacobians: missing one or more GARS checks")
    elif max(float(value) for value in jacobians.values()) >= 1e-5:
        errors.append("jacobians: finite-difference error exceeds 1e-5")

    gars = result.get("gars")
    if not isinstance(gars, dict) or set(gars) != EXPECTED_GARS:
        errors.append("gars: expected exactly Borda, BT, and Rank Centrality")
        return errors

    for name in sorted(EXPECTED_GARS):
        summary = gars[name]
        if not isinstance(summary, dict):
            errors.append(f"{name}: summary is not an object")
            continue
        plugin_width = float(summary.get("plugin_width_median", math.nan))
        debiased_width = float(summary.get("debiased_width_median", math.nan))
        plugin_widths = summary.get("plugin_width_by_model")
        debiased_widths = summary.get("debiased_width_by_model")
        recorded_ratio = float(
            summary.get("plugin_to_debiased_median_width_ratio", math.nan)
        )
        if not (
            isinstance(plugin_widths, list)
            and isinstance(debiased_widths, list)
            and len(plugin_widths) == len(debiased_widths) == 20
        ):
            errors.append(f"{name}: expected 20 plugin and debiased widths")
            continue
        plugin_widths = [float(value) for value in plugin_widths]
        debiased_widths = [float(value) for value in debiased_widths]
        recomputed_ratio = statistics.median(
            plugin / max(debiased, 1e-5)
            for plugin, debiased in zip(plugin_widths, debiased_widths)
        )
        if not (
            all(value > 0 for value in plugin_widths)
            and all(value > 0 for value in debiased_widths)
            and recorded_ratio < 0.5
            and bool(summary.get("contract_passed"))
        ):
            errors.append(f"{name}: scientific width contract did not pass")
        if not math.isclose(
            plugin_width,
            statistics.median(plugin_widths),
            rel_tol=1e-12,
            abs_tol=1e-12,
        ):
            errors.append(f"{name}: plugin median does not match raw widths")
        if not math.isclose(
            debiased_width,
            statistics.median(debiased_widths),
            rel_tol=1e-12,
            abs_tol=1e-12,
        ):
            errors.append(f"{name}: debiased median does not match raw widths")
        if not math.isclose(
            recorded_ratio, recomputed_ratio, rel_tol=1e-12, abs_tol=1e-12
        ):
            errors.append(
                f"{name}: recorded ratio {recorded_ratio} != "
                f"recomputed {recomputed_ratio}"
            )
        if not (
            summary.get("mechanism_omit_eif_matches_plugin") is True
            and float(
                summary.get(
                    "mechanism_omit_eif_width_max_abs_difference_from_plugin",
                    math.inf,
                )
            )
            == 0
        ):
            errors.append(f"{name}: omit-EIF mechanism audit did not match plugin")
        if not (
            math.isclose(
                float(
                    summary.get(
                        "negative_control_disabled_eif_width_ratio", math.nan
                    )
                ),
                1.0,
                rel_tol=0,
                abs_tol=1e-12,
            )
            and summary.get("negative_control_disabled_eif_contract_passed")
            is False
            and summary.get("negative_control_disabled_eif_rejected") is True
        ):
            errors.append(f"{name}: disabled-EIF negative control was not rejected")

    return errors


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("outputs/arena_full_results.json"),
    )
    parser.add_argument(
        "--corrupt",
        action="store_true",
        help="Corrupt the scale field; used only by the checker self-test.",
    )
    args = parser.parse_args()

    result = json.loads(args.input.read_text(encoding="utf-8"))
    if args.corrupt:
        result = copy.deepcopy(result)
        result["n_contexts"] = 3000

    errors = validate(result)
    output = {
        "checker": "independent serialized-contract reconstruction",
        "corruption_injected": args.corrupt,
        "errors": errors,
        "passed": not errors,
    }
    print("CLAIM6_CHECKER=" + json.dumps(output, sort_keys=True), flush=True)
    return 0 if not errors else 1


if __name__ == "__main__":
    raise SystemExit(main())
