"""Independent final-status checker for the six-claim evidence bundle."""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
ARTIFACTS = ROOT / ".openresearch/artifacts"


def _load(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def validate(*, corrupt: bool = False) -> dict[str, object]:
    exact = _load(ARTIFACTS / "claim1/raw/exact_theory.json")
    claim3_routes = [
        _load(ARTIFACTS / "claim3/routes/route1_p2_raw.json"),
        _load(ARTIFACTS / "claim3/routes/route2_p5_raw.json"),
        _load(ARTIFACTS / "claim3/routes/route3_raw.json"),
        _load(ARTIFACTS / "claim3/routes/route4_raw.json"),
    ]
    claim5_routes = [
        _load(ARTIFACTS / f"claim5/routes/route{route}_raw.json")
        for route in range(1, 5)
    ]
    claim6 = _load(ARTIFACTS / "claim6/raw/final_with_width_vectors.json")
    claim6_checker = _load(
        ARTIFACTS / "claim6/raw/independent_checker.json"
    )
    if corrupt:
        claim6 = copy.deepcopy(claim6)
        claim6["n_contexts"] = 3000

    errors: dict[str, list[str]] = {str(index): [] for index in range(1, 7)}

    claim1 = exact.get("claim_1", {})
    c1_checks = claim1.get("checks", {}) if isinstance(claim1, dict) else {}
    if claim1.get("status") != "VERIFIED":
        errors["1"].append("exact-theory status is not VERIFIED")
    if float(c1_checks.get("bt_recovers_centered_strengths_max_abs", 1)) >= 2e-9:
        errors["1"].append("BT reconstruction exceeds tolerance")
    if float(c1_checks.get("rc_stationarity_max_abs", 1)) >= 2e-9:
        errors["1"].append("Rank Centrality stationarity exceeds tolerance")
    if float(c1_checks.get("negative_control_wrong_bt_difference", 0)) <= 0.1:
        errors["1"].append("wrong-symmetrization control was not separated")

    claim2 = exact.get("claim_2", {})
    c2_checks = claim2.get("checks", {}) if isinstance(claim2, dict) else {}
    if claim2.get("status") != "VERIFIED":
        errors["2"].append("exact-theory status is not VERIFIED")
    if float(c2_checks.get("max_pathwise_identity_abs", 1)) >= 1e-12:
        errors["2"].append("pathwise identity exceeds tolerance")
    if float(c2_checks.get("negative_control_max_identity_error", 0)) <= 0.01:
        errors["2"].append("omitted-IPW control was not rejected")

    if (
        len(claim3_routes) != 4
        or claim3_routes[2].get("route") != 3
        or claim3_routes[3].get("route") != 4
        or "five-covariate" not in str(
            claim3_routes[1].get("source_interpretation", "")
        )
    ):
        errors["3"].append("four distinct routes are not present")
    if any(route.get("status") != "BLOCKED" for route in claim3_routes):
        errors["3"].append("every Claim 3 route must remain BLOCKED")
    if claim3_routes[-1].get("valid_assumption_satisfying_counterexample") is not False:
        errors["3"].append("falsification route did not reject exact counterexample")
    if not claim3_routes[-1].get("unblocker"):
        errors["3"].append("external unblocker is missing")

    claim4 = exact.get("claim_4", {})
    c4_checks = claim4.get("checks", {}) if isinstance(claim4, dict) else {}
    if claim4.get("status") != "VERIFIED":
        errors["4"].append("exact-theory status is not VERIFIED")
    if float(c4_checks.get("budget_abs_error", 1)) >= 1e-10:
        errors["4"].append("budget does not bind")
    if float(c4_checks.get("independent_slsqp_max_abs", 1)) >= 1e-6:
        errors["4"].append("independent optimizer disagrees")
    if float(c4_checks.get("negative_control_objective_gap", 0)) <= 1:
        errors["4"].append("inverted-information control was not rejected")

    if [route.get("route") for route in claim5_routes] != [1, 2, 3, 4]:
        errors["5"].append("four distinct routes are not present")
    if any(route.get("status") != "BLOCKED" for route in claim5_routes):
        errors["5"].append("every Claim 5 route must remain BLOCKED")
    if claim5_routes[-1].get("valid_exact_source_counterexamples") != []:
        errors["5"].append("falsification route claims an exact counterexample")
    if claim5_routes[-1].get("statistically_supported_clean_room_reversals") != []:
        errors["5"].append("clean-room reversal audit is inconsistent")
    if not claim5_routes[-1].get("unblocker"):
        errors["5"].append("external unblocker is missing")

    exact_claim6 = {
        "status": "VERIFIED",
        "raw_rows": 33000,
        "n_contexts": 32980,
        "k_models": 20,
        "feature_dimension": 102,
    }
    for field, expected in exact_claim6.items():
        if claim6.get(field) != expected:
            errors["6"].append(
                f"{field}: expected {expected!r}, observed {claim6.get(field)!r}"
            )
    gars = claim6.get("gars", {})
    if not isinstance(gars, dict) or set(gars) != {
        "borda",
        "bt",
        "rank_centrality",
    }:
        errors["6"].append("three GARS summaries are not present")
    else:
        for name, summary in gars.items():
            if not (
                summary.get("contract_passed") is True
                and float(
                    summary.get(
                        "plugin_to_debiased_median_width_ratio", 1
                    )
                )
                < 0.5
                and summary.get("negative_control_disabled_eif_rejected")
                is True
            ):
                errors["6"].append(f"{name}: result/control contract failed")
    if claim6_checker.get("passed") is not True:
        errors["6"].append("independent checker did not pass")

    verdicts = {
        "1": "VERIFIED",
        "2": "VERIFIED",
        "3": "BLOCKED",
        "4": "VERIFIED",
        "5": "BLOCKED",
        "6": "VERIFIED",
    }
    return {
        "corruption_injected": corrupt,
        "errors": errors,
        "passed": not any(errors.values()),
        "verdicts": verdicts,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--corrupt", action="store_true")
    args = parser.parse_args()
    output = validate(corrupt=args.corrupt)
    print(
        "EVIDENCE_BUNDLE_CHECKER=" + json.dumps(output, sort_keys=True),
        flush=True,
    )
    return 0 if output["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
