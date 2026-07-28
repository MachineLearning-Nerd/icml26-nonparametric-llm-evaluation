"""Claim 3 route 3: independently audit Table 1's undefined error metric."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys
import time

import numpy as np
from scipy.stats import norm


ROOT = Path(__file__).resolve().parents[2]
RAW = (
    ROOT
    / ".openresearch/artifacts/claim3/routes/route2_p5_raw.json"
)
PAPER = {
    "plugin": {"mean": 0.38, "half_width": 0.08},
    "debiased": {"mean": 0.15, "half_width": 0.03},
}


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _metrics(point: np.ndarray, truth: np.ndarray) -> dict[str, float]:
    absolute = np.abs(point - truth)
    squared = absolute**2
    return {
        "l1": float(absolute.sum()),
        "l2": float(np.sqrt(squared.sum())),
        "rmse": float(np.sqrt(squared.mean())),
        "mae": float(absolute.mean()),
        "linf": float(absolute.max()),
        "mse": float(squared.mean()),
    }


def _summary(values: np.ndarray) -> dict[str, float]:
    return {
        "mean": float(values.mean()),
        "monte_carlo_95_half_width": float(
            norm.ppf(0.975) * values.std(ddof=1) / np.sqrt(len(values))
        ),
    }


def _overlap(observed: dict[str, float], paper: dict[str, float]) -> bool:
    return (
        observed["mean"] - observed["monte_carlo_95_half_width"]
        <= paper["mean"] + paper["half_width"]
        and paper["mean"] - paper["half_width"]
        <= observed["mean"] + observed["monte_carlo_95_half_width"]
    )


def main() -> int:
    started = time.monotonic()
    source = json.loads(RAW.read_text(encoding="utf-8"))
    truth = np.asarray(source["truth"], dtype=float)
    summaries: dict[str, dict[str, dict[str, float]]] = {}
    matches: dict[str, bool] = {}
    scale_intersections: dict[str, object] = {}

    for metric in ("l1", "l2", "rmse", "mae", "linf", "mse"):
        summaries[metric] = {}
        scale_ranges = []
        for estimator in ("plugin", "debiased"):
            values = np.asarray(
                [
                    _metrics(
                        np.asarray(row[f"{estimator}_point"], dtype=float),
                        truth,
                    )[metric]
                    for row in source["raw_repetitions"]
                ]
            )
            summaries[metric][estimator] = _summary(values)
            paper = PAPER[estimator]
            scale_ranges.append(
                (
                    (paper["mean"] - paper["half_width"]) / values.mean(),
                    (paper["mean"] + paper["half_width"]) / values.mean(),
                )
            )
        matches[metric] = all(
            _overlap(summaries[metric][estimator], PAPER[estimator])
            for estimator in PAPER
        )
        lower = max(interval[0] for interval in scale_ranges)
        upper = min(interval[1] for interval in scale_ranges)
        scale_intersections[metric] = {
            "plugin_range": list(scale_ranges[0]),
            "debiased_range": list(scale_ranges[1]),
            "common_scale_exists": lower <= upper,
            "common_scale_intersection": [lower, upper] if lower <= upper else None,
        }

    recorded_l2 = {
        estimator: float(source["observed"][f"{estimator}_error"]["mean"])
        for estimator in PAPER
    }
    independent_l2_error = max(
        abs(summaries["l2"][estimator]["mean"] - recorded_l2[estimator])
        for estimator in PAPER
    )

    corrupted = json.loads(json.dumps(source["raw_repetitions"][0]))
    corrupted["plugin_point"][0] += 0.1
    original_metric = _metrics(
        np.asarray(source["raw_repetitions"][0]["plugin_point"]), truth
    )["l2"]
    corrupted_metric = _metrics(
        np.asarray(corrupted["plugin_point"]), truth
    )["l2"]
    corruption_detected = abs(corrupted_metric - original_metric) > 1e-3

    matching_metrics = [name for name, value in matches.items() if value]
    passed = (
        bool(matching_metrics)
        and independent_l2_error < 1e-12
        and corruption_detected
    )
    result = {
        "schema_version": 1,
        "claim_id": 3,
        "route": 3,
        "status": "VERIFIED" if passed else "BLOCKED",
        "question": (
            "Can a standard, unscaled vector-error definition reconcile both "
            "Table 1 Borda error rows with the route-2 raw estimates?"
        ),
        "paper": PAPER,
        "standard_metric_summaries": summaries,
        "standard_metric_matches_both_paper_intervals": matches,
        "matching_standard_metrics": matching_metrics,
        "undisclosed_common_scale_audit": scale_intersections,
        "independent_checker": {
            "recorded_l2_means": recorded_l2,
            "max_abs_recomputation_error": independent_l2_error,
            "passed": independent_l2_error < 1e-12,
        },
        "negative_control": {
            "name": "corrupt one plugin coordinate by +0.1",
            "original_first_l2": original_metric,
            "corrupted_first_l2": corrupted_metric,
            "corruption_detected": corruption_detected,
        },
        "raw_source": str(RAW.relative_to(ROOT)),
        "raw_source_sha256": _sha256(RAW),
        "runtime_seconds": round(time.monotonic() - started, 3),
        "limitation": (
            "The paper does not define Table 1's error norm or disclose a "
            "scale multiplier. A post-hoc scale is not accepted as verification."
        ),
    }
    output = ROOT / "outputs/claim3_metric_audit.json"
    output.parent.mkdir(exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print("CLAIM3_METRIC_AUDIT=" + json.dumps(result, sort_keys=True), flush=True)
    return 0 if passed else 1


if __name__ == "__main__":
    sys.exit(main())
