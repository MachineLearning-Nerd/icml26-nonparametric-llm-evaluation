"""Claim 5 route 2: pilot-learned reconstruction of Table 2."""

from __future__ import annotations

import json
import os
from pathlib import Path
import sys
import time

from joblib import parallel_backend
from lightgbm import LGBMClassifier
import numpy as np
from scipy.special import expit, logit
from scipy.stats import norm
from sklearn.model_selection import RandomizedSearchCV

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from repro.src.table2_oracle import (
    ALPHA,
    BUDGET,
    EPS,
    K_ITEMS,
    N_CONTEXTS,
    PAPER,
    P_FEATURES,
    REPETITIONS,
    ROOT,
    SEED,
    BTMisspecDGP,
    _aopt_policy,
    _estimate,
    _overlap,
    _precompute,
    _summary,
    _truth,
)


PARAMETER_GRID = {
    "n_estimators": [100, 300, 600],
    "learning_rate": [0.03, 0.07, 0.1],
    "num_leaves": [15, 31, 63],
    "min_child_samples": [10, 30, 60],
    "subsample": [0.7, 0.9, 1.0],
    "colsample_bytree": [0.7, 0.9, 1.0],
    "reg_alpha": [0.0, 0.1, 1.0],
    "reg_lambda": [0.0, 0.1, 0.5, 1.0],
    "max_depth": [-1, 6, 10],
}
PILOT_CONTEXTS = 1500
PILOT_BASE_RATE = 0.3
PILOT_MIXING = 0.1
PILOT_MINIMUM = 0.01
PILOT_MAXIMUM = 0.5


def _available_cpus() -> int:
    if hasattr(os, "sched_getaffinity"):
        return len(os.sched_getaffinity(0))
    return os.cpu_count() or 1


def _pilot_propensity(
    dgp: BTMisspecDGP, x_value: np.ndarray
) -> np.ndarray:
    utility = (
        x_value @ dgp.linear.T
        + dgp.intercept
        + x_value[:, :2] ** 2 @ dgp.quadratic.T
        + 0.6
        * np.sin(2 * np.pi * x_value[:, [0]] + dgp.phase[None, :])
    )
    difference = utility[:, :, None] - utility[:, None, :]
    nonlinear = expit(
        logit(PILOT_BASE_RATE)
        - 0.8 * np.abs(difference)
        + 0.4 * (x_value[:, 0] - 0.5)[:, None, None]
    )
    propensity = np.clip(
        (1 - PILOT_MIXING) * PILOT_BASE_RATE
        + PILOT_MIXING * nonlinear,
        PILOT_MINIMUM,
        PILOT_MAXIMUM,
    )
    diagonal = np.arange(K_ITEMS)
    propensity[:, diagonal, diagonal] = 0.0
    return propensity


def _draw_outcomes(
    mu: np.ndarray, uniforms: np.ndarray
) -> np.ndarray:
    return (uniforms >= mu[..., 0]).astype(int)


def _training_rows(
    x_value: np.ndarray,
    selected: np.ndarray,
    outcomes: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    context, j_value, k_value = np.nonzero(selected)
    rows = np.column_stack(
        [x_value[context], j_value, k_value]
    ).astype(np.float32)
    return rows, outcomes[context, j_value, k_value]


def _fit_mu(
    rows: np.ndarray,
    labels: np.ndarray,
    seed: int,
) -> tuple[LGBMClassifier, dict[str, object]]:
    classifier = LGBMClassifier(
        objective="binary",
        n_jobs=1,
        random_state=seed,
        verbosity=-1,
        subsample_freq=1,
    )
    search = RandomizedSearchCV(
        classifier,
        PARAMETER_GRID,
        n_iter=30,
        scoring="neg_log_loss",
        cv=3,
        random_state=seed,
        n_jobs=_available_cpus(),
        refit=True,
    )
    with parallel_backend("threading", n_jobs=_available_cpus()):
        search.fit(
            rows,
            labels,
            categorical_feature=[P_FEATURES, P_FEATURES + 1],
        )
    if not np.array_equal(search.best_estimator_.classes_, np.arange(2)):
        raise AssertionError(
            f"Missing nuisance class: {search.best_estimator_.classes_.tolist()}"
        )
    return search.best_estimator_, {
        "best_params": search.best_params_,
        "best_cv_neg_log_loss": float(search.best_score_),
    }


def _predict_all_pairs(
    classifier: LGBMClassifier, x_value: np.ndarray
) -> np.ndarray:
    pairs = np.asarray(
        [
            (j_value, k_value)
            for j_value in range(K_ITEMS)
            for k_value in range(K_ITEMS)
            if j_value != k_value
        ],
        dtype=np.float32,
    )
    rows = np.column_stack(
        [
            np.repeat(x_value, len(pairs), axis=0),
            np.tile(pairs, (len(x_value), 1)),
        ]
    ).astype(np.float32)
    predicted = classifier.predict_proba(rows).reshape(
        len(x_value), len(pairs), 2
    )
    mu_hat = np.full(
        (len(x_value), K_ITEMS, K_ITEMS, 2), 0.5
    )
    for position, (j_value, k_value) in enumerate(pairs):
        mu_hat[:, int(j_value), int(k_value)] = np.clip(
            predicted[:, position], EPS, 1 - EPS
        )
    mu_hat /= mu_hat.sum(axis=-1, keepdims=True)
    return mu_hat


def _anti_policy(q_value: np.ndarray) -> np.ndarray:
    maximum = q_value.max(axis=(1, 2), keepdims=True)
    inverted = maximum - q_value
    diagonal = np.arange(K_ITEMS)
    inverted[:, diagonal, diagonal] = 0.0
    return _aopt_policy(inverted)[0]


def main() -> int:
    started = time.monotonic()
    dgp = BTMisspecDGP()
    truth = _truth(dgp)
    mse = {
        name: {"aopt": [], "random": []}
        for name in truth
    }
    raw: list[dict[str, object]] = []
    budget_errors: list[float] = []
    objective_checks: list[bool] = []
    negative_control_checks: list[bool] = []
    q_minima: list[float] = []

    random_probability = BUDGET / (
        N_CONTEXTS * K_ITEMS * (K_ITEMS - 1)
    )
    random_policy = np.full(
        (N_CONTEXTS, K_ITEMS, K_ITEMS), random_probability
    )
    diagonal = np.arange(K_ITEMS)
    random_policy[:, diagonal, diagonal] = 0.0

    for repetition in range(REPETITIONS):
        rng = np.random.default_rng(SEED + 400_000 + repetition)
        pilot_x = rng.uniform(
            0.0, 1.0, size=(PILOT_CONTEXTS, P_FEATURES)
        )
        pilot_mu = dgp.probabilities(pilot_x)
        pilot_propensity = _pilot_propensity(dgp, pilot_x)
        pilot_selected = rng.random(pilot_propensity.shape) < pilot_propensity
        pilot_selected[:, diagonal, diagonal] = False
        pilot_outcomes = _draw_outcomes(
            pilot_mu, rng.random(pilot_mu.shape[:-1])
        )
        rows, labels = _training_rows(
            pilot_x, pilot_selected, pilot_outcomes
        )
        classifier, tuning = _fit_mu(
            rows, labels, SEED + 410_000 + repetition
        )

        x_value = rng.uniform(
            0.0, 1.0, size=(N_CONTEXTS, P_FEATURES)
        )
        true_mu = dgp.probabilities(x_value)
        mu_hat = _predict_all_pairs(classifier, x_value)
        f_values, jacobians, q_values = _precompute(mu_hat)
        q_minima.extend(float(values.min()) for values in q_values.values())
        selection_uniform = rng.random(
            (N_CONTEXTS, K_ITEMS, K_ITEMS)
        )
        outcome_uniform = rng.random(
            (N_CONTEXTS, K_ITEMS, K_ITEMS)
        )

        row: dict[str, object] = {
            "repetition": repetition,
            "seed": SEED + 400_000 + repetition,
            "pilot_labeled_rows": int(pilot_selected.sum()),
            "pilot_tuning": tuning,
            "gars": {},
        }
        for name in truth:
            aopt, lam, used = _aopt_policy(q_values[name])
            aopt_estimate, aopt_labels = _estimate(
                f_values[name],
                jacobians[name],
                true_mu,
                aopt,
                selection_uniform,
                outcome_uniform,
            )
            random_estimate, random_labels = _estimate(
                f_values[name],
                jacobians[name],
                true_mu,
                random_policy,
                selection_uniform,
                outcome_uniform,
            )
            aopt_mse = float(
                100 * np.mean((aopt_estimate - truth[name]) ** 2)
            )
            random_mse = float(
                100 * np.mean((random_estimate - truth[name]) ** 2)
            )
            mse[name]["aopt"].append(aopt_mse)
            mse[name]["random"].append(random_mse)
            budget_errors.append(abs(used - BUDGET))

            aopt_objective = float(
                np.sum(q_values[name] / np.maximum(aopt, EPS))
            )
            random_objective = float(
                np.sum(
                    q_values[name] / np.maximum(random_policy, EPS)
                )
            )
            anti = _anti_policy(q_values[name])
            anti_objective = float(
                np.sum(q_values[name] / np.maximum(anti, EPS))
            )
            objective_checks.append(
                aopt_objective <= random_objective + 1e-8
            )
            negative_control_checks.append(
                anti_objective > aopt_objective + 1e-8
                and abs(float(anti.sum()) - BUDGET) < 1e-6
            )
            row["gars"][name] = {
                "lambda": lam,
                "expected_budget": used,
                "aopt_realized_labels": aopt_labels,
                "random_realized_labels": random_labels,
                "aopt_estimate": aopt_estimate.tolist(),
                "random_estimate": random_estimate.tolist(),
                "aopt_mse_x100": aopt_mse,
                "random_mse_x100": random_mse,
                "aopt_variance_objective": aopt_objective,
                "random_variance_objective": random_objective,
                "anti_policy_variance_objective": anti_objective,
            }
        raw.append(row)
        print(
            "TABLE2_LEARNED_PROGRESS="
            + json.dumps(
                {
                    "repetition": repetition + 1,
                    "pilot_labeled_rows": int(pilot_selected.sum()),
                    "mse_x100": {
                        name: {
                            policy: mse[name][policy][-1]
                            for policy in ("aopt", "random")
                        }
                        for name in mse
                    },
                },
                sort_keys=True,
            ),
            flush=True,
        )

    observed = {
        name: {
            policy: _summary(np.asarray(values))
            for policy, values in policies.items()
        }
        for name, policies in mse.items()
    }
    direction = {
        name: observed[name]["aopt"]["mean"]
        < observed[name]["random"]["mean"]
        for name in observed
    }
    overlap = {
        name: {
            policy: _overlap(
                observed[name][policy], PAPER[name][policy]
            )
            for policy in ("aopt", "random")
        }
        for name in observed
    }
    passed = (
        all(direction.values())
        and all(
            all(policy_checks.values())
            for policy_checks in overlap.values()
        )
        and max(budget_errors) < 1e-6
        and min(q_minima) >= -1e-12
        and all(objective_checks)
        and all(negative_control_checks)
    )
    result = {
        "schema_version": 1,
        "claim_id": 5,
        "route": 2,
        "status": "VERIFIED" if passed else "BLOCKED",
        "interpretation": (
            "Appendix Table 5 nctx=3000 is a 1500-context random pilot "
            "plus the Section 7 n=1500 acquisition sample."
        ),
        "paper_v1_mse_x100": PAPER,
        "observed_mse_x100": observed,
        "paper_observed_95_interval_overlap": overlap,
        "aopt_lower_than_random": direction,
        "independent_checker": {
            "max_expected_budget_abs_error": max(budget_errors),
            "minimum_influence_variance_weight": min(q_minima),
            "all_q_nonnegative": min(q_minima) >= -1e-12,
            "aopt_objective_no_greater_than_random_all": all(
                objective_checks
            ),
        },
        "negative_control": {
            "name": "budget-matched inverted-information policy",
            "rejected_every_repetition_and_gars": all(
                negative_control_checks
            ),
        },
        "truth": {
            name: value.tolist() for name, value in truth.items()
        },
        "pilot_contexts": PILOT_CONTEXTS,
        "acquisition_contexts": N_CONTEXTS,
        "repetitions": REPETITIONS,
        "budget": BUDGET,
        "alpha": ALPHA,
        "estimated_required_cores": 64,
        "logical_cpu_allocation": _available_cpus(),
        "runtime_seconds": round(time.monotonic() - started, 3),
        "deviations": [
            "The paper does not state how its Section 7 n=1500 and Appendix Table 5 nctx=3000 are related; this route interprets the extra 1500 contexts as an independent nuisance-training pilot.",
            "One pilot-trained nuisance model supplies independent predictions for policy design and EIF estimation; this is sample splitting rather than repeated cross-fitting within the acquisition sample.",
            "The author DGP seed and several coefficient distributions are unavailable; all clean-room draws are explicit.",
            "The judged v1 values are retained even though the current PDF revises Table 2 to 50 runs and different numbers.",
        ],
        "raw_repetitions": raw,
    }
    output = ROOT / "outputs/table2_learned_results.json"
    output.parent.mkdir(exist_ok=True)
    output.write_text(
        json.dumps(result, indent=2) + "\n", encoding="utf-8"
    )
    print(
        "TABLE2_LEARNED_RESULTS="
        + json.dumps(result, sort_keys=True),
        flush=True,
    )
    return 0 if passed else 1


if __name__ == "__main__":
    sys.exit(main())
