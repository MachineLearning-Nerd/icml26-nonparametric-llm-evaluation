"""Claim 5 route 3: cross-fitted acquired-data reconstruction of Table 2."""

from __future__ import annotations

import json
import os
from pathlib import Path
import sys
import time

import numpy as np
from sklearn.model_selection import KFold

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from repro.src.table2_learned import (  # noqa: E402
    _anti_policy,
    _draw_outcomes,
    _fit_mu,
    _predict_all_pairs,
    _training_rows,
)
from repro.src.table2_oracle import (  # noqa: E402
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
    _overlap,
    _precompute,
    _summary,
    _truth,
)


def _available_cpus() -> int:
    if hasattr(os, "sched_getaffinity"):
        return len(os.sched_getaffinity(0))
    return os.cpu_count() or 1


def _crossfit_estimates(
    x_value: np.ndarray,
    true_mu: np.ndarray,
    policy: np.ndarray,
    selection_uniform: np.ndarray,
    outcome_uniform: np.ndarray,
    repetition: int,
    dataset_code: int,
) -> tuple[dict[str, np.ndarray], int, list[dict[str, object]]]:
    selected = selection_uniform < policy
    diagonal = np.arange(K_ITEMS)
    selected[:, diagonal, diagonal] = False
    outcomes = _draw_outcomes(true_mu, outcome_uniform)
    summands = {
        name: np.zeros((len(x_value), K_ITEMS))
        for name in ("borda", "bt", "rank_centrality")
    }
    diagnostics: list[dict[str, object]] = []
    folds = KFold(
        n_splits=2,
        shuffle=True,
        random_state=SEED + 600_000 + 100 * repetition + dataset_code,
    )
    for fold, (train, test) in enumerate(folds.split(x_value)):
        rows, labels = _training_rows(
            x_value[train], selected[train], outcomes[train]
        )
        classifier, tuning = _fit_mu(
            rows,
            labels,
            SEED
            + 610_000
            + 1000 * repetition
            + 10 * dataset_code
            + fold,
        )
        mu_hat = _predict_all_pairs(classifier, x_value[test])
        f_values, jacobians, _ = _precompute(mu_hat)
        one_hot = np.eye(2)[outcomes[test]]
        residual = one_hot - mu_hat
        weight = np.divide(
            selected[test],
            policy[test],
            out=np.zeros_like(policy[test]),
            where=policy[test] > 0,
        )
        for name in summands:
            correction = np.einsum(
                "ndjkc,njk,njkc->nd",
                jacobians[name],
                weight,
                residual,
            )
            summands[name][test] = f_values[name] + correction
        diagnostics.append(
            {
                "fold": fold,
                "training_contexts": int(len(train)),
                "held_out_contexts": int(len(test)),
                "labeled_training_rows": int(len(rows)),
                "tuning": tuning,
            }
        )
    return (
        {
            name: values.mean(axis=0)
            for name, values in summands.items()
        },
        int(selected.sum()),
        diagnostics,
    )


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
        rng = np.random.default_rng(SEED + 600_000 + repetition)
        aopt_x = rng.uniform(
            0.0, 1.0, size=(N_CONTEXTS, P_FEATURES)
        )
        random_x = rng.uniform(
            0.0, 1.0, size=(N_CONTEXTS, P_FEATURES)
        )
        aopt_mu = dgp.probabilities(aopt_x)
        random_mu = dgp.probabilities(random_x)
        _, _, design_q = _precompute(aopt_mu)
        q_minima.extend(float(values.min()) for values in design_q.values())

        aopt_selection_uniform = rng.random(
            (N_CONTEXTS, K_ITEMS, K_ITEMS)
        )
        aopt_outcome_uniform = rng.random(
            (N_CONTEXTS, K_ITEMS, K_ITEMS)
        )
        random_selection_uniform = rng.random(
            (N_CONTEXTS, K_ITEMS, K_ITEMS)
        )
        random_outcome_uniform = rng.random(
            (N_CONTEXTS, K_ITEMS, K_ITEMS)
        )

        random_estimates, random_labels, random_diagnostics = (
            _crossfit_estimates(
                random_x,
                random_mu,
                random_policy,
                random_selection_uniform,
                random_outcome_uniform,
                repetition,
                dataset_code=0,
            )
        )
        row: dict[str, object] = {
            "repetition": repetition,
            "seed": SEED + 600_000 + repetition,
            "random_realized_labels": random_labels,
            "random_fold_diagnostics": random_diagnostics,
            "gars": {},
        }

        for position, name in enumerate(truth, start=1):
            aopt, lam, used = _aopt_policy(design_q[name])
            aopt_estimates, aopt_labels, aopt_diagnostics = (
                _crossfit_estimates(
                    aopt_x,
                    aopt_mu,
                    aopt,
                    aopt_selection_uniform,
                    aopt_outcome_uniform,
                    repetition,
                    dataset_code=position,
                )
            )
            aopt_mse = float(
                100
                * np.mean(
                    (aopt_estimates[name] - truth[name]) ** 2
                )
            )
            random_mse = float(
                100
                * np.mean(
                    (random_estimates[name] - truth[name]) ** 2
                )
            )
            mse[name]["aopt"].append(aopt_mse)
            mse[name]["random"].append(random_mse)
            budget_errors.append(abs(used - BUDGET))

            aopt_objective = float(
                np.sum(design_q[name] / np.maximum(aopt, EPS))
            )
            random_objective = float(
                np.sum(
                    design_q[name]
                    / np.maximum(random_policy, EPS)
                )
            )
            anti = _anti_policy(design_q[name])
            anti_objective = float(
                np.sum(design_q[name] / np.maximum(anti, EPS))
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
                "aopt_estimate": aopt_estimates[name].tolist(),
                "random_estimate": random_estimates[name].tolist(),
                "aopt_mse_x100": aopt_mse,
                "random_mse_x100": random_mse,
                "aopt_variance_objective": aopt_objective,
                "random_variance_objective": random_objective,
                "anti_policy_variance_objective": anti_objective,
                "aopt_fold_diagnostics": aopt_diagnostics,
            }
        raw.append(row)
        print(
            "TABLE2_CROSSFIT_PROGRESS="
            + json.dumps(
                {
                    "repetition": repetition + 1,
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
        "route": 3,
        "status": "VERIFIED" if passed else "BLOCKED",
        "interpretation": (
            "Appendix Table 5 nctx=3000 is two independent "
            "1500-context acquisitions, one per policy; exact synthetic "
            "mu designs the policy and held-out acquired labels fit the "
            "estimator nuisance."
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
        "contexts_per_policy": N_CONTEXTS,
        "total_contexts_across_two_policies": 2 * N_CONTEXTS,
        "cross_fitting_folds": 2,
        "repetitions": REPETITIONS,
        "budget": BUDGET,
        "alpha": ALPHA,
        "estimated_required_cores": 64,
        "logical_cpu_allocation": _available_cpus(),
        "runtime_seconds": round(time.monotonic() - started, 3),
        "deviations": [
            "This route interprets Appendix nctx=3000 as two independent n=1500 policy datasets; the paper does not explicitly resolve whether contexts are shared.",
            "Exact synthetic mu is used only for policy design. The paper says practical policy design requires historical or external estimates but does not disclose which source its synthetic Table 2 used.",
            "Every estimator nuisance prediction is two-fold cross-fitted on that policy's acquired labels with tuning inside the training fold.",
            "The author DGP seed and several coefficient distributions are unavailable; all clean-room draws are explicit.",
            "The judged v1 numbers remain the target although the current PDF revises Table 2.",
        ],
        "raw_repetitions": raw,
    }
    output = ROOT / "outputs/table2_crossfit_results.json"
    output.parent.mkdir(exist_ok=True)
    output.write_text(
        json.dumps(result, indent=2) + "\n", encoding="utf-8"
    )
    print(
        "TABLE2_CROSSFIT_RESULTS="
        + json.dumps(result, sort_keys=True),
        flush=True,
    )
    return 0 if passed else 1


if __name__ == "__main__":
    sys.exit(main())
