"""Claim 5 route 1: oracle calibration of the published Table 2 experiment."""

from __future__ import annotations

import json
import math
import os
from pathlib import Path
import sys
import time

import numpy as np
from scipy.special import expit, logit
from scipy.stats import norm


ROOT = Path(__file__).resolve().parents[2]
SEED = 260121816
REPETITIONS = 15
N_CONTEXTS = 1500
K_ITEMS = 3
P_FEATURES = 2
BUDGET = 2000.0
ALPHA = 0.01
EPS = 1e-5
W1 = np.array([1.0, 0.0])
W2 = np.array([0.0, 1.0])
PAPER = {
    "borda": {
        "aopt": {"mean": 0.108, "half_width": 0.041},
        "random": {"mean": 0.133, "half_width": 0.060},
    },
    "bt": {
        "aopt": {"mean": 2.565, "half_width": 1.046},
        "random": {"mean": 2.910, "half_width": 1.143},
    },
    "rank_centrality": {
        "aopt": {"mean": 0.010, "half_width": 0.008},
        "random": {"mean": 0.017, "half_width": 0.007},
    },
}


def _available_cpus() -> int:
    if hasattr(os, "sched_getaffinity"):
        return len(os.sched_getaffinity(0))
    return os.cpu_count() or 1


class BTMisspecDGP:
    """Appendix J.2.2, with explicit draws for omitted coefficient defaults."""

    def __init__(self, seed: int = SEED):
        rng = np.random.default_rng(seed)
        self.linear = rng.normal(0.0, 1.0, size=(K_ITEMS, P_FEATURES))
        self.intercept = rng.normal(0.0, 1.0, size=K_ITEMS)
        self.quadratic = rng.normal(0.0, 0.6, size=(K_ITEMS, 2))
        self.phase = rng.uniform(0.0, 2 * np.pi, size=K_ITEMS)
        self.cycle = np.zeros((K_ITEMS, K_ITEMS))
        for j_value, k_value in ((0, 1), (1, 2), (2, 0)):
            self.cycle[j_value, k_value] = 1.0
            self.cycle[k_value, j_value] = -1.0

    def probabilities(self, x_value: np.ndarray) -> np.ndarray:
        utility = (
            x_value @ self.linear.T
            + self.intercept
            + x_value[:, :2] ** 2 @ self.quadratic.T
            + 0.6
            * np.sin(2 * np.pi * x_value[:, [0]] + self.phase[None, :])
        )
        difference = utility[:, :, None] - utility[:, None, :]
        first = expit(difference + self.cycle[None, :, :])
        mu = np.stack([first, 1 - first], axis=-1)
        diagonal = np.arange(K_ITEMS)
        mu[:, diagonal, diagonal] = 0.5
        return mu


def _incidence() -> tuple[np.ndarray, list[tuple[int, int]], np.ndarray]:
    pairs = [(j, k) for j in range(K_ITEMS) for k in range(j + 1, K_ITEMS)]
    matrix = np.zeros((len(pairs), K_ITEMS))
    for edge, (j_value, k_value) in enumerate(pairs):
        matrix[edge, j_value], matrix[edge, k_value] = 1.0, -1.0
    h_matrix = np.vstack(
        [np.eye(K_ITEMS - 1), -np.ones(K_ITEMS - 1)]
    )
    projection = h_matrix @ np.linalg.solve(
        h_matrix.T @ matrix.T @ matrix @ h_matrix,
        h_matrix.T @ matrix.T,
    )
    return matrix, pairs, projection


B_MATRIX, UNORDERED_PAIRS, PROJECTION = _incidence()
EDGE_INDEX = {pair: index for index, pair in enumerate(UNORDERED_PAIRS)}


def _gars_batch(mu: np.ndarray) -> dict[str, np.ndarray]:
    sym = 0.5 * (mu @ W1 + np.swapaxes(mu @ W2, 1, 2))
    diagonal = np.arange(K_ITEMS)
    sym[:, diagonal, diagonal] = 0.0
    borda = sym.sum(axis=2) / (K_ITEMS - 1)

    logits = []
    for j_value, k_value in UNORDERED_PAIRS:
        logits.append(
            0.5
            * (
                logit(np.clip(mu[:, j_value, k_value] @ W1, EPS, 1 - EPS))
                + logit(np.clip(mu[:, k_value, j_value] @ W2, EPS, 1 - EPS))
            )
        )
    bt = np.column_stack(logits) @ PROJECTION.T

    r_matrix = np.swapaxes(sym, 1, 2)
    degrees = np.maximum(r_matrix.sum(axis=2), EPS)
    transition = r_matrix / degrees[:, :, None]
    transition[:, diagonal, diagonal] = 0.0
    system = (
        np.eye(K_ITEMS)[None]
        - np.swapaxes(transition, 1, 2)
        + np.ones((1, K_ITEMS, K_ITEMS))
    )
    rc = np.linalg.solve(
        system, np.ones((len(mu), K_ITEMS, 1))
    )[..., 0]
    return {"borda": borda, "bt": bt, "rank_centrality": rc}


def _selected_jacobians(
    mu: np.ndarray, selected_j: int, selected_k: int
) -> dict[str, np.ndarray]:
    sym = 0.5 * (mu @ W1 + (mu @ W2).T)
    np.fill_diagonal(sym, 0.0)

    j_borda = np.zeros((K_ITEMS, 2))
    j_borda[selected_j] += W1 / (2 * (K_ITEMS - 1))
    j_borda[selected_k] += W2 / (2 * (K_ITEMS - 1))

    a_value, b_value = sorted((selected_j, selected_k))
    projection_column = PROJECTION[:, EDGE_INDEX[(a_value, b_value)]]
    if selected_j < selected_k:
        probability = np.clip(mu[selected_j, selected_k] @ W1, EPS, 1 - EPS)
        factor = W1 / (2 * probability * (1 - probability))
    else:
        probability = np.clip(mu[selected_j, selected_k] @ W2, EPS, 1 - EPS)
        factor = W2 / (2 * probability * (1 - probability))
    j_bt = np.outer(projection_column, factor)

    r_matrix = sym.T.copy()
    degrees = np.maximum(r_matrix.sum(axis=1), EPS)
    transition = r_matrix / degrees[:, None]
    np.fill_diagonal(transition, 0.0)
    system = (
        np.eye(K_ITEMS) - transition.T + np.ones((K_ITEMS, K_ITEMS))
    )
    rc = np.linalg.solve(system, np.ones(K_ITEMS))
    j_rc = np.zeros((K_ITEMS, 2))
    for category in range(2):
        d_r = np.zeros_like(r_matrix)
        d_r[selected_k, selected_j] += 0.5 * W1[category]
        d_r[selected_j, selected_k] += 0.5 * W2[category]
        d_degree = d_r.sum(axis=1)
        d_transition = (
            d_r * degrees[:, None] - r_matrix * d_degree[:, None]
        ) / degrees[:, None] ** 2
        np.fill_diagonal(d_transition, 0.0)
        j_rc[:, category] = np.linalg.solve(
            system, d_transition.T @ rc
        )
    return {"borda": j_borda, "bt": j_bt, "rank_centrality": j_rc}


def _precompute(
    mu: np.ndarray,
) -> tuple[
    dict[str, np.ndarray],
    dict[str, np.ndarray],
    dict[str, np.ndarray],
]:
    f_values = _gars_batch(mu)
    jacobians = {
        name: np.zeros((len(mu), K_ITEMS, K_ITEMS, K_ITEMS, 2))
        for name in f_values
    }
    q_values = {
        name: np.zeros((len(mu), K_ITEMS, K_ITEMS)) for name in f_values
    }
    for context in range(len(mu)):
        for j_value in range(K_ITEMS):
            for k_value in range(K_ITEMS):
                if j_value == k_value:
                    continue
                selected = _selected_jacobians(
                    mu[context], j_value, k_value
                )
                variance = (
                    np.diag(mu[context, j_value, k_value])
                    - np.outer(
                        mu[context, j_value, k_value],
                        mu[context, j_value, k_value],
                    )
                )
                for name, jacobian in selected.items():
                    jacobians[name][context, :, j_value, k_value] = jacobian
                    q_values[name][context, j_value, k_value] = float(
                        np.sum((jacobian @ variance) * jacobian)
                    )
    return f_values, jacobians, q_values


def _aopt_policy(q_value: np.ndarray) -> tuple[np.ndarray, float, float]:
    def policy(lam: float) -> np.ndarray:
        result = np.clip(np.sqrt(q_value / lam), ALPHA, 1.0)
        diagonal = np.arange(K_ITEMS)
        result[:, diagonal, diagonal] = 0.0
        return result

    lower, upper = 1e-12, 1e12
    for _ in range(160):
        middle = math.sqrt(lower * upper)
        if policy(middle).sum() > BUDGET:
            lower = middle
        else:
            upper = middle
    lam = math.sqrt(lower * upper)
    result = policy(lam)
    return result, lam, float(result.sum())


def _estimate(
    f_value: np.ndarray,
    jacobian: np.ndarray,
    mu: np.ndarray,
    policy: np.ndarray,
    selected_uniform: np.ndarray,
    outcome_uniform: np.ndarray,
) -> tuple[np.ndarray, int]:
    selected = selected_uniform < policy
    diagonal = np.arange(K_ITEMS)
    selected[:, diagonal, diagonal] = False
    outcome = (outcome_uniform >= mu[..., 0]).astype(int)
    one_hot = np.eye(2)[outcome]
    residual = one_hot - mu
    weight = np.divide(
        selected,
        policy,
        out=np.zeros_like(policy),
        where=policy > 0,
    )
    correction = np.einsum(
        "ndjkc,njk,njkc->nd", jacobian, weight, residual
    )
    return (f_value + correction).mean(axis=0), int(selected.sum())


def _truth(dgp: BTMisspecDGP) -> dict[str, np.ndarray]:
    accumulator = {
        name: np.zeros(K_ITEMS)
        for name in ("borda", "bt", "rank_centrality")
    }
    rng = np.random.default_rng(SEED + 10)
    total, chunk = 1_000_000, 50_000
    for _ in range(total // chunk):
        x_value = rng.uniform(0.0, 1.0, size=(chunk, P_FEATURES))
        values = _gars_batch(dgp.probabilities(x_value))
        for name in accumulator:
            accumulator[name] += values[name].sum(axis=0)
    return {name: value / total for name, value in accumulator.items()}


def _summary(values: np.ndarray) -> dict[str, float]:
    return {
        "mean": float(values.mean()),
        "monte_carlo_95_half_width": float(
            norm.ppf(0.975) * values.std(ddof=1) / np.sqrt(len(values))
        ),
        "standard_deviation": float(values.std(ddof=1)),
    }


def _overlap(observed: dict[str, float], paper: dict[str, float]) -> bool:
    return bool(
        observed["mean"] - observed["monte_carlo_95_half_width"]
        <= paper["mean"] + paper["half_width"]
        and paper["mean"] - paper["half_width"]
        <= observed["mean"] + observed["monte_carlo_95_half_width"]
    )


def main() -> int:
    started = time.monotonic()
    dgp = BTMisspecDGP()
    truth = _truth(dgp)
    raw: list[dict[str, object]] = []
    mse = {
        name: {"aopt": [], "random": []}
        for name in truth
    }
    budget_errors, objective_checks, q_minima = [], [], []

    random_probability = BUDGET / (
        N_CONTEXTS * K_ITEMS * (K_ITEMS - 1)
    )
    random_policy = np.full(
        (N_CONTEXTS, K_ITEMS, K_ITEMS), random_probability
    )
    diagonal = np.arange(K_ITEMS)
    random_policy[:, diagonal, diagonal] = 0.0

    for repetition in range(REPETITIONS):
        rng = np.random.default_rng(SEED + 200_000 + repetition)
        x_value = rng.uniform(0.0, 1.0, size=(N_CONTEXTS, P_FEATURES))
        mu = dgp.probabilities(x_value)
        f_values, jacobians, q_values = _precompute(mu)
        q_minima.extend(float(values.min()) for values in q_values.values())
        selection_uniform = rng.random((N_CONTEXTS, K_ITEMS, K_ITEMS))
        outcome_uniform = rng.random((N_CONTEXTS, K_ITEMS, K_ITEMS))
        row: dict[str, object] = {
            "repetition": repetition,
            "seed": SEED + 200_000 + repetition,
            "gars": {},
        }
        for name in truth:
            aopt, lam, used = _aopt_policy(q_values[name])
            aopt_estimate, aopt_labels = _estimate(
                f_values[name],
                jacobians[name],
                mu,
                aopt,
                selection_uniform,
                outcome_uniform,
            )
            random_estimate, random_labels = _estimate(
                f_values[name],
                jacobians[name],
                mu,
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
                np.sum(q_values[name] / np.maximum(random_policy, EPS))
            )
            objective_checks.append(aopt_objective <= random_objective + 1e-8)
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
            }
        raw.append(row)
        print(
            "TABLE2_PROGRESS="
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
            policy: _overlap(observed[name][policy], PAPER[name][policy])
            for policy in ("aopt", "random")
        }
        for name in observed
    }
    passed = (
        all(direction.values())
        and all(all(values.values()) for values in overlap.values())
        and max(budget_errors) < 1e-6
        and all(objective_checks)
        and min(q_minima) >= -1e-12
    )
    result = {
        "schema_version": 1,
        "claim_id": 5,
        "route": 1,
        "status": "VERIFIED" if passed else "BLOCKED",
        "paper_v1_mse_x100": PAPER,
        "observed_mse_x100": observed,
        "paper_observed_95_interval_overlap": overlap,
        "aopt_lower_than_random": direction,
        "independent_checker": {
            "max_expected_budget_abs_error": max(budget_errors),
            "aopt_objective_no_greater_than_random_all": all(objective_checks),
            "minimum_influence_variance_weight": min(q_minima),
            "all_q_nonnegative": min(q_minima) >= -1e-12,
        },
        "negative_control": {
            "name": "uniform random budget-matched policy",
            "rejected_for_all_gars": all(direction.values()),
        },
        "truth": {name: value.tolist() for name, value in truth.items()},
        "n_contexts": N_CONTEXTS,
        "repetitions": REPETITIONS,
        "budget": BUDGET,
        "alpha": ALPHA,
        "estimated_required_cores": 8,
        "logical_cpu_allocation": _available_cpus(),
        "runtime_seconds": round(time.monotonic() - started, 3),
        "deviations": [
            "This theorem-calibration route uses oracle true mu for both policy design and EIF estimation; it isolates the A-optimal allocation effect but is not the paper's unavailable learned-nuisance realization.",
            "The v1 judged claim reports 15 runs and 0.108/2.565/0.010; the current arXiv PDF reports 50 runs and revised values 0.130/2.861/0.017.",
            "The author DGP seed and several coefficient distributions are not published; all clean-room draws are explicit.",
        ],
        "raw_repetitions": raw,
    }
    output = ROOT / "outputs/table2_oracle_results.json"
    output.parent.mkdir(exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print("TABLE2_ORACLE_RESULTS=" + json.dumps(result, sort_keys=True), flush=True)
    return 0 if passed else 1


if __name__ == "__main__":
    sys.exit(main())
