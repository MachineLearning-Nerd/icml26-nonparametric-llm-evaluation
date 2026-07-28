"""Claim 3: clean-room reconstruction of Table 1's n=1000 Borda row.

This implements the published NonlinearTie DGP and the nuisance-estimation
protocol from Appendices I and J. The expired author archive prevents a
seed-identical reproduction, so agreement is assessed against the paper's
reported 95% Monte Carlo intervals rather than exact bitwise equality.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
import sys
import time

from joblib import parallel_backend
from lightgbm import LGBMClassifier
import numpy as np
from scipy.special import expit, logit, softmax
from scipy.stats import norm
from sklearn.model_selection import KFold, RandomizedSearchCV


ROOT = Path(__file__).resolve().parents[2]
SEED = 260121816
REPETITIONS = 100
N_CONTEXTS = 1000
K_ITEMS = 3
P_FEATURES = 2
W1 = np.array([1.0, 0.0, 0.5])
W2 = np.array([0.0, 1.0, 0.5])
PARAMETER_GRID = {
    "n_estimators": [50, 100, 300, 600],
    "learning_rate": [0.03, 0.07, 0.1],
    "num_leaves": [15, 31, 63],
    "min_child_samples": [10, 30, 60, 80],
    "subsample": [0.7, 0.9, 1.0],
    "colsample_bytree": [0.7, 0.9, 1.0],
    "reg_alpha": [0.0, 0.1, 1.0],
    "reg_lambda": [0.0, 0.1, 0.5, 1.0],
    "max_depth": [-1, 6, 10, 15],
}
PAPER = {
    "plugin_error": {"mean": 0.38, "half_width": 0.08},
    "plugin_coverage": {"mean": 0.17, "half_width": 0.09},
    "debiased_error": {"mean": 0.15, "half_width": 0.03},
    "debiased_coverage": {"mean": 0.94, "half_width": 0.06},
}


def _available_cpus() -> int:
    if hasattr(os, "sched_getaffinity"):
        return len(os.sched_getaffinity(0))
    return os.cpu_count() or 1


class NonlinearTieDGP:
    """Appendix J.2.1 with explicit clean-room draws for omitted defaults."""

    def __init__(self, seed: int = SEED):
        rng = np.random.default_rng(seed)
        self.linear = rng.normal(0.0, 1.0, size=(K_ITEMS, P_FEATURES))
        self.intercept = rng.normal(0.0, 1.0, size=K_ITEMS)
        self.quadratic = rng.normal(0.0, 0.6, size=(K_ITEMS, 2))
        self.phase = rng.uniform(0.0, 2 * np.pi, size=K_ITEMS)
        self.selection_bias = rng.normal(0.0, 0.25, size=K_ITEMS)

    def utility(self, x_value: np.ndarray) -> np.ndarray:
        return (
            x_value @ self.linear.T
            + self.intercept
            + x_value[:, :2] ** 2 @ self.quadratic.T
            + 0.6
            * np.sin(
                2 * np.pi * x_value[:, [0]] + self.phase[None, :]
            )
        )

    def probabilities(
        self, x_value: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        utility = self.utility(x_value)
        difference = utility[:, :, None] - utility[:, None, :]
        tie_logit = (
            0.2
            - 1.2 * np.abs(difference)
            + 0.4 * np.cos(2 * np.pi * x_value[:, 1])[:, None, None]
        )
        logits = np.stack([difference, -difference, tie_logit], axis=-1)
        mu = softmax(logits, axis=-1)
        mu = np.maximum(mu, 0.05)
        mu /= mu.sum(axis=-1, keepdims=True)

        base = 0.3
        nonlinear = expit(
            logit(base)
            - 0.8 * np.abs(difference)
            + 0.4 * (x_value[:, 0] - 0.5)[:, None, None]
            + self.selection_bias[None, :, None]
            + self.selection_bias[None, None, :]
        )
        propensity = np.clip(0.9 * base + 0.1 * nonlinear, 0.05, 0.5)
        diagonal = np.arange(K_ITEMS)
        propensity[:, diagonal, diagonal] = 0.0
        return mu, propensity


def _borda(mu: np.ndarray) -> np.ndarray:
    s1 = mu @ W1
    s2 = mu @ W2
    symmetric = 0.5 * (s1 + np.swapaxes(s2, 1, 2))
    diagonal = np.arange(K_ITEMS)
    symmetric[:, diagonal, diagonal] = 0.0
    return symmetric.sum(axis=2) / (K_ITEMS - 1)


def _training_rows(
    x_value: np.ndarray,
    selected: np.ndarray,
    outcomes: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    context, j_value, k_value = np.nonzero(selected)
    rows = np.column_stack(
        [x_value[context], j_value, k_value]
    ).astype(np.float32)
    labels = outcomes[context, j_value, k_value]
    return rows, labels


def _fit_mu(
    rows: np.ndarray,
    labels: np.ndarray,
    seed: int,
) -> tuple[LGBMClassifier, dict[str, object]]:
    classifier = LGBMClassifier(
        objective="multiclass",
        num_class=3,
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
    if not np.array_equal(search.best_estimator_.classes_, np.arange(3)):
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
    pairs = np.array(
        [(j, k) for j in range(K_ITEMS) for k in range(K_ITEMS) if j != k],
        dtype=np.float32,
    )
    rows = np.column_stack(
        [
            np.repeat(x_value, len(pairs), axis=0),
            np.tile(pairs, (len(x_value), 1)),
        ]
    ).astype(np.float32)
    predicted = classifier.predict_proba(rows).reshape(
        len(x_value), len(pairs), 3
    )
    mu_hat = np.full((len(x_value), K_ITEMS, K_ITEMS, 3), 1 / 3)
    for position, (j_value, k_value) in enumerate(pairs):
        mu_hat[:, int(j_value), int(k_value)] = predicted[:, position]
    return mu_hat


def _estimate(
    x_value: np.ndarray,
    selected: np.ndarray,
    outcomes: np.ndarray,
    propensity: np.ndarray,
    repetition: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, list[dict[str, object]]]:
    plugin_summands = np.zeros((N_CONTEXTS, K_ITEMS))
    eif_summands = np.zeros_like(plugin_summands)
    diagnostics: list[dict[str, object]] = []
    folds = KFold(n_splits=2, shuffle=True, random_state=SEED + repetition)
    for fold, (train, test) in enumerate(folds.split(x_value)):
        rows, labels = _training_rows(
            x_value[train], selected[train], outcomes[train]
        )
        classifier, tuning = _fit_mu(
            rows, labels, SEED + 1000 * repetition + fold
        )
        mu_hat = _predict_all_pairs(classifier, x_value[test])
        f_value = _borda(mu_hat)
        correction = np.zeros_like(f_value)
        for local, context in enumerate(test):
            for j_value in range(K_ITEMS):
                for k_value in range(K_ITEMS):
                    if j_value == k_value or not selected[context, j_value, k_value]:
                        continue
                    residual = (
                        np.eye(3)[outcomes[context, j_value, k_value]]
                        - mu_hat[local, j_value, k_value]
                    )
                    correction[local, j_value] += (
                        W1 @ residual
                        / (2 * (K_ITEMS - 1) * propensity[context, j_value, k_value])
                    )
                    correction[local, k_value] += (
                        W2 @ residual
                        / (2 * (K_ITEMS - 1) * propensity[context, j_value, k_value])
                    )
        plugin_summands[test] = f_value
        eif_summands[test] = f_value + correction
        diagnostics.append(
            {
                "fold": fold,
                "labeled_training_rows": int(len(rows)),
                "tuning": tuning,
            }
        )

    plugin = plugin_summands.mean(axis=0)
    debiased = eif_summands.mean(axis=0)
    plugin_cov = np.cov(plugin_summands, rowvar=False, ddof=1) / N_CONTEXTS
    debiased_cov = np.cov(eif_summands, rowvar=False, ddof=1) / N_CONTEXTS
    return plugin, plugin_cov, debiased, debiased_cov, diagnostics


def _ground_truth(dgp: NonlinearTieDGP) -> np.ndarray:
    accumulator = np.zeros(K_ITEMS)
    rng = np.random.default_rng(SEED + 10)
    total = 1_000_000
    chunk = 50_000
    for _ in range(total // chunk):
        x_value = rng.uniform(0.0, 1.0, size=(chunk, P_FEATURES))
        mu, _ = dgp.probabilities(x_value)
        accumulator += _borda(mu).sum(axis=0)
    return accumulator / total


def _summary(values: np.ndarray) -> dict[str, float]:
    return {
        "mean": float(values.mean()),
        "monte_carlo_95_half_width": float(
            norm.ppf(0.975) * values.std(ddof=1) / np.sqrt(len(values))
        ),
        "standard_deviation": float(values.std(ddof=1)),
    }


def _intervals_overlap(left: dict[str, float], right: dict[str, float]) -> bool:
    return (
        left["mean"] - left["monte_carlo_95_half_width"]
        <= right["mean"] + right["half_width"]
        and right["mean"] - right["half_width"]
        <= left["mean"] + left["monte_carlo_95_half_width"]
    )


def main() -> int:
    started = time.monotonic()
    dgp = NonlinearTieDGP()
    truth = _ground_truth(dgp)
    z_bonferroni = norm.ppf(1 - 0.05 / (2 * K_ITEMS))
    raw: list[dict[str, object]] = []
    plugin_errors, debiased_errors = [], []
    plugin_coverage, debiased_coverage = [], []

    for repetition in range(REPETITIONS):
        rng = np.random.default_rng(SEED + 100_000 + repetition)
        x_value = rng.uniform(0.0, 1.0, size=(N_CONTEXTS, P_FEATURES))
        mu, propensity = dgp.probabilities(x_value)
        selected = rng.random(propensity.shape) < propensity
        diagonal = np.arange(K_ITEMS)
        selected[:, diagonal, diagonal] = False
        uniforms = rng.random(mu.shape[:-1])[..., None]
        outcomes = np.argmax(uniforms <= np.cumsum(mu, axis=-1), axis=-1)
        plugin, plugin_cov, debiased, debiased_cov, diagnostics = _estimate(
            x_value, selected, outcomes, propensity, repetition
        )
        plugin_error = float(np.linalg.norm(plugin - truth))
        debiased_error = float(np.linalg.norm(debiased - truth))
        plugin_half = z_bonferroni * np.sqrt(np.maximum(np.diag(plugin_cov), 0))
        debiased_half = z_bonferroni * np.sqrt(np.maximum(np.diag(debiased_cov), 0))
        plugin_hit = bool(np.all(np.abs(plugin - truth) <= plugin_half))
        debiased_hit = bool(np.all(np.abs(debiased - truth) <= debiased_half))
        plugin_errors.append(plugin_error)
        debiased_errors.append(debiased_error)
        plugin_coverage.append(plugin_hit)
        debiased_coverage.append(debiased_hit)
        raw.append(
            {
                "repetition": repetition,
                "seed": SEED + 100_000 + repetition,
                "labels": int(selected.sum()),
                "plugin_error_l2": plugin_error,
                "debiased_error_l2": debiased_error,
                "plugin_joint_coverage": plugin_hit,
                "debiased_joint_coverage": debiased_hit,
                "plugin_point": plugin.tolist(),
                "debiased_point": debiased.tolist(),
                "fold_diagnostics": diagnostics,
            }
        )
        print(
            "TABLE1_PROGRESS="
            + json.dumps(
                {
                    "repetition": repetition + 1,
                    "plugin_error": plugin_error,
                    "debiased_error": debiased_error,
                    "plugin_coverage": plugin_hit,
                    "debiased_coverage": debiased_hit,
                },
                sort_keys=True,
            ),
            flush=True,
        )

    observed = {
        "plugin_error": _summary(np.asarray(plugin_errors)),
        "plugin_coverage": _summary(np.asarray(plugin_coverage, dtype=float)),
        "debiased_error": _summary(np.asarray(debiased_errors)),
        "debiased_coverage": _summary(np.asarray(debiased_coverage, dtype=float)),
    }
    overlap = {
        metric: _intervals_overlap(observed[metric], PAPER[metric])
        for metric in PAPER
    }
    direction_checks = {
        "debiased_error_lower": observed["debiased_error"]["mean"]
        < observed["plugin_error"]["mean"],
        "debiased_coverage_higher": observed["debiased_coverage"]["mean"]
        > observed["plugin_coverage"]["mean"],
        "debiased_coverage_contains_0.95": (
            observed["debiased_coverage"]["mean"]
            - observed["debiased_coverage"]["monte_carlo_95_half_width"]
            <= 0.95
            <= observed["debiased_coverage"]["mean"]
            + observed["debiased_coverage"]["monte_carlo_95_half_width"]
        ),
    }
    passed = all(overlap.values()) and all(direction_checks.values())
    result = {
        "schema_version": 1,
        "claim_id": 3,
        "status": "VERIFIED" if passed else "BLOCKED",
        "paper": PAPER,
        "observed": observed,
        "paper_observed_95_interval_overlap": overlap,
        "direction_checks": direction_checks,
        "truth": truth.tolist(),
        "n_contexts": N_CONTEXTS,
        "repetitions": REPETITIONS,
        "k_items": K_ITEMS,
        "p_features": P_FEATURES,
        "cross_fitting_folds": 2,
        "tuning_iterations": 30,
        "tuning_cv_folds": 3,
        "estimated_required_cores": 16,
        "logical_cpu_allocation": _available_cpus(),
        "runtime_seconds": round(time.monotonic() - started, 3),
        "negative_control": {
            "name": "omit EIF correction (plugin)",
            "rejected": bool(
                observed["plugin_error"]["mean"]
                > observed["debiased_error"]["mean"]
                and observed["plugin_coverage"]["mean"]
                < observed["debiased_coverage"]["mean"]
            ),
        },
        "limitations": [
            "The author archive returned HTTP 410, so its DGP seed, omitted coefficient distributions, and synthetic cross-fit fold count are unavailable.",
            "The clean-room coefficient draws are explicit; the published K=3, p=2, n=1000, DGP equations, propensities, nuisance grid, 100 repetitions, and one-million-context truth calculation are preserved.",
            "The main text says five covariates while Appendix Table 5 says p=2; this run follows the more specific Appendix Table 5.",
        ],
        "raw_repetitions": raw,
    }
    output = ROOT / "outputs/table1_borda_results.json"
    output.parent.mkdir(exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print("TABLE1_BORDA_RESULTS=" + json.dumps(result, sort_keys=True), flush=True)
    return 0 if passed else 1


if __name__ == "__main__":
    sys.exit(main())
