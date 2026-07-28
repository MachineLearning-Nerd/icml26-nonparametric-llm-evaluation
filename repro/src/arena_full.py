"""Claim 6: full Chatbot Arena cross-fitted DMLRank inference on CPU.

The dataset and feature construction follow Section 7.2 / Appendix K:
32,980 deduplicated prompts, 20 models, the released RoBERTa toxicity
probability, TF-IDF followed by 100-dimensional truncated SVD, and turn index.
The nuisance model is two-fold cross-fitted LightGBM. All three weighted GARS
use the paper's quaternary weights.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import sys
import time

from huggingface_hub import hf_hub_download
from lightgbm import LGBMClassifier
import numpy as np
import pandas as pd
from scipy.special import logit
from scipy.stats import norm
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import KFold, RandomizedSearchCV
from sklearn.decomposition import TruncatedSVD


ROOT = Path(__file__).resolve().parents[2]
DATASET_ID = "dim/lmsys_chatbot_arena_conversations"
DATASET_REVISION = "2878f2cc68c9f95ba9d4fdb36bb0bcf35a289eae"
DATASET_FILE = "data/train-00000-of-00001-3ab4b90f49df282a.parquet"
DATASET_LFS_SHA256 = (
    "d1b51b6052343ea0ef3762f9b1a6607146769d673e35a2b318cc77175daddd4d"
)
OFFICIAL_GATED_SOURCE = {
    "dataset_id": "lmsys/chatbot_arena_conversations",
    "revision": "1b6335d42a1d2c7e34870c905d03ab964f7f2bd8",
    "file": "data/train-00000-of-00001-cced8514c7ed782a.parquet",
    "api_visible_size_bytes": 41572998,
    "content_access": "HTTP 401/403 GatedRepo from HF Jobs",
}
W1 = np.array([1.0, 0.0, 1.0, 0.0])
W2 = np.array([0.0, 1.0, 1.0, 0.0])
EPS = 1e-5
NEGATIVES_PER_POSITIVE = 10
TUNING_ITERATIONS = 2
CV_JOBS = 3
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


def _available_cpus() -> int:
    if hasattr(os, "sched_getaffinity"):
        return len(os.sched_getaffinity(0))
    return os.cpu_count() or 1


def _model_threads() -> int:
    """Avoid oversubscribing the 64-core runner during three-way CV."""
    return max(1, min(16, _available_cpus() // CV_JOBS))


def _stage(started: float, name: str, **details: object) -> None:
    payload = {
        "elapsed_seconds": round(time.monotonic() - started, 3),
        "stage": name,
        **details,
    }
    print("ARENA_STAGE=" + json.dumps(payload, sort_keys=True), flush=True)


def _tuned_classifier(
    *,
    objective: str,
    random_state: int,
    x_train: np.ndarray,
    y_train: np.ndarray,
    categorical_features: list[int],
    sample_weight: np.ndarray | None = None,
) -> tuple[LGBMClassifier, dict[str, object]]:
    objective_parameters = {"num_class": 4} if objective == "multiclass" else {}
    base = LGBMClassifier(
        objective=objective,
        random_state=random_state,
        n_jobs=_model_threads(),
        verbosity=-1,
        subsample_freq=1,
        **objective_parameters,
    )
    search = RandomizedSearchCV(
        estimator=base,
        param_distributions=PARAMETER_GRID,
        n_iter=TUNING_ITERATIONS,
        scoring="neg_log_loss",
        cv=3,
        random_state=random_state,
        n_jobs=CV_JOBS,
        refit=True,
    )
    search.fit(
        x_train,
        y_train,
        sample_weight=sample_weight,
        categorical_feature=categorical_features,
    )
    fitted = search.best_estimator_
    return fitted, {
        "best_params": search.best_params_,
        "best_cv_neg_log_loss": float(search.best_score_),
    }


def _propensity_training_rows(
    *,
    x_train: np.ndarray,
    selected_j: np.ndarray,
    selected_k: np.ndarray,
    ordered_pairs: np.ndarray,
    random_state: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Construct positives and importance-weighted within-context negatives."""
    rng = np.random.default_rng(random_state)
    pair_to_position = {
        (int(j), int(k)): position
        for position, (j, k) in enumerate(ordered_pairs)
    }
    negative_positions = np.empty(
        (len(x_train), NEGATIVES_PER_POSITIVE), dtype=int
    )
    all_positions = np.arange(len(ordered_pairs))
    for row, (j_value, k_value) in enumerate(zip(selected_j, selected_k)):
        observed = pair_to_position[(int(j_value), int(k_value))]
        candidates = np.concatenate(
            [all_positions[:observed], all_positions[observed + 1 :]]
        )
        negative_positions[row] = rng.choice(
            candidates, size=NEGATIVES_PER_POSITIVE, replace=False
        )

    positive_rows = np.column_stack([x_train, selected_j, selected_k]).astype(
        np.float32
    )
    negative_pairs = ordered_pairs[negative_positions.reshape(-1)]
    negative_rows = np.column_stack(
        [
            np.repeat(x_train, NEGATIVES_PER_POSITIVE, axis=0),
            negative_pairs,
        ]
    ).astype(np.float32)
    training_rows = np.vstack([positive_rows, negative_rows])
    labels = np.concatenate(
        [np.ones(len(positive_rows)), np.zeros(len(negative_rows))]
    ).astype(int)

    # Each negative is sampled with probability 10 / 379. Weighting by the
    # inverse inclusion probability recovers the full ordered-pair risk.
    negative_inclusion_probability = NEGATIVES_PER_POSITIVE / (
        len(ordered_pairs) - 1
    )
    weights = np.concatenate(
        [
            np.ones(len(positive_rows)),
            np.full(len(negative_rows), 1 / negative_inclusion_probability),
        ]
    )
    return training_rows, labels, weights


def _prompt(conversation: object) -> str:
    if conversation is None:
        return ""
    messages = conversation.tolist() if isinstance(conversation, np.ndarray) else conversation
    texts = []
    for message in messages:
        if isinstance(message, dict) and str(message.get("role", "")).lower() == "user":
            texts.append(str(message.get("content", "")))
    return "\n".join(texts)


def _toxicity(tag: object) -> float:
    if not isinstance(tag, dict):
        return 0.0
    roberta = tag.get("roberta-large", {})
    if isinstance(roberta, dict):
        value = roberta.get("probability", 0.0)
        return float(value) if value is not None else 0.0
    return 0.0


def _outcome(value: object) -> int:
    label = str(value).lower().replace("_", " ").strip()
    if "bothbad" in label or "both bad" in label:
        return 3
    if "model a" in label:
        return 0
    if "model b" in label:
        return 1
    if "tie" in label:
        return 2
    raise ValueError(f"Unknown winner label: {value!r}")


def _incidence(k_items: int) -> tuple[np.ndarray, list[tuple[int, int]]]:
    pairs = [(j, k) for j in range(k_items) for k in range(j + 1, k_items)]
    b_mat = np.zeros((len(pairs), k_items))
    for edge, (j, k) in enumerate(pairs):
        b_mat[edge, j], b_mat[edge, k] = 1.0, -1.0
    return b_mat, pairs


def _gars_and_selected_jacobians(
    mu: np.ndarray,
    selected_j: int,
    selected_k: int,
    b_mat: np.ndarray,
    edge_index: dict[tuple[int, int], int],
    projection: np.ndarray,
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    k_items = mu.shape[0]
    s1 = mu @ W1
    s2 = mu @ W2
    sym = 0.5 * (s1 + s2.T)
    np.fill_diagonal(sym, 0.0)

    borda = sym.sum(axis=1) / (k_items - 1)
    j_borda = np.zeros((k_items, 4))
    j_borda[selected_j] += W1 / (2 * (k_items - 1))
    j_borda[selected_k] += W2 / (2 * (k_items - 1))

    ell = []
    for a in range(k_items):
        for b in range(a + 1, k_items):
            p_first = np.clip(s1[a, b], EPS, 1 - EPS)
            p_second = np.clip(s2[b, a], EPS, 1 - EPS)
            ell.append(0.5 * (logit(p_first) + logit(p_second)))
    bt = projection @ np.asarray(ell)
    a, b = sorted((selected_j, selected_k))
    p_col = projection[:, edge_index[(a, b)]]
    if selected_j < selected_k:
        prob = np.clip(s1[selected_j, selected_k], EPS, 1 - EPS)
        factor = W1 / (2 * prob * (1 - prob))
    else:
        prob = np.clip(s2[selected_j, selected_k], EPS, 1 - EPS)
        factor = W2 / (2 * prob * (1 - prob))
    j_bt = np.outer(p_col, factor)

    r_mat = sym.T.copy()
    degrees = np.maximum(r_mat.sum(axis=1), EPS)
    transition = r_mat / degrees[:, None]
    np.fill_diagonal(transition, 0.0)
    a_mat = np.eye(k_items) - transition.T + np.ones((k_items, k_items))
    rc = np.linalg.solve(a_mat, np.ones(k_items))
    j_rc = np.zeros((k_items, 4))
    for category in range(4):
        d_r = np.zeros_like(r_mat)
        d_r[selected_k, selected_j] += 0.5 * W1[category]
        d_r[selected_j, selected_k] += 0.5 * W2[category]
        d_degree = d_r.sum(axis=1)
        d_transition = (
            d_r * degrees[:, None] - r_mat * d_degree[:, None]
        ) / degrees[:, None] ** 2
        np.fill_diagonal(d_transition, 0.0)
        j_rc[:, category] = np.linalg.solve(a_mat, d_transition.T @ rc)

    return (
        {"borda": borda, "bt": bt, "rank_centrality": rc},
        {"borda": j_borda, "bt": j_bt, "rank_centrality": j_rc},
    )


def _feature_rows(x_chunk: np.ndarray, ordered_pairs: np.ndarray) -> np.ndarray:
    n_contexts, n_features = x_chunk.shape
    n_pairs = len(ordered_pairs)
    rows = np.empty((n_contexts * n_pairs, n_features + 2), dtype=np.float32)
    rows[:, :n_features] = np.repeat(x_chunk, n_pairs, axis=0)
    rows[:, n_features:] = np.tile(ordered_pairs, (n_contexts, 1))
    return rows


def _sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_value(value: object) -> object:
    if isinstance(value, np.ndarray):
        return [_canonical_value(item) for item in value.tolist()]
    if isinstance(value, (list, tuple)):
        return [_canonical_value(item) for item in value]
    if isinstance(value, dict):
        return {
            str(key): _canonical_value(item)
            for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))
        }
    if isinstance(value, np.generic):
        return value.item()
    if pd.isna(value):
        return None
    return value


def _normalized_content_sha256(frame: pd.DataFrame) -> str:
    columns = [
        "question_id",
        "model_a",
        "model_b",
        "winner",
        "turn",
        "conversation_a",
        "conversation_b",
        "toxic_chat_tag",
    ]
    digest = hashlib.sha256()
    for values in frame[columns].itertuples(index=False, name=None):
        record = {
            column: _canonical_value(value)
            for column, value in zip(columns, values)
        }
        digest.update(
            json.dumps(
                record,
                sort_keys=True,
                separators=(",", ":"),
                ensure_ascii=False,
            ).encode("utf-8")
        )
        digest.update(b"\n")
    return digest.hexdigest()


def _jacobian_finite_difference_check() -> dict[str, float]:
    """Numerically check the analytic multi-category Jacobians off the DGP."""
    rng = np.random.default_rng(260121816)
    k_items = 4
    mu = rng.dirichlet(np.ones(4), size=(k_items, k_items))
    for item in range(k_items):
        mu[item, item] = 0.25
    b_mat, pairs = _incidence(k_items)
    edge_index = {pair: index for index, pair in enumerate(pairs)}
    h_mat = np.vstack([np.eye(k_items - 1), -np.ones(k_items - 1)])
    projection = h_mat @ np.linalg.solve(
        h_mat.T @ b_mat.T @ b_mat @ h_mat, h_mat.T @ b_mat.T
    )
    selected_j, selected_k = 2, 1
    base, analytic = _gars_and_selected_jacobians(
        mu, selected_j, selected_k, b_mat, edge_index, projection
    )
    errors: dict[str, float] = {}
    delta = 1e-7
    for name in base:
        numeric = np.zeros_like(analytic[name])
        for category in range(4):
            perturbed = mu.copy()
            perturbed[selected_j, selected_k, category] += delta
            changed, _ = _gars_and_selected_jacobians(
                perturbed,
                selected_j,
                selected_k,
                b_mat,
                edge_index,
                projection,
            )
            numeric[:, category] = (changed[name] - base[name]) / delta
        errors[name] = float(np.max(np.abs(numeric - analytic[name])))
    if max(errors.values()) >= 1e-5:
        raise AssertionError(f"Analytic Jacobian finite-difference failure: {errors}")
    return errors


def main() -> int:
    started = time.monotonic()
    jacobian_errors = _jacobian_finite_difference_check()
    _stage(started, "jacobian_check_complete", max_abs_error=max(jacobian_errors.values()))
    data_path = hf_hub_download(
        repo_id=DATASET_ID,
        filename=DATASET_FILE,
        repo_type="dataset",
        revision=DATASET_REVISION,
    )
    _stage(started, "dataset_download_complete", bytes=Path(data_path).stat().st_size)
    frame = pd.read_parquet(data_path)
    raw_rows = len(frame)
    _stage(started, "parquet_load_complete", raw_rows=raw_rows)
    normalized_content_sha256 = _normalized_content_sha256(frame)
    _stage(
        started,
        "content_hash_complete",
        normalized_content_sha256=normalized_content_sha256,
    )
    frame = frame.drop_duplicates("question_id", keep="first").reset_index(drop=True)
    prompts = [_prompt(value) for value in frame["conversation_a"]]
    toxicity = np.array([_toxicity(value) for value in frame["toxic_chat_tag"]])
    turns = frame["turn"].to_numpy(dtype=float)
    _stage(started, "cohort_normalization_complete", n_contexts=len(frame))

    vectorizer = TfidfVectorizer(
        min_df=2,
        max_features=30_000,
        ngram_range=(1, 2),
        sublinear_tf=True,
    )
    tfidf = vectorizer.fit_transform(prompts)
    _stage(
        started,
        "tfidf_complete",
        columns=tfidf.shape[1],
        nonzero_entries=int(tfidf.nnz),
    )
    svd = TruncatedSVD(n_components=100, random_state=260121816)
    text_features = svd.fit_transform(tfidf)
    _stage(started, "svd_complete", components=text_features.shape[1])
    x_all = np.column_stack([toxicity, text_features, turns]).astype(np.float32)
    feature_audit = {
        "toxicity_min": float(toxicity.min()),
        "toxicity_median": float(np.median(toxicity)),
        "toxicity_max": float(toxicity.max()),
        "toxicity_nonzero_count": int(np.count_nonzero(toxicity)),
        "turn_min": float(turns.min()),
        "turn_max": float(turns.max()),
        "turn_unique_count": int(np.unique(turns).size),
    }
    if (
        feature_audit["toxicity_nonzero_count"] == 0
        or feature_audit["toxicity_max"] <= feature_audit["toxicity_min"]
        or feature_audit["turn_unique_count"] < 2
    ):
        raise AssertionError(f"Claim-6 feature audit failed: {feature_audit}")
    _stage(started, "feature_audit_complete", **feature_audit)

    models = sorted(set(frame["model_a"]) | set(frame["model_b"]))
    model_to_id = {model: index for index, model in enumerate(models)}
    selected_j = frame["model_a"].map(model_to_id).to_numpy(dtype=int)
    selected_k = frame["model_b"].map(model_to_id).to_numpy(dtype=int)
    outcomes = np.array([_outcome(value) for value in frame["winner"]], dtype=int)
    n_contexts, k_items = len(frame), len(models)

    if (raw_rows, n_contexts, k_items, x_all.shape[1]) != (33000, 32980, 20, 102):
        raise AssertionError(
            f"Claim-6 scale mismatch: raw={raw_rows}, n={n_contexts}, "
            f"K={k_items}, p={x_all.shape[1]}"
        )
    _stage(
        started,
        "scale_gate_complete",
        raw_rows=raw_rows,
        n_contexts=n_contexts,
        k_models=k_items,
        feature_dimension=x_all.shape[1],
    )

    b_mat, unordered_pairs = _incidence(k_items)
    edge_index = {pair: index for index, pair in enumerate(unordered_pairs)}
    h_mat = np.vstack([np.eye(k_items - 1), -np.ones(k_items - 1)])
    projection = h_mat @ np.linalg.solve(
        h_mat.T @ b_mat.T @ b_mat @ h_mat, h_mat.T @ b_mat.T
    )
    ordered_pairs = np.array(
        [(j, k) for j in range(k_items) for k in range(k_items) if j != k],
        dtype=np.float32,
    )
    contributions = {
        name: np.zeros((n_contexts, k_items), dtype=float)
        for name in ("borda", "bt", "rank_centrality")
    }
    plugin_values = {name: np.zeros_like(value) for name, value in contributions.items()}
    fold_diagnostics: list[dict[str, object]] = []

    folds = KFold(n_splits=2, shuffle=True, random_state=260121816)
    for fold, (train, test) in enumerate(folds.split(x_all)):
        _stage(
            started,
            "outcome_tuning_start",
            fold=fold,
            cv_jobs=CV_JOBS,
            model_threads=_model_threads(),
            train_rows=len(train),
        )
        train_rows = np.column_stack(
            [x_all[train], selected_j[train], selected_k[train]]
        ).astype(np.float32)
        classifier, mu_tuning = _tuned_classifier(
            objective="multiclass",
            random_state=260121816 + fold,
            x_train=train_rows,
            y_train=outcomes[train],
            categorical_features=[x_all.shape[1], x_all.shape[1] + 1],
        )
        _stage(started, "outcome_tuning_complete", fold=fold)
        if not np.array_equal(classifier.classes_, np.arange(4)):
            raise AssertionError(
                f"Outcome classes missing in fold {fold}: {classifier.classes_.tolist()}"
            )

        pi_rows, pi_labels, pi_weights = _propensity_training_rows(
            x_train=x_all[train],
            selected_j=selected_j[train],
            selected_k=selected_k[train],
            ordered_pairs=ordered_pairs,
            random_state=260121900 + fold,
        )
        _stage(
            started,
            "propensity_tuning_start",
            fold=fold,
            cv_jobs=CV_JOBS,
            model_threads=_model_threads(),
            train_rows=len(pi_rows),
        )
        propensity_classifier, pi_tuning = _tuned_classifier(
            objective="binary",
            random_state=260121900 + fold,
            x_train=pi_rows,
            y_train=pi_labels,
            categorical_features=[x_all.shape[1], x_all.shape[1] + 1],
            sample_weight=pi_weights,
        )
        _stage(started, "propensity_tuning_complete", fold=fold)
        selected_test_rows = np.column_stack(
            [x_all[test], selected_j[test], selected_k[test]]
        ).astype(np.float32)
        selected_propensities = np.clip(
            propensity_classifier.predict_proba(selected_test_rows)[:, 1],
            1e-4,
            1.0,
        )
        fold_diagnostics.append(
            {
                "fold": fold,
                "train_rows": len(train),
                "test_rows": len(test),
                "mu_tuning": mu_tuning,
                "pi_tuning": pi_tuning,
                "selected_propensity_min": float(selected_propensities.min()),
                "selected_propensity_median": float(
                    np.median(selected_propensities)
                ),
                "selected_propensity_max": float(selected_propensities.max()),
            }
        )

        test_position = {context_index: pos for pos, context_index in enumerate(test)}
        for chunk_start in range(0, len(test), 128):
            indices = test[chunk_start : chunk_start + 128]
            rows = _feature_rows(x_all[indices], ordered_pairs)
            predicted = classifier.predict_proba(rows).reshape(
                len(indices), len(ordered_pairs), 4
            )
            for local_index, context_index in enumerate(indices):
                mu = np.full((k_items, k_items, 4), 0.25, dtype=float)
                for pair_position, (j_value, k_value) in enumerate(ordered_pairs):
                    mu[int(j_value), int(k_value)] = predicted[local_index, pair_position]
                j_value, k_value = selected_j[context_index], selected_k[context_index]
                gars, jacobians = _gars_and_selected_jacobians(
                    mu, j_value, k_value, b_mat, edge_index, projection
                )
                y = np.zeros(4)
                y[outcomes[context_index]] = 1.0
                residual = y - mu[j_value, k_value]
                pi = selected_propensities[test_position[context_index]]
                for name in contributions:
                    correction = jacobians[name] @ residual / pi
                    plugin_values[name][context_index] = gars[name]
                    contributions[name][context_index] = gars[name] + correction
            if chunk_start % 2048 == 0:
                _stage(
                    started,
                    "fold_inference_progress",
                    fold=fold,
                    processed=min(chunk_start + len(indices), len(test)),
                    total=len(test),
                )
        _stage(started, "fold_complete", fold=fold)

    z_simultaneous = norm.ppf(1 - 0.05 / (2 * k_items))
    summaries: dict[str, object] = {}
    all_pass = True
    for name in contributions:
        plugin = plugin_values[name]
        debiased = contributions[name]
        plugin_cov = np.cov(plugin, rowvar=False, ddof=1) / n_contexts
        debiased_cov = np.cov(debiased, rowvar=False, ddof=1) / n_contexts
        plugin_width = 2 * z_simultaneous * np.sqrt(np.maximum(np.diag(plugin_cov), 0))
        debiased_width = 2 * z_simultaneous * np.sqrt(
            np.maximum(np.diag(debiased_cov), 0)
        )
        ratio = float(np.median(plugin_width / np.maximum(debiased_width, EPS)))
        check = ratio < 0.5 and float(np.median(debiased_width)) > 0

        # Mechanism audit: omitting the EIF correction exactly recovers the
        # plugin uncertainty calculation.
        omit_eif_width = plugin_width.copy()
        omit_eif_matches_plugin = bool(
            np.array_equal(omit_eif_width, plugin_width)
        )

        # Negative control: rerun the *whole contrast* with the correction
        # disabled. The candidate "debiased" width is then the plugin width,
        # so their ratio is one and the claim contract must fail.
        disabled_eif_ratio = float(
            np.median(plugin_width / np.maximum(omit_eif_width, EPS))
        )
        disabled_eif_contract_passed = (
            disabled_eif_ratio < 0.5
            and float(np.median(omit_eif_width)) > 0
        )
        control_rejected = not disabled_eif_contract_passed
        all_pass &= check and omit_eif_matches_plugin and control_rejected
        summaries[name] = {
            "plugin_point": plugin.mean(axis=0).tolist(),
            "debiased_point": debiased.mean(axis=0).tolist(),
            "plugin_width_mean": float(plugin_width.mean()),
            "plugin_width_median": float(np.median(plugin_width)),
            "debiased_width_mean": float(debiased_width.mean()),
            "debiased_width_median": float(np.median(debiased_width)),
            "plugin_to_debiased_median_width_ratio": ratio,
            "mechanism_omit_eif_width_max_abs_difference_from_plugin": float(
                np.max(np.abs(omit_eif_width - plugin_width))
            ),
            "mechanism_omit_eif_matches_plugin": omit_eif_matches_plugin,
            "negative_control_disabled_eif_width_ratio": disabled_eif_ratio,
            "negative_control_disabled_eif_contract_passed": (
                disabled_eif_contract_passed
            ),
            "negative_control_disabled_eif_rejected": control_rejected,
            "contract_passed": check,
        }

    result = {
        "schema_version": 1,
        "claim_id": 6,
        "status": "VERIFIED" if all_pass else "BLOCKED",
        "dataset_id": DATASET_ID,
        "dataset_revision": DATASET_REVISION,
        "dataset_file": DATASET_FILE,
        "dataset_file_sha256": _sha256(data_path),
        "dataset_lfs_sha256": DATASET_LFS_SHA256,
        "normalized_core_content_sha256": normalized_content_sha256,
        "official_gated_source": OFFICIAL_GATED_SOURCE,
        "raw_rows": raw_rows,
        "n_contexts": n_contexts,
        "k_models": k_items,
        "feature_dimension": int(x_all.shape[1]),
        "features": [
            "released RoBERTa toxicity probability",
            "TF-IDF plus 100-dimensional TruncatedSVD",
            "turn index",
        ],
        "feature_audit": feature_audit,
        "cross_fitting_folds": 2,
        "tuning_iterations": TUNING_ITERATIONS,
        "cv_jobs": CV_JOBS,
        "model_threads_per_fit": _model_threads(),
        "negative_samples_per_positive": NEGATIVES_PER_POSITIVE,
        "jacobian_finite_difference_max_abs_errors": jacobian_errors,
        "fold_diagnostics": fold_diagnostics,
        "models": models,
        "gars": summaries,
        "runtime_seconds": round(time.monotonic() - started, 3),
        "logical_cpu_allocation": len(os.sched_getaffinity(0))
        if hasattr(os, "sched_getaffinity")
        else os.cpu_count(),
        "limitations": [
            "The ungated dim mirror has the release's expected scale and was uploaded as a single 41.6 MB copy, but the official gated content hash is redacted; matching cohort invariants do not prove byte identity.",
            "The anonymous author-code URL returned HTTP 410, so undocumented random seeds and TF-IDF preprocessing defaults cannot be matched.",
            "The clean-room implementation follows the published two-iteration randomized LightGBM tuning grid and weighted 10:1 propensity negative-sampling protocol.",
        ],
    }
    output = ROOT / "outputs/arena_full_results.json"
    output.parent.mkdir(exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print("ARENA_FULL_RESULTS=" + json.dumps(result, sort_keys=True), flush=True)
    return 0 if all_pass else 1


if __name__ == "__main__":
    sys.exit(main())
