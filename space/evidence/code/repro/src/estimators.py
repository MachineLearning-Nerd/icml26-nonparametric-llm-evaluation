"""Estimators and A-optimal labeling policy for DMLRank."""
from __future__ import annotations
import numpy as np
from dmlrank import C, EPS, F_MAP, jacobian_for


# --------------------------------------------------------------------------- #
#  Observation model
# --------------------------------------------------------------------------- #
def sample_observations(mu_true_ctx, pi_label, rng):
    """Given true mu (n,K,K,C) and labeling policy pi (n,K,K) in [0,1] over ordered
    pairs, draw S_{i,jk} ~ Bern(pi) and Y_{i,jk} ~ Cat(mu_true[i,j,k]).

    Returns S (n,K,K) bool, Y (n,K,K,C) one-hot.
    """
    n, K, _, _ = mu_true_ctx.shape
    S = rng.random(mu_true_ctx.shape[:3]) < pi_label                # (n,K,K)
    # categorical draws
    cum = np.cumsum(mu_true_ctx, axis=-1)
    u = rng.random(mu_true_ctx.shape[:3] + (1,))
    Y = (u > cum[..., :1]).astype(int)                              # placeholder, redo properly
    # proper one-hot via argmax of (u <= cum)
    draw = np.argmax(u <= cum, axis=-1)                             # (n,K,K) in {0,1,2}
    Y = np.zeros_like(mu_true_ctx)
    ii, jj, kk = np.indices((n, K, K))
    Y[ii, jj, kk, draw] = 1.0
    Y = Y * S[..., None]
    return S, Y


# --------------------------------------------------------------------------- #
#  Plugin and debiased one-step EIF estimators
# --------------------------------------------------------------------------- #
def plugin_estimator(mu_hat_ctx):
    """theta_hat_plugin = (1/n) sum_i F(mu_hat(x_i)).  Returns (d,)."""
    F = F_MAP[_FNAME[0]]
    return np.mean([F(mu_hat_ctx[i]) for i in range(mu_hat_ctx.shape[0])], axis=0)


_FNAME = ["borda"]   # set by callers via set_F


def set_F(name: str):
    _FNAME[0] = name


def debiased_estimator(mu_hat_ctx, S, Y, pi_label, K, F_name="borda"):
    """One-step debiased EIF estimator (Eq 11) with plug-in mu_hat (cross-fit optional).

    theta_hat = (1/n) sum_i [ F(mu_hat(x_i))
                + sum_{j!=k} (S_{i,jk}/pi_{i,jk}) * J_{jk}(mu_hat(x_i)) (Y_{i,jk}-mu_hat_{i,jk}) ]

    Also returns influence-function contributions phi_i (n,d) for variance Sigma_hat.
    """
    F = F_MAP[F_name]
    n = mu_hat_ctx.shape[0]
    acc = np.zeros(F(mu_hat_ctx[0]).shape)
    phis = np.zeros((n, acc.shape[0]))
    for i in range(n):
        mu_x = mu_hat_ctx[i]
        Fx = F(mu_x)
        J = jacobian_for(F_name, mu_x, K)                       # (d,K,K,C)
        resid = Y[i] - mu_x                                     # (K,K,C)  NOT masked
        wt = np.where(pi_label[i] > EPS, S[i] / np.maximum(pi_label[i], EPS), 0.0)  # (K,K)
        # correction[d] = sum_{j,k} (S_{jk}/pi_{jk}) sum_c J[d,j,k,c] * (Y_{jk,c}-mu_{jk,c})
        corr = np.einsum('jk,djkc,jkc->d', wt, J, resid)
        phi_i = Fx + corr                                       # influence contribution
        phis[i] = phi_i
        acc += phi_i
    theta_hat = acc / n
    Sigma_hat = (phis - phis.mean(axis=0, keepdims=True))
    Sigma_hat = Sigma_hat.T @ Sigma_hat / n
    return theta_hat, Sigma_hat, phis


def plugin_mean_var(mu_hat_ctx):
    F = F_MAP[_FNAME[0]]
    Fs = np.stack([F(mu_hat_ctx[i]) for i in range(mu_hat_ctx.shape[0])], axis=0)
    n = Fs.shape[0]
    cov = np.cov(Fs, rowvar=False)
    cov = np.atleast_2d(cov)
    return cov / n


# --------------------------------------------------------------------------- #
#  A-optimal labeling policy  (Thm 6.2, Eq 17)
# --------------------------------------------------------------------------- #
def a_optimal_policy(mu_hat_ctx, cost_mat, alpha, budget, F_name="borda", lam_grid=None):
    """pi*_{jk}(x) = clip_{[alpha,1]} sqrt( tr(J V J^T) / (lambda * c_{jk}) ),
    with V_{jk}=Diag(mu)-mu mu^T (variance of one-hot Y), lambda chosen so that
    E[ sum_{j!=k} c_{jk} pi*_{jk}(X) ] = budget.  Returns pi (n,K,K) and lambda.

    mu_hat_ctx: (n,K,K,C).  cost_mat: (K,K) per-query costs.
    """
    F = F_MAP[F_name]
    n, K, _, _ = mu_hat_ctx.shape
    # numerator q_{i,jk} = tr(J_{jk} V_{jk} J_{jk}^T)  (scalar, = || J_{jk} sqrt(V) ||_F^2)
    q = np.zeros((n, K, K))
    for i in range(n):
        J = jacobian_for(F_name, mu_hat_ctx[i], K)              # (d,K,K,C)
        for j in range(K):
            for k in range(K):
                if j == k:
                    continue
                muv = mu_hat_ctx[i, j, k]
                V = np.diag(muv) - np.outer(muv, muv)           # (C,C)
                JV = J[:, j, k, :] @ V                          # (d,C)
                q[i, j, k] = (JV * J[:, j, k, :]).sum()         # tr(J V J^T)
    sqrtq = np.sqrt(np.maximum(q, 0.0))
    invc = 1.0 / np.maximum(cost_mat, EPS)                      # (K,K)

    def policy_for_lambda(lam):
        val = sqrtq * np.sqrt(invc) / np.sqrt(max(lam, EPS))    # (n,K,K)
        return np.clip(val, alpha, 1.0)

    def budget_usage(lam):
        pi = policy_for_lambda(lam)
        for i in range(n):
            np.fill_diagonal(pi[i], 0.0)
        # TOTAL expected cost over the n contexts:  sum_i sum_{j!=k} c_{jk} pi_{jk}(x_i)
        return (pi * cost_mat[None]).sum()

    # bisection on lambda: larger lambda -> smaller pi -> smaller cost.
    lo, hi = 1e-8, 1e6
    # ensure hi gives cost <= budget (under-budget) and lo gives cost >= budget
    for _ in range(60):
        mid = np.sqrt(lo * hi)
        if budget_usage(mid) > budget:
            lo = mid
        else:
            hi = mid
    lam = np.sqrt(lo * hi)
    pi = policy_for_lambda(lam)
    # zero diagonal explicitly
    for i in range(n):
        np.fill_diagonal(pi[i], 0.0)
    return pi, lam, budget_usage(lam)


def random_policy(K, p_query, rng, n):
    pi = np.full((n, K, K), p_query)
    for i in range(n):
        np.fill_diagonal(pi[i], 0.0)
    return pi
