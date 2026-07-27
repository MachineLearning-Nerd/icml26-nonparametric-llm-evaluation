"""DMLRank — Nonparametric LLM Evaluation from Preference Data (clean-room repro).

Implements the Generalized Average Ranking Scores (GARS) functionals and their
Jacobians (claim 1), the one-step debiased efficient-influence-function estimator
with cross-fitting (claim 2 / Thm 5.1), and the A-optimal labeling policy (claim 4 /
Thm 6.2) for the paper "Nonparametric LLM Evaluation from Preference Data"
(Frauen, Deviyani, van der Schaar, Feuerriegel), arXiv:2601.21816.

Notation
--------
K  : number of models (items).
C=3: preference outcomes per ordered pair (j,k):  [j-win, tie, k-win].
mu : tensor (K, K, C);  mu[j,k,:] = E[Y_{jk} | X=x] in the simplex.
     By truth symmetry mu[j,k,0]/mu[j,k,2] = exp(2*a*(u_j-u_k)).
F  : a GARS functional  mu(x) -> R^d  (d = K for Borda/BT/RC vertex scores).
theta = E_X[ F(mu(X)) ].

All CPU / numpy+scipy.  Deterministic given a seed.
"""
from __future__ import annotations
import numpy as np
from scipy.special import logit, expit  # expit = sigmoid

C = 3  # [j-win, tie, k-win]
EPS = 1e-9


# --------------------------------------------------------------------------- #
#  Data-generating process
# --------------------------------------------------------------------------- #
def make_dgp(K: int, p: int, a: float, tau: float, seed: int):
    """Latent utilities u_j(x) = w_j . x + b_j ; pairwise mu via softmax-tie model.

    mu_{jk}(x) = softmax( [ a*(u_j-u_k),  tau,  a*(u_k-u_j) ] )
    so that mu[j,k,0]/mu[j,k,2] = exp(2*a*(u_j-u_k)) and a tie mass controlled by tau.
    Returns a closure sampling contexts and computing mu / theta.
    """
    rng = np.random.default_rng(seed)
    W = rng.standard_normal((K, p)) * 0.8      # per-model feature weights
    b = rng.standard_normal(K) * 1.0           # per-model offsets (baseline strength)
    params = dict(K=K, p=p, a=a, tau=tau, W=W, b=b)

    def utilities(X):
        # X: (n,p) -> U: (n,K)
        return X @ W.T + b[None, :]

    def mu_from_U(U):
        # U: (n,K) -> mu: (n,K,K,C)
        n = U.shape[0]
        dj = U[:, :, None] - U[:, None, :]          # (n,K,K)  u_j - u_k
        win = a * dj                                  # (n,K,K)
        logits = np.stack([win, np.full_like(win, tau), -win], axis=-1)  # (n,K,K,3)
        logits = logits - logits.max(axis=-1, keepdims=True)
        e = np.exp(logits)
        mu = e / e.sum(axis=-1, keepdims=True)
        mu = np.clip(mu, 1e-3, 1 - 1e-3)            # stabilise BT logit / Jacobian
        return mu / mu.sum(axis=-1, keepdims=True)   # renormalise to simplex

    def sample_X(n, r):
        return r.standard_normal((n, p))

    return params, utilities, mu_from_U, sample_X


def true_theta(F_func, mu_from_U, utilities, sample_X, n_mc=200_000, seed=0):
    """Ground-truth theta = E_X[F(mu(X))] via large Monte-Carlo over X."""
    r = np.random.default_rng(seed)
    X = sample_X(n_mc, r)
    U = utilities(X)
    mu = mu_from_U(U)                                 # (n,K,K,C)
    Fs = np.stack([F_func(mu[i]) for i in range(n_mc)], axis=0)  # (n_mc,d)
    return Fs.mean(axis=0)


def true_theta_all(mu_from_U, utilities, sample_X, n_mc=60_000, seed=11):
    """Ground truth for ALL three GARS in one MC pass.  Returns dict name->theta."""
    r = np.random.default_rng(seed)
    X = sample_X(n_mc, r)
    U = utilities(X)
    mu = mu_from_U(U)
    out = {}
    for name, F in F_MAP.items():
        Fs = np.stack([F(mu[i]) for i in range(n_mc)], axis=0)
        out[name] = Fs.mean(axis=0)
    return out


# --------------------------------------------------------------------------- #
#  GARS functionals  (claim 1)
# --------------------------------------------------------------------------- #
def F_borda(mu_x: np.ndarray) -> np.ndarray:
    """Borda score: F_j = mean_{k!=j} P(j beats k) = mean_{k!=j} mu[j,k,0]."""
    K = mu_x.shape[0]
    win = mu_x[:, :, 0]                                  # (K,K)  P(j beats k)
    off = win * (1 - np.eye(K))
    return off.sum(axis=1) / (K - 1)


def F_bt(mu_x: np.ndarray) -> np.ndarray:
    """Bradley-Terry log-odds least-squares projection (zero-sum vertex scores).

    edge e=(j,k), j<k:  ell_e = log(mu[j,k,0]/mu[j,k,2]) = logit-ish win/loss log-odds.
    Solve  min_s || B s - ell ||^2  with 1^T s = 0  ->  s = L0^+ B^T ell  (then center).
    """
    K = mu_x.shape[0]
    idx = np.triu_indices(K, k=1)
    j_idx, k_idx = idx
    ell = np.log(mu_x[j_idx, k_idx, 0] / mu_x[j_idx, k_idx, 2])   # (#edges,)
    B = np.zeros((len(j_idx), K))
    B[np.arange(len(j_idx)), j_idx] = 1.0
    B[np.arange(len(j_idx)), k_idx] = -1.0
    L0 = B.T @ B
    s = np.linalg.pinv(L0) @ (B.T @ ell)
    return s - s.mean()                                          # enforce zero-mean


def F_rc(mu_x: np.ndarray) -> np.ndarray:
    """Rank-Centrality stationary distribution (paper Eq 6).

    Row-stochastic transition  T[i,j] = P(j beats i) / d_i ,  d_i = sum_{l!=i} P(l beats i).
    Stationary pi (pi^T T = pi^T) recovered via the PageRank-style linear solve
        (I - T^T + 1 1^T) pi = 1 .
    """
    K = mu_x.shape[0]
    pbeats = mu_x[:, :, 0]                       # pbeats[j,i] = P(j beats i)
    d = pbeats.sum(axis=0) - np.diag(pbeats)     # d[i] = sum_{l!=i} P(l beats i)
    d = np.maximum(d, EPS)
    T = (pbeats / d[None, :]).T.copy()           # T[i,j] = P(j beats i)/d_i  (row-stochastic)
    np.fill_diagonal(T, 0.0)
    rs = T.sum(axis=1, keepdims=True)
    T = T / np.maximum(rs, EPS)                  # rows sum to 1
    M = np.eye(K) - T.T + np.ones((K, K))
    pi = np.linalg.solve(M, np.ones(K))          # = M^{-1} 1, already sums to 1 & >0
    return pi


# --------------------------------------------------------------------------- #
#  Jacobians  J_{jk}(mu) = dF/dmu_{jk}  in R^{d x C}
# --------------------------------------------------------------------------- #
def jac_borda(mu_x: np.ndarray, K: int) -> np.ndarray:
    """dF_a/dmu[j,k,c]:  F_j gets +1/(K-1) from mu[j,k,0] only (j != k)."""
    J = np.zeros((K, K, K, C))
    for a in range(K):
        for k in range(K):
            if k != a:
                J[a, a, k, 0] = 1.0 / (K - 1)     # dF_a / d mu[a,k,0]
    return J


def jac_bt(mu_x: np.ndarray, K: int) -> np.ndarray:
    """Analytic Jacobian of F_bt w.r.t. mu[j,k,{0,2}] (tie component c=1 -> 0)."""
    idx = np.triu_indices(K, k=1)
    j_idx, k_idx = idx
    B = np.zeros((len(j_idx), K))
    B[np.arange(len(j_idx)), j_idx] = 1.0
    B[np.arange(len(j_idx)), k_idx] = -1.0
    L0 = B.T @ B
    M = np.linalg.pinv(L0)                                  # maps B^T ell -> s
    # dF/dell_e = column e of (M @ B^T) then center  -> A in R^{K x #edges}
    A = M @ B.T
    A = A - A.mean(axis=0, keepdims=True)                  # effect of zero-mean centering
    J = np.zeros((K, K, K, C))
    for e, (j, k) in enumerate(zip(j_idx, k_idx)):
        d0 = 1.0 / np.maximum(mu_x[j, k, 0], EPS)          # d ell/d mu[j,k,0]
        d2 = -1.0 / np.maximum(mu_x[j, k, 2], EPS)         # d ell/d mu[j,k,2]
        J[:, j, k, 0] = A[:, e] * d0
        J[:, j, k, 2] = A[:, e] * d2
    # ordered pair (k,j) shares the SAME edge in the BT projection only if we also
    # include it; here F_bt is defined on unordered edges so (k,j) derivatives are 0.
    return J


def jac_numeric(F_func, mu_x: np.ndarray, K: int, h: float = 1e-5) -> np.ndarray:
    """Generic central-difference Jacobian dF/dmu[j,k,c].  (Reference / tests only.)"""
    J = np.zeros((K, K, K, C))
    for j in range(K):
        for k in range(K):
            if j == k:
                continue
            for c in range(C):
                mp = mu_x.copy(); mp[j, k, c] += h
                mm = mu_x.copy(); mm[j, k, c] -= h
                J[:, j, k, c] = (F_func(mp) - F_func(mm)) / (2 * h)
    return J


def jac_rc(mu_x: np.ndarray, K: int) -> np.ndarray:
    """Analytic Jacobian of F_rc (stationary distribution) w.r.t. mu[a,b,0] (win prob).

    pi = M^{-1} 1,  M = I - T^T + 1 1^T,  T[i,j]=pbeats[j,i]/d_i.
    dpi_alpha/dmu[a,b,0] = pi_b * (M^{-1} g)_alpha,  g_beta=(1[beta=a] d_b - mu[beta,b,0])/d_b^2.
    Only the win-probability component c=0 enters RC.
    """
    pbeats = mu_x[:, :, 0]                          # pbeats[j,i] = P(j beats i)
    d = pbeats.sum(axis=0) - np.diag(pbeats)
    d = np.maximum(d, EPS)
    T = (pbeats / d[None, :]).T.copy()
    np.fill_diagonal(T, 0.0)
    rs = T.sum(axis=1, keepdims=True)
    T = T / np.maximum(rs, EPS)
    M = np.eye(K) - T.T + np.ones((K, K))
    Minv = np.linalg.inv(M)
    pi = Minv @ np.ones(K)                        # matches F_rc exactly
    J = np.zeros((K, K, K, C))
    eye = np.eye(K)
    for a in range(K):
        for b in range(K):
            if a == b:
                continue
            g = (eye[a] * d[b] - pbeats[:, b]) / (d[b] ** 2)   # (K,)
            g[b] = 0.0     # T[b,b] is pinned to zero -> derivative is zero
            J[:, a, b, 0] = pi[b] * (Minv @ g)
    return J


def jacobian_for(F_name: str, mu_x: np.ndarray, K: int) -> np.ndarray:
    if F_name == "borda":
        return jac_borda(mu_x, K)
    if F_name == "bt":
        return jac_bt(mu_x, K)
    if F_name == "rc":
        return jac_rc(mu_x, K)
    raise ValueError(F_name)


F_MAP = {"borda": F_borda, "bt": F_bt, "rc": F_rc}


# --------------------------------------------------------------------------- #
#  Nuisance mu_hat : a data-estimated / black-box predictor with sampling error.
#
#  Faithful to the paper's "black-box machine-learning nuisance" (and the Fig 4a
#  external-judge simulation): mu_hat is a per-rep-random, systematically-biased
#  estimate of the true mu.  We realize it through perturbed latent utilities
#      u_hat_j(x) = (1 - kappa) * u_j(x) + delta_j
#  where kappa>0 is a shrinkage (regularised / under-fit nuisance -> first-order
#  bias that the debiased estimator removes) and delta_j ~ N(0, s_u^2) is a
#  per-rep model offset (refit variability -> makes the plugin's reported variance
#  under-estimate, so its CIs under-cover).  mu_hat(x) still varies with x.
# --------------------------------------------------------------------------- #
def nuisance_utilities(U, kappa, delta, nu, rng, params):
    """U: (n,K) true latent utilities -> (n,K) perturbed nuisance utilities.

    u_hat_j(x_i) = (1-kappa)*u_j(x_i) + delta_j + eta_{i,j}
      kappa>0 : shrinkage (regularised/under-fit nuisance -> first-order bias,
                removed by the debiased estimator; visible for NONLINEAR F).
      delta_j : per-rep model offset (refit variability -> plugin CIs under-cover).
      eta_{i,j}: per-context prediction noise (black-box nuisance error -> inflates
                both estimators' variance; captured by the influence variance).
    """
    n, K = U.shape
    eta = rng.standard_normal((n, K)) * nu
    return (1.0 - kappa) * U + delta[None, :] + eta


def draw_nuisance_perturbation(K, kappa_mean, kappa_sd, s_u, rng):
    """Per-rep nuisance perturbation: (kappa, delta)."""
    kappa = max(rng.normal(kappa_mean, kappa_sd), 0.0)
    delta = rng.standard_normal(K) * s_u
    return kappa, delta
