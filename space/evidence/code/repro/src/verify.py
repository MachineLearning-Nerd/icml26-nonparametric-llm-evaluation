"""Reproduce Table 1 & Table 2 and verify claims 2/3/4/5.

Observation model: n contexts x_i; each context labels (queries) ordered pairs (j,k)
independently with probability pi_{jk}(x_i); outcome Y_{jk} ~ Categorical(mu_{jk}(x_i)).
For Table 1 the labeling is uniform (pi = p_q); for Table 2 it is the A-optimal policy
vs a budget-matched random policy.  The nuisance mu_hat is a per-rep-random, biased
black-box estimate (utility shrinkage kappa + refit offset delta + per-context noise nu).

Claim 2 (Thm 5.1): debiased one-step is sqrt(n)-asymptotically normal, attains the
    semiparametric efficiency bound (var == influence variance), valid coverage.
Claim 3 (Table 1): plugin high-error/invalid-coverage vs debiased low-error/valid.
Claim 4 (Thm 6.2): A-optimal policy feasible (budget exact) & A-optimal.
Claim 5 (Table 2): A-optimal beats random MSE.
"""
from __future__ import annotations
import sys, os, json, time
sys.path.insert(0, os.path.dirname(__file__))
import numpy as np
from scipy.stats import norm
from dmlrank import (C, EPS, F_MAP, make_dgp, true_theta_all,
                     nuisance_utilities, draw_nuisance_perturbation)
from estimators import (sample_observations, debiased_estimator, plugin_mean_var,
                        a_optimal_policy, random_policy, set_F)

Z95 = norm.ppf(0.975)


def _label_uniform(n, K, p_q):
    pi = np.full((n, K, K), p_q)
    for i in range(n):
        np.fill_diagonal(pi[i], 0.0)
    return pi


def one_rep(F_name, theta, params, utilities, mu_from_U, sample_X, n,
            kappa, s_u, nu, pi, seed):
    set_F(F_name)
    F = F_MAP[F_name]
    K = params["K"]
    rng = np.random.default_rng(seed)
    X = sample_X(n, rng)
    U = utilities(X)
    mu_true = mu_from_U(U)
    kp, dl = draw_nuisance_perturbation(K, kappa, 0.0, s_u, rng)
    U_hat = nuisance_utilities(U, kp, dl, nu, rng, params)
    mu_hat = mu_from_U(U_hat)
    S, Y = sample_observations(mu_true, pi, rng)
    plug = np.mean([F(mu_hat[i]) for i in range(n)], axis=0)
    plug_cov = plugin_mean_var(mu_hat)
    deb, Sig, _ = debiased_estimator(mu_hat, S, Y, pi, K, F_name)
    return plug, plug_cov, deb, Sig


def run_table1(F_name, theta, params, utilities, mu_from_U, sample_X,
               n=1000, R=20, kappa=0.0, s_u=0.7, nu=0.4, p_q=0.05, base_seed=1000):
    K = params["K"]
    pi = _label_uniform(n, K, p_q)
    d = K
    pe = np.zeros((R, d)); de = np.zeros((R, d))
    pc = np.zeros((R, d, d)); dc = np.zeros((R, d, d))
    for r in range(R):
        p, pcv, db, Sig = one_rep(F_name, theta, params, utilities, mu_from_U, sample_X,
                                  n, kappa, s_u, nu, pi, base_seed + r)
        pe[r] = p; de[r] = db; pc[r] = pcv; dc[r] = Sig
    perr = np.sqrt(((pe - theta) ** 2).mean(1)); derr = np.sqrt(((de - theta) ** 2).mean(1))

    def cov(ests, covs):
        out = np.zeros(R)
        for r in range(R):
            half = Z95 * np.sqrt(np.maximum(np.diag(covs[r]), 0) / n)
            out[r] = np.mean((theta >= ests[r] - half) & (theta <= ests[r] + half))
        return out
    return dict(F=F_name, n=n, R=R, theta=theta.tolist(),
                plugin_err=perr.mean(), plugin_err_sd=perr.std(),
                plugin_cov=cov(pe, pc).mean(), plugin_cov_sd=cov(pe, pc).std(),
                plugin_bias=np.sqrt(((pe - theta).mean(0) ** 2).mean()),
                debiased_err=derr.mean(), debiased_err_sd=derr.std(),
                debiased_cov=cov(de, dc).mean(), debiased_cov_sd=cov(de, dc).std(),
                debiased_bias=np.sqrt(((de - theta).mean(0) ** 2).mean()),
                eff_emp=de.var(0, ddof=1).mean(),
                eff_Sig=np.mean([np.diag(dc[r]).mean() for r in range(R)]) / n,
                norm_z=((de - theta) / np.sqrt(np.maximum(
                    np.stack([np.diag(dc[r]) for r in range(R)]) / n, EPS))).std())


def run_table2(F_name, theta, params, utilities, mu_from_U, sample_X,
               n=1500, R=15, kappa=0.0, s_u=0.5, nu=0.3, budget=2000.0, alpha=0.05,
               base_seed=5000):
    K = params["K"]
    cost = np.ones((K, K))
    p_q = min(max(budget / (n * K * (K - 1)), 0.0), 1.0)
    mse_a = np.zeros(R); mse_r = np.zeros(R); ub = np.zeros(R)
    for r in range(R):
        rng = np.random.default_rng(base_seed + r)
        X = sample_X(n, rng); U = utilities(X); mu_true = mu_from_U(U)
        kp, dl = draw_nuisance_perturbation(K, kappa, 0.0, s_u, rng)
        U_hat = nuisance_utilities(U, kp, dl, nu, rng, params)
        mu_hat = mu_from_U(U_hat)
        pi_star, lam, used = a_optimal_policy(mu_hat, cost, alpha, budget, F_name)
        pi_rand = random_policy(K, p_q, rng, n)
        for pi, store in [(pi_star, mse_a), (pi_rand, mse_r)]:
            S, Y = sample_observations(mu_true, pi, rng)
            db, Sig, _ = debiased_estimator(mu_hat, S, Y, pi, K, F_name)
            store[r] = ((db - theta) ** 2).mean()
        ub[r] = (pi_star * cost[None]).sum() / n
    return dict(F=F_name, n=n, R=R, budget=budget,
                aopt_mse=mse_a.mean(), aopt_mse_sd=mse_a.std(),
                rand_mse=mse_r.mean(), rand_mse_sd=mse_r.std(),
                aopt_budget=ub.mean())


def main():
    t0 = time.time()
    K, p, a, tau = 6, 5, 2.0, 0.4
    params, utilities, mu_from_U, sample_X = make_dgp(K, p, a, tau, seed=7)
    theta_all = true_theta_all(mu_from_U, utilities, sample_X, n_mc=80_000, seed=11)
    out = {"table1": {}, "table2": {}}
    # Per-functional nuisance strengths.  kappa = shrinkage (first-order plugin bias,
    # removed by debiasing; bounded Jacobian keeps BT/RC stable).  s_u/nu kept small
    # for the sensitive nonlinear functionals; Borda (linear) tolerates aggressive noise.
    NUIS = {"borda": dict(kappa=0.12, s_u=0.70, nu=0.35),
            "bt":    dict(kappa=0.45, s_u=0.04, nu=0.02),
            "rc":    dict(kappa=0.45, s_u=0.06, nu=0.03)}
    PQ = {"borda": 0.06, "bt": 0.25, "rc": 0.20}

    print("=" * 72)
    print("TABLE 1  (n=1000, plugin vs debiased; err=RMSE, cov=marginal-95%)")
    print("=" * 72)
    paper = {"borda": ("0.38 / 0.17", "0.15 / 0.94"),
             "bt":    ("0.62 / 0.09", "0.25 / 0.90"),
             "rc":    ("0.52 / 0.12", "0.27 / 0.91")}
    for F_name in ["borda", "bt", "rc"]:
        r = run_table1(F_name, theta_all[F_name], params, utilities, mu_from_U, sample_X,
                       n=1000, R=20, p_q=PQ[F_name], **NUIS[F_name])
        out["table1"][F_name] = r
        print(f"\n[{F_name.upper()}]  (paper plugin err/cov={paper[F_name][0]} ; debiased={paper[F_name][1]})")
        print(f"  plugin    err {r['plugin_err']:.3f}±{r['plugin_err_sd']:.3f}  cov {r['plugin_cov']:.3f}±{r['plugin_cov_sd']:.3f}  bias {r['plugin_bias']:.3f}")
        print(f"  debiased  err {r['debiased_err']:.3f}±{r['debiased_err_sd']:.3f}  cov {r['debiased_cov']:.3f}±{r['debiased_cov_sd']:.3f}  bias {r['debiased_bias']:.3f}")
        print(f"  claim2: eff emp {r['eff_emp']:.4f} vs Sig/n {r['eff_Sig']:.4f}  | std-z {r['norm_z']:.3f}(~1)")

    print("\n" + "=" * 72)
    print("TABLE 2  (n=1500, beta=2000, A-optimal vs random MSE)")
    print("=" * 72)
    paper2 = {"borda": "0.108 / 0.133", "bt": "2.565 / 2.910", "rc": "0.010 / 0.017"}
    for F_name in ["borda", "bt", "rc"]:
        r = run_table2(F_name, theta_all[F_name], params, utilities, mu_from_U, sample_X,
                       n=1500, R=24, budget=2000.0, alpha=0.01, **NUIS[F_name])
        out["table2"][F_name] = r
        tot = r['aopt_budget'] * 1500
        print(f"\n[{F_name.upper()}]  (paper A-opt/random MSE = {paper2[F_name]})")
        print(f"  A-optimal MSE {r['aopt_mse']:.4f}±{r['aopt_mse_sd']:.4f}  (total budget {tot:.0f}/{r['budget']:.0f})")
        print(f"  random    MSE {r['rand_mse']:.4f}±{r['rand_mse_sd']:.4f}")
        print(f"  A-optimal beats random: {r['aopt_mse'] < r['rand_mse']}  "
              f"(reduction {(1 - r['aopt_mse'] / r['rand_mse']) * 100:.1f}%)")

    out["elapsed_sec"] = round(time.time() - t0, 1)
    os.makedirs("outputs", exist_ok=True)
    with open("outputs/verify_results.json", "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nSaved outputs/verify_results.json  ({out['elapsed_sec']}s)")


if __name__ == "__main__":
    main()
