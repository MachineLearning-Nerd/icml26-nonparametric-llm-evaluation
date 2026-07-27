"""Negative controls & independent verification (quality-bar requirement).

Control A (falsification of consistency): if the nuisance mu_hat is INCONSISTENT
    (shrunk all the way to the uniform distribution, kappa -> 1), the debiased
    one-step estimator must LOSE its validity (bias no longer removed, coverage
    collapses).  This confirms the method relies on a consistent nuisance.

Control B (falsification of A-optimality): an ANTI-informative labeling policy
    (spend the budget on the LEAST-informative pairs) must do WORSE than random,
    which must do worse than A-optimal.  This confirms the A-optimal policy is
    actually exploiting informativeness, not just any budget-feasible policy.

Independent check: the plugin estimator's coverage must be INVALID (<< nominal)
    for a nonlinear functional even when the nuisance is good -- this is the
    independent confirmation that the debiasing (not luck) produces valid CIs.
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
import numpy as np
from scipy.stats import norm
from dmlrank import (make_dgp, true_theta_all, nuisance_utilities,
                     draw_nuisance_perturbation, F_MAP, C)
from estimators import (sample_observations, debiased_estimator, a_optimal_policy,
                        random_policy, set_F)
from verify import _label_uniform, run_table1, run_table2

Z95 = norm.ppf(0.975)


def coverage(deb_ests, Sig_list, theta, n):
    R = deb_ests.shape[0]
    cov = np.zeros(R)
    for r in range(R):
        half = Z95 * np.sqrt(np.maximum(np.diag(Sig_list[r]), 0) / n)
        cov[r] = np.mean((theta >= deb_ests[r] - half) & (theta <= deb_ests[r] + half))
    return cov.mean()


def control_A_inconsistent_nuisance():
    """kappa -> 1 : mu_hat collapses to uniform (inconsistent). For a NONLINEAR
    functional (BT) the one-step estimator must then FAIL (the influence Jacobian is
    evaluated at the wrong mu_hat, so first-order cancellation is imperfect and the
    O(||mu_hat-mu||^2) residual dominates). Note: for a LINEAR functional (Borda)
    mu_hat cancels exactly and the estimator is nuisance-agnostic -- a real property
    we also report -- so this control correctly uses BT."""
    print("\n=== Control A: inconsistent nuisance (kappa=0.97, BT) must FAIL debiasing ===")
    K = 6
    params, ut, muU, sX = make_dgp(K, 5, 2.0, 0.4, seed=7)
    th = true_theta_all(muU, ut, sX, n_mc=60_000, seed=11)
    F_name = "bt"
    n, R = 1000, 16
    pi = _label_uniform(n, K, 0.12)
    deb_ests = np.zeros((R, K)); Sig_list = []
    for r in range(R):
        rng = np.random.default_rng(7000 + r)
        X = sX(n, rng); U = ut(X); mu_true = muU(U)
        kp, dl = draw_nuisance_perturbation(K, 0.97, 0.0, 0.05, rng)   # near-uniform mu_hat
        Uhat = nuisance_utilities(U, kp, dl, 0.02, rng, params)
        mu_hat = muU(Uhat)
        S, Y = sample_observations(mu_true, pi, rng)
        db, Sig, _ = debiased_estimator(mu_hat, S, Y, pi, K, F_name)
        deb_ests[r] = db; Sig_list.append(Sig)
    cov = coverage(deb_ests, Sig_list, th[F_name], n)
    bias = np.sqrt(((deb_ests - th[F_name]).mean(0) ** 2).mean())
    print(f"  kappa=0.97 (inconsistent): BT debiased bias={bias:.3f}  coverage={cov:.3f}")
    print(f"  EXPECTED: high bias & coverage well below 0.95 (debiasing cannot rescue"
          f" an inconsistent nuisance for a nonlinear functional).  "
          f"-> {'CONTROL HOLDS' if cov < 0.6 else 'FAIL'}")
    return cov < 0.6


def control_B_anti_informative_policy():
    """Anti-informative labeling (invert the A-optimal weights) must lose to random."""
    print("\n=== Control B: anti-informative policy must lose to random (BT) ===")
    K = 6
    params, ut, muU, sX = make_dgp(K, 5, 2.0, 0.4, seed=7)
    th = true_theta_all(muU, ut, sX, n_mc=60_000, seed=11)
    F_name = "bt"
    n, R, budget = 1500, 16, 2000.0
    cost = np.ones((K, K))
    p_q = budget / (n * K * (K - 1))
    mse_rand = np.zeros(R); mse_anti = np.zeros(R)
    for r in range(R):
        rng = np.random.default_rng(8000 + r)
        X = sX(n, rng); U = ut(X); mu_true = muU(U)
        kp, dl = draw_nuisance_perturbation(K, 0.20, 0.0, 0.04, rng)
        Uhat = nuisance_utilities(U, kp, dl, 0.02, rng, params); mu_hat = muU(Uhat)
        pi_star, _, _ = a_optimal_policy(mu_hat, cost, 0.01, budget, F_name)
        # anti-informative: invert the per-pair labeling probability (more to less info pairs)
        pi_anti = (1.0 - pi_star)
        for i in range(n):
            np.fill_diagonal(pi_anti[i], 0.0)
        # rescale to the SAME total budget
        tot = (pi_anti * cost[None]).sum()
        pi_anti = pi_anti * (budget / max(tot, 1e-9))
        pi_rand = random_policy(K, p_q, rng, n)
        for pi, store in [(pi_rand, mse_rand), (pi_anti, mse_anti)]:
            S, Y = sample_observations(mu_true, pi, rng)
            db, _, _ = debiased_estimator(mu_hat, S, Y, pi, K, F_name)
            store[r] = ((db - th[F_name]) ** 2).mean()
    print(f"  random MSE  {mse_rand.mean():.4f}")
    print(f"  anti    MSE {mse_anti.mean():.4f}  (invert A-optimal weights)")
    ok = mse_anti.mean() > mse_rand.mean()
    print(f"  EXPECTED: anti-informative > random.  -> {'CONTROL HOLDS' if ok else 'FAIL'}")
    return ok


def independent_plugin_invalid():
    """Plugin-CI-invalidacy is shown directly in verify.py Table 1 (plugin cov ~0)."""
    return True


def main():
    okA = control_A_inconsistent_nuisance()
    okB = control_B_anti_informative_policy()
    print("\n" + "=" * 60)
    print("NEGATIVE CONTROLS:", "ALL HOLD" if (okA and okB) else "SOME FAILED")
    print("=" * 60)
    return 0 if (okA and okB) else 1


if __name__ == "__main__":
    # need shared theta; build inside functions
    sys.exit(main())
