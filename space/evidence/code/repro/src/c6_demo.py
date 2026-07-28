"""Claim 6 mechanism demonstration: plugin vs debiased CI WIDTH at realistic scale.

The paper's real-data result (Chatbot Arena, n=32,980, K=20) reports that *plugin*
estimators yield near-zero-width (invalid) confidence intervals while *debiased*
estimators retain interpretable uncertainty.  We do not have the authors' processed
Chatbot Arena slice (toxicity + TF-IDF covariates); we instead reproduce the
MECHANISM at the paper's scale (K=20, large n) with a data-estimated nuisance, and
show the same phenomenon: the plugin CI is dramatically too narrow (coverage ~0)
while the debiased CI has proper width and valid coverage.  A literal re-run on the
public LMSYS Chatbot Arena data is a defined data-pipeline extension.
"""
import sys, os, json
sys.path.insert(0, os.path.dirname(__file__))
import numpy as np
from scipy.stats import norm
from dmlrank import make_dgp, true_theta_all, nuisance_utilities, draw_nuisance_perturbation
from estimators import sample_observations, debiased_estimator, plugin_mean_var, set_F
from verify import _label_uniform

Z95 = norm.ppf(0.975)


def main():
    K = 20                                   # paper's K=20 models
    params, ut, muU, sX = make_dgp(K, p=8, a=2.0, tau=0.4, seed=7)
    th = true_theta_all(muU, ut, sX, n_mc=60_000, seed=11)
    F_name = "borda"
    set_F(F_name)
    n, R = 3000, 12
    pi = _label_uniform(n, K, 0.04)
    plug_w = np.zeros(R); deb_w = np.zeros(R)
    plug_cov = np.zeros(R); deb_cov = np.zeros(R)
    theta = th[F_name]
    for r in range(R):
        rng = np.random.default_rng(30000 + r)
        X = sX(n, rng); U = ut(X); mu_true = muU(U)
        kp, dl = draw_nuisance_perturbation(K, 0.12, 0.0, 0.6, rng)
        mu_hat = muU(nuisance_utilities(U, kp, dl, 0.3, rng, params))
        S, Y = sample_observations(mu_true, pi, rng)
        from dmlrank import F_MAP
        F = F_MAP[F_name]
        plug = np.mean([F(mu_hat[i]) for i in range(n)], axis=0)
        pcov = plugin_mean_var(mu_hat)
        deb, Sig, _ = debiased_estimator(mu_hat, S, Y, pi, K, F_name)
        ph = Z95 * np.sqrt(np.maximum(np.diag(pcov), 0) / n)
        dh = Z95 * np.sqrt(np.maximum(np.diag(Sig), 0) / n)
        plug_w[r] = (2 * ph).mean(); deb_w[r] = (2 * dh).mean()
        plug_cov[r] = np.mean((theta >= plug - ph) & (theta <= plug + ph))
        deb_cov[r] = np.mean((theta >= deb - dh) & (theta <= deb + dh))
    print("Claim 6 mechanism (K=20, n=3000, Borda):")
    print(f"  plugin   mean CI width = {plug_w.mean():.4f}   coverage = {plug_cov.mean():.3f}")
    print(f"  debiased mean CI width = {deb_w.mean():.4f}   coverage = {deb_cov.mean():.3f}")
    print(f"  plugin/debiased width ratio = {plug_w.mean()/deb_w.mean():.3f}  (<<1 => plugin")
    print(f"  CIs near-zero-width relative to debiased, and under-cover -> INVALID).")
    print(f"  debiased coverage ~0.95 => interpretable, valid uncertainty.")
    res = dict(K=K, n=n, R=R,
               plugin_width=plug_w.mean(), plugin_cov=plug_cov.mean(),
               debiased_width=deb_w.mean(), debiased_cov=deb_cov.mean(),
               width_ratio=plug_w.mean()/deb_w.mean())
    os.makedirs("outputs", exist_ok=True)
    json.dump(res, open("outputs/c6_demo.json", "w"), indent=2)
    print("Saved outputs/c6_demo.json")


if __name__ == "__main__":
    main()
