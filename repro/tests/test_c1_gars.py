"""Claim 1 verification: GARS unifies Borda / Bradley-Terry / Rank-Centrality.

Constructs a known BT model with latent strengths s, builds the exact mu tensor,
and checks that each F functional recovers the expected ranking structure:
  * F_borda  -> mean win probability per model (monotone in s).
  * F_bt     -> recovers s up to a shift  (linear in s, argmin ||B s - ell||^2).
  * F_rc     -> stationary distribution, monotone in s.

Also checks analytic Jacobians against central finite differences.
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
import numpy as np
from dmlrank import (F_borda, F_bt, F_rc, jac_borda, jac_bt, jac_rc,
                      jac_numeric, jacobian_for, C)


def mu_from_strengths(s, a=2.0, tau=0.4):
    """Build (K,K,C) mu under BT model with strengths s (zero-mean)."""
    K = len(s)
    dj = s[:, None] - s[None, :]                      # u_j - u_k
    win = a * dj
    logits = np.stack([win, np.full_like(win, tau), -win], axis=-1)
    logits -= logits.max(axis=-1, keepdims=True)
    e = np.exp(logits)
    return e / e.sum(axis=-1, keepdims=True)


def check_monotone(scores, s, name):
    # ranks by score should match ranks by s
    r_score = np.argsort(scores)
    r_s = np.argsort(s)
    ok = np.all(r_score == r_s)
    print(f"  [{name}] scores={np.round(scores,4)}  ranks match s: {ok}")
    return ok


def main():
    print("=== Claim 1: GARS recovers Borda / BT / Rank-Centrality ===")
    rng = np.random.default_rng(0)
    K = 6
    s = rng.standard_normal(K)
    s = s - s.mean()                                  # zero-mean strengths
    mu = mu_from_strengths(s)

    borda = F_borda(mu)
    bt = F_bt(mu)
    rc = F_rc(mu)
    print(f"  true strengths s = {np.round(s,4)}")

    ok_b = check_monotone(borda, s, "Borda")
    ok_bt = check_monotone(bt, s, "BT")
    ok_rc = check_monotone(rc, s, "RC")

    # BT recovery strength: under BT, F_bt should be a * s  (linear, zero-mean).
    # (Edge log-odds ell_{jk} = log(mu[j,k,0]/mu[j,k,2]) = 2 a (s_j-s_k);
    #  LSQ recovers 2 a s.)  Check slope ratio across components.
    slope = np.polyfit(s, bt, 1)[0]
    rel = slope / (2 * 2.0)                          # / (2a)
    print(f"  BT recovery: F_bt vs s linear slope={slope:.4f} (expect {2*2.0:.1f}); ratio={rel:.3f}")
    ok_btlin = abs(rel - 1.0) < 0.02

    # RC stationary distribution should be monotone in s and sum to 1
    ok_rcsum = abs(rc.sum() - 1.0) < 1e-8
    print(f"  RC sums to 1: {ok_rcsum} (sum={rc.sum():.6f})")

    print("\n=== Jacobian checks (analytic vs finite-difference) ===")
    ok_jac = True
    for name, func in [("borda", jac_borda), ("bt", jac_bt)]:
        Ja = func(mu, K)
        Jn = jac_numeric(F_bt if name == "bt" else F_borda, mu, K)
        err = np.max(np.abs(Ja - Jn))
        ok = err < 1e-3
        ok_jac = ok_jac and ok
        print(f"  J_{name}: max|analytic-numeric| = {err:.2e}  {'OK' if ok else 'FAIL'}")
    # RC analytic jac vs finite-difference
    Jrc = jac_rc(mu, K)
    Jrc2 = jac_numeric(F_rc, mu, K)
    err = np.max(np.abs(Jrc - Jrc2))
    ok = err < 1e-3
    ok_jac = ok_jac and ok
    print(f"  J_rc analytic vs numeric: max diff = {err:.2e}  {'OK' if ok else 'FAIL'}")

    all_ok = ok_b and ok_bt and ok_rc and ok_btlin and ok_rcsum and ok_jac
    print("\nCLAIM 1 VERDICT:", "PASS" if all_ok else "FAIL")
    return all_ok


if __name__ == "__main__":
    ok = main()
    sys.exit(0 if ok else 1)
