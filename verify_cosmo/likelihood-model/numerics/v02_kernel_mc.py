"""Independent check of the blended kernel table G (data_cosmo/G_fid_b03rates.npz) by direct Monte Carlo of the
compound-Poisson sum: for selected (theta node, q_A node), draw Poisson counts in EVERY rate-histogram bin
(data/blending/analytic.npz, all three mass classes, x < q_A only), delta = sum n_j xbar_j, y = N(0,1) + delta - mu.
No FFT, no Gaussian small-x approximation.  Compare the CDF of y with G/P_dom, and P_dom with exp(-Lambda(x>=q_A)).
Also report the sign/size of the median shift and the skewness."""
import os
os.environ["OMP_NUM_THREADS"] = "2"
import numpy as np

ROOT = "/Users/boris/GitHub/sz-poisson-blending"
A = np.load(f"{ROOT}/data/blending/analytic.npz")
K = np.load(f"{ROOT}/data_cosmo/G_fid_b03rates.npz")
G, Pdom, mu, th, QN, TG = K["G"], K["Pdom"], K["mu"], K["th"], K["qnodes"], K["tgrid"]
W = A["W"].sum(1); WX = A["WX"].sum(1); CAP = A["CAP"].sum(1)
mu_chk = A["WX"].sum((1, 2)) + A["CAPX"].sum(1)
print(f"mu in G file vs sum(WX)+CAPX from rates: max rel diff {np.abs(mu/mu_chk-1).max():.2e}")
rng = np.random.default_rng(12345)
NS, CH = 200_000, 10_000
for i in (5, 15, 25, 35):
    with np.errstate(invalid="ignore", divide="ignore"):
        xb = np.where(W[i] > 0, WX[i] / W[i], 0.0)
    for qa in (2.0, 5.0, 10.0):
        k = int(np.argmin(np.abs(QN - qa)))
        keep = (W[i] > 0) & (xb < QN[k])
        lam_above = W[i][(W[i] > 0) & (xb >= QN[k])].sum() + CAP[i]
        pd_mine = np.exp(-lam_above)
        Wk, xk = W[i][keep], xb[keep]
        ys = []
        for c in range(NS // CH):
            n = rng.poisson(Wk[None, :], size=(CH, Wk.size))
            ys.append(n @ xk + rng.standard_normal(CH) - mu[i])
        y = np.sort(np.concatenate(ys))
        F_mc_at = lambda t: np.searchsorted(y, t, side="right") / y.size
        tt = np.quantile(y, [0.01, 0.05, 0.16, 0.5, 0.84, 0.95, 0.99, 0.999])
        Fg = np.interp(tt, TG, G[i, k] / Pdom[i, k])
        Fm = F_mc_at(tt)
        err = np.sqrt(Fm * (1 - Fm) / y.size)
        med = np.median(y); sk = np.mean((y - y.mean()) ** 3) / y.std() ** 3
        # median from G
        med_G = np.interp(0.5, G[i, k] / Pdom[i, k], TG)
        print(f"theta={th[i]:6.2f}' q_A={QN[k]:.2f}: P_dom G {Pdom[i,k]:.5f} mine {pd_mine:.5f} | CDF diff (G - MC) at MC quantiles "
              f"{np.array2string(Fg - Fm, precision=4)} | max |pull| {np.max(np.abs(Fg-Fm)/np.maximum(err,1e-6)):.2f} | "
              f"median MC {med:+.4f} G {med_G:+.4f}; mean MC {y.mean():+.4f}; skew MC {sk:.2f}", flush=True)
