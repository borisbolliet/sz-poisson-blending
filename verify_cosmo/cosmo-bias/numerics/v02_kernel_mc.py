"""Monte Carlo check of the catalogue blended kernel K_b(theta, q_A; e) = P(dominant and q_obs > e) of c03 (FFT).
Neighbours with x > 0.02 drawn explicitly (Poisson number, x from the W-weighted bin means), x < 0.02 as a Gaussian,
unit noise, map mean mu_exact removed; lost if any neighbour has x >= q_A."""
import numpy as np, time
from scipy.special import ndtr
ROOT = "/Users/boris/GitHub/sz-poisson-blending"
d = np.load(f"{ROOT}/data/blending/analytic.npz")
k = np.load(f"{ROOT}/data_cosmo/kernel_catalogue.npz")
Kb, QA, E = k["Kb"], k["QA"], k["EDGES"]
rng = np.random.default_rng(5)
NS = 400_000
t0 = time.time()
maxdiff = 0
for i in (5, 20, 30, 39):
    W, WX, WX2 = d["W"][i].sum(0), d["WX"][i].sum(0), d["WX2"][i].sum(0)
    mu = d["mu_exact"][i].sum()
    xb = np.where(W > 0, WX / np.where(W > 0, W, 1), 0.0)
    small = (W > 0) & (xb < 0.02); big = (W > 0) & ~small
    ms, vs = WX[small].sum(), WX2[small].sum()
    lam = W[big].sum(); pb = W[big] / lam; xbig = xb[big]
    n = rng.poisson(lam, NS)
    idx = rng.choice(xbig.size, n.sum(), p=pb)
    x = xbig[idx]
    sid = np.repeat(np.arange(NS), n)
    tot = np.bincount(sid, weights=x, minlength=NS)
    xmax = np.zeros(NS); np.maximum.at(xmax, sid, x)
    delta = tot + ms + np.sqrt(vs) * rng.standard_normal(NS) + rng.standard_normal(NS) - mu
    for qa in (3.0, 4.5, 6.0, 10.0, 30.0):
        j = int(np.argmin(np.abs(QA - qa)))
        qq = QA[j]
        dom = xmax < qq
        for e in (5.0, 6.0, 7.0):
            ie = int(np.argmin(np.abs(E - e)))
            Kmc = np.mean(dom & (qq + delta > e)); se = np.sqrt(Kmc * (1 - Kmc) / NS)
            dif = (Kb[i, j, ie] - Kmc) / max(se, 1e-6)
            maxdiff = max(maxdiff, abs(dif))
            print(f"theta {d['th'][i]:6.2f}' q_A {qq:5.2f} e {e}: K_fft {Kb[i,j,ie]:.5f}  K_mc {Kmc:.5f} +- {se:.5f} "
                  f"({dif:+.1f} se)  naive Phi {ndtr(qq-e):.5f}  Pdom fft {k['Pdom'][i,j]:.5f} mc {dom.mean():.5f}")
    print(f"[{time.time()-t0:.0f}s]", flush=True)
print(f"max |K_fft - K_mc| / se = {maxdiff:.1f}")
