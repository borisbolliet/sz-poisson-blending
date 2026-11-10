"""V2: independent full-sky Poisson MC (own sampler, own pair counter, new seeds) for Q1 and N_det (C1, C2).
Floor M>1e14 as in q1_configs_floor.py; NSKY skies per config; errors = scatter of per-sky ratios / sqrt(NSKY)
(sky = independent unit) and, for comparison, the claimed numbers."""
import os, sys, time
os.environ.setdefault("NUMBA_NUM_THREADS", "2"); os.environ.setdefault("OMP_NUM_THREADS", "2")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
from scipy.stats import poisson, chi2
import indep_model as im
from paircount import count_within

NSKY = int(sys.argv[1]) if len(sys.argv) > 1 else 20
yP = 33e-6 / (2.7255 * abs(143 / 56.7798 / np.tanh(143 / 56.7798 / 2) - 4))
CFG = {"base": (1.4, 4e-6, 0.173, "tinker08", 3.0), "noise_x0.5": (1.4, 2e-6, 0.173, "tinker08", 3.0),
       "beam_5": (5.0, 4e-6, 0.173, "tinker08", 3.0), "planck_like": (7.3, yP, 0.173, "tinker08", 3.0),
       "zmax_1": (1.4, 4e-6, 0.173, "tinker08", 1.0), "hmf_bocquet16": (1.4, 4e-6, 0.173, "bocquet16", 3.0)}
CLAIM = {("base", 1, 1): (1.757e-3, 3.2e-5), ("base", 5, 2): (9.598e-4, 2.5e-5), ("base", 5, 1): (4.336e-2, 1.6e-4),
         ("planck_like", 30, 1): (7.109e-2, 9.1e-4), ("planck_like", 1, 1): (1.17e-4, 3.9e-5), ("planck_like", 5, 1): (1.67e-3, 1.5e-4),
         ("noise_x0.5", 1, 1): (4.789e-3, None), ("noise_x0.5", 5, 1): (0.1134, None), ("beam_5", 1, 1): (6.51e-4, None),
         ("beam_5", 5, 1): (1.671e-2, None), ("zmax_1", 1, 1): (1.650e-3, None), ("zmax_1", 5, 1): (3.868e-2, None),
         ("hmf_bocquet16", 1, 1): (1.621e-3, None), ("hmf_bocquet16", 5, 1): (3.989e-2, None)}
TH = np.array([1.0, 2.0, 5.0, 10.0, 30.0])
ARC = im.ARC
zs = []
t0 = time.time()
for ic, (name, (beam, noise, sig, model, zmax)) in enumerate(CFG.items()):
    inst = im.Instrument(beam, noise)
    Nan = im.N_det(inst, 5.0, sig=sig, model=model, Mmin=1e14, zmax=zmax)
    Sig = Nan / (4 * np.pi)  # per sr
    Nd, num, den = [], {}, []
    for s in range(NSKY):
        c = im.draw(910000 + 1000 * ic + s, inst, sig=sig, Mmin=1e14, zmax=zmax, model=model)
        sel = c["q"] + c["n"] > 5.0
        Nd.append(sel.sum())
        cnt = count_within(c["ra"][sel], c["dec"][sel], TH)
        den.append(sel.sum())
        for it, th in enumerate(TH):
            for k in (1, 2, 3):
                num.setdefault((th, k), []).append(np.sum(cnt[:, it] >= k))
    Nd = np.array(Nd, float); den = np.array(den, float)
    print(f"{name:14s} N_det(M>1e14,q>5): own MC {Nd.mean():9.1f} +- {Nd.std(ddof=1)/np.sqrt(NSKY):.1f} vs own analytic {Nan:9.1f} "
          f"({(Nd.mean()-Nan)/np.sqrt(Nan/NSKY):+.2f} sigma_Poisson); Sigma_det = {Nan/(4*np.pi)*ARC**2*3600:.4f}/deg2  [{time.time()-t0:.0f}s]", flush=True)
    for th in TH:
        lam = Sig * 2 * np.pi * (1 - np.cos(th * ARC))
        for k in (1, 2, 3):
            nu = np.array(num[(th, k)], float)
            if nu.sum() < 10:
                continue
            r = nu.sum() / den.sum()
            # per-sky jackknife
            rj = (nu.sum() - nu) / (den.sum() - den)
            e = np.sqrt((NSKY - 1) / NSKY * np.sum((rj - rj.mean()) ** 2))
            pa = poisson.sf(k - 1, lam); z = (r - pa) / e; zs.append(z)
            cl = CLAIM.get((name, int(th), k))
            cls = ""
            if cl:
                d = r - cl[0]; ee = np.hypot(e, cl[1]) if cl[1] else e
                cls = f" | claimed MC {cl[0]:.4e} (own-claimed {d/ee:+.2f} sigma)"
            print(f"     theta={th:4.0f}' k>={k}: own MC {r:.4e} +- {e:.1e}  Poisson {pa:.4e} ({z:+.2f} sigma){cls}")
zs = np.array(zs)
print(f"\nall: {len(zs)} tests, chi2 = {np.sum(zs**2):.1f}/{len(zs)} (p = {chi2.sf(np.sum(zs**2), len(zs)):.3f}), max|z| = {np.abs(zs).max():.2f}; total {time.time()-t0:.0f}s")
