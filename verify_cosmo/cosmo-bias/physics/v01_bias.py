"""Independent recomputation of the Asimov cosmology bias (naive model fitted to the blended expectation).
Uses vmodel.py (own model). Blending changes Delta taken from data_cosmo/truth_catalogue.npz (catalogue P(D))
and data_cosmo/truth_maps.npz (maps), as relative changes applied to MY naive expectation.
usage: v01_bias.py [qcut] [configs] [routes]
"""
import os, sys, time
os.environ.setdefault("OMP_NUM_THREADS", "2")
import numpy as np
from scipy.optimize import minimize
from scipy.stats import chi2, ncx2
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import vmodel as V

DC = "/Users/boris/GitHub/sz-poisson-blending/data_cosmo"
qc = float(sys.argv[1]) if len(sys.argv) > 1 else 5.0
CONF = {"a": ([0, 1], {}), "b": ([0, 1, 2], {}), "c": ([0, 1, 2, 4], {}),
        "d": ([0, 1, 2, 3, 4], {2: 0.032}), "e": ([0, 1, 2, 3, 4], {})}
confs = sys.argv[2].split(",") if len(sys.argv) > 2 else list(CONF)
routes = sys.argv[3].split(",") if len(sys.argv) > 3 else ["self", "cat", "clean", "all"]
s = f"{qc:.0f}"
t0 = time.time()
N0 = V.counts(V.FID, qc)
print(f"q>{qc}: my naive full-sky total {N0.sum():,.1f}  [{time.time()-t0:.1f}s]")
tc = np.load(f"{DC}/truth_catalogue.npz"); tm = np.load(f"{DC}/truth_maps.npz")
sc = float(tm["fullsky_deg2"] / tm["area_deg2"])
Nn = tc[f"Nn_q{s}"]; N0t = tc[f"N0_q{s}"]
print(f"their naive total {N0t.sum():,.1f}; mine/theirs-1 = {N0.sum()/N0t.sum()-1:+.2e}; per-bin max|rel| (mu>10) "
      f"{np.abs(N0/N0t-1)[N0t>10].max():.2e}")
rel = {"cat": tc[f"D_q{s}"] / Nn,
       "clean": (tm[f"clean{s}"] - tm[f"iso{s}"]).sum(0) * sc / N0t,
       "all": (tm[f"map{s}"] - tm[f"iso{s}"]).sum(0) * sc / N0t}
for k, r in rel.items():
    print(f"  route {k}: total blending change {np.nansum(r*N0)/N0.sum():+.4f}; per z: {np.round(np.nansum(r*N0,1)/N0.sum(1),4)}")
DATA = {"self": N0}
for k, r in rel.items():
    DATA[k] = np.clip(N0 * (1 + np.nan_to_num(r)), 0, None)


def nll(th, d, fsky, pri):
    mu = V.counts(th, qc, fsky)
    with np.errstate(divide="ignore", invalid="ignore"):
        t = np.where(d > 0, d * np.log(d / mu), 0.0)
    dev = 2 * np.sum(mu - d + t)
    return dev + sum(((th[i] - V.FID[i]) / sp) ** 2 for i, sp in pri.items()), dev


SV = sys.argv[4].split(",") if len(sys.argv) > 4 else ["SO40", "full"]
for fsky, sv in [(f_, s_) for f_, s_ in [(0.4, "SO40"), (1.0, "full")] if s_ in SV]:
    for cn in confs:
        free, pri = CONF[cn]
        mu0 = V.counts(V.FID, qc, fsky).ravel()
        J = V.jac(V.FID.copy(), free, qc, fsky)
        F = J.T @ (J / mu0[:, None])
        for i, sp in pri.items():
            k = free.index(i); F[k, k] += 1 / sp ** 2
        C = np.linalg.inv(F); sig = np.sqrt(np.diag(C))
        gS = np.array([0.5 * V.S8(*V.FID[:2]) / V.FID[0], V.S8(*V.FID[:2]) / V.FID[1]] + [0] * (len(free) - 2))
        sS8 = np.sqrt(gS @ C @ gS)
        print(f"  [{sv} {cn}] Fisher sigma at truth (free {[V.NAMES[i] for i in free]}, priors {pri}): " + ", ".join(f"{V.NAMES[i]} {sig[k]:.5f}" for k, i in enumerate(free)) + f", S8 {sS8:.5f}", flush=True)
        for dn in routes:
            d = DATA[dn] * fsky
            g = J.T @ ((d.ravel() - mu0) / mu0)
            xlin = C @ g
            lo = np.array([{0: 0.2, 1: 0.6, 2: 0.3, 3: 1.0, 4: 1e-3}[i] for i in free])
            hi = np.array([{0: 0.5, 1: 1.05, 2: 1.5, 3: 2.6, 4: 1.0}[i] for i in free])
            best = None
            for start in (V.FID[free], np.clip(V.FID[free] + xlin, lo + 1e-6, hi - 1e-6)):
                def f(u):
                    th = V.FID.copy(); th[free] = start + u * sig
                    return nll(th, d, fsky, pri)[0]
                r = minimize(f, np.zeros(len(free)), method="Powell", bounds=list(zip((lo - start) / sig, (hi - start) / sig)),
                             options=dict(xtol=1e-3, ftol=1e-9, maxfev=3000))
                if best is None or r.fun < best[0]:
                    best = (r.fun, start + r.x * sig)
            th = V.FID.copy(); th[free] = best[1]
            atb = [V.NAMES[i] for k, i in enumerate(free) if best[1][k] <= lo[k] + 1e-3 * sig[k]]
            _, dev = nll(th, d, fsky, pri)
            ndof = d.size - len(free)
            pw = float(ncx2.sf(chi2.ppf(0.95, ndof), ndof, dev)) if dev > 1e-8 else 0.05
            dS8 = V.S8(th[0], th[1]) - V.S8(*V.FID[:2])
            bb = (th[free] - V.FID[free]) / sig
            nuis = " ".join(f"{V.NAMES[i]}{th[i]-V.FID[i]:+.4f}({bb[k]:+.2f})" for k, i in enumerate(free) if i >= 2)
            print(f"q{s} {sv} {cn} {dn:5s}: sig(Om,s8,S8)=({sig[0]:.4f},{sig[1]:.4f},{sS8:.4f}) | "
                  f"bOm {bb[0]:+.2f} bs8 {bb[1]:+.2f} bS8 {dS8/sS8:+.2f} | lin bOm {xlin[0]/sig[0]:+.2f} bs8 {xlin[1]/sig[1]:+.2f} "
                  f"bS8 {gS@xlin/sS8:+.2f} | ABS dOm {th[0]-V.FID[0]:+.4f} ds8 {th[1]-V.FID[1]:+.4f} dS8 {dS8:+.4f} | {nuis} | "
                  f"chi2min {dev:.2f} ndof {ndof} power {pw:.2f} atbound {atb}  [{time.time()-t0:.0f}s]", flush=True)
print(f"total {time.time()-t0:.0f}s")
