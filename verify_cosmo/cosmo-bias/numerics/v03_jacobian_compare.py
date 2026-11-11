"""Why do config-b Fisher errors differ by ~2x between cosmolib (HMF grid) and the independent direct-colossus model?
Compare Jacobian columns (per bin) and Fisher matrices, with several finite-difference steps for both models."""
import os, sys
os.environ["OMP_NUM_THREADS"] = "2"
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, "/Users/boris/GitHub/sz-poisson-blending/analysis/cosmo/bias")
import cosmolib as CL
import v01_indep_model as V

model = CL.NaiveModel(5.0)
FIDv = np.array([V.FID[k] for k in V.NAMES])
fsky = 0.4
mu = model.counts(CL.FID).ravel() * fsky
muI = V.counts(V.FID).ravel() * fsky
sel = mu > 10


def J_cl(k, h):
    e = np.zeros(5); e[k] = h
    return (model.counts(CL.FID + e) - model.counts(CL.FID - e)).ravel() / (2 * h) * fsky


def J_ind(k, h):
    pp = dict(V.FID); pm = dict(V.FID); n = V.NAMES[k]; pp[n] += h; pm[n] -= h
    return (V.counts(pp) - V.counts(pm)).ravel() / (2 * h) * fsky


cols = {}
for k, steps in ((0, (4e-4, 2e-3, 6e-3)), (1, (4e-4, 2e-3, 6e-3)), (2, (1e-3, 4e-3, 1.6e-2)), (4, (1e-3, 1e-2))):
    for h in steps:
        cols[("cl", k, h)] = J_cl(k, h); cols[("ind", k, h)] = J_ind(k, h)
    ref = cols[("ind", k, steps[1])]
    for (src, kk, h), v in cols.items():
        if kk != k:
            continue
        r = (v / ref - 1)[sel & (np.abs(ref) > 1e-3 * np.abs(ref).max())]
        print(f"d mu/d {V.NAMES[k]:5s} {src:3s} step {h:.0e}: sum {v.sum():+.4e}; vs ind(step {steps[1]:.0e}) per-bin rel: "
              f"median {np.median(r):+.3e}, max|.| {np.abs(r).max():.3e}")


def fisher(J, m):
    return J.T @ (J / m[:, None])


def report(tag, J, m):
    C = np.linalg.inv(fisher(J, m)); s = np.sqrt(np.diag(C))
    S8t = CL.S8_of(*CL.FID[:2]); g = np.zeros(J.shape[1]); g[0] = 0.5 * S8t / CL.FID[0]; g[1] = S8t / CL.FID[1]
    w, U = np.linalg.eigh(fisher(J, m) * np.outer(1 / np.sqrt(np.diag(fisher(J, m))), 1 / np.sqrt(np.diag(fisher(J, m)))))
    print(f"  {tag}: sigma(Om,s8,omb)=({s[0]:.4f},{s[1]:.4f},{s[2]:.4f}) sigma(S8)={np.sqrt(g@C@g):.4f}; "
          f"normalised-Fisher eigenvalues {w}")


print("config b Fisher (fsky 0.4):")
for hO, hb in ((4e-4, 1e-3), (2e-3, 4e-3), (6e-3, 1.6e-2)):
    report(f"cosmolib steps ({hO},{hb})", np.stack([cols[("cl", 0, hO)], cols[("cl", 1, hO)], cols[("cl", 2, hb)]], 1), mu)
    report(f"indep    steps ({hO},{hb})", np.stack([cols[("ind", 0, hO)], cols[("ind", 1, hO)], cols[("ind", 2, hb)]], 1), muI)
report("hybrid: cosmolib Om,s8 (4e-4) + indep omb", np.stack([cols[("cl", 0, 4e-4)], cols[("cl", 1, 4e-4)], cols[("ind", 2, 4e-3)]], 1), mu)
report("hybrid: indep Om,s8 + cosmolib omb (1e-3)", np.stack([cols[("ind", 0, 2e-3)], cols[("ind", 1, 2e-3)], cols[("cl", 2, 1e-3)]], 1), mu)
# ---- profile likelihood along s8 for config b with cosmolib model (Asimov self data): is the curvature as Fisher says?
print("profile check (cosmolib, self data, config b): fix s8 = fid + k*sigma_F, refit Om, omb; Delta chi2 = 2 Delta lnL")
F_ = CL.Fitter(model, mu.reshape(model.nzb, model.nqb), ["Om", "s8", "omb"], fsky=fsky)
sF, _ = CL.derived_sigma(F_.fisher(CL.FID[:3]), [0, 1, 2])
for kk in (-1.0, 1.0, 2.0):
    s8v = CL.FID[1] + kk * sF["s8"]
    G = CL.Fitter(model, mu.reshape(model.nzb, model.nqb), ["Om", "omb"], fsky=fsky)
    base = CL.FID.copy(); base[1] = s8v
    G.full = (lambda b: (lambda x: np.concatenate([[x[0]], [b[1]], [x[1]], b[3:]])))(base)
    x, L, it = G.fit(x0=[CL.FID[0], CL.FID[2]])
    print(f"   s8 = fid {kk:+.0f} sigma_F ({s8v:.4f}): best Om {x[0]:.4f}, omb {x[1]:.4f}, Delta chi2 = {-2*L:.3f} (Gaussian: {kk**2:.1f})")
