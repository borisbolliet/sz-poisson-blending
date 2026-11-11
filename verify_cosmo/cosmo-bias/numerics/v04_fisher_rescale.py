"""(1) Fisher sigmas of cosmolib with SAFE finite-difference steps (>= the ln q0 interpolation spacing) vs the
    independent direct-colossus model vs the stored (default-step) sigmas in fits_final.json.
(2) Re-express the stored ML shifts (fits_final.json d_<p>) in units of the corrected Fisher sigma.
(3) Poisson realisations, config b, SO40, self data: scatter of the TRUE ML (Nelder-Mead on cosmolib lnL),
    vs the default-step Fisher sigma and the safe-step one.
"""
import os, sys, json, time
os.environ["OMP_NUM_THREADS"] = "2"
import numpy as np
from scipy.optimize import minimize
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, "/Users/boris/GitHub/sz-poisson-blending/analysis/cosmo/bias")
import cosmolib as CL
import v01_indep_model as V

t0 = time.time()
J_ind = np.load("/private/tmp/claude-501/-Users-boris-GitHub/0935154e-1cde-4eab-ab64-4515ca671a83/scratchpad/v01_jac.npz")
Jind, muind = J_ind["J"], J_ind["mu"]
CFG = {"a": (["Om", "s8"], {}), "b": (["Om", "s8", "omb"], {}), "c": (["Om", "s8", "omb", "slnY"], {}),
       "d": (["Om", "s8", "omb", "alpha", "slnY"], {"omb": 0.032}), "d3": (["Om", "s8", "omb", "alpha", "slnY"], {"omb": 0.024}),
       "d5": (["Om", "s8", "omb", "alpha", "slnY"], {"omb": 0.040}), "e": (["Om", "s8", "omb", "alpha", "slnY"], {})}
SAFE = np.array([6e-3, 6e-3, 1.6e-2, 2e-2, 2e-2])
R = json.load(open(f"{CL.DC}/fits_final.json"))
S8t = CL.S8_of(*CL.FID[:2])


def sigmas(J, mu, free, pri):
    F = J.T @ (J / mu[:, None])
    for k, s in pri.items():
        F[free.index(k), free.index(k)] += 1 / s ** 2
    C = np.linalg.inv(F); s = dict(zip(free, np.sqrt(np.diag(C))))
    g = np.zeros(len(free)); g[0] = 0.5 * S8t / CL.FID[0]; g[1] = S8t / CL.FID[1]; s["S8"] = np.sqrt(g @ C @ g)
    return s


safe_sig = {}
for qc in (5.0, 6.0):
    model = CL.NaiveModel(qc)
    N0 = model.counts(CL.FID)
    Jc = []
    for i in range(5):
        e = np.zeros(5); e[i] = SAFE[i]
        Jc.append(((model.counts(CL.FID + e) - model.counts(CL.FID - e)) / (2 * SAFE[i])).ravel())
    Jc = np.stack(Jc, 1)
    print(f"\n===== q_cut {qc} =====")
    for sv, fsky in (("SO40", 0.4), ("full", 1.0)):
        for cn, (free, pri) in CFG.items():
            idx = [CL.PNAMES.index(p) for p in free]
            ss = sigmas(Jc[:, idx] * fsky, N0.ravel() * fsky, free, pri)
            safe_sig[(qc, sv, cn)] = ss
            line = f"{sv} {cn:2s} sigma safe-step cosmolib: " + ", ".join(f"{p} {ss[p]:.4f}" for p in free + ["S8"])
            if qc == 5.0:
                si = sigmas(Jind[..., idx].reshape(-1, len(idx)) * fsky, muind.ravel() * fsky, free, pri)
                line += " | indep: " + ", ".join(f"{p} {si[p]:.4f}" for p in free + ["S8"])
            r0 = next(r for r in R if r["qcut"] == qc and r["survey"] == sv and r["config"] == cn and r["data"] == "cat")
            line += " | stored (default step): " + ", ".join(f"{p} {r0['s_'+p]:.4f}" for p in free + ["S8"])
            print(line)
            for dn in ("cat", "map_clean", "map", "map_raw"):
                r = next(r for r in R if r["qcut"] == qc and r["survey"] == sv and r["config"] == cn and r["data"] == dn)
                print(f"      {dn:9s}: stored bias " + " ".join(f"{p} {r['b_'+p]:+.2f}" for p in free + ["S8"]) +
                      "  ->  in safe-step sigma: " + " ".join(f"{p} {r['d_'+p]/ss[p]:+.2f}" for p in free + ["S8"]))
print(f"[{time.time()-t0:.0f}s]", flush=True)

# ---- (3) realisations, config b, SO40, true ML by Nelder-Mead
model = CL.NaiveModel(5.0); fsky = 0.4; N0 = model.counts(CL.FID)
rng = np.random.default_rng(123)
free = ["Om", "s8", "omb"]
ss = safe_sig[(5.0, "SO40", "b")]
sc = np.array([ss[p] for p in free])
res, res_fs = [], []
NR = int(os.environ.get("NR", "60"))
for r in range(NR):
    d = rng.poisson(N0 * fsky).astype(float)
    F = CL.Fitter(model, d, free, fsky=fsky)
    x_fs, _, _ = F.fit()
    best = None
    for start in (CL.FID[:3], x_fs):
        o = minimize(lambda u: -F.lnL(start + u * sc), np.zeros(3), method="Nelder-Mead",
                     options=dict(xatol=1e-3, fatol=1e-6, maxiter=3000, initial_simplex=np.vstack([np.zeros(3), np.eye(3)])))
        if best is None or o.fun < best[1]:
            best = (start + o.x * sc, o.fun)
    res.append(best[0]); res_fs.append(x_fs)
res = np.array(res); res_fs = np.array(res_fs)
r0 = next(r for r in R if r["qcut"] == 5.0 and r["survey"] == "SO40" and r["config"] == "b" and r["data"] == "cat")
for k, p in enumerate(free + ["S8"]):
    v = res[:, k] if p != "S8" else CL.S8_of(res[:, 0], res[:, 1])
    vf = res_fs[:, k] if p != "S8" else CL.S8_of(res_fs[:, 0], res_fs[:, 1])
    print(f"config b SO40 self, {NR} Poisson realisations: {p:4s} std(NM ML) {v.std(ddof=1):.4f}  std(Fisher-scoring fit) "
          f"{vf.std(ddof=1):.4f} | default-step Fisher {r0['s_'+p]:.4f} | safe-step Fisher {ss[p]:.4f}  "
          f"=> std(NM)/default {v.std(ddof=1)/r0['s_'+p]:.2f}, std(NM)/safe {v.std(ddof=1)/ss[p]:.2f} (+-{1/np.sqrt(2*(NR-1)):.2f})")
print(f"total {time.time()-t0:.0f}s")
