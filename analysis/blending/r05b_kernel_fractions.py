"""[repair after verification] Kernel-interpolation systematic, step B: full-sky blending fractions with the
production (NT=40) and finer (NT=79) cross-response tables, same primary population H, same per-node
algebra as b03 (vectorised re-implementation; the NT=40 column must reproduce b03_analytic.log).
Also: mu, Var (x<500), Lambda(x>1) and P(delta>t) at the random-point nodes under both kernels."""
import os, sys, time
os.environ.setdefault("NUMBA_NUM_THREADS", "2"); os.environ.setdefault("OMP_NUM_THREADS", "2")
import numpy as np
from scipy.special import ndtr

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import blendlib as bl

t0 = time.time()
A = dict(np.load(os.path.join(bl.DATA, "analytic.npz")))
R79 = dict(np.load(os.path.join(bl.DATA, "rate_nt79.npz")))
TH = A["th"]; H = A["H"]; QN = A["QNODES"]
KER = {"NT=40": dict(W=A["W"], WX=A["WX"], WX2=A["WX2"], CAP=A["CAP"], mu=A["mu_exact"]),
       "NT=79": dict(W=R79["W"], WX=R79["WX"], WX2=R79["WX2"], CAP=R79["CAP"], mu=R79["mu"])}

# ---- random-point quantities
pdc = bl.PD(N=2 ** 20, dd=0.001, x_small=0.002)
print("random-point quantities, floor 1e13 (all classes) and 1e14:")
for t in (0.524, 0.946, 1.983, 4.153, 7.505):
    i = int(np.argmin(np.abs(TH - t)))
    s = f"  thA={TH[i]:.3f}':"
    for fl in ("1e13", "1e14"):
        cl = list(bl.FLOORS[fl]); vals = {}
        for nm, K in KER.items():
            Wf, WXf, WX2f = K["W"][i, cl].sum(0), K["WX"][i, cl].sum(0), K["WX2"][i, cl].sum(0)
            p = pdc.pdf(Wf, WXf, WX2f) * np.exp(-K["CAP"][i, cl].sum())
            tail = 1 - np.cumsum(p)
            vals[nm] = np.array([K["mu"][i, cl].sum(), WX2f.sum(), bl.PD.lam_above(Wf, WXf, 1.0) + K["CAP"][i, cl].sum()]
                                + [np.interp(tt, pdc.grid + 0.5 * pdc.dd, tail) for tt in (0.5, 1, 2, 5, 10, 20)])
        r = vals["NT=79"] / vals["NT=40"] - 1
        s += f" [floor {fl}: NT79/NT40-1: mu {r[0]:+.4f} var {r[1]:+.4f} Lam(x>1) {r[2]:+.4f} P(>0.5,1,2,5,10,20) " + " ".join(f"{x:+.4f}" for x in r[3:]) + "]"
    print(s, flush=True)

# ---- full-sky fractions
pd_small = bl.PD(N=2 ** 18, dd=0.001, x_small=0.002)
pd_big = bl.PD(N=2 ** 19, dd=0.002, x_small=0.004)
QC = (4.0, 5.0, 7.0); EPS = (0.1, 0.2, 0.5)
res = {}
for nm, K in KER.items():
    for fl in ("1e13", "1e14"):
        cl = list(bl.FLOORS[fl])
        acc = {}
        for i in range(len(TH)):
            if H[i].sum() == 0:
                continue
            Wf, WXf, WX2f = K["W"][i, cl].sum(0), K["WX"][i, cl].sum(0), K["WX2"][i, cl].sum(0)
            mu = K["mu"][i, cl].sum(); capf = K["CAP"][i, cl].sum()
            for k, qa in enumerate(QN):
                h = H[i, k]
                if h == 0:
                    continue
                xcut = min(qa, bl.X_CAP)
                pdom = np.exp(-(bl.PD.lam_above(Wf, WXf, xcut) + capf))
                pd_ = pd_small if xcut < 20 else pd_big
                p = pd_.pdf(Wf, WXf, WX2f, xcut=xcut); p /= p.sum()
                nz = int(np.nonzero(p > 1e-18)[0].max()) + 1; p = p[:nz]; g = pd_.grid[:nz]
                for c in (0, 1):
                    sh = c * mu; dpr = g - sh
                    for qc in QC:
                        Ph = ndtr(qa + dpr - qc); P0 = ndtr(qa - qc)
                        def add(key, v):
                            acc[key] = acc.get(key, 0.0) + h * pdom * v
                        add((c, qc, "det"), np.sum(p * Ph))
                        add((c, qc, "single"), P0 / pdom)       # N_single = sum h Phi0 (undo pdom)
                        add((c, qc, "B3"), max(np.sum((p * (Ph - P0))[dpr > 0]), 0.0))
                        add((c, qc, "B3d_1.0"), max(np.sum((p * (Ph - P0))[dpr > 1.0]), 0.0))
                        add((c, qc, "lost"), max(np.sum((p * (P0 - Ph))[dpr <= 0]), 0.0))
                        for e in EPS:
                            add((c, qc, f"B2_{e}"), np.sum((p * Ph)[dpr > e * qa]))
                        if c == 1:
                            add((c, qc, "B2neg_0.1"), np.sum((p * Ph)[dpr <= -0.1 * qa]))
        for c in (0, 1):
            for qc in QC:
                Nd = acc[(c, qc, "det")]; Ns = acc[(c, qc, "single")]
                res[(nm, fl, c, qc, "ratio")] = Nd / Ns
                for key in ["B3", "B3d_1.0"] + [f"B2_{e}" for e in EPS] + (["B2neg_0.1"] if c == 1 else []):
                    res[(nm, fl, c, qc, key)] = acc[(c, qc, key)] / Nd
                res[(nm, fl, c, qc, "lost")] = acc[(c, qc, "lost")] / Ns
        print(f"  {nm} floor {fl} done {time.time()-t0:.0f}s", flush=True)

print("\nfull-sky fractions: NT=40 (this re-implementation) | NT=79 | relative change")
keys = sorted(set(k[1:] for k in res))
for fl, c, qc, st in keys:
    a = res[("NT=40", fl, c, qc, st)]; b = res[("NT=79", fl, c, qc, st)]
    print(f"  floor {fl} c={c} q_cut={qc:.0f} {st:10s}: {a:.5f} | {b:.5f} | {b/a-1:+.4f}")
rel = np.array([res[("NT=79",) + k] / res[("NT=40",) + k] - 1 for k in keys if res[("NT=40",) + k] > 1e-3])
print(f"max |relative change| over {len(rel)} fractions > 1e-3: {np.abs(rel).max():.4f}; median {np.median(np.abs(rel)):.4f}")
print(f"total {time.time()-t0:.0f}s")
