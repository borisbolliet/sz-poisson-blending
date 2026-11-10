"""V04: blending statistics from the independent torus MC (v02) vs the analysis' analytic values (analytic.npz)."""
import os, glob, sys
import numpy as np
import indep_model as im

A = dict(np.load("/Users/boris/GitHub/sz-poisson-blending/data/blending/analytic.npz"))
H = A["H"]; QC = list(A["QCUTS"]); EPSA = list(A["EPS"]); FL = list(A["FLOORS"])
FLOORS = {"1e13": [0, 1, 2], "3e13": [1, 2], "1e14": [2]}
files = sorted(glob.glob(os.path.join(im.HERE, "torus", "patch_*.npz")))
d = [dict(np.load(f)) for f in files]
npch = len(d)
L = 600.0
area_deg = npch * (L / 60) ** 2
print(f"{npch} patches, {area_deg:.0f} deg^2, halos {sum(x['nhalo'] for x in d):,}")
# ensemble ybar from my model
ze, lme, dens = im.hmf_grid(); cc = im.cell_counts(ze, lme, dens)
zc = 0.5 * (ze[1:] + ze[:-1]); Mc = np.exp(0.5 * (lme[1:] + lme[:-1])); Z, MM = np.meshgrid(zc, Mc, indexing="ij")
Yc = im.Ymed(MM, Z) * np.exp(0.5 * im.SIG_LNY ** 2)
cls = np.searchsorted([1e13, 3e13, 1e14], MM, side="right") - 1
YB = np.array([(cc * Yc)[cls == c].sum() for c in range(3)])
yr = np.array([x["ybar"] for x in d])
print("ybar ensemble mine", YB, "; realised mean over patches", yr.mean(0), "±", yr.std(0, ddof=1) / np.sqrt(npch))
MUF = lambda th: (im.sigmaY_fast(th) * im.RATIO ** 2 / im.NW)[:, None] * YB[None, :]  # (n,3)


def jk(num_p, den_p):
    num_p = np.asarray(num_p, float); den_p = np.asarray(den_p, float)
    r = num_p.sum() / den_p.sum()
    rj = (num_p.sum() - num_p) / (den_p.sum() - den_p)
    n = len(num_p)
    return r, np.sqrt((n - 1) / n * ((rj - rj.mean()) ** 2).sum())


zs = []
def rep(name, num, den, an):
    r, e = jk(num, den)
    z = (r - an) / e if e > 0 else np.nan
    if np.isfinite(z) and np.sum(num) >= 20:
        zs.append((name, z))
    return f"{name} {r:.4f}±{e:.4f}|{an:.4f}({z:+.1f})"

# ---------------- random points
print("\nRandom points (exact filter at theta_A): delta = Yhat/sigma (raw). mine vs analysis analytic")
rth = d[0]["rand_th"]
for k, t in enumerate(rth):
    inode = int(np.argmin(np.abs(A["th"] - t)))
    for fi, fl in enumerate(FL):
        cl = FLOORS[fl]
        per = [x[f"rD_{k}"][:, cl].sum(1) for x in d]
        allv = np.concatenate(per)
        mu_e = (im.sigmaY_fast(np.array([t])) * im.RATIO ** 2 / im.NW)[0] * YB[cl].sum()
        mm = np.array([p.mean() for p in per])
        s = f"  thA={t:.4f}' floor {fl}: mean {mm.mean():.4f}±{mm.std(ddof=1)/np.sqrt(npch):.4f} (my ens mu {mu_e:.4f}; analysis mu {A['mu_exact'][inode, cl].sum():.4f}); "
        s += f"median {np.median(allv):.4f} (analysis {A['QUANT'][fi, inode, 1]:.4f}, node th {A['th'][inode]:.4f}); "
        for it, tt in enumerate(A["CCDF_T"]):
            if tt in (0.1, 0.5, 1.0, 3.0, 10.0):
                pp = np.array([(p > tt).mean() for p in per])
                s += f"P>{tt:g} {pp.mean():.4e}±{pp.std(ddof=1)/np.sqrt(npch):.1e}|{A['CCDF'][fi, inode, it]:.4e} "
        print(s)
    nx1 = np.concatenate([x[f"rX1_{k}"] for x in d])
    print(f"     neighbours with x>1 per filter: total {nx1.sum(1).mean():.4f}±{nx1.sum(1).std()/np.sqrt(len(nx1)):.4f}; per class {np.round(nx1.mean(0), 4)}")
    if abs(t - 1.0975) < 1e-3:
        dd = np.concatenate([x[f"rD_{k}"] for x in d])
        print(f"     sample variance per class (raw delta) {np.round(dd.var(0), 3)}, total {dd.sum(1).var():.3f} (heavy-tailed)")

# ---------------- primaries
print("\nPrimaries q_A>0.5: fractions (mine, patch jackknife) | analysis analytic")
q = [x["q"] for x in d]; n = [x["n"] for x in d]; th = [x["th"] for x in d]
for fi, fl in enumerate(FL):
    cl = FLOORS[fl]
    for c in (0, 1):
        for ic, qc in enumerate(QC):
            N = dict(det=[], single=[], B3=[], B3d1=[], lost=[], sat=[], **{f"B2_{e}": [] for e in (0.1, 0.2, 0.5, 1.0)}, B2n=[])
            for x in d:
                dl = x["D"][:, cl].sum(1); xm = x["xmax"][:, cl].max(1)
                mu = MUF(x["th"])[:, cl].sum(1)
                dp = dl - c * mu
                qo = x["q"] + x["n"]; qb = qo + dp
                dom = x["q"] > xm
                det = (qb > qc) & dom; single = qo > qc
                N["det"].append(det.sum()); N["single"].append(single.sum())
                N["B3"].append((det & (qo < qc)).sum()); N["B3d1"].append((det & (qo < qc) & (dp > 1)).sum())
                N["lost"].append((single & dom & (qb < qc)).sum()); N["sat"].append((single & ~dom).sum())
                for e in (0.1, 0.2, 0.5, 1.0):
                    N[f"B2_{e}"].append((det & (dp / x["q"] > e)).sum())
                N["B2n"].append((det & (dp / x["q"] < -0.1)).sum())
            Ns_an = (H * A["Phi_single"][ic][None]).sum(); Nd_an = (H * A["Pdet"][fi, c, ic]).sum()
            f = lambda arr: (H * arr).sum() / Nd_an
            s = f" floor {fl} c={c} qcut={qc:.0f}: Nsingle/deg2 {np.sum(N['single'])/area_deg:.3f}|{Ns_an/41252.96:.3f} "
            s += rep("ratio", N["det"], N["single"], Nd_an / Ns_an) + " "
            for e in (0.1, 0.2, 0.5, 1.0):
                s += rep(f"B2_{e}", N[f"B2_{e}"], N["det"], f(A["Pb2"][fi, c, ic, :, :, EPSA.index(e)])) + " "
            if c == 1:
                s += rep("B2neg0.1", N["B2n"], N["det"], f(A["Pb2n"][fi, c, ic, :, :, EPSA.index(0.1)])) + " "
            s += rep("B3", N["B3"], N["det"], f(A["Pb3"][fi, c, ic])) + " "
            s += rep("B3d1", N["B3d1"], N["det"], f(A["Pb3d"][fi, c, ic, :, :, 1])) + " "
            s += rep("lost", N["lost"], N["single"], (H * A["Plost"][fi, c, ic]).sum() / Ns_an) + " "
            s += rep("sat", N["sat"], N["single"], (H * A["Psat"][fi, ic]).sum() / Ns_an)
            print(s, flush=True)
z = np.array([v for _, v in zs])
print(f"\nall tests with >=20 events: {len(z)}, rms {np.sqrt(np.mean(z**2)):.2f}, max|z| {np.abs(z).max():.2f}, frac|z|>2 {np.mean(np.abs(z)>2):.3f}")
for nm in sorted(set(k for k, _ in zs)):
    zz = np.array([v for k, v in zs if k == nm]); print(f"   {nm:9s} n={len(zz):3d} rms {np.sqrt(np.mean(zz**2)):.2f} mean {zz.mean():+.2f}")
# B2 vs q_A bins (floor 1e13, c=1, q_cut=5, eps=0.1)
print("\nB2(eps=0.1) vs q_A, floor 1e13, c=1, q_cut=5 (claimed analytic: 0.788 [0.5,3), 0.385 [3,5), 0.141 [5,7), 0.067 [10,15), 0.035 [20,40), 0.017 [40,100))")
for lo, hi in ((0.5, 3), (3, 5), (5, 7), (7, 10), (10, 15), (20, 40), (40, 100)):
    nu, de = [], []
    for x in d:
        dl = x["D"].sum(1); xm = x["xmax"].max(1); dp = dl - MUF(x["th"]).sum(1)
        det = (x["q"] + x["n"] + dp > 5) & (x["q"] > xm); sel = (x["q"] >= lo) & (x["q"] < hi)
        nu.append((det & sel & (dp / x["q"] > 0.1)).sum()); de.append((det & sel).sum())
    r, e = jk(nu, de); print(f"   [{lo},{hi}): {r:.4f}±{e:.4f} (n_det={np.sum(de)})")
