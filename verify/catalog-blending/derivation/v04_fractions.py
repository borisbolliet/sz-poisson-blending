"""Blending fractions from the verifier's own neighbour sums (v03_sky*.npz), compared with the CLAIMED
analytic numbers.  Definitions re-implemented from the claim text (C1).  Errors: delete-one-region
jackknife (HEALPix nside=2, 48 regions per sky, pooled over skies).  mu from v02 (own colossus ybar)."""
import sys, glob
import numpy as np, healpy as hp
import v01_kernel as K
YBAR = np.array([8.97647742e-08, 1.20306859e-07, 1.48898975e-07])  # v02_population.log
FL = {"1e13": [0, 1, 2], "3e13": [1, 2], "1e14": [2]}
files = sorted(glob.glob("v03_sky*.npz"))
print("skies:", files)
d0 = np.load("v01_kernel.npz"); TH, SIG = d0["TH"], d0["SIG"]
sig = lambda t: np.exp(np.interp(np.log(t), np.log(TH), np.log(SIG)))
NREG = 48

def jk(num, den, lab, nl):
    a = np.bincount(lab, weights=num, minlength=nl); b = np.bincount(lab, weights=den, minlength=nl)
    r = a.sum() / b.sum(); ok = b > 0; rj = (a.sum() - a[ok]) / (b.sum() - b[ok]); n = ok.sum()
    return r, np.sqrt((n - 1) / n * ((rj - rj.mean()) ** 2).sum())

P = {k: [] for k in ("q", "qo", "th", "D", "XM", "lab")}
RP = {}
ybar_mc = []
for isk, f in enumerate(files):
    d = np.load(f)
    s = sig(d["p_th"])
    P["q"].append(d["p_q"].astype(float)); P["qo"].append(d["p_qo"].astype(float)); P["th"].append(d["p_th"])
    P["D"].append(d["D"] / s[:, None]); P["XM"].append(d["XM"] / s[:, None])
    P["lab"].append(hp.vec2pix(2, d["p_x"], d["p_y"], d["p_z"]) + NREG * isk)
    ybar_mc.append(d["ybar"])
    for key in d.files:
        if key.startswith("rD_"):
            tA = float(key[3:]); v = d[f"rv_{tA}"]
            RP.setdefault(tA, []).append((d[key] / sig(tA), hp.vec2pix(2, v[:, 0], v[:, 1], v[:, 2]) + NREG * isk))
P = {k: np.concatenate(v) for k, v in P.items()}
nl = NREG * len(files)
ybar_mc = np.mean(ybar_mc, axis=0)
print(f"realised ybar/analytic per class: {np.round(ybar_mc / YBAR, 4)}")
mu_c = YBAR[None, :] * sig(P["th"])[:, None] * K.RATIO ** 2 / K.NW

print("\n(a) random points: delta = D/sigma_A")
for tA in sorted(RP):
    D = np.concatenate([a for a, _ in RP[tA]]); lab = np.concatenate([b for _, b in RP[tA]])
    sA = sig(tA)
    for fl, cl in FL.items():
        dl = D[:, cl].sum(1)
        mu_an = (YBAR[cl] * sA * K.RATIO ** 2 / K.NW).sum()
        mu_cond = (ybar_mc[cl] * sA * K.RATIO ** 2 / K.NW).sum()
        m, e = jk(dl, np.ones_like(dl), lab, nl)
        med = np.median(dl)
        # bootstrap-free median error via jackknife of regions
        meds = np.array([np.median(dl[lab != r]) for r in range(nl)])
        emed = np.sqrt((nl - 1) / nl * ((meds - meds.mean()) ** 2).sum())
        cc = [f"P(>{tt})={(dl > tt).mean():.4e}" for tt in (1, 10, 50)]
        print(f"  thA={tA}' floor {fl}: mean {m:.4f}±{e:.4f} vs analytic {mu_an:.4f} ({(m-mu_an)/e:+.2f}σ), "
              f"given realised ybar {mu_cond:.4f} ({(m-mu_cond)/e:+.2f}σ); median {med:.4f}±{emed:.4f}; {' '.join(cc)}")

CLAIM = {  # (floor, c, qcut): dict of claimed analytic values (b03 log / claim text)
 ("1e13", 1, 4): dict(B2_0_1=0.18468, B2_0_2=0.12426, B2_0_5=0.05053, B2_1_0=0.00685, B2n_0_1=0.11504, B3=0.08723, lost=0.10935, sat=0.00949, ratio=0.96537),
 ("1e13", 1, 5): dict(B2_0_1=0.1623, B2_0_2=0.10514, B2_0_5=0.04146, B2_1_0=0.00499, B2n_0_1=0.05733, B3=0.08122, B3d_1=0.05192, lost=0.09028, sat=0.00595, ratio=0.98366),
 ("1e13", 1, 6): dict(B2_0_1=0.145, B2_0_2=0.09116, B2_0_5=0.03502, B2_1_0=0.0038, B2n_0_1=0.02902, B3=0.07554, B3d_1=0.05237, lost=0.07887, sat=0.00436, ratio=0.99169),
 ("1e13", 1, 7): dict(B2_0_1=0.13128, B2_0_2=0.08054, B2_0_5=0.0303, B2_1_0=0.00301, B2n_0_1=0.01573, B3=0.07075, B3d_1=0.0513, lost=0.07092, sat=0.00341, ratio=0.99615),
 ("1e13", 0, 4): dict(B2_0_1=0.59998, B2_0_2=0.36538, B3=0.29048, ratio=1.39602),
 ("1e13", 0, 5): dict(B2_0_1=0.51622, B2_0_2=0.28848, B2_0_5=0.10844, B3=0.25938, B3d_1=0.1471, ratio=1.34218),
 ("1e13", 0, 6): dict(B2_0_1=0.45433, B2_0_2=0.23938, B3=0.23704, ratio=1.30497),
 ("1e13", 0, 7): dict(B2_0_1=0.40653, B2_0_2=0.20514, B3=0.22048, ratio=1.27847),
 ("3e13", 1, 5): dict(B2_0_1=0.15994, B2_0_2=0.10401, B3d_1=0.05131),
 ("1e14", 1, 5): dict(B2_0_1=0.12763, B2_0_2=0.08756, B3d_1=0.04279),
 ("3e13", 0, 5): dict(B2_0_2=0.21237, B3=0.19575, ratio=1.23600),
 ("1e14", 0, 5): dict(B2_0_2=0.11962, B3=0.09841, ratio=1.10254),
}
print("\n(b) primaries q_A>0.5; verifier MC vs CLAIMED analytic")
zs = []
for (fl, c, qc), cl_vals in CLAIM.items():
    cl = FL[fl]
    dl = P["D"][:, cl].sum(1); xm = P["XM"][:, cl].max(1); mu = mu_c[:, cl].sum(1)
    dp = dl - c * mu; dom = P["q"] > xm; qb = P["qo"] + dp
    single = P["qo"] > qc; det = (qb > qc) & dom
    flags = dict(B3=(det & (P["qo"] < qc), det), B3d_1=(det & (P["qo"] < qc) & (dp > 1), det),
                 lost=(single & dom & (qb < qc), single), sat=(single & ~dom, single), ratio=(det, single))
    for e in (0.1, 0.2, 0.5, 1.0):
        flags[f"B2_{str(e).replace('.', '_')}"] = (det & (dp / P["q"] > e), det)
        flags[f"B2n_{str(e).replace('.', '_')}"] = (det & (dp / P["q"] < -e), det)
    line = f" floor {fl} c={c} q_cut={qc}: N_single/sky {single.sum()/len(files):,.0f}, N_det_b/sky {det.sum()/len(files):,.0f};"
    for name, val in cl_vals.items():
        num, den = flags[name]
        r, e = jk(num.astype(float), den.astype(float), P["lab"], nl)
        z = (r - val) / e
        zs.append(z)
        line += f" {name} {r:.4f}±{e:.4f}|claim {val:.4f} ({z:+.1f})"
    print(line)
zs = np.array(zs)
print(f"\n  {len(zs)} tests: rms z {np.sqrt(np.mean(zs**2)):.2f}, max|z| {np.abs(zs).max():.2f}, mean z {zs.mean():+.2f}")

print("\n(c) P(B2(0.1)|det_b) vs q_A, floor 1e13, c=1, q_cut=5 (claim 0.788/0.385/0.141/0.067/0.035/0.017)")
dl = P["D"].sum(1); xm = P["XM"].max(1); mu = mu_c.sum(1); dp = dl - mu
det = (P["qo"] + dp > 5) & (P["q"] > xm)
for lo, hi in ((0.5, 3), (3, 5), (5, 7), (10, 15), (20, 40), (40, 100)):
    s = (P["q"] >= lo) & (P["q"] < hi)
    r, e = jk((det & (dp / P["q"] > 0.1))[s].astype(float), det[s].astype(float), P["lab"][s], nl)
    print(f"  q_A in [{lo},{hi}): {r:.4f}±{e:.4f}  (n_det={det[s].sum()})")
