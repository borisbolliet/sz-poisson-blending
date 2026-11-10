"""Compare the analytic compound-Poisson predictions (b03) with the Monte Carlo neighbour sums (b02).

(a) P(D) at random sky points with fixed filter scale (CCDF of delta = D/sigma_A, mean)
(b) blended-catalogue statistics for primaries q_A > 0.5: N_single, N_det_b, B2(eps), B2neg(eps), B3,
    lost, satellites; for q_cut in {4,5,6,7}, both map-mean conventions, neighbour floors 1e13/3e13/1e14
(c) B2 vs the primary's own (true) SNR q_A
Errors: delete-one-region jackknife, HEALPix nside=2 regions (48 per sky) x skies.
"""
import os, sys, glob, json
os.environ.setdefault("OMP_NUM_THREADS", "2")
import numpy as np
import healpy as hp

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import blendlib as bl
from szsim import signal as sg, cosmo_hmf as ch

m = sg.SZModel()
A = dict(np.load(os.path.join(bl.DATA, "analytic.npz")))
FLOORS = list(A["FLOORS"]); QCUTS = A["QCUTS"]; EPS = A["EPS"]; QN = A["QNODES"]
H = A["H"]
files = sorted(glob.glob(os.path.join(bl.DATA, "mc_sky*.npz")), key=lambda f: int(f.split("sky")[1][:-4]))
seeds = [int(f.split("sky")[1][:-4]) for f in files]
print(f"MC skies: {seeds}")
NREG = 48
MU_MODE = sys.argv[1] if len(sys.argv) > 1 else "expected"
print("monopole mode:", MU_MODE)


def jk(num, den, lab, nlab):
    Nn = np.bincount(lab, weights=num, minlength=nlab); Nd = np.bincount(lab, weights=den, minlength=nlab)
    r = Nn.sum() / Nd.sum()
    ok = Nd > 0
    rj = (Nn.sum() - Nn[ok]) / (Nd.sum() - Nd[ok])
    n = ok.sum()
    return r, np.sqrt((n - 1) / n * ((rj - rj.mean()) ** 2).sum())


def jk_mean(x, lab, nlab):
    return jk(x, np.ones_like(x), lab, nlab)


zlist = []
out = {}

# ---------------------------------------------------------------- (a) random points
print("\n(a) P(D) at random points: P(delta > t), delta = D/sigma_Y(thA); MC (jackknife) vs FFT compound Poisson")
tab = bl.load_rho_table()
d0 = np.load(files[0]); rthA = d0["rand_thA"]; rn = [len(d0[f"rand_D_{i}"]) for i in range(len(rthA))]
for i, thA in enumerate(rthA):
    inode = int(np.argmin(np.abs(tab.th - thA)))
    sigA = float(m.sigma_Y(thA))
    Ds, labs = [], []
    for isky, (f, s) in enumerate(zip(files, seeds)):
        d = np.load(f)
        rng = np.random.default_rng(10_000 + s)
        for j, (t_, n_) in enumerate(zip(rthA, rn)):
            ra = 360 * rng.random(n_); dec = np.degrees(np.arcsin(2 * rng.random(n_) - 1))
            if j == i:
                Ds.append(d[f"rand_D_{i}"]); labs.append(hp.ang2pix(2, ra, dec, lonlat=True) + NREG * isky)
                break
    D = np.concatenate(Ds); lab = np.concatenate(labs); nlab = NREG * len(files)
    ybar_mc = np.mean([np.load(f)["ybar"] for f in files], axis=0)
    for fi, fl in enumerate(FLOORS):
        cl = list(bl.FLOORS[fl])
        dl = D[:, cl].sum(1) / sigA
        mu_an = A["mu_exact"][inode, cl].sum()
        mu_mc, mu_e = jk_mean(dl, lab, nlab)
        mu_cond = (A["mu_exact"][inode, cl] * ybar_mc[cl] / A["ybar"][cl]).sum()
        line = [f"  thA={thA:.3f}' floor {fl}: mean MC {mu_mc:.4f}±{mu_e:.4f} vs an {mu_an:.4f} ({(mu_mc-mu_an)/mu_e:+.2f}σ), "
                f"vs an. given the realised ybar {mu_cond:.4f} ({(mu_mc-mu_cond)/mu_e:+.2f}σ);"]
        zlist.append(("mean", (mu_mc - mu_an) / mu_e))
        for it, t in enumerate(A["CCDF_T"]):
            p, e = jk_mean((dl > t).astype(float), lab, nlab)
            pa = A["CCDF"][fi, inode, it]
            if p * len(dl) >= 20 and e > 0 and (1 - p) * len(dl) >= 20:
                z = (p - pa) / e
                zlist.append(("ccdf", z))
                line.append(f"t={t:g}: {p:.4e}±{e:.1e}|{pa:.4e} ({z:+.1f})")
            out[("ccdf", round(thA, 4), fl, t)] = (p, e, pa)
        med_mc = np.median(dl)
        line.append(f"median MC {med_mc:.4f} an {A['QUANT'][fi, inode, 1]:.4f}")
        out[("median", round(thA, 4), fl)] = (med_mc, A["QUANT"][fi, inode, 1])
        print(" ".join(line))
zc = np.array([z for k, z in zlist if k == "ccdf"]); zm = np.array([z for k, z in zlist if k == "mean"])
print(f"  random-point CCDF tests: {len(zc)} z-scores, rms {np.sqrt(np.mean(zc**2)):.2f}, max|z| {np.abs(zc).max():.2f}; "
      f"mean tests: {len(zm)}, rms {np.sqrt(np.mean(zm**2)):.2f}, max|z| {np.abs(zm).max():.2f}")

# ---------------------------------------------------------------- (b) primaries
print("\n(b) blended catalogue, primaries q_A>0.5 (full-sky numbers per sky; fractions)")
P = {k: [] for k in ("th", "q", "qo", "D", "xm", "lab", "mu")}
for isky, f in enumerate(files):
    d = np.load(f)
    th = d["p_theta500"].astype(np.float64)
    sig = m.sigma_Y(th)
    P["th"].append(th); P["q"].append(d["p_q"].astype(np.float64)); P["qo"].append(d["p_q_obs"].astype(np.float64))
    P["D"].append(d["D"] / sig[:, None]); P["xm"].append(d["xmax"] / sig[:, None])
    # monopole shift: mean of delta under the same interpolated kernel (analytic mu per class, nodes in
    # ln theta) rescaled by this sky's realised mean Y density per class (what a map-mean removal subtracts)
    lt = np.log(th)
    # MU_MODE='expected': subtract the ensemble-mean monopole (same as the analytic); 'realised': subtract
    # this sky's own mean (rescaled by its realised ybar).  The sky-to-sky ybar scatter (~0.5%) is a global
    # per-sky shift that the region jackknife cannot see, so the analytic test uses 'expected'.
    yfac = (d["ybar"] / A["ybar"]) if MU_MODE == "realised" else np.ones(3)
    P["mu"].append(np.column_stack([np.interp(lt, np.log(A["th"]), A["mu_exact"][:, c]) * yfac[c]
                                    for c in range(3)]))
    P["lab"].append(hp.ang2pix(2, d["p_ra"].astype(float), d["p_dec"].astype(float), lonlat=True) + NREG * isky)
P = {k: np.concatenate(v) for k, v in P.items()}
nsky = len(files); nlab = NREG * nsky
regs_per_sky = np.arange(nlab) // NREG


def an_frac(fi, c, ic, num_arr, den="det", Hbin=None):
    w = H if Hbin is None else Hbin
    Nnum = (w * num_arr).sum()
    if den == "det":
        Nden = (w * A["Pdet"][fi, c, ic]).sum()
    else:
        Nden = (w * A["Phi_single"][ic][None, :]).sum()
    return Nnum / Nden


ztab = []
for fi, fl in enumerate(FLOORS):
    cl = list(bl.FLOORS[fl])
    dl = P["D"][:, cl].sum(1); xm = P["xm"][:, cl].max(1); mu = P["mu"][:, cl].sum(1)
    dom = P["q"] > xm
    for c in (0, 1):
        dprime = dl - c * mu
        qb = P["qo"] + dprime
        for ic, qc in enumerate(QCUTS):
            single = P["qo"] > qc
            det = (qb > qc) & dom
            Ns_mc = single.sum() / nsky; Nd_mc = det.sum() / nsky
            Ns_an = (H * A["Phi_single"][ic][None, :]).sum(); Nd_an = (H * A["Pdet"][fi, c, ic]).sum()
            # per-sky count scatter -> error on mean
            cnt = np.bincount(P["lab"][det], minlength=nlab).reshape(nsky, NREG).sum(1)
            cnts = np.bincount(P["lab"][single], minlength=nlab).reshape(nsky, NREG).sum(1)
            eNd = max(cnt.std(ddof=1), np.sqrt(Nd_an)) / np.sqrt(nsky) if nsky > 1 else np.sqrt(Nd_an)
            eNs = max(cnts.std(ddof=1), np.sqrt(Ns_an)) / np.sqrt(nsky) if nsky > 1 else np.sqrt(Ns_an)
            # ratio N_det_b / N_single (paired, jackknife)
            r_mc, r_e = jk(det.astype(float), single.astype(float), P["lab"], nlab)
            r_an = Nd_an / Ns_an
            row = dict(floor=fl, c=c, qcut=qc, Ns_mc=Ns_mc, Ns_an=Ns_an, Nd_mc=Nd_mc, Nd_an=Nd_an,
                       ratio=(r_mc, r_e, r_an))
            ztab.append(("ratio", (r_mc - r_an) / r_e))
            for name, flag, arr, den in [("B3", det & (P["qo"] < qc), A["Pb3"][fi, c, ic], "det"),
                                         ("lost", single & dom & (qb < qc), A["Plost"][fi, c, ic], "single"),
                                         ("sat", single & ~dom, A["Psat"][fi, ic], "single")]:
                dd = det if den == "det" else single
                r, e = jk(flag.astype(float), dd.astype(float), P["lab"], nlab)
                ra = an_frac(fi, c, ic, arr, den)
                row[name] = (r, e, ra)
                if flag.sum() >= 20:
                    ztab.append((name, (r - ra) / e))
            for idm, dm in enumerate(A["DMIN"]):
                flag = det & (P["qo"] < qc) & (dprime > dm)
                r, e = jk(flag.astype(float), det.astype(float), P["lab"], nlab)
                ra = an_frac(fi, c, ic, A["Pb3d"][fi, c, ic, :, :, idm], "det")
                row[f"B3d_{dm}"] = (r, e, ra)
                if flag.sum() >= 20:
                    ztab.append(("B3d", (r - ra) / e))
            for ie, eps in enumerate(EPS):
                for name, flag, arr in [("B2", det & (dprime / P["q"] > eps), A["Pb2"][fi, c, ic, :, :, ie]),
                                        ("B2neg", det & (dprime / P["q"] < -eps), A["Pb2n"][fi, c, ic, :, :, ie])]:
                    r, e = jk(flag.astype(float), det.astype(float), P["lab"], nlab)
                    ra = an_frac(fi, c, ic, arr, "det")
                    row[f"{name}_{eps}"] = (r, e, ra)
                    if flag.sum() >= 20:
                        ztab.append((name, (r - ra) / e))
            out[("prim", fl, c, qc)] = row
            print(f" floor {fl} c={c} q_cut={qc:.0f}: N_single MC {Ns_mc:,.1f} an {Ns_an:,.1f} ({(Ns_mc-Ns_an)/eNs:+.2f}σ); "
                  f"N_det_b MC {Nd_mc:,.1f} an {Nd_an:,.1f} ({(Nd_mc-Nd_an)/eNd:+.2f}σ); "
                  f"N_det_b/N_single MC {r_mc:.5f}±{r_e:.5f} an {r_an:.5f} ({(r_mc-r_an)/r_e:+.2f}σ)")
            ztab.append(("Ns", (Ns_mc - Ns_an) / eNs)); ztab.append(("Nd", (Nd_mc - Nd_an) / eNd))
            s = "    "
            for name in ["B3", "lost", "sat"] + [f"B3d_{dm}" for dm in A["DMIN"]] + [f"B2_{e}" for e in EPS] + ([f"B2neg_{e}" for e in EPS] if c == 1 else []):
                r, e, ra = row[name]
                zz = (r - ra) / e if e > 0 else np.nan
                s += f"{name}: {r:.5f}±{e:.5f}|{ra:.5f}({zz:+.1f}) "
            print(s)

zt = np.array([z for _, z in ztab if np.isfinite(z)])
print(f"\n  primary-level tests: {len(zt)} z-scores (>=20 MC events), rms {np.sqrt(np.mean(zt**2)):.2f}, "
      f"max|z| {np.abs(zt).max():.2f}; fraction |z|>2: {(np.abs(zt)>2).mean():.3f}, |z|>3: {(np.abs(zt)>3).mean():.3f}")
for name in sorted(set(k for k, _ in ztab)):
    zz = np.array([z for k, z in ztab if k == name and np.isfinite(z)])
    print(f"     {name:6s}: n={len(zz):3d} rms {np.sqrt(np.mean(zz**2)):.2f} max|z| {np.abs(zz).max():.2f}")

# ---------------------------------------------------------------- (c) B2 vs own SNR
print("\n(c) P(B2 | det_b, q_A bin): floor 1e13, q_cut=5")
QB = [0.5, 3, 5, 7, 10, 15, 20, 40, 100, 3000]
from b03_analytic import primary_population
HBIN = {lo: primary_population(tab, lo, hi) for lo, hi in zip(QB[:-1], QB[1:])}  # population restricted to the bin
fi = 0; ic = list(QCUTS).index(5.0)
cl = list(bl.FLOORS["1e13"])
dl = P["D"][:, cl].sum(1); xm = P["xm"][:, cl].max(1); mu = P["mu"][:, cl].sum(1); dom = P["q"] > xm
for c in (0, 1):
    dprime = dl - c * mu; det = (P["qo"] + dprime > 5.0) & dom
    for eps_i in (1, 2):
        eps = EPS[eps_i]
        flag = det & (dprime / P["q"] > eps)
        s = f"  c={c} eps={eps}: "
        for lo, hi in zip(QB[:-1], QB[1:]):
            sel = (P["q"] >= lo) & (P["q"] < hi)
            r, e = jk(flag[sel].astype(float), det[sel].astype(float), P["lab"][sel], nlab)
            ra = an_frac(fi, c, ic, A["Pb2"][fi, c, ic, :, :, eps_i], "det", HBIN[lo])
            out[("B2vsq", c, eps, lo)] = (r, e, ra, det[sel].sum())
            s += f"[{lo},{hi}): {r:.4f}±{e:.4f}|{ra:.4f} ; "
        print(s)

np.save(os.path.join(bl.DATA, "compare_out.npy" if MU_MODE == "expected" else "compare_out_realised.npy"), np.array([out], dtype=object), allow_pickle=True)
