"""[repair after verification] Medians / small-t CCDF of the neighbour sum at random points with a
high-resolution P(D) (blendlib.pd_cdf_fine), compared with the production (coarse, b03) values and the MC.

MC errors: (i) region jackknife (48 regions x 8 skies) for CCDFs, as in b05; (ii) between-sky scatter
(std of the 8 per-sky values / sqrt 8) for CCDFs and medians.  z uses max(e_jk, e_sky).
Also writes the corrected quantiles / small-t CCDF into data/blending/analytic.npz (keys QUANT, CCDF;
the previous production values are kept as QUANT_coarse, CCDF_coarse) for the figures (b06).
"""
import os, sys, glob, time
os.environ.setdefault("OMP_NUM_THREADS", "2"); os.environ.setdefault("NUMBA_NUM_THREADS", "2")
import numpy as np
import healpy as hp

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import blendlib as bl
from szsim import signal as sg

t0 = time.time()
m = sg.SZModel()
AP = os.path.join(bl.DATA, "analytic.npz")
A = dict(np.load(AP))
FLOORS = list(A["FLOORS"]); TH = A["th"]; CT = A["CCDF_T"]
W, WX, WX2, CAP = A["W"], A["WX"], A["WX2"], A["CAP"]
QCOARSE = A.get("QUANT_coarse", A["QUANT"]); CCOARSE = A.get("CCDF_coarse", A["CCDF"])
PROBS = np.array([0.16, 0.5, 0.84, 0.977, 0.9987])


def fine(i, fi, dd):
    cl = list(bl.FLOORS[FLOORS[fi]])
    te, cdf, xc = bl.pd_cdf_fine(W[i, cl].sum(0), WX[i, cl].sum(0), WX2[i, cl].sum(0), CAP[i, cl].sum(), dd)
    return te, cdf, xc


def dd_for(med):
    return float(np.clip(med / 500.0, 1e-5, 5e-4))


# ---------------- convergence of the fine grid at the worst case + all nodes
NT = len(TH)
QF = QCOARSE.copy(); CF = CCOARSE.copy()
conv = []
for fi in range(len(FLOORS)):
    for i in range(NT):
        dd = dd_for(QCOARSE[fi, i, 1])
        te, cdf, xc = fine(i, fi, dd)
        q = np.interp(PROBS, cdf, te)
        ok = q < xc
        QF[fi, i, ok] = q[ok]
        s = CT < xc
        CF[fi, i, s] = 1.0 - np.interp(CT[s], te, cdf)
        if i in (np.searchsorted(TH, 0.5), np.searchsorted(TH, 0.9), np.searchsorted(TH, 7.0)):
            te2, cdf2, _ = fine(i, fi, dd / 2)
            q2 = np.interp(0.5, cdf2, te2)
            conv.append((FLOORS[fi], TH[i], dd, q[1], q2))
print("fine-grid convergence of the median (dd -> dd/2):")
for fl, th, dd, a, b in conv:
    print(f"   floor {fl} thA={th:.3f}' dd={dd:.2e}: median {a:.6f} -> {b:.6f} (rel {b/a-1:+.1e})")
print(f"fine P(D) at all {NT} nodes x 3 floors done {time.time()-t0:.1f}s")

# ---------------- MC at random points
files = sorted(glob.glob(os.path.join(bl.DATA, "mc_sky*.npz")), key=lambda f: int(f.split("sky")[1][:-4]))
seeds = [int(f.split("sky")[1][:-4]) for f in files]
nsky = len(files); NREG = 48
d0 = np.load(files[0]); rthA = d0["rand_thA"]; rn = [len(d0[f"rand_D_{i}"]) for i in range(len(rthA))]
DS = {i: [] for i in range(len(rthA))}; LAB = {i: [] for i in range(len(rthA))}
for isky, (f, s) in enumerate(zip(files, seeds)):
    d = np.load(f)
    rng = np.random.default_rng(10_000 + s)          # same region labels as b05
    for j, n_ in enumerate(rn):
        ra = 360 * rng.random(n_); dec = np.degrees(np.arcsin(2 * rng.random(n_) - 1))
        DS[j].append(d[f"rand_D_{j}"]); LAB[j].append(hp.ang2pix(2, ra, dec, lonlat=True) + NREG * isky)


def jk(x, lab, nlab):
    Nn = np.bincount(lab, weights=x, minlength=nlab); Nd = np.bincount(lab, minlength=nlab).astype(float)
    ok = Nd > 0
    rj = (Nn.sum() - Nn[ok]) / (Nd.sum() - Nd[ok]); n = ok.sum()
    return Nn.sum() / Nd.sum(), np.sqrt((n - 1) / n * ((rj - rj.mean()) ** 2).sum())


print("\nmedians of delta at random points: MC pooled ± between-sky error | coarse (b03) | fine")
zmed_c, zmed_f, zcc, zcf, zcc_jk, rows = [], [], [], [], [], []
for i, thA in enumerate(rthA):
    inode = int(np.argmin(np.abs(TH - thA))); sigA = float(m.sigma_Y(thA))
    lab = np.concatenate(LAB[i]); nlab = NREG * nsky
    for fi, fl in enumerate(FLOORS):
        cl = list(bl.FLOORS[fl])
        per = [Dk[:, cl].sum(1) / sigA for Dk in DS[i]]
        dl = np.concatenate(per)
        med = np.median(dl); meds = np.array([np.median(p) for p in per]); emed = meds.std(ddof=1) / np.sqrt(nsky)
        qc_, qf_ = QCOARSE[fi, inode, 1], QF[fi, inode, 1]
        zmed_c.append((med - qc_) / emed); zmed_f.append((med - qf_) / emed)
        print(f"  thA={thA:.3f}' floor {fl}: MC {med:.5f}±{emed:.5f} | coarse {qc_:.5f} ({(qc_/med-1)*100:+.1f}%, {(med-qc_)/emed:+.1f}σ) "
              f"| fine {qf_:.5f} ({(qf_/med-1)*100:+.2f}%, {(med-qf_)/emed:+.1f}σ)")
        for it, t in enumerate(CT):
            p, ejk = jk((dl > t).astype(float), lab, nlab)
            ps = np.array([(pp > t).mean() for pp in per]); esky = ps.std(ddof=1) / np.sqrt(nsky)
            if p * len(dl) < 20 or (1 - p) * len(dl) < 20 or ejk == 0:
                continue
            e = max(ejk, esky)
            zc = (p - CCOARSE[fi, inode, it]) / e; zf = (p - CF[fi, inode, it]) / e
            zcc.append(zc); zcf.append(zf); zcc_jk.append((p - CCOARSE[fi, inode, it]) / ejk)
            rows.append((thA, fl, t, p, ejk, esky, CCOARSE[fi, inode, it], CF[fi, inode, it]))
zmed_c, zmed_f, zcc_jk = map(np.array, (zmed_c, zmed_f, zcc_jk))
print(f"\nmedian tests (15): coarse rms z {np.sqrt(np.mean(zmed_c**2)):.2f} max {np.abs(zmed_c).max():.2f}; "
      f"fine rms z {np.sqrt(np.mean(zmed_f**2)):.2f} max {np.abs(zmed_f).max():.2f}")
zcf = np.array([(r[3] - r[7]) / max(r[4], r[5]) for r in rows])
zcc = np.array([(r[3] - r[6]) / max(r[4], r[5]) for r in rows])
tt = np.array([r[2] for r in rows])
print(f"CCDF tests: n={len(rows)}; jackknife-only z (coarse, as b05): rms {np.sqrt(np.mean(zcc_jk**2)):.2f} max {np.abs(zcc_jk).max():.2f}")
print(f"   z with max(e_jk, e_sky): coarse rms {np.sqrt(np.mean(zcc**2)):.2f} max {np.abs(zcc).max():.2f}; "
      f"fine rms {np.sqrt(np.mean(zcf**2)):.2f} max {np.abs(zcf).max():.2f}")
for lo, hi in ((0, 0.11), (0.11, 9.9), (9.9, 100)):
    s = (tt > lo) & (tt < hi)
    print(f"   t in ({lo},{hi}): n={s.sum()} coarse rms {np.sqrt(np.mean(zcc[s]**2)):.2f} mean {zcc[s].mean():+.2f} | "
          f"fine rms {np.sqrt(np.mean(zcf[s]**2)):.2f} mean {zcf[s].mean():+.2f}; "
          f"max |MC - an| fine {np.max(np.abs([r[3]-r[7] for r, k in zip(rows, s) if k])):.1e}, coarse {np.max(np.abs([r[3]-r[6] for r, k in zip(rows, s) if k])):.1e}")
esr = np.array([r[5] / r[4] for r in rows])
print(f"   between-sky / jackknife error ratio for CCDFs: median {np.median(esr):.2f}, max {esr.max():.2f}")

if "QUANT_coarse" not in A:
    A["QUANT_coarse"] = A["QUANT"]; A["CCDF_coarse"] = A["CCDF"]
A["QUANT"] = QF; A["CCDF"] = CF
np.savez(AP, **A)
print(f"analytic.npz updated (QUANT, CCDF fine; *_coarse kept). total {time.time()-t0:.1f}s")
