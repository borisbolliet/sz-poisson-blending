"""Verifier random-point CCDF of delta (3 skies, own kernel) vs the analysis' analytic FFT P(D) CCDF.
Errors: region jackknife (48 regions x 3 skies)."""
import glob, numpy as np, healpy as hp
d0 = np.load("v01_kernel.npz"); TH, SIG = d0["TH"], d0["SIG"]
sig = lambda t: np.exp(np.interp(np.log(t), np.log(TH), np.log(SIG)))
A = np.load("/Users/boris/GitHub/sz-poisson-blending/data/blending/analytic.npz")
FL = {"1e13": [0, 1, 2], "3e13": [1, 2], "1e14": [2]}
files = sorted(glob.glob("v03_sky*.npz")); ds = [np.load(f) for f in files]
zs = []; ztail = []
for tA in (0.524, 0.946, 1.983, 4.153, 7.505):
    i = np.argmin(abs(A["th"] - tA))
    D = np.concatenate([d[f"rD_{tA}"] for d in ds]) / sig(tA)
    lab = np.concatenate([hp.vec2pix(2, *d[f"rv_{tA}"].T.astype(float)) + 48 * k for k, d in enumerate(ds)])
    for fi, (fl, cl) in enumerate(FL.items()):
        dl = D[:, cl].sum(1); line = f"thA={tA} {fl}:"
        for it, t in enumerate(A["CCDF_T"]):
            x = (dl > t).astype(float)
            if x.sum() < 20 or (1 - x).sum() < 20: continue
            a = np.bincount(lab, weights=x, minlength=144); b = np.bincount(lab, minlength=144)
            r = a.sum() / b.sum(); rj = (a.sum() - a) / (b.sum() - b); e = np.sqrt(143 / 144 * ((rj - rj.mean()) ** 2).sum())
            z = (r - A["CCDF"][fi, i, it]) / e; zs.append(z)
            if t >= 10: ztail.append(z)
            if t in (0.1, 1, 10, 50): line += f" t={t:g}: {r:.4e}±{e:.1e}|{A['CCDF'][fi,i,it]:.4e} ({z:+.1f})"
        print(line)
zs = np.array(zs); ztail = np.array(ztail)
print(f"{len(zs)} CCDF tests: rms {np.sqrt(np.mean(zs**2)):.2f}, max|z| {np.abs(zs).max():.2f}; t>=10: n={len(ztail)} rms {np.sqrt(np.mean(ztail**2)):.2f} mean {ztail.mean():+.2f}")
