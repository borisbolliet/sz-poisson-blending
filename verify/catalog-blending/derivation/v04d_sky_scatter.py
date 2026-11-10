"""Sky-to-sky scatter of monopole-convention statistics (q_cut=4,5; floor 1e13; c=1) over the analysis' 8 MC skies
(their stored sums; mu = verifier exact formula).  Compares the between-sky error of the mean with the
region-jackknife error, to judge whether the quoted jackknife errors capture sky-level fluctuations."""
import numpy as np, healpy as hp
import v01_kernel as K
YBAR = np.array([8.97647742e-08, 1.20306859e-07, 1.48898975e-07])
d0 = np.load("v01_kernel.npz"); TH, SIG = d0["TH"], d0["SIG"]
sig = lambda t: np.exp(np.interp(np.log(t), np.log(TH), np.log(SIG)))
CLAIM = {4: dict(lost=0.10935, ratio=0.96537, B2n=0.11504, B2=0.12426, B3=0.08723), 5: dict(lost=0.09028, ratio=0.98366, B2n=0.05733, B2=0.10514, B3=0.08122)}
per = {qc: {k: [] for k in CLAIM[qc]} for qc in CLAIM}
pooled = {qc: {k: ([], []) for k in CLAIM[qc]} for qc in CLAIM}
labs = []
for s in range(1, 9):
    d = np.load(f"/Users/boris/GitHub/sz-poisson-blending/data/blending/mc_sky{s}.npz")
    th = d["p_theta500"].astype(float); sA = sig(th)
    q = d["p_q"].astype(float); qo = d["p_q_obs"].astype(float)
    dl = d["D"].sum(1) / sA; xm = d["xmax"].max(1) / sA
    mu = (YBAR[None, :] * sA[:, None] * K.RATIO ** 2 / K.NW).sum(1)
    dp = dl - mu; dom = q > xm
    lab = hp.ang2pix(2, d["p_ra"].astype(float), d["p_dec"].astype(float), lonlat=True) + 48 * (s - 1)
    labs.append(lab)
    for qc in CLAIM:
        single = qo > qc; det = (qo + dp > qc) & dom
        fl = dict(lost=(single & dom & (qo + dp < qc), single), ratio=(det, single), B2n=(det & (dp / q < -0.1), det),
                  B2=(det & (dp / q > 0.2), det), B3=(det & (qo < qc), det))
        for k, (a, b) in fl.items():
            per[qc][k].append(a.sum() / b.sum())
            pooled[qc][k][0].append(a); pooled[qc][k][1].append(b)
lab = np.concatenate(labs)
def jk(num, den):
    a = np.bincount(lab, weights=num, minlength=384); b = np.bincount(lab, weights=den, minlength=384)
    r = a.sum() / b.sum(); rj = (a.sum() - a) / (b.sum() - b); n = 384
    return r, np.sqrt((n - 1) / n * ((rj - rj.mean()) ** 2).sum())
for qc in CLAIM:
    for k in CLAIM[qc]:
        v = np.array(per[qc][k]); a = np.concatenate(pooled[qc][k][0]).astype(float); b = np.concatenate(pooled[qc][k][1]).astype(float)
        r, e = jk(a, b)
        es = v.std(ddof=1) / np.sqrt(len(v))
        print(f"q_cut={qc} {k:5s}: pooled {r:.5f}, region-jk err {e:.5f}, between-sky err {es:.5f} (ratio {es/e:.2f}); "
              f"claim {CLAIM[qc][k]:.5f}: z_jk {(r-CLAIM[qc][k])/e:+.2f}, z_sky {(r-CLAIM[qc][k])/es:+.2f}; per-sky {np.round(v,4)}")
