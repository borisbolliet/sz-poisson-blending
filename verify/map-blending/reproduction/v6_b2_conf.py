"""V6 (C7, C12): (a) confusion variance of filtered maps of q_true<2 / q_true<5 / [1e13,1e14) halos in my patches vs my
analytic Poisson one-halo variance (own HMF grid, W_k = 2 pi int r R_k^2 dr from my response table) and vs the
realised-population sum; [1e12,1e13) analytic.  (b) single-neighbour Poisson formula 1-exp(-lambda) with lambda
built ANALYTICALLY from my HMF grid + lognormal scatter (no histogram of realised halos), against my catalogue-level
unselected halos and against my map's SNR-selected real detections."""
import glob, json
import numpy as np
from scipy.special import ndtr
import indep_lib as I

SIG = I.templates()[1]; RT, RR = I.resp_table(SIG); SIGY = I.sigY_interp()
files = sorted(glob.glob(f"{I.OUT}/patches/p*.npz")); NP = len(files)
O = [np.load(f) for f in files]
AREA = I.LBOX ** 2
# W_k(theta node)
W = np.zeros((I.K, I.NT))
for it in range(I.NT):
    r = np.linspace(0, RR[it], I.NRR)
    for k in range(I.K):
        W[k, it] = np.trapezoid(2 * np.pi * r * RT[k, it] ** 2, r)
Wf = lambda k, th: np.exp(np.interp(np.log(th), np.log(I.TNODES), np.log(W[k])))
s = I.SIGLNY


def var_an(grid, qthr=None):
    dz = np.diff(grid["ze"]); dl = np.diff(grid["le"])
    Z, LM = np.meshgrid(grid["zc"], grid["lc"], indexing="ij")
    th = I.theta500(np.exp(LM), Z); Ym = I.Ymed(np.exp(LM), Z)
    mom = Ym ** 2 * np.exp(2 * s * s)
    if qthr is not None:
        mom = mom * ndtr((np.log(qthr * SIGY(th) / Ym) - 2 * s * s) / s)
    w = grid["dens"] * dz[:, None] * dl[None, :] * I.ARC ** 2       # per arcmin^2
    return np.array([(w * mom * Wf(k, th)).sum() for k in range(I.K)])


g13 = I.hmf_grid()
g12 = I.hmf_grid(Mmin=1e12, Mmax=1e13, nz=600, nM=300)
g1314 = I.hmf_grid(Mmin=1e13, Mmax=1e14, nz=600, nM=300)
an = {"q<2": var_an(g13, 2.0), "q<5": var_an(g13, 5.0), "[1e13,1e14)": var_an(g1314), "[1e12,1e13)": var_an(g12)}
V = np.array([o["var"] for o in O]) / SIG[None, :, None] ** 2          # (NP, K, 5): N, A, Q2, Q5, C
pop = np.zeros(I.K)
for o in O:
    m = o["pop_M"] < 1e14
    pop += np.array([(o["pop_Y"][m].astype(float) ** 2 * Wf(k, o["pop_th"][m].astype(float))).sum() for k in range(I.K)]) / AREA / NP
claimQ2 = {0: (0.0533, 0.0532), 11: (0.1697, 0.1696)}; claimQ5 = {0: (0.1188, 0.1194), 11: (0.4612, 0.4663)}
print(f"(a) confusion variance / sigma_k^2, {NP} independent patches")
print("  k theta | noise Var/sig^2      | q<2: map           analytic (dev)  | q<5: map            analytic (dev)  | [1e13,1e14): map     analytic pop-sum | [1e12,1e13) an")
for k in range(I.K):
    row = []
    for j in range(5):
        row.append((V[:, k, j].mean(), V[:, k, j].std(ddof=1) / np.sqrt(NP)))
    s2 = SIG[k] ** 2
    print(f"  {k:2d} {I.THK[k]:5.2f} | {row[0][0]:.5f}+-{row[0][1]:.5f} | {row[2][0]:.4f}+-{row[2][1]:.4f} {an['q<2'][k]/s2:.4f} ({(row[2][0]-an['q<2'][k]/s2)/row[2][1]:+.2f}) |"
          f" {row[3][0]:.4f}+-{row[3][1]:.4f} {an['q<5'][k]/s2:.4f} ({(row[3][0]-an['q<5'][k]/s2)/row[3][1]:+.2f}) |"
          f" {row[1][0]:.4f}+-{row[1][1]:.4f} {an['[1e13,1e14)'][k]/s2:.4f} {pop[k]/s2:.4f} | {an['[1e12,1e13)'][k]/s2:.5f}"
          f"   sig_eff/sig q<2 {np.sqrt(1+row[2][0]):.4f} q<5 {np.sqrt(1+row[3][0]):.4f}")
print("  claimed: q<2 0.0533 vs 0.0532 (0.5'), 0.1697 vs 0.1696 (10'); q<5 0.1188 vs 0.1194, 0.4612 vs 0.4663; sig_eff 1.026-1.082 / 1.058-1.209;"
      " [1e12,1e13) 0.0008-0.0024")

# ---------------------------------------------------------------- (b) analytic lambda
dz = np.diff(g13["ze"]); dl = np.diff(g13["le"])
Z, LM = np.meshgrid(g13["zc"], g13["lc"], indexing="ij")
th = I.theta500(np.exp(LM), Z).ravel(); lY0 = np.log(I.Ymed(np.exp(LM), Z)).ravel()
w0 = (g13["dens"] * dz[:, None] * dl[None, :] * I.ARC ** 2).ravel()
xh, wh = np.polynomial.hermite_e.hermegauss(16); wh = wh / wh.sum()
t = np.clip((np.log(th) - I.LNT0) / I.DLNT, 0, I.NT - 1.000001); i0 = t.astype(int); fw = t - i0
lYe = np.linspace(lY0.min() - 4 * s - 0.01, lY0.max() + 4 * s + 0.01, 401); lYc = 0.5 * (lYe[1:] + lYe[:-1])
Hn = np.zeros((I.NT, lYc.size))
for e, we in zip(xh, wh):
    iy = np.clip(np.searchsorted(lYe, lY0 + s * e) - 1, 0, lYc.size - 1)
    np.add.at(Hn, (i0, iy), w0 * we * (1 - fw)); np.add.at(Hn, (i0 + 1, iy), w0 * we * fw)
print(f"\n(b) analytic population: total density {Hn.sum()*3600:.2f} /deg^2 (patch lambda/100 = {g13['dens'].sum()*0 + O[0]['lam']/100:.2f})")
lt = np.linspace(np.log(1e-9), np.log(1e-2), 200)
LAM = np.zeros((I.K, lt.size))
Yb = np.exp(lYc)
for k in range(I.K):
    for it in range(I.NT):
        nz = Hn[it] > 0
        if not nz.any():
            continue
        r = np.linspace(0, RR[it], I.NRR); env = np.minimum.accumulate(RT[k, it])
        x = np.exp(lt)[:, None] / Yb[nz][None, :]
        rs = np.interp(-x, -env, r, left=0.0, right=0.0); rs[x >= env[0]] = 0.0
        LAM[k] += (np.pi * rs ** 2 * Hn[it, nz][None, :]).sum(1)
lam = lambda k, tt: np.exp(np.interp(np.log(tt), lt, np.log(np.maximum(LAM[k], 1e-300))))
cat = np.load(f"{I.OUT}/catrows.npy")      # p, k_own, own, maxnb, net, q
rr = np.load(f"{I.OUT}/realrows.npy")      # p, qc, k, cP
rng = np.random.default_rng(5); BOOT = rng.integers(0, NP, (500, NP))
claim_u = {4: (15.39, 0.23, 15.13), 5: (13.01, 0.24, 12.53), 6: (10.91, 0.29, 10.69), 7: (9.70, 0.34, 9.28)}
A = np.load(f"{I.OUT}/A_metrics.npy"); MET = json.load(open(f"{I.OUT}/A_metrics_names.json")); mi = {m: i for i, m in enumerate(MET)}
for iq, qc in enumerate([4, 5, 6, 7]):
    sel = cat[:, 5] > qc; c = cat[sel]
    for f in (0.1, 0.2):
        hit = (c[:, 3] > f * c[:, 2]).astype(float)
        pidx = c[:, 0].astype(int)
        num = np.bincount(pidx, hit, NP); den = np.bincount(pidx, None, NP)
        meas = num.sum() / den.sum(); err = (num[BOOT].sum(1) / den[BOOT].sum(1)).std(ddof=1)
        pred = np.mean(1 - np.exp(-np.array([lam(int(k), f * ow) for k, ow in zip(c[:, 1], c[:, 2])])))
        extra = f"  claimed {claim_u[qc][0]} +- {claim_u[qc][1]} vs formula {claim_u[qc][2]}" if f == 0.1 else ""
        print(f"  UNSELECTED q>{qc} f={f}: measured {100*meas:.2f}+-{100*err:.2f}%  analytic 1-exp(-lam) {100*pred:.2f}%  ({(meas-pred)/err:+.2f} sigma){extra}")
    s_ = rr[:, 1] == qc
    pr = np.mean(1 - np.exp(-np.array([lam(int(k), 0.1 * cp) for k, cp in zip(rr[s_, 2], rr[s_, 3])])))
    a = A[:, iq, mi["nc10r"]]; b = A[:, iq, mi["n_real"]]
    m = a.sum() / b.sum(); e = (a[BOOT].sum(1) / b[BOOT].sum(1)).std(ddof=1)
    print(f"  SELECTED (map real detections) q>{qc}: measured >=1 nb >10% c_P {100*m:.2f}+-{100*e:.2f}%  formula {100*pr:.2f}%  ratio {m/pr:.3f}"
          f"  (claimed {[21.93,20.81,18.56,17.02][iq]} vs {[18.20,15.38,12.78,10.92][iq]})")
