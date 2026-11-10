"""Diagnostic: is the verifier MC's realised mean neighbour boost consistent with the analytic mu used for c=1?
mean(delta)/mu per theta_A bin over all primaries of the 3 skies (Slivnyak: same as random points), and the
sensitivity of 'lost' and N_det_b/N_single at q_cut=4 to a coherent rescaling of mu."""
import glob, numpy as np, healpy as hp
import v01_kernel as K
YBAR = np.array([8.97647742e-08, 1.20306859e-07, 1.48898975e-07])
d0 = np.load("v01_kernel.npz"); TH, SIG = d0["TH"], d0["SIG"]
sig = lambda t: np.exp(np.interp(np.log(t), np.log(TH), np.log(SIG)))
q, qo, th, D, XM, lab, yb = [], [], [], [], [], [], []
for i, f in enumerate(sorted(glob.glob("v03_sky*.npz"))):
    d = np.load(f); s = sig(d["p_th"])
    q.append(d["p_q"].astype(float)); qo.append(d["p_qo"].astype(float)); th.append(d["p_th"])
    D.append(d["D"] / s[:, None]); XM.append(d["XM"] / s[:, None]); yb.append(d["ybar"])
    lab.append(hp.vec2pix(2, d["p_x"], d["p_y"], d["p_z"]) + 48 * i)
q, qo, th, D, XM, lab = map(np.concatenate, (q, qo, th, D, XM, lab))
yb = np.mean(yb, 0)
mu_c = YBAR[None, :] * sig(th)[:, None] * K.RATIO ** 2 / K.NW
mu_r = yb[None, :] * sig(th)[:, None] * K.RATIO ** 2 / K.NW
print("theta bin   n      mean(delta_c)/mu_c  [classes 0,1]  (expected from realised ybar: %.4f %.4f)" % tuple(yb[:2] / YBAR[:2]))
for lo, hi in ((0.3, 0.7), (0.7, 1.2), (1.2, 2.5), (2.5, 6), (6, 80)):
    s = (th >= lo) & (th < hi)
    r = [D[s, c].mean() / mu_c[s, c].mean() for c in range(2)]
    # error from region scatter
    rr = np.array([[D[s & (lab == L), c].sum() / mu_c[s & (lab == L), c].sum() for c in range(2)] for L in range(144) if (s & (lab == L)).sum() > 50])
    e = rr.std(0) / np.sqrt(len(rr))
    print(f"[{lo},{hi})  {s.sum():8d}  {r[0]:.4f}±{e[0]:.4f}  {r[1]:.4f}±{e[1]:.4f}")
for name, mu in (("ensemble mu", mu_c.sum(1)), ("realised-ybar mu", mu_r.sum(1))):
    for scale in (1.0, 1.005, 1.01):
        dp = D.sum(1) - scale * mu; dom = q > XM.max(1)
        single = qo > 4; det = (qo + dp > 4) & dom
        lost = (single & dom & (qo + dp < 4)).sum() / single.sum()
        print(f"{name} x{scale}: q_cut=4 lost {lost:.4f}, N_det_b/N_single {det.sum()/single.sum():.4f}, B2neg(0.1) {(det & (dp/q < -0.1)).sum()/det.sum():.4f}")
