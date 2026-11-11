"""Physics caveat (verifier): the S1/S2 priors on the scaling relation are centred on the TRUE (unblended) relation, but a
weak-lensing calibration of detected clusters measures the BLENDED observable.  Estimate where such a calibration
would centre (1-b): average fractional observable shift of the detected (q_obs > 5) population from the P(D) kernel,
   Delta ln Y ~ <t_blend> / q_A,     t_blend = q_obs - q_A - n  (= delta_tr - mu),
using (a) the kernel mean of t (noise has zero mean) and (b) the median shift (median of n+delta-mu, noise symmetric),
each weighted by the expected number of blended detections of that population point.  Then
   Delta ln(1-b)_cal = Delta ln Y / alpha   (Y ~ [(1-b) M]^alpha).
Prints the implied prior centre; the fits with that centre are run with c05b (env PRIOR_OMB)."""
import os
os.environ.setdefault("NUMBA_NUM_THREADS", "1"); os.environ.setdefault("OMP_NUM_THREADS", "1")
import numpy as np
import cclib as cc

K = np.load(f"{cc.OUT}/G_fid_b03rates.npz")
G, Pdom, th_n = K["G"].astype(np.float64), K["Pdom"], K["th"]
T = cc.TGRID
C = G / np.maximum(Pdom[..., None], 1e-30)                      # CDF of t = n + delta_tr - mu
dC = np.diff(C, axis=2); tm = 0.5 * (T[1:] + T[:-1])
mean_t = (dC * tm).sum(2) / dC.sum(2)                           # (ntheta, nq)
med_t = np.array([[np.interp(0.5, C[i, k], T) for k in range(C.shape[1])] for i in range(C.shape[0])])
pop = cc.population(cc.FID)
lth = np.clip(np.log(pop["th"]), np.log(th_n[0]), np.log(th_n[-1])); q = np.clip(pop["q"], cc.QNODES[0], cc.QNODES[-1])
from scipy.interpolate import RegularGridInterpolator as RGI
pts = np.column_stack([lth, q])
mt = RGI((np.log(th_n), cc.QNODES), mean_t)(pts); md = RGI((np.log(th_n), cc.QNODES), med_t)(pts)
pd = np.clip(RGI((np.log(th_n), cc.QNODES), Pdom)(pts), 0, 1)
# detection weight: P_dom * P(q_obs > 5) under the blended kernel (approximated by the naive Phi with the mean shift)
from scipy.stats import norm
for qc in (5.0,):
    wdet = pop["w"] * pd * norm.sf(qc - pop["q"] - md)
    for lab, s in (("mean", mt), ("median", md)):
        dlnY = np.sum(wdet * s / pop["q"]) / wdet.sum()
        dlnY_hi = np.sum((wdet * s / pop["q"])[pop["q"] > 5]) / wdet[pop["q"] > 5].sum()
        print(f"q_obs>{qc}: {lab} blend shift: <Delta ln Y> = {dlnY:+.4f} (all detected), {dlnY_hi:+.4f} (q_A>5 only);"
              f" implied (1-b)_cal = 0.8 x exp(<dlnY>/1.79) = {0.8*np.exp(dlnY/1.79):.4f} ({0.8*np.exp(dlnY_hi/1.79):.4f});"
              f" shift in prior sigma (0.04): {(0.8*np.exp(dlnY/1.79)-0.8)/0.04:+.2f} ({(0.8*np.exp(dlnY_hi/1.79)-0.8)/0.04:+.2f})")
print("mean t at q=5 for th nodes", np.round(th_n[::6], 2), ":", np.round(mean_t[::6, 49], 3), " median:", np.round(med_t[::6, 49], 3))
