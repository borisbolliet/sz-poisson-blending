"""f10 (step-3b verification, lens 'fits'): bin-by-bin agreement of the estimators (claim K16), with MY bootstrap
resamples (f02; the tent R_s of f07 uses the same resamples).  Bins with the build's N_exp >= 10, q_cut 5; chi^2 =
sum over bins of (A - B)^2 / var_boot(A - B) (diagonal).  N_iso vs N_exp: per bin N_iso / N_exp - 1, its resamples as
R / R_p (= N_iso / N_exp per resample where N_det > 0).  Also the N_exp-weighted mean of R_p - R_s per q bin.
Usage: f10_k16.py
"""
import os, sys
os.environ["VF_THREADS"] = "1"
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import numpy as np
import f00lib as L


def chi2(A, AB, Bv, BB, ok):
    d = A - Bv; dB = AB - BB
    s = dB.std(0)
    m = ok & (s > 0)
    return float(np.sum((d[m] / s[m]) ** 2)), int(m.sum())


print("q_cut 5, bins with the build's N_exp >= 10: chi^2 (diagonal, my bootstrap errors of the difference) / number of bins")
for setup in ("prod_none", "szifi3000_none", "prod_fof10"):
    for mode in L.MODES:
        m = np.load(os.path.join(L.OUT, f"dv_{setup}_{mode}.npz"))
        t = np.load(os.path.join(L.OUT, f"f07_{setup}_{mode}.npz"))
        b = np.load(os.path.join(L.D3B, f"datavec_{setup}_{mode}.npz"))
        ok = b["Nexp_q5"] >= 10
        Rs, RsB = m["Rs_q5"], m["Rsboot_q5"].astype(float)
        Rt, RtB = t["Rt_q5"], t["Rtboot_q5"].astype(float)
        Rp, RpB = m["Rp_q5"], m["Rpboot_q5"].astype(float)
        R, RB = m["R_bilin_q5"], m["Rboot_bilin_q5"].astype(float)
        Nd = m["Ndet_q5"]
        okd = ok & (Nd > 0)
        Q = np.where(okd, m["Niso_q5"] / np.where(m["Nexp_bilin_q5"] > 0, m["Nexp_bilin_q5"], 1), 1.0)
        QB = np.where(okd[None], RB / np.where(RpB > 0, RpB, 1), 1.0)
        c = [("R_p vs tent R_s", chi2(Rp, RpB, Rt, RtB, ok)), ("R_p vs top-hat R_s", chi2(Rp, RpB, Rs, RsB, ok)),
             ("R vs tent R_s", chi2(R, RB, Rt, RtB, ok)), ("R vs top-hat R_s", chi2(R, RB, Rs, RsB, ok)),
             ("N_iso vs N_exp", chi2(Q, QB, np.ones_like(Q), np.ones_like(QB), okd)), ("top-hat vs tent R_s", chi2(Rs, RsB, Rt, RtB, ok))]
        w = m["Nexp_bilin_q5"]
        mq = [np.sum(w[:, k] * (Rp[:, k] - Rt[:, k])) / max(w[:, k].sum(), 1e-300) for k in range(9)]
        print(f"  {setup:15s} {mode:8s}: " + "; ".join(f"{lab} {v[0]:.1f}/{v[1]}" for lab, v in c)
              + "\n      N_exp-weighted mean R_p - tent R_s per q bin [%]: " + " ".join(f"{100 * v:+.3f}" for v in mq), flush=True)
