"""REPAIR checks on the noise / background claims (C3, C4, C5, C6), on three patch sets:
S1 = original 108 patches (skies 1-3), S2 = 216 new patches (skies 4-9, new noise seeds), ALL = 324 patches.
 a) Var(Y_hat_noise)/sigma_k^2 per set
 b) nearest-pixel single-scale noise peak counts vs Kac-Rice (MC evaluation with 4e6 Hessian draws, two seeds)
 c) pure-noise multi-scale (maxQ, 2' greedy merge) false-detection rate per set, S1-S2 difference, per-sky dispersion test
 d) mean tSZ background ybar of the painted M>=1e13 halos: patch 0, ensemble per set, analytic; l=0 bias ybar*kappa/sigma
"""
import os, sys, json
import numpy as np
from scipy.stats import chi2 as chi2d
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import q2lib as L
from analysis_noise_kr import kr_counts
from szsim import cosmo_hmf as ch, signal as sg

SETS = {"S1(0-107)": list(range(0, 108)), "S2(108-323)": list(range(108, 324)), "ALL(0-323)": list(range(324))}
ALLP = SETS["ALL(0-323)"]
O = {}
for p in ALLP:
    with np.load(f"{L.DATA}/patch_out/patch_{p:03d}.npz") as f:
        O[p] = {k: f[k] for k in ("sig", "st_var_N", "means", "npk_0", "npk_6", "npk_11", "cand_V0", "candv_V0")}
sig = O[0]["sig"]; K = len(sig)
A2 = (L.LBOX ** 2) / 3600.0
sky = lambda p: p // 36

print("a) Var(Y_hat_noise)/sigma_k^2")
for name, P in SETS.items():
    v = np.array([O[p]["st_var_N"] for p in P]) / sig ** 2
    m, e = v.mean(0), v.std(0, ddof=1) / np.sqrt(len(P))
    print(f"   {name:12s}: " + "  ".join(f"{L.THETA_K[k]:.2f}': {m[k]:.5f}+-{e[k]:.5f} ({(m[k]-1)/e[k]:+.2f}s)" for k in (0, 5, 11)))

print("\nb) nearest-pixel single-scale noise peaks per patch vs Kac-Rice")
nus = np.array([3.0, 3.5, 4.0, 4.5])
for k in (0, 6, 11):
    kr1 = kr_counts(k, nus, nmc=4_000_000, seed=11) * L.LBOX ** 2
    kr2 = kr_counts(k, nus, nmc=4_000_000, seed=12) * L.LBOX ** 2
    kr = 0.5 * (kr1 + kr2)
    print(f"   scale {L.THETA_K[k]:.2f}': Kac-Rice (two MC seeds) " + "  ".join(f"nu>{n}: {a:.3f}/{b:.3f}" for n, a, b in zip(nus, kr1, kr2)))
    for name, P in SETS.items():
        c = np.array([[(O[p][f"npk_{k}"] >= nu).sum() for nu in nus] for p in P], float)
        m, e = c.mean(0), np.sqrt(c.sum(0)) / len(P)
        print(f"     {name:12s}: " + "  ".join(f"nu>{n}: {mm:.2f}+-{ee:.2f} ({100*(mm/kk-1):+.1f}%, {(mm-kk)/ee:+.1f}s)"
                                         for n, mm, ee, kk in zip(nus, m, e, kr)))

print("\nc) pure-noise (V0) multi-scale maxQ detections, greedy 2' merge, per deg^2")
QC = [4, 5, 6, 7]
cnt = np.zeros((324, 4))
for p in ALLP:
    c, cv = O[p]["cand_V0"], O[p]["candv_V0"]
    s = cv[:, 0] >= 4.0
    c, cv = c[s], cv[s]
    x = (c[:, 1] + 0.5) * L.PIX; y = (c[:, 0] + 0.5) * L.PIX
    acc = L.merge_greedy(x, y, L.THETA_K[c[:, 2]].astype(np.float64), cv[:, 0].copy(), 2.0, 0.0, L.LBOX)
    for i, qc in enumerate(QC):
        cnt[p, i] = (cv[acc, 0] >= qc).sum()
rates = {}
for name, P in SETS.items():
    t = cnt[P].sum(0); a = len(P) * A2
    rates[name] = (t / a, np.sqrt(t) / a, t)
    print(f"   {name:12s} ({a:.0f} deg^2): " + "  ".join(f"q>{qc}: {t[i]:.0f} ev -> {t[i]/a:.4e}+-{np.sqrt(t[i])/a:.1e}" for i, qc in enumerate(QC)))
for i, qc in enumerate(QC[:2]):
    r1, e1, _ = rates["S1(0-107)"]; r2, e2, _ = rates["S2(108-323)"]
    print(f"   q>{qc}: S1 - S2 = {(r1[i]-r2[i])/np.hypot(e1[i], e2[i]):+.2f} sigma (Poisson)")
# per-sky dispersion of the q>4 counts (9 skies of 36 patches): chi2 vs Poisson
for i, qc in enumerate(QC[:2]):
    ns = np.array([cnt[s * 36:(s + 1) * 36, i].sum() for s in range(9)])
    mu = ns.mean(); ch2 = ((ns - mu) ** 2 / mu).sum()
    pc = np.array([cnt[p, i] for p in ALLP])
    print(f"   q>{qc}: per-sky counts {ns.astype(int).tolist()}  chi2/dof = {ch2:.2f}/8 (p={chi2d.sf(ch2, 8):.3f});"
          f"  per-patch var/mean = {pc.var(ddof=1)/pc.mean():.3f} +- {np.sqrt(2/323):.3f}")
kr0 = np.array([kr_counts(0, np.array([4.0, 5.0, 6.0]), nmc=4_000_000, seed=11)]) * 3600
krs = np.array([kr_counts(k, np.array([4.0, 5.0, 6.0]), nmc=4_000_000, seed=11) for k in range(K)]) * 3600
print(f"   Kac-Rice single scale 0.5': q>4/5/6 = {kr0[0,0]:.4f} / {kr0[0,1]:.3e} / {kr0[0,2]:.3e} per deg^2;"
      f"  sum over 12 scales {krs[:,0].sum():.4f} / {krs[:,1].sum():.3e} / {krs[:,2].sum():.3e}")

print("\nd) mean tSZ background of the painted M>=1e13 halos (l=0 mode)")
yb = np.array([O[p]["means"][:3].sum() for p in ALLP]); ycomp = np.array([O[p]["means"][:3] for p in ALLP])
print(f"   patch 0: ybar = {yb[0]:.4e}  components [1e13,3e13) {ycomp[0,0]:.4e}, [3e13,1e14) {ycomp[0,1]:.4e}, >=1e14 {ycomp[0,2]:.4e}")
for name, P in SETS.items():
    v = yb[P]
    print(f"   {name:12s}: mean {v.mean():.4e} +- {v.std(ddof=1)/np.sqrt(len(v)):.1e}  median {np.median(v):.4e}"
          f"  16-84% [{np.percentile(v,16):.4e}, {np.percentile(v,84):.4e}]  components mean "
          + " ".join(f"{a:.4e}" for a in ycomp[P].mean(0)))
m = L.MODEL; T = ch.default_table()
Zg, LMg = np.meshgrid(T.z, T.lnM, indexing="ij")
YT = m.Y500_med(np.exp(LMg), Zg) * np.exp(0.5 * m.sigma_lnY ** 2) * sg.RATIO_5R500
FULLSKY_ARCMIN2 = 4 * np.pi / ch.ARCMIN ** 2
def yan(lo, hi):
    wM = T._weights(T.lnM, np.log(lo), np.log(hi)); wz = T._weights(T.z, T.z[0], T.z[-1])
    return wz @ (T.d2N * YT) @ wM / FULLSKY_ARCMIN2
ya = [yan(1e13, 3e13), yan(3e13, 1e14), yan(1e14, 1e16)]
print(f"   analytic (HMF table, <Y> incl. lognormal scatter, x RATIO_5R500={sg.RATIO_5R500:.4f}): ybar = {sum(ya):.4e}"
      f"  components " + " ".join(f"{a:.4e}" for a in ya))
for lab, y in [("patch 0", yb[0]), ("ALL mean", yb.mean()), ("analytic", sum(ya)),
               ("ALL 16%", np.percentile(yb, 16)), ("ALL 84%", np.percentile(yb, 84))]:
    bias = y * L.kappa_mean(sig) / sig
    print(f"   l=0 bias ybar*kappa_k/sigma_k for {lab:9s} (ybar={y:.3e}): 0.5' {bias[0]:.3f}  1.95' {bias[5]:.3f}  10' {bias[11]:.3f}")
