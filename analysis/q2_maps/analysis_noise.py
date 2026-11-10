"""Noise-level checks and tSZ confusion noise.

1. Empirical matched-filter sigma_Y per scale in the maps (noise component) vs analytic Hankel sigma_Y.
2. Single-scale noise-only peak counts vs exact Gaussian-field Kac-Rice expectation (and the EC approximation).
3. Multi-scale (max over 12 scales) noise-only false-detection rate after merging, per q_cut.
4. Confusion noise from unresolved halos: map variance of the filtered [1e13,1e14) and [1e13,3e13) halo maps vs the
   analytic Poisson (one-halo) variance  Var_k = int dN/dOmega <Y^2> W_k(theta),
   W_k(theta) = (sigma_k^2/N_w)^2 (1/2pi) int dl l B_l^4 g(l theta_k)^2 g(l theta)^2 ;
   extrapolation below 1e13 with colossus; effective noise sigma_eff/sigma_inst.
"""
import os, json
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import q2lib as L
from szsim import cosmo_hmf as ch, signal as sg

FIG = f"{L.ROOT}/figures/q2_maps"; os.makedirs(FIG, exist_ok=True)
NP = 108
O = [dict(np.load(f"{L.DATA}/patch_out/patch_{p:03d}.npz")) for p in range(NP)]
K = len(L.THETA_K); sig = O[0]["sig"]
tab = L.build_tables(); sigH = tab["sigk_arcmin2"]
AREA_ARCMIN2 = L.LBOX ** 2; AREA_DEG2 = AREA_ARCMIN2 / 3600
out = {}

# ------------------------------------------------------------- 1. sigma_Y
vN = np.array([o["st_var_N"] for o in O]) / sig[None, :] ** 2
r = vN.mean(0); e = vN.std(0, ddof=1) / np.sqrt(NP)
print("1) matched-filter noise: empirical Var(Y_hat_noise)/sigma_k^2 (108 patches), grid sigma vs Hankel sigma")
for k in range(K):
    print(f"   theta_k={L.THETA_K[k]:6.3f}'  sigma_Hankel={sigH[k]:.4e}  grid/Hankel-1={sig[k]/sigH[k]-1:+.1e}"
          f"  Var_emp/sigma^2 = {r[k]:.5f} +- {e[k]:.5f}  ({(r[k]-1)/e[k]:+.2f} sigma)")
out["sigma_emp_ratio"] = dict(ratio=r.tolist(), err=e.tolist())
chi2 = (((r - 1) / e) ** 2).sum()
print(f"   chi2 over 12 scales = {chi2:.1f} (scales are correlated)")

# ------------------------------------------------------------- 2. single-scale noise peaks
f, fr = np.fft.fftfreq(L.NPIX, d=L.PIX) * 2 * np.pi, None
lg = np.linspace(0, 12 / L.SIGB, 200001)
def moments(k):
    T2 = (sg.template_hankel(lg * L.THETA_K[k]) * np.exp(-0.5 * (lg * L.SIGB) ** 2)) ** 2
    s = [np.trapezoid(lg ** (1 + 2 * j) * T2, lg) / (2 * np.pi) for j in range(3)]
    return s[1] / (2 * s[0]), s[2] / s[0]          # lambda = Var(d_i f) for unit-var f;  s2 = sigma2^2/sigma0^2

rng = np.random.default_rng(1)
def kac_rice_above(nus, lam, s2, nmc=400000):
    """Expected density of local maxima with height > nu (per arcmin^2), unit-variance isotropic 2D Gaussian field."""
    nug = np.linspace(min(nus), 9.0, 241)
    a, b = 3 * s2 / 8, s2 / 8
    C = np.array([[a - lam ** 2, b - lam ** 2], [b - lam ** 2, a - lam ** 2]])
    Lc = np.linalg.cholesky(C)
    z = rng.standard_normal((nmc, 2)) @ Lc.T; h12 = rng.standard_normal(nmc) * np.sqrt(b)
    dens = []
    for nu in nug:
        H11 = -lam * nu + z[:, 0]; H22 = -lam * nu + z[:, 1]
        det = H11 * H22 - h12 ** 2
        ok = (H11 < 0) & (det > 0)
        dens.append(np.mean(det * ok) * np.exp(-nu ** 2 / 2) / np.sqrt(2 * np.pi) / (2 * np.pi * lam))
    dens = np.array(dens)
    cum = np.array([np.trapezoid(dens[nug >= nu], nug[nug >= nu]) for nu in nus])
    return cum

nus = np.array([3.0, 3.5, 4.0, 4.5, 5.0])
print("\n2) single-scale noise-only local maxima (3x3) per 10x10 deg patch: MC (108 patches) vs Kac-Rice vs EC")
pk = {}
for k in [0, 2, 4, 6, 8, 11]:
    lam, s2 = moments(k)
    kr = kac_rice_above(nus, lam, s2) * AREA_ARCMIN2
    ec = lam / (2 * np.pi) ** 1.5 * nus * np.exp(-nus ** 2 / 2) * AREA_ARCMIN2
    cnt = np.array([[(o[f"npk_{k}"] >= nu).sum() for nu in nus] for o in O])
    m, er = cnt.mean(0), np.sqrt(cnt.sum(0)) / NP
    for i, nu in enumerate(nus):
        print(f"   k={k:2d} (theta={L.THETA_K[k]:5.2f}') nu>{nu:.1f}: MC {m[i]:9.3f} +- {er[i]:.3f}   Kac-Rice {kr[i]:9.3f}"
              f" ({(m[i]-kr[i])/er[i]:+.2f} sigma)   EC {ec[i]:9.3f}")
    pk[k] = dict(mc=m.tolist(), err=er.tolist(), kr=kr.tolist(), ec=ec.tolist())
out["noise_peaks_single_scale"] = pk

# ------------------------------------------------------------- 3. multi-scale noise-only false detections (merged)
def merge(c, cv, rule):
    """Greedy: sort by Q desc, reject a candidate closer than r_merge to an already accepted one."""
    o_ = np.argsort(-cv[:, 0]); acc = []
    xy = (c[:, [1, 0]] + 0.5) * L.PIX; th = L.THETA_K[c[:, 2]]
    for j in o_:
        ok = True
        for a in acc:
            dx = (xy[j] - xy[a] + L.LBOX / 2) % L.LBOX - L.LBOX / 2
            rm = rule(th[a])
            if dx[0] ** 2 + dx[1] ** 2 < rm * rm:
                ok = False; break
        if ok:
            acc.append(j)
    return np.array(acc, int)

RULES = {"R1": lambda t: 1.0, "R2": lambda t: 2.0, "R3": lambda t: max(2.0, t)}
print("\n3) noise-only (V0) multi-scale detections after merging, per deg^2 (10800 deg^2)")
fr_out = {}
for rn, rule in RULES.items():
    cnts = np.zeros((NP, 4))
    for p, o in enumerate(O):
        c, cv = o["cand_V0"], o["candv_V0"]
        s = cv[:, 0] >= 4.0
        acc = merge(c[s], cv[s], rule)
        for i, qc in enumerate([4, 5, 6, 7]):
            cnts[p, i] = (cv[s][acc, 0] >= qc).sum()
    tot = cnts.sum(0)
    print(f"   {rn}: " + "  ".join(f"q>{qc}: {tot[i]:.0f} -> {tot[i]/(NP*AREA_DEG2):.3e} +- {np.sqrt(tot[i])/(NP*AREA_DEG2):.1e} /deg^2"
                                   for i, qc in enumerate([4, 5, 6, 7])))
    fr_out[rn] = (tot / (NP * AREA_DEG2)).tolist()
# single-scale bounds at q=4,5 for comparison
lam0, s20 = moments(0)
krs = np.array([kac_rice_above(np.array([4.0, 5.0, 6.0]), *moments(k)) for k in range(K)]) * 3600
print(f"   Kac-Rice single scale theta=0.5': q>4 {krs[0,0]:.3e}, q>5 {krs[0,1]:.3e}, q>6 {krs[0,2]:.3e} /deg^2;"
      f"  sum over 12 scales: {krs[:,0].sum():.3e}, {krs[:,1].sum():.3e}, {krs[:,2].sum():.3e} /deg^2")
out["noise_false_rate_per_deg2"] = fr_out

# ------------------------------------------------------------- 4. confusion noise
lw = np.linspace(0, 10 / L.SIGB, 40001)
B4 = np.exp(-2 * (lw * L.SIGB) ** 2)
thg = np.geomspace(0.005, 120, 400)
W = np.zeros((K, thg.size))
for k in range(K):
    gk = sg.template_hankel(lw * L.THETA_K[k])
    for i, t in enumerate(thg):
        W[k, i] = (sigH[k] ** 2 / L.NW) ** 2 * np.trapezoid(lw * B4 * gk ** 2 * sg.template_hankel(lw * t) ** 2, lw) / (2 * np.pi)
Wf = lambda k, th: np.exp(np.interp(np.log(th), np.log(thg), np.log(W[k])))
m = L.MODEL
FULLSKY_ARCMIN2 = 4 * np.pi / ch.ARCMIN ** 2
e2s = np.exp(2 * m.sigma_lnY ** 2)

T = ch.default_table()
Zg, LMg = np.meshgrid(T.z, T.lnM, indexing="ij")
thT = m.theta500(np.exp(LMg), Zg); YT = m.Y500_med(np.exp(LMg), Zg)
def var_analytic(Mlo, Mhi):
    wM = T._weights(T.lnM, np.log(Mlo), np.log(Mhi)); wz = T._weights(T.z, T.z[0], T.z[-1])
    return np.array([wz @ (T.d2N * YT ** 2 * e2s * Wf(k, thT)) @ wM for k in range(K)]) / FULLSKY_ARCMIN2

def var_low(Mlo, Mhi, nz=120, nM=50):
    zg = np.linspace(0.01, 3.0, nz); lg_ = np.linspace(np.log(Mlo), np.log(Mhi), nM)
    acc = np.zeros(K)
    D = np.array([ch.d2N_dzdlnM_fullsky(np.exp(lg_), z) for z in zg])
    Z2, L2 = np.meshgrid(zg, lg_, indexing="ij")
    th = m.theta500(np.exp(L2), Z2); Y = m.Y500_med(np.exp(L2), Z2)
    for k in range(K):
        integrand = D * Y ** 2 * e2s * Wf(k, th)
        acc[k] = np.trapezoid(np.trapezoid(integrand, lg_, axis=1), zg)
    return acc / FULLSKY_ARCMIN2

vAB_an = var_analytic(1e13, 1e14); vA_an = var_analytic(1e13, 3e13); vB_an = var_analytic(3e13, 1e14)
vC_an = var_analytic(1e14, 1e16)
v12 = var_low(1e12, 1e13); v11 = var_low(1e11, 1e12)
# realised-population version (sum over the halos actually in the patches)
vAB_pop = np.zeros(K); vA_pop = np.zeros(K)
for p in range(NP):
    d = L.load_patch(p)
    for lo, hi, acc in [(1e13, 1e14, vAB_pop), (1e13, 3e13, vA_pop)]:
        s = (d["M500"] >= lo) & (d["M500"] < hi)
        for k in range(K):
            acc[k] += (d["Y500"][s] ** 2 * Wf(k, d["theta500"][s])).sum() / AREA_ARCMIN2 / NP

def stat(key):
    a = np.array([o[key] for o in O]) / sig[None, :] ** 2
    return a.mean(0), a.std(0, ddof=1) / np.sqrt(NP)
mAB, eAB = stat("st_var_AB"); mA, eA = stat("st_var_A"); mB, eB = stat("st_var_B"); mC, eC = stat("st_var_C")
cov, ecov = stat("st_cov_ABN")
print("\n4) confusion variance in units of sigma_k^2 (instrumental): map (108 patches) vs analytic Poisson one-halo term")
print("   theta_k | [1e13,1e14): map        analytic  pop-sum  (dev) | [1e13,3e13): map       analytic (dev) | [1e14,1e16): map  analytic | <1e13 analytic [1e12,1e13) [1e11,1e12) | cov(AB,N)")
for k in range(K):
    s2 = sig[k] ** 2
    print(f"   {L.THETA_K[k]:6.3f}  | {mAB[k]:.4f}+-{eAB[k]:.4f}  {vAB_an[k]/s2:.4f}  {vAB_pop[k]/s2:.4f} ({(mAB[k]-vAB_an[k]/s2)/eAB[k]:+.2f}s) |"
          f" {mA[k]:.4f}+-{eA[k]:.4f}  {vA_an[k]/s2:.4f} ({(mA[k]-vA_an[k]/s2)/eA[k]:+.2f}s) | {mC[k]:.3f}+-{eC[k]:.3f} {vC_an[k]/s2:.3f} |"
          f" {v12[k]/s2:.4f} {v11[k]/s2:.5f} | {cov[k]:+.1e}+-{ecov[k]:.1e}")
out["confusion"] = dict(theta_k=L.THETA_K.tolist(), map_AB=mAB.tolist(), err_AB=eAB.tolist(), an_AB=(vAB_an / sig ** 2).tolist(),
                        map_A=mA.tolist(), err_A=eA.tolist(), an_A=(vA_an / sig ** 2).tolist(),
                        an_12_13=(v12 / sig ** 2).tolist(), an_11_12=(v11 / sig ** 2).tolist(),
                        map_C=mC.tolist(), an_C=(vC_an / sig ** 2).tolist())
print("\n   effective noise sigma_eff/sigma_inst = sqrt(1 + Var_conf/sigma^2):")
for k in range(K):
    s2 = sig[k] ** 2
    print(f"   theta_k={L.THETA_K[k]:6.3f}':  M in [1e13,1e14) map {np.sqrt(1+mAB[k]):.4f}   analytic {np.sqrt(1+vAB_an[k]/s2):.4f};"
          f"  adding [1e11,1e13) analytic {np.sqrt(1+(vAB_an[k]+v12[k]+v11[k])/s2):.4f}")
# robust widths
def rstat(key):
    a = np.array([o[key] for o in O]) / sig[None, :]
    return a.mean(0), a.std(0, ddof=1) / np.sqrt(NP)
mN, _ = rstat("st_mad_N"); mABN, eABN = rstat("st_mad_ABN"); mALL, eALL = rstat("st_mad_ABCN"); mCN, _ = rstat("st_mad_CN")
print("\n   robust (MAD) width of the filtered map in units of sigma_k:  noise only | +[1e14,..) | +[1e13,1e14) | all halos")
for k in range(K):
    print(f"   theta_k={L.THETA_K[k]:6.3f}':  {mN[k]:.4f} | {mCN[k]:.4f} | {mABN[k]:.4f}+-{eABN[k]:.4f} | {mALL[k]:.4f}+-{eALL[k]:.4f}")
pct = np.mean([o["st_pct_AB"] for o in O], 0) / sig[:, None]
print("\n   percentiles of the filtered [1e13,1e14) halo map (no noise) in units of sigma_k, pct =", O[0]["pct_q"])
for k in [0, 3, 6, 9, 11]:
    print(f"   theta_k={L.THETA_K[k]:6.3f}': " + " ".join(f"{v:+.3f}" for v in pct[k]))
out["robust"] = dict(mad_N=mN.tolist(), mad_ABN=mABN.tolist(), mad_ALL=mALL.tolist())
json.dump(out, open(f"{L.DATA}/noise_confusion.json", "w"), indent=1)

# ------------------------------------------------------------- figure
fig, ax = plt.subplots(1, 2, figsize=(11, 4.2))
ax[0].errorbar(L.THETA_K, r, e, fmt="o", label="map noise: Var/sigma_Y^2 (108 patches)")
ax[0].axhline(1, color="k", lw=0.8)
ax[0].set_xscale("log"); ax[0].set_xlabel("filter scale theta_k [arcmin]"); ax[0].set_ylabel("Var(Y_hat_noise) / sigma_Y,Hankel^2")
ax[0].legend(fontsize=8)
ax[1].errorbar(L.THETA_K, mAB, eAB, fmt="o", label="map: halos 1e13-1e14")
ax[1].plot(L.THETA_K, vAB_an / sig ** 2, "-", label="analytic Poisson 1e13-1e14")
ax[1].errorbar(L.THETA_K, mA, eA, fmt="s", ms=4, label="map: halos 1e13-3e13")
ax[1].plot(L.THETA_K, vA_an / sig ** 2, "--", label="analytic 1e13-3e13")
ax[1].plot(L.THETA_K, v12 / sig ** 2, ":", label="analytic 1e12-1e13 (not painted)")
ax[1].set_xscale("log"); ax[1].set_yscale("log"); ax[1].set_xlabel("filter scale theta_k [arcmin]")
ax[1].set_ylabel("confusion variance / sigma_Y^2"); ax[1].legend(fontsize=7)
plt.tight_layout(); plt.savefig(f"{FIG}/noise_and_confusion.png", dpi=130)
print("saved", f"{FIG}/noise_and_confusion.png")
