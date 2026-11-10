"""(a) y-confusion noise from the *undetected* halo population (q_true < q_thr, any mass >= 1e13):
       map variance of the filtered map vs analytic Poisson one-halo variance with a truncated lognormal:
       <Y^2 1(Y < q_thr sigma_Y(theta500))> = Y_med^2 e^{2s^2} Phi((ln(Ycut/Y_med) - 2 s^2)/s)
(b) the HMF-analytic vs realised-population confusion variance for [1e13,1e14) on the three full skies
    (heavy low-z tail check).
(c) pixel-sampling bias of noise peak heights: 3x3 quadratic sub-pixel refinement vs Kac-Rice, scale 0.5'.
"""
import os
os.environ.setdefault("OMP_NUM_THREADS", "4")
import numpy as np
from scipy import fft as sfft
from scipy.special import ndtr
import q2lib as L
from szsim import cosmo_hmf as ch, signal as sg

tab = L.build_tables(); sigH = tab["sigk_arcmin2"]; K = len(L.THETA_K)
psis, sig = L.filter_bank()
m = L.MODEL; s = m.sigma_lnY
lw = np.linspace(0, 10 / L.SIGB, 40001); B4 = np.exp(-2 * (lw * L.SIGB) ** 2)
thg = np.geomspace(0.005, 120, 400)
W = np.array([[(sigH[k] ** 2 / L.NW) ** 2 * np.trapezoid(lw * B4 * sg.template_hankel(lw * L.THETA_K[k]) ** 2 *
               sg.template_hankel(lw * t) ** 2, lw) / (2 * np.pi) for t in thg] for k in range(K)])
Wf = lambda k, th: np.exp(np.interp(np.log(th), np.log(thg), np.log(W[k])))
FULL = 4 * np.pi / ch.ARCMIN ** 2
T = ch.default_table()
Zg, LMg = np.meshgrid(T.z, T.lnM, indexing="ij")
thT = m.theta500(np.exp(LMg), Zg); YT = m.Y500_med(np.exp(LMg), Zg); sYT = m.sigma_Y(thT)
wz = T._weights(T.z, T.z[0], T.z[-1])

def var_an(qthr, Mlo=1e13, Mhi=1e16):
    wM = T._weights(T.lnM, np.log(Mlo), np.log(Mhi))
    Ycut = qthr * sYT
    mom = YT ** 2 * np.exp(2 * s * s) * ndtr((np.log(Ycut / YT) - 2 * s * s) / s)
    return np.array([wz @ (T.d2N * mom * Wf(k, thT)) @ wM for k in range(K)]) / FULL

# ---------------- (a)
NPa = 40
QTHR = [2.0, 4.0, 5.0]
v_map = {q: [] for q in QTHR}; mad_map = {q: [] for q in QTHR}
for p in range(NPa):
    d = L.load_patch(p)
    fts = {}
    for q in QTHR:
        sel = d["q"] < q
        mp = np.zeros((L.NPIX, L.NPIX))
        L.paint_map(mp, d["x"][sel], d["y"][sel], d["Y500"][sel], d["theta500"][sel], tab["paint"], tab["rP"], L.LNTH0, L.DLNTH, L.PIX)
        fts[q] = sfft.rfft2(mp, workers=4)
    vv = {q: [] for q in QTHR}; mm = {q: [] for q in QTHR}
    for k in range(K):
        for q in QTHR:
            F = sfft.irfft2(fts[q] * psis[k], s=(L.NPIX, L.NPIX), workers=4)
            vv[q].append(F.var()); a = F[::2, ::2]
            mm[q].append(1.482602 * np.median(np.abs(a - np.median(a))))
    for q in QTHR:
        v_map[q].append(vv[q]); mad_map[q].append(mm[q])
print(f"(a) confusion variance / sigma_k^2 from halos with q_true < q_thr (all M >= 1e13), {NPa} patches")
for q in QTHR:
    a = np.array(v_map[q]) / sig ** 2; mu, er = a.mean(0), a.std(0, ddof=1) / np.sqrt(NPa)
    an = var_an(q) / sig ** 2
    md = np.array(mad_map[q]).mean(0) / sig
    print(f"  q_thr={q}:")
    for k in range(K):
        print(f"    theta_k={L.THETA_K[k]:6.3f}'  map {mu[k]:.4f}+-{er[k]:.4f}  analytic {an[k]:.4f}  ({(mu[k]-an[k])/er[k]:+.2f} sigma)"
              f"  sigma_eff/sigma = {np.sqrt(1+mu[k]):.4f} (analytic {np.sqrt(1+an[k]):.4f});  MAD(conf)/sigma {md[k]:.4f}")

# ---------------- (b)
print("\n(b) [1e13,1e14) confusion variance / sigma_k^2: realised full-sky catalogues vs HMF analytic")
wM = T._weights(T.lnM, np.log(1e13), np.log(1e14))
an = np.array([wz @ (T.d2N * YT ** 2 * np.exp(2 * s * s) * Wf(k, thT)) @ wM for k in range(K)]) / FULL / sig ** 2
rows = []
for seed in (1, 2, 3):
    with np.load(f"{L.ROOT}/data/catalogs/cat_M1e13_seed{seed}.npz") as f:
        M = f["M500"]; sel = M < 1e14; del M
        Y = f["Y500"][sel].astype(np.float64); th = f["theta500"][sel].astype(np.float64); zz = f["z"][sel]
    rows.append([(Y ** 2 * Wf(k, th)).sum() / FULL / sig[k] ** 2 for k in range(K)])
    if seed == 1:
        lowz = [(Y[zz < zc] ** 2 * Wf(K - 1, th[zz < zc])).sum() / FULL / sig[K - 1] ** 2 for zc in (0.02, 0.05, 0.1, 0.2)]
    del Y, th, zz
rows = np.array(rows)
for k in range(K):
    print(f"    theta_k={L.THETA_K[k]:6.3f}'  skies {rows[:, k].round(5)}  mean/analytic = {rows[:, k].mean()/an[k]:.4f}")
print(f"    seed 1, theta_k=10': share of the variance from z<0.02, 0.05, 0.1, 0.2: "
      + ", ".join(f"{v/rows[0, -1]:.3f}" for v in lowz))

# ---------------- (c)
print("\n(c) noise-only single-scale (0.5') peaks: nearest-pixel height vs 3x3 quadratic sub-pixel refinement")
rng = np.random.default_rng(777)
nus = np.array([3.0, 3.5, 4.0, 4.5])
c_pix = np.zeros(nus.size); c_ref = np.zeros(nus.size); NPc = 60
for p in range(NPc):
    nm = rng.standard_normal((L.NPIX, L.NPIX)) * (m.noise_y_arcmin / L.PIX)
    F = sfft.irfft2(sfft.rfft2(nm, workers=4) * psis[0], s=(L.NPIX, L.NPIX), workers=4) / sig[0]
    iy, ix = L.local_max(F.astype(np.float32), np.float32(2.7))
    n = L.NPIX
    g = lambda dy, dx: F[(iy + dy) % n, (ix + dx) % n]
    f0 = g(0, 0); fx = (g(0, 1) - g(0, -1)) / 2; fy = (g(1, 0) - g(-1, 0)) / 2
    fxx = g(0, 1) - 2 * f0 + g(0, -1); fyy = g(1, 0) - 2 * f0 + g(-1, 0)
    fxy = (g(1, 1) - g(1, -1) - g(-1, 1) + g(-1, -1)) / 4
    Hm = np.stack([np.stack([fxx, fxy], -1), np.stack([fxy, fyy], -1)], -2)
    gr = np.stack([fx, fy], -1)
    dlt = -np.linalg.solve(Hm, gr[..., None])[..., 0]
    dlt = np.clip(dlt, -0.5, 0.5)
    fref = f0 + (gr * dlt).sum(-1) + 0.5 * np.einsum("ni,nij,nj->n", dlt, Hm, dlt)
    c_pix += (f0[:, None] >= nus[None, :]).sum(0); c_ref += (fref[:, None] >= nus[None, :]).sum(0)
from analysis_noise_kr import kr_counts
kr = kr_counts(0, nus) * L.LBOX ** 2
for i, nu in enumerate(nus):
    e = np.sqrt(c_pix[i]) / NPc
    print(f"   nu>{nu}: pixel {c_pix[i]/NPc:8.3f}+-{e:.3f}  refined {c_ref[i]/NPc:8.3f}+-{np.sqrt(c_ref[i])/NPc:.3f}"
          f"  Kac-Rice {kr[i]:8.3f}  (refined: {(c_ref[i]/NPc-kr[i])/(np.sqrt(c_ref[i])/NPc):+.2f} sigma)")
