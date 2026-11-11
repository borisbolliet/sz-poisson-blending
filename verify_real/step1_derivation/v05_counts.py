"""v05: expected full-sky detections (no blending) with MY sigma_Y (2D grid, method a), by two independent routes:
  (i)  my own HMF integral: colossus Tinker08 (500c, planck18) called directly, Gauss-Legendre in z and ln M,
       Gauss-Hermite over the lognormal Y scatter (no szsim HMF table, no SZModel);
  (ii) a FRESH sampled sky (szsim.sampler, seeds 20261111/2): sum over halos of Phi(Y500/sigma_Y(theta500) - q_cut)
       (exact noise marginalisation per halo; Y500 includes the sampled scatter).
Also: wide-range sigma_Y tables (0.01'-500') saved for v06-v08; check of my Y-M / theta500 code vs the sampler.
"""
import os, sys, time, json
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import vlib as V
V.set_threads(4)
import numpy as np
from scipy.special import ndtr
from scipy.interpolate import CubicSpline

t0 = time.time()
AM = V.AM
QC = [4.0, 5.0, 6.0, 7.0]

# ---------------- wide sigma_Y tables (method a)
TH = np.geomspace(0.01, 500.0, 241)
G = V.LGrid2D(dl=10.0)
tabs = {}
for mode in ("baseline", "goal"):
    iv = V.inv_Ny_chunked(G.l, mode=mode)
    tabs[mode] = np.array([1 / np.sqrt(G.sum(V.g_u(G.l * t * AM) ** 2 * iv)) / AM ** 2 for t in TH])
    del iv
del G
Gw = V.LGrid2D(dl=20.0, lmin=0.0, lmax=40000.0)
ivw = V.inv_Ny_white(Gw.l)
tabs["white"] = np.array([1 / np.sqrt(Gw.sum(V.g_u(Gw.l * t * AM) ** 2 * ivw)) / AM ** 2 for t in TH])
del Gw, ivw
np.savez(os.path.join(HERE, "sigmaY_mine_wide.npz"), theta500=TH, **tabs)
print(f"sigma_Y tables (0.01-500', 241 nodes) done {time.time()-t0:.0f}s")
SP = {k: CubicSpline(np.log(TH), np.log(v)) for k, v in tabs.items()}


def sigY(mode, th):
    return np.exp(SP[mode](np.log(np.clip(th, TH[0], TH[-1]))))


# ---------------- (i) own HMF integral
from colossus.lss import mass_function
c = V.cosmo(); h = c.h
C_KMS = 299792.458


def gl(a, b, n):
    x, w = np.polynomial.legendre.leggauss(n)
    return 0.5 * (b - a) * x + 0.5 * (a + b), 0.5 * (b - a) * w


zs, wz = zip(*[gl(a, b, 48) for a, b in ((0.01, 0.1), (0.1, 0.4), (0.4, 1.0), (1.0, 2.0), (2.0, 3.0))])
zs, wz = np.concatenate(zs), np.concatenate(wz)
lms, wm = zip(*[gl(np.log(a), np.log(b), 40) for a, b in ((1e13, 1e14), (1e14, 1e15), (1e15, 1e16))])
lms, wm = np.concatenate(lms), np.concatenate(wm)
M = np.exp(lms)
xg, wg = np.polynomial.hermite_e.hermegauss(40); wg = wg / wg.sum()
Ntot = 0.0; N1e14 = 0.0
Ndet = {k: np.zeros(len(QC)) for k in tabs}; Ndet14 = {k: np.zeros(len(QC)) for k in tabs}
for z, w_z in zip(zs, wz):
    dn = mass_function.massFunction(M * h, z, mdef="500c", model="tinker08", q_out="dndlnM") * h ** 3   # Mpc^-3
    chi = c.comovingDistance(0.0, z) / h
    dV = C_KMS / (100 * h * c.Ez(z)) * chi ** 2 * 4 * np.pi                                        # Mpc^3 per unit z, full sky
    d2N = dn * dV * w_z * wm
    Ntot += d2N.sum(); N1e14 += d2N[M >= 1e14].sum()
    Ym = V.Y500_med_am2(M, z); th = V.theta500_am(M, z)
    for k in tabs:
        q = (Ym / sigY(k, th))[:, None] * np.exp(0.173 * xg)[None, :]
        for j, qc in enumerate(QC):
            P = (ndtr(q - qc) * wg).sum(1)
            Ndet[k][j] += (d2N * P).sum(); Ndet14[k][j] += (d2N * P)[M >= 1e14].sum()
print(f"(i) own HMF integral: N(M>1e13) = {Ntot:,.1f} (szsim README 29,295,487); N(M>1e14) = {N1e14:,.1f} (396,387)  "
      f"[{time.time()-t0:.0f}s]")
bj = json.load(open(os.path.join(V.ROOT, "data_real", "expected_detections.json")))
bmap = {"white": "white (old)", "baseline": "SO baseline lmin100", "goal": "SO goal lmin100"}
for k in tabs:
    b = np.array(bj[bmap[k]])
    print(f"   {k:9s}: q>4..7 = " + " / ".join(f"{v:,.1f}" for v in Ndet[k]) + "   builder: " +
          " / ".join(f"{v:,.1f}" for v in b) + "   mine/builder-1: " + " ".join(f"{v:+.1e}" for v in Ndet[k] / b - 1))
for k in ("baseline", "goal"):
    print(f"   {k:9s}: M>1e14 only, q>5: {Ndet14[k][1]:,.1f} (builder analytic: {'81,428.5' if k=='baseline' else '126,740.4'})")
print(f"   f_sky=0.4 at q>5: " + ", ".join(f"{k} {0.4*Ndet[k][1]:,.0f}" for k in tabs) + "  (builder 34,365 / 32,682 / 51,267)")

# ---------------- (ii) fresh catalogue sums
from szsim import sampler
for seed in (20261111, 20261112):
    ts = time.time()
    cat = sampler.draw_catalog(seed, fields=("z", "M500", "Y500", "theta500"))
    z = cat["z"].astype(np.float64); Mc = cat["M500"].astype(np.float64)
    Y = cat["Y500"].astype(np.float64); th = cat["theta500"].astype(np.float64)
    del cat
    # my Y-M / theta500 code vs the sampler (scatter included in the sampler's Y)
    sub = slice(0, None, 97)
    r = np.log(Y[sub] / V.Y500_med_am2(Mc[sub], z[sub]))
    tr = th[sub] / V.theta500_am(Mc[sub], z[sub]) - 1
    print(f"\n(ii) sky seed {seed}: N = {len(z):,} (draw {time.time()-ts:.0f}s); ln(Y_cat/Y_med,mine): mean {r.mean():+.4f} "
          f"std {r.std():.4f} (expect 0, 0.173); theta500 cat/mine - 1: max |.| {np.max(np.abs(tr)):.1e}")
    for k in tabs:
        q = Y / sigY(k, th)
        out = []
        for qc in QC:
            p = ndtr(q - qc)
            out.append((p.sum(), np.sqrt((p * p).sum()), p[Mc >= 1e14].sum()))
        print(f"   {k:9s}: q>4..7 = " + " / ".join(f"{a:,.0f}+-{e:,.0f}" for a, e, _ in out) +
              "   pulls vs my HMF integral: " + " ".join(f"{(a - Ndet[k][j]) / e:+.2f}" for j, (a, e, _) in enumerate(out)) +
              f"   [M>1e14, q>5: {out[1][2]:,.0f}]")
    del z, Mc, Y, th, q, p
print(f"\ntotal {time.time()-t0:.0f}s")
