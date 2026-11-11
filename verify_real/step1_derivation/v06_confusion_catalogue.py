"""v06: Campbell confusion variance and mean by DIRECT SUMMATION over a freshly sampled Poisson sky
(szsim.sampler seed 20261111, M500 > 1e13, ~2.93e7 halos), with my own filter quantities:
   var_conf(A) / sigma_A^2 = (1/A_sky) sum_B Y_B^2 I_A(theta_B) / sigma_A^2,   I_A(theta_B) = int d^2s rho_AB^2,
   mu(A) / sigma_A = (1/A_sky) sum_B Y_B DC_A / sigma_A,  DC_A = int d^2s rho_AB = sigma_A^2 RATIO^2 / N_y(l=0),
I_A from my 2D l-grid (dl = 10, MMF; dl = 20 l <= 40000 incl. l = 0 cell, white); per-halo cubic-spline lookup.
No HMF integral, no neighbour_population, no Hankel.  Also: tSZ Poisson power D_3000 from the same sky, and the
'tSZ-aware' filter (N_y -> N_y + C_l^yy) sigma_Y ratio.  Compared with data_real/confusion_var.npz (data only).
"""
import os, sys, time
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import vlib as V
V.set_threads(4)
import numpy as np
from scipy.interpolate import CubicSpline

t0 = time.time()
AM = V.AM
A_SKY = 4 * np.pi / AM ** 2                     # arcmin^2
BV = np.load(os.path.join(V.ROOT, "data_real", "confusion_var.npz"))
THA = BV["thA"]
SW = np.load(os.path.join(HERE, "sigmaY_mine_wide.npz"))
SPs = {k: CubicSpline(np.log(SW["theta500"]), np.log(SW[k])) for k in ("baseline", "goal", "white")}
sigY = lambda k, th: np.exp(SPs[k](np.log(np.clip(th, SW["theta500"][0], SW["theta500"][-1]))))
THB = np.geomspace(0.01, 700.0, 220)

# 2D half-plane grids with identical |l| terms grouped (exactly the 2D grid sum; see vlib.LGridRadial)
grids = {"mmf": V.LGridRadial(dl=10.0), "white": V.LGridRadial(dl=20.0, lmin=0.0, lmax=40000.0)}
inv = {"baseline": V.inv_Ny(grids["mmf"].l, "baseline"), "goal": V.inv_Ny(grids["mmf"].l, "goal"),
       "white": V.inv_Ny_white(grids["white"].l)}
print(f"2D grids: mmf {grids['mmf'].nmodes:,} modes ({len(grids['mmf'].l):,} distinct |l|); white {grids['white'].nmodes:,} "
      f"({len(grids['white'].l):,})")
gname = {"baseline": "mmf", "goal": "mmf", "white": "white"}


def I_table(mode, inv_=None):
    """I_A(theta_B) [arcmin^2] and sigma_A [arcmin^2] on the THA x THB nodes."""
    G = grids[gname[mode]]; iv = inv[mode] if inv_ is None else inv_
    I = np.empty((len(THA), len(THB))); sA = np.empty(len(THA))
    base = np.empty((len(THA), len(G.l)))
    for i, ta in enumerate(THA):
        gA2 = V.g_u(G.l * ta * AM) ** 2
        s2 = 1.0 / G.sum(gA2 * iv)                                  # sr^2
        sA[i] = np.sqrt(s2) / AM ** 2
        base[i] = s2 * s2 * G.norm * G.m * gA2 * iv * iv / AM ** 2
    for j, tb in enumerate(THB):
        I[:, j] = base @ (V.g_u(G.l * tb * AM) ** 2)
    return I, sA


from szsim import sampler
cat = sampler.draw_catalog(20261111, fields=("M500", "Y500", "theta500"))
Y = cat["Y500"].astype(np.float64); th = cat["theta500"].astype(np.float64); M = cat["M500"]
cls = np.searchsorted(np.array([1e13, 3e13, 1e14, 1e17]), M, side="right") - 1
del cat, M
lth = np.log(th)
Y2 = Y * Y
print(f"sky seed 20261111: N = {len(Y):,}; theta500 range {th.min():.3f}-{th.max():.1f}'; "
      f"N(theta>100') = {(th > 100).sum()}, N(theta>700', clamped) = {(th > 700).sum()}; [{time.time()-t0:.0f}s]")
ybar = Y.sum() / A_SKY
print(f"ybar (sum Y500 / A_sky) = {ybar:.4e} (builder: 3.5895e-07); mean y incl. 5R500 = {ybar*V.RATIO:.3e}")

out = {}
show = [int(np.argmin(np.abs(THA - s))) for s in (0.52, 0.95, 1.98, 4.15, 7.5, 15.72, 32.94)]
for mode in ("white", "baseline", "goal"):
    ts = time.time()
    I, sA = I_table(mode)
    qB = Y / sigY(mode, th)
    sub = qB < 5.0
    res = {k: np.empty(len(THA)) for k in ("all", "c0", "c1", "c2", "subthr")}
    for i in range(len(THA)):
        sp = CubicSpline(np.log(THB), np.log(I[i]))
        Ib = np.exp(sp(np.clip(lth, np.log(THB[0]), np.log(THB[-1]))))
        Iw = Y2 * Ib
        tot = Iw.sum()
        res["all"][i] = tot / A_SKY / sA[i] ** 2
        for c in range(3):
            res[f"c{c}"][i] = Iw[cls == c].sum() / A_SKY / sA[i] ** 2
        res["subthr"][i] = Iw[sub].sum() / A_SKY / sA[i] ** 2
    if mode == "white":
        DC = sA ** 2 * V.RATIO ** 2 / (4e-6) ** 2                     # arcmin^2 (N_w = (4e-6 arcmin)^2)
        mu = ybar * DC / sA
    else:
        DC = np.zeros(len(THA)); mu = np.zeros(len(THA))
        # explicit: the l = 0 cell has 1/N_y(0) = 0; the response at the lowest non-zero grid mode l = 10:
        l10 = np.array([10.0]); iv10 = V.inv_Ny(l10, mode)[0]
        near = sA ** 2 * AM ** 4 * V.RATIO * V.g_u(10 * THA * AM) * iv10 / AM ** 2
        print(f"[{mode}] 1/N_y(0) = {V.inv_Ny(np.array([0.0]), mode)[0]}; response to a l=10 mode (per unit y amplitude) at "
              f"thA = 1, 4, 16': {np.interp([1, 4, 16], THA, near)} arcmin^2 (cf. white DC 10.5-937)")
    out[mode] = dict(res, mu=mu, sA=sA, I=I)
    b = BV
    print(f"\n=== {mode}: var_conf / sigma_A^2 (catalogue sum) vs builder (Parseval + HMF population) [{time.time()-ts:.0f}s] ===")
    print("   theta_A [']   " + " ".join(f"{THA[i]:8.2f}" for i in show))
    for k in ("all", "c0", "c1", "c2", "subthr"):
        bk = f"{mode}_var_{k}"
        print(f"   {k:7s} mine  " + " ".join(f"{res[k][i]:8.4g}" for i in show))
        print(f"   {k:7s} bldr  " + " ".join(f"{b[bk][i]:8.4g}" for i in show) +
              f"   max|mine/bldr-1| (40 nodes) = {np.max(np.abs(res[k]/b[bk]-1)):.3f}")
    print(f"   mu/sigma mine " + " ".join(f"{mu[i]:8.4g}" for i in show))
    print(f"   mu/sigma bldr " + " ".join(f"{b[mode+'_mu'][i]:8.4g}" for i in show))
    print(f"   sigma_A mine/bldr-1 (40 nodes): max {np.max(np.abs(sA/b[mode+'_sigA']-1)):.1e}")
    frac14 = res["c2"] / res["all"]
    print(f"   fraction of var from M>1e14: " + " ".join(f"{frac14[i]:.3f}" for i in show))
    if mode == "white":
        print(f"   DC response at 1/4/16': {np.interp([1, 4, 16], THA, DC)} arcmin^2")

print("\n=== tSZ Poisson power of the sampled population (direct sum, g from 2D FFT) ===")
for l in (3000.0,):
    g2 = V.g_u(l * th * AM) ** 2
    Cl = np.sum((Y * AM ** 2) ** 2 * g2) / (4 * np.pi)                 # sr
    Dy = l * (l + 1) * Cl / (2 * np.pi)
    for nu in (145.0, 150.0):
        print(f"   l={l:.0f}: 1e12 l(l+1)C/2pi = {1e12*Dy:.4f}; D_l at {nu:.0f} GHz = {Dy*V.f_uK(nu)**2:.3f} uK^2")
    for mode in ("white", "baseline", "goal"):
        sub = (Y / sigY(mode, th)) < 5
        print(f"   [{mode}] sub-threshold share of the Y^2 power at l=3000: {np.sum((Y2*g2)[sub])/np.sum(Y2*g2):.3f}")
print("   builder: 1.128; 8.65 uK^2 (145 GHz), 7.62 (150 GHz); shares 0.310 / 0.318 / 0.260; SPT 3.42 +- 0.54 uK^2")

print("\n=== tSZ-aware filter: N_y -> N_y + C_l^yy (catalogue C_l, binned in ln theta) ===")
lg = np.arange(0.0, 20001.0, 5.0)
edges = np.linspace(lth.min() - 1e-9, lth.max() + 1e-9, 2001)
idx = np.digitize(lth, edges) - 1
for mode in ("baseline", "goal"):
    G = grids["mmf"]
    for tag, selmask in (("subthr", (Y / sigY(mode, th)) < 5), ("all", np.ones(len(Y), bool))):
        S2 = np.bincount(idx[selmask], weights=Y2[selmask], minlength=len(edges) - 1)
        S2t = np.bincount(idx[selmask], weights=(Y2 * lth)[selmask], minlength=len(edges) - 1)
        nz = S2 > 0; thc = np.exp(S2t[nz] / S2[nz])
        Cyy = np.zeros(len(lg))
        for a in range(0, nz.sum(), 100):
            Cyy += (S2[nz][a:a + 100, None] * V.g_u(np.outer(thc[a:a + 100] * AM, lg)) ** 2).sum(0)
        Cyy *= AM ** 4 / (4 * np.pi)                                        # sr
        with np.errstate(divide="ignore"):
            ivp = np.where(inv[mode] > 0, 1.0 / (1.0 / inv[mode] + np.interp(G.l, lg, Cyy)), 0.0)
        ratio = []
        for i in show[:5]:
            gA2 = V.g_u(G.l * THA[i] * AM) ** 2
            ratio.append(np.sqrt(G.sum(gA2 * inv[mode]) / G.sum(gA2 * ivp)))
        v = out[mode]["subthr" if tag == "subthr" else "all"]
        print(f"   [{mode}] filter knows C^yy({tag}): sigma'/sigma at thA = " + " ".join(f"{THA[i]:.2f}" for i in show[:5]) +
              ": " + " ".join(f"{r:.3f}" for r in ratio) + "   sqrt(1+v): " + " ".join(f"{np.sqrt(1+v[i]):.3f}" for i in show[:5]))
np.savez(os.path.join(HERE, "confusion_mine.npz"), thA=THA, thB=THB,
         **{f"{m}_{k}": out[m][k] for m in out for k in ("all", "c0", "c1", "c2", "subthr", "mu", "sA", "I")})
print(f"\ntotal {time.time()-t0:.0f}s")
