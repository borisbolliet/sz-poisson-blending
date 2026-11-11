"""v02: own Arnaud template + painting tables, each checked by two independent routes.

(1) Fourier template g(u) by the 3D projection-slice theorem (sinc transform of the truncated 3D GNFW; no projected
    profile, no J0) vs the BUILD's filter template szsim.signal.template_hankel (2D Hankel of the projected
    profile, linear interpolation on a 701-point log grid).  RATIO_5R500 from our own quadrature.
(2) Projected profile f(x) by line-of-sight quadrature: normalisation int 2 pi x f dx = RATIO, and vs the project's
    szsim.signal.projected_shape at a few radii.
(3) Painting table P_s(r; theta) = (f_theta conv Gaussian FWHM 0.75')(r), real-space I0-kernel route, for
    theta = 0.03'..300' (dln theta = 0.02, 1024 radii to 5 theta + 7 sigma_s); normalisation per node; and vs the
    Hankel route (1/2pi) int dl l g(l theta) G_s(l) J0(l r) with OUR g(u) at six nodes (independent routes).
(4) Painted single clusters on the 0.25' grid: sum of pixels x pix^2 = Y RATIO; DFT vs g(l theta) G_s(l).
"""
import os, sys, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import vlib as V                                    # noqa: E402
import numpy as np                                  # noqa: E402
from scipy.special import i0e, j0                    # noqa: E402

log = V.Tee(os.path.join(V.HERE, "v02_template_paint.log"))
t0 = time.time()
from szsim import signal as sg                      # noqa: E402

# ------------------------------------------------------------------ (1) template
log(f"(1) RATIO_5R500: ours {V.RATIO:.8f}   project sg.RATIO_5R500 {sg.RATIO_5R500:.8f}   rel {V.RATIO/sg.RATIO_5R500-1:+.1e}")
ub, gb = sg.template_hankel()
sel = ub <= 3000
gm = V.g_projslice(ub[sel])
d = np.abs(gm - gb[sel]) / gm[0]
for lo, hi in ((0, 1), (1, 10), (10, 100), (100, 3000)):
    s = (ub[sel] >= lo) & (ub[sel] <= hi)
    log(f"    u in [{lo:4d},{hi:4d}] (on the build's {s.sum()} table nodes): max |g_ours - g_build| / g(0) = {d[s].max():.2e}")
# between the build's log nodes the build interpolates linearly: test at mid points
um = np.sqrt(ub[1:-1] * ub[2:])
um = um[(um > 0.05) & (um < 300)]
gmm = V.g_projslice(um)
gbm = sg.template_hankel(um)
for lo, hi in ((0.05, 1), (1, 10), (10, 30), (30, 100), (100, 300)):
    s = (um >= lo) & (um < hi)
    log(f"    mid-nodes u in [{lo},{hi}): max |g_ours - g_build(interp)| / g(0) = {np.abs(gmm[s]-gbm[s]).max()/gm[0]:.2e}"
        f"   (|g_ours|/g(0) there up to {np.abs(gmm[s]).max()/gm[0]:.1e})")
log(f"    [{time.time()-t0:.0f}s]")

# ------------------------------------------------------------------ (2) projected profile
xs = np.unique(np.concatenate([[0.0], np.geomspace(1e-6, 0.05, 400),
                               V.XT * (1 - (1 - np.linspace(0, 1, 6001)) ** 2)]))
xs = xs[xs < V.XT]
xs = np.append(xs, V.XT)
fx = V.proj_profile(xs)
fx[-1] = 0.0
# normalisation: substitution-free trapezoid on the non-uniform grid + exact tail of the sqrt edge
I = np.trapezoid(2 * np.pi * xs * fx, xs)
log(f"(2) projected profile on {xs.size} nodes: int 2 pi x f dx = {I:.6f} vs RATIO {V.RATIO:.6f} (rel {I/V.RATIO-1:+.1e})")
xc = np.array([0.0, 0.01, 0.1, 0.5, 1.0, 2.0, 4.0, 4.9])
fp = sg.projected_shape(xc)
fo = V.proj_profile(xc)
log("    f_ours / f_project - 1 at x = " + ", ".join(f"{a}: {b/c-1:+.1e}" for a, b, c in zip(xc, fo, fp)))
log(f"    [{time.time()-t0:.0f}s]")

# ------------------------------------------------------------------ (3) painting table (I0 route)
zz = np.concatenate([np.linspace(0, 14.99, 300), np.linspace(15, 400, 300)])
ie = np.empty_like(zz)
# check the numba i0e used inside _smooth_profile against scipy, through a one-node smoothing of a delta-like ring
lnth0, dlnth = np.log(0.03), 0.02
nth = int(np.ceil(np.log(300 / 0.03) / dlnth)) + 1
th_nodes = np.exp(lnth0 + dlnth * np.arange(nth))
NR = 1024
rmax = V.rmax_paint(th_nodes)
tab = np.zeros((nth, NR))
for i, th in enumerate(th_nodes):
    r = np.linspace(0, rmax[i], NR)
    V._smooth_profile(xs, fx, th, V.SIG_S, r, tab[i])
log(f"(3) painting table: {nth} theta nodes {th_nodes[0]:.3f}-{th_nodes[-1]:.1f}', NR = {NR}, sigma_s = {V.SIG_S:.4f}' "
    f"[{time.time()-t0:.0f}s]")
norm = np.array([np.trapezoid(2 * np.pi * np.linspace(0, rmax[i], NR) * tab[i], np.linspace(0, rmax[i], NR))
                 for i in range(nth)]) / V.RATIO - 1
log(f"    normalisation int 2 pi r P_s dr / RATIO - 1: max |.| = {np.abs(norm).max():.1e} "
    f"(at theta = {th_nodes[np.argmax(np.abs(norm))]:.3f}'); median {np.median(np.abs(norm)):.1e}")
# numba i0e check
rr = np.array([0.0, 0.05, 0.3, 1.0, 3.0])
for th in (0.05, 1.0):
    s_ = np.linspace(0, 5 * th, 20001)
    f_ = np.interp(s_ / th, xs, fx) / th ** 2
    ref = [np.trapezoid(s_ / V.SIG_S ** 2 * f_ * np.exp(-0.5 * (r_ - s_) ** 2 / V.SIG_S ** 2) * i0e(r_ * s_ / V.SIG_S ** 2), s_)
           for r_ in rr]
    out = np.empty(rr.size)
    V._smooth_profile(xs, fx, th, V.SIG_S, rr, out)
    ref = np.array(ref)
    log(f"    numba (own i0e) vs numpy/scipy i0e at theta={th}', r = {rr}': P = {np.array2string(out, precision=3)}; "
        f"|diff|/P(0) = {np.array2string(np.abs(out-ref)/ref[0], precision=1)}")

# Hankel route with our projection-slice g(u)
uf = np.concatenate([np.linspace(0, 2, 401)[:-1], np.arange(2.0, 1600.0, 0.02)])
log(f"    tabulating our g(u) on {uf.size} nodes for the Hankel cross-check ... [{time.time()-t0:.0f}s]")
gf = V.g_projslice(uf)
log(f"    done [{time.time()-t0:.0f}s]")
worst = 0.0
for th in (0.05, 0.3, 1.0, 4.0, 16.0, 60.0):
    i = int(np.argmin(np.abs(th_nodes - th)))
    th = th_nodes[i]
    r = np.linspace(0, rmax[i], NR)
    lmx = min(9.0 / V.SIG_S, uf[-1] / th)            # arcmin^-1
    dl = min(0.25 / rmax[i], lmx / 4000)
    l = np.arange(0, lmx + dl, dl)
    w = np.full(l.size, dl); w[0] *= 0.5; w[-1] *= 0.5
    gl = np.interp(l * th, uf, gf)
    kern = w * l * gl * np.exp(-0.5 * (l * V.SIG_S) ** 2) / (2 * np.pi)
    PH = np.array([np.sum(kern * j0(l * rk)) for rk in r[::8]])
    PI = tab[i, ::8]
    e = np.abs(PH - PI).max() / PI[0]
    worst = max(worst, e)
    log(f"    theta={th:7.3f}': max |P_I0 - P_Hankel| / P(0) = {e:.1e}  (P(0) = {PI[0]:.4e} arcmin^-2)")
log(f"    worst I0 vs Hankel: {worst:.1e}   [{time.time()-t0:.0f}s]")
np.savez(os.path.join(V.OUT, "paint_tables.npz"), tab=tab, rmax=rmax, lnth0=lnth0, dlnth=dlnth, sig_s=V.SIG_S,
         th_nodes=th_nodes, xs=xs, fx=fx, u_fine=uf, g_fine=gf)

# ------------------------------------------------------------------ (4) painted clusters on the grid
G = V.Grid()
P = V.Painter()
rng = np.random.default_rng(7)
Gs = P.presmooth(G)
for th in (0.3, 0.5, 1.0, 5.0, 20.0, 50.0):
    for pos in ("centre", "random"):
        x0, y0 = (300.125, 300.125) if pos == "centre" else tuple(rng.random(2) * G.L)
        m = P.paint(G, np.array([x0]), np.array([y0]), np.array([1.0]), np.array([th]))
        tot = m.sum() * G.pix ** 2
        # DFT vs analytic g(l theta) G_s(l) (with the position phase)
        mk = G.rfft(m) * G.pix ** 2                                         # ~ continuous FT [arcmin^2 per Y]
        # pixel j is centred at (j + 1/2) pix: DFT x pix^2 = FT x exp(+i l pix/2) per axis
        ph = np.exp(-1j * (G.ly[:, None] * (y0 - 0.5 * G.pix) + G.lx[None, :] * (x0 - 0.5 * G.pix)) * V.ARCMIN)
        ana = np.interp(G.ell * th * V.ARCMIN, uf, gf, right=0.0) * Gs * ph
        band = G.ell <= 20000
        err = np.abs(mk - ana)[band].max() / V.RATIO
        log(f"(4) theta={th:5.1f}' {pos:6s}: sum(map) pix^2 / RATIO - 1 = {tot/V.RATIO-1:+.1e};  "
            f"max |DFT - g(l theta) G_s(l)| / RATIO over l <= 20000: {err:.1e}")
log(f"[v02] done {time.time()-t0:.0f}s")
