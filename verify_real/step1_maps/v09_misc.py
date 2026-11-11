"""v09: smaller claims checked with our own code / maps.

(1) SED: g(nu) at the six band centres and the tSZ null (claim 'sed': g = -1.963, -1.922, -1.572, -1.016, +0.116,
    +1.002; null 217.51 GHz).
(2) Compensation (claim 'compensation'): 1/N_y(l) at low l from OUR covariance with l_min = 0 (log-slope, value at
    l -> 0); DC response of the old white filter sigma^2 RATIO^2 / N_w at theta500 = 1, 4, 16' (claim 10.5, 69, 937
    arcmin^2) by direct grid sums of the build's white filter kernel on our 2D grid (k = 0 mode).
(3) tSZ power of the population (claim 'tsz_power': 1e12 l(l+1) C_l^yy / 2pi = 1.128 at l = 3000, D_3000 = 8.65 uK^2
    at 145 GHz, 7.62 at 150 GHz): measured from the painted halo-sky maps of the first NPS v05 patches (same seeds),
    pre-smoothing deconvolved, annulus 2850 < l < 3150.
"""
import os, sys, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import vlib as V                                    # noqa: E402
import numpy as np                                  # noqa: E402
from scipy.optimize import brentq                   # noqa: E402

log = V.Tee(os.path.join(V.HERE, "v09_misc.log"))
t0 = time.time()
NPS = int(os.environ.get("NPS", 60))
# (1) SED
g = V.sed_uK(V.FREQ) / (V.TCMB * 1e6)
nu0 = brentq(lambda nu: V.sed_uK(nu), 150, 300)
log("(1) tSZ SED g(nu) = Delta T / (T_CMB y) at " + ", ".join(f"{n:.0f}: {x:+.4f}" for n, x in zip(V.FREQ, g))
    + f"; null at {nu0:.2f} GHz")
log("    claim: -1.963, -1.922, -1.572, -1.016, +0.116, +1.002; null 217.51 GHz")

# (2) compensation with our covariance, l_min = 0
cmb = V.CMB()
for mode in ("baseline", "goal"):
    l = np.array([1.0, 2.0, 5.0, 10.0, 20.0, 50.0, 100.0])
    C = V.channel_cov(l, mode, cmb(l))
    a = np.stack([V.beam(l, f) * V.sed_uK(nu) for f, nu in zip(V.FWHM, V.FREQ)], axis=1)
    inv = np.einsum("li,li->l", a, np.linalg.solve(C, a[..., None])[..., 0])
    sl = np.log(inv[2] / inv[0]) / np.log(5.0)
    log(f"(2) [{mode}] OUR 1/N_y at l = {l.astype(int).tolist()}: {np.array2string(inv, precision=3)} sr^-1; "
        f"log-slope l=1..5: {sl:.2f}  (-> 0 as l -> 0: no response to a uniform y)")
G = V.Grid()
F = V.BuildMMF(G, "white")
for th in (1.0, 4.0, 16.0):
    s = F.sigma(th) * V.ARCMIN ** 2
    # kernel at l = 0: Y_hat per unit uniform y = sigma^2 g(0) inv(0) [sr]
    dc_grid = s * s * F.template(th)[0, 0] * F.inv2d[0, 0] / V.ARCMIN ** 2
    dc_formula = s * s * V.RATIO ** 2 / (4e-6 * V.ARCMIN) ** 2 / V.ARCMIN ** 2 / V.RATIO
    log(f"    old white filter, theta500 = {th:4.1f}': DC response (Y_hat per unit uniform y) = {dc_grid:8.2f} arcmin^2 "
        f"(sigma^2 RATIO / N_w: {dc_formula:8.2f}; x RATIO = response to a uniform y of the template normalisation: "
        f"{dc_grid*V.RATIO:8.2f})")
for mode in ("baseline", "goal"):
    Fm = V.BuildMMF(G, mode)
    log(f"    [{mode}] build MMF kernel at l = 0 on the grid: inv_Ny(0) = {Fm.inv2d[0,0]:.1e} -> DC response 0")
    del Fm

# (3) tSZ power from painted maps
P = V.Painter()
Gs = P.presmooth(G)
S = V.PatchSampler(G.L)
ann = (G.ell > 2850) & (G.ell < 3150)
ann2 = (G.ell > 950) & (G.ell < 1050)
vals, vals1 = [], []
for p in range(NPS):
    rng = np.random.default_rng([2026, 11, 11, p])
    h = S.draw(rng)
    keep = h["th"] <= 60.0
    m = P.paint(G, h["x"][keep], h["y"][keep], h["Y"][keep], h["th"][keep])
    yk = G.rfft(m) * G.omega / Gs                                # continuous FT [sr], pre-smoothing removed
    Pk = np.abs(yk) ** 2 / G.area                                # C_l estimate [sr]
    vals.append(np.average(Pk[ann], weights=G.mult[ann]))
    vals1.append(np.average(Pk[ann2], weights=G.mult[ann2]))
vals, vals1 = np.array(vals), np.array(vals1)
D = 1e12 * 3000 * 3001 / (2 * np.pi) * vals
D1 = 1e12 * 1000 * 1001 / (2 * np.pi) * vals1
gs = V.sed_uK(np.array([145.0, 150.0]))
log(f"(3) tSZ power of the painted halo skies ({NPS} patches = {NPS*100} deg^2): 1e12 l(l+1)C_l^yy/2pi at l=3000: "
    f"{D.mean():.3f} +- {D.std(ddof=1)/np.sqrt(NPS):.3f} (claim 1.128); at l=1000: {D1.mean():.3f} +- {D1.std(ddof=1)/np.sqrt(NPS):.3f}")
log(f"    D_3000 at 145 GHz = {D.mean()*1e-12*gs[0]**2:.2f} +- {D.std(ddof=1)/np.sqrt(NPS)*1e-12*gs[0]**2:.2f} uK^2 (claim 8.65); "
    f"at 150 GHz = {D.mean()*1e-12*gs[1]**2:.2f} uK^2 (claim 7.62); SPT (Reichardt+21) 3.42 +- 0.54 uK^2 at 143 GHz")
log(f"[v09] done {time.time()-t0:.0f}s")
