"""v03: empirical filtered-noise std of the BUILD's MMF on our own simulated SO LAT skies (no clusters).

Patch 2400^2 x 0.25' (10 x 10 deg, periodic; fundamental l = 36, axis Nyquist l = 43200).  Each realisation:
our CAMB CMB (beam-convolved per channel), pixel-space white noise, atmosphere (common mode + independent per tube,
r = 0.9).  Baseline and goal share the CMB, atmosphere and unit white fields (white levels differ).
The build's filter (szsim.instrument: inv_Ny(l), ILC weights w_i(l), sigma_Y(theta) by its Hankel integral) is
applied to the six channel maps; Y_hat map variance over all pixels -> sigma_emp(theta500).

Two methods for the expected value:
  (M1) Monte Carlo maps (this script);
  (M2) semi-analytic: the build's filter applied to OUR channel covariance C_ours(l) (independently coded noise,
       our CAMB CMB), summed over the grid modes:  Var = sigma^4 sum_l g^2 inv^2 (w^T C_ours w) / A.
and the claim: sigma_Y(theta500) at 0.5, 1, 2, 5, 10, 20' (claim 'sigmaY_mmf') to < 3%.
Control: the old single-frequency white filter on a white y-noise map (4e-6 arcmin, 1.4' beam).
"""
import os, sys, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import vlib as V                                    # noqa: E402
import numpy as np                                  # noqa: E402

log = V.Tee(os.path.join(V.HERE, "v03_noise_sigmaY.log"))
NSIM = int(os.environ.get("NSIM", 48))
THS = [0.5, 1.0, 2.0, 5.0, 10.0, 20.0]
CLAIM = {"baseline": [4.51e-6, 6.21e-6, 1.09e-5, 3.85e-5, 1.43e-4, 6.55e-4],
         "goal": [3.36e-6, 4.72e-6, 8.59e-6, 3.19e-5, 1.24e-4, 5.94e-4],
         "white": [5.63e-6, 7.23e-6, 1.08e-5, 2.26e-5, 4.31e-5, 8.47e-5]}
t0 = time.time()
G = V.Grid()
cmb = V.CMB()
sim = V.SkySim(G, cmb)
F = {m: V.BuildMMF(G, m) for m in ("baseline", "goal", "white")}
log(f"[v03] grid {G.n}^2 x {G.pix}' ; NSIM = {NSIM}; set-up {time.time()-t0:.0f}s")

# ------------------------------------------------------------ (M2) semi-analytic on our covariance
l1 = np.arange(100.0, 20001.0)
log("(M2a) N_y: build inv_Ny(l) vs 1/(a^T C_ours^-1 a) from our independently coded covariance and SED/beams")
for mode in ("baseline", "goal"):
    C = V.channel_cov(l1, mode, cmb(l1))
    a = np.stack([V.beam(l1, f) * V.sed_uK(nu) for f, nu in zip(V.FWHM, V.FREQ)], axis=1)        # (nl, 6)
    x = np.linalg.solve(C, a[..., None])[..., 0]
    inv_ours = np.einsum("li,li->l", a, x)
    inv_b = np.interp(l1, F[mode].inst.ell, F[mode].inst.inv_Ny)
    wb = np.stack([np.interp(l1, F[mode].inst.ell, F[mode].inst.weights[:, i]) for i in range(6)], axis=1)
    peff = np.einsum("li,lij,lj->l", wb, C, wb)        # variance of the build's ILC combination under OUR covariance
    resp = np.einsum("li,li->l", wb, a)               # build weights x our a: unit response?
    r = inv_ours / inv_b - 1
    log(f"   [{mode}] max|inv_ours/inv_build - 1| = {np.abs(r).max():.2e} (at l={l1[np.argmax(np.abs(r))]:.0f}); "
        f"max|w_build^T C_ours w_build x inv_build - 1| = {np.abs(peff*inv_b-1).max():.2e}; "
        f"max|sum_i w_i a_i(ours) - 1| = {np.abs(resp-1).max():.2e}")
    for lv in (100, 300, 1000, 3000, 10000, 20000):
        k = int(lv - 100)
        log(f"      l={lv:6d}: N_y(ours) = {1/inv_ours[k]:.3e} sr   build {1/inv_b[k]:.3e}")

log("(M2b) predicted std of the build's filter on OUR covariance, summed over the grid modes, vs the build's sigma_Y")
pred = {}
for mode in ("baseline", "goal"):
    f = F[mode]
    L = G.ell
    sel = f.inv2d > 0
    Ls = L[sel]
    Cs = V.channel_cov(Ls, mode, cmb(Ls))                                        # (nsel, 6, 6)
    ws = f.w2d[:, sel].T                                                         # (nsel, 6)
    Peff = np.zeros(L.shape)
    Peff[sel] = np.einsum("li,lij,lj->l", ws, Cs, ws)
    del Cs
    pred[mode] = []
    for th in THS:
        s4 = (f.sigma(th) * V.ARCMIN ** 2) ** 4
        gt = f.template(th)
        var = s4 * G.modesum(gt * gt * f.inv2d ** 2 * Peff)
        pred[mode].append(np.sqrt(var) / V.ARCMIN ** 2)
    log(f"   [{mode}] theta500: " + "  ".join(f"{th:5.1f}'" for th in THS))
    log(f"   [{mode}] sigma_Y build (Hankel)  : " + " ".join(f"{f.sigma(th):.4e}" for th in THS))
    log(f"   [{mode}] sigma_Y claimed (rounded): " + " ".join(f"{v:.4e}" for v in CLAIM[mode]))
    log(f"   [{mode}] M2 grid prediction      : " + " ".join(f"{v:.4e}" for v in pred[mode]))
    log(f"   [{mode}] M2 / build - 1          : " + " ".join(f"{p/f.sigma(th)-1:+.2e}" for p, th in zip(pred[mode], THS)))
    log(f"   [{mode}] grid-normalisation sigma_disc / sigma_Hankel - 1: "
        + " ".join(f"{f.sigma_disc(th)/f.sigma(th)-1:+.1e}" for th in THS))

# ------------------------------------------------------------ (M1) Monte Carlo maps
rng = np.random.default_rng(20261111)
var = {m: np.zeros((NSIM, len(THS))) for m in ("baseline", "goal", "white")}
mean = {m: np.zeros((NSIM, len(THS))) for m in ("baseline", "goal", "white")}
kurt = {m: np.zeros((NSIM, len(THS))) for m in ("baseline", "goal")}
Nw_y = (4e-6 * V.ARCMIN) ** 2
Bw = V.beam(G.ell, 1.4)
for k in range(NSIM):
    X = sim.realise(rng)
    for mode in ("baseline", "goal"):
        Z = F[mode].ycomb(X[mode])
        for j, th in enumerate(THS):
            Yh = F[mode].filt(Z, th)
            mean[mode][k, j] = Yh.mean()
            v = Yh.var()
            var[mode][k, j] = v
            kurt[mode][k, j] = np.mean((Yh - Yh.mean()) ** 4) / v ** 2 - 3
    del X
    # control: single-frequency white y map, 4e-6 arcmin, 1.4' beam (old setup); data = n only
    n = G.rfft(rng.standard_normal((G.n, G.n)) * (4e-6 / G.pix))
    Z = Bw / Nw_y * n                                         # inv_Ny * w * D with w = 1/B, inv_Ny = B^2/N_w
    for j, th in enumerate(THS):
        Yh = F["white"].filt(Z, th)
        mean["white"][k, j] = Yh.mean()
        var["white"][k, j] = Yh.var()
    if k % 8 == 0 or k == NSIM - 1:
        log(f"   realisation {k+1}/{NSIM} [{time.time()-t0:.0f}s]")

log(f"\n(M1) Monte Carlo: {NSIM} patches of 100 deg^2; sigma_emp = sqrt(mean over patches of the pixel variance);"
    f" error from the patch-to-patch scatter")
res = {}
for mode in ("baseline", "goal", "white"):
    f = F[mode]
    v = var[mode]
    se = np.sqrt(v.mean(0))
    err = 0.5 * v.std(0, ddof=1) / np.sqrt(NSIM) / v.mean(0)          # relative error on sigma
    log(f"   [{mode}]")
    for j, th in enumerate(THS):
        sb = f.sigma(th)
        line = (f"      theta500={th:5.1f}': sigma_emp = {se[j]:.4e} +- {err[j]*100:.2f}%   build sigma_Y = {sb:.4e}"
                f"   emp/build - 1 = {se[j]/sb-1:+.4f} ({(se[j]/sb-1)/err[j]:+.1f} sig)   emp/claim - 1 = {se[j]/CLAIM[mode][j]-1:+.4f}")
        if mode != "white":
            line += f"   emp/M2 - 1 = {se[j]/pred[mode][j]-1:+.4f}   excess kurtosis {kurt[mode][:, j].mean():+.3f}"
        log(line)
    res[mode] = dict(se=se, err=err)
worst = max(abs(res[m]["se"][j] / CLAIM[m][j] - 1) for m in ("baseline", "goal") for j in range(len(THS)))
log(f"\n   worst |sigma_emp / sigma_claimed - 1| over baseline+goal and theta500 = 0.5-20': {worst:.4f} (criterion < 0.03)")
np.savez(os.path.join(V.OUT, "v03_noise.npz"), ths=THS, **{f"var_{m}": var[m] for m in var},
         **{f"pred_{m}": np.array(pred[m]) for m in pred})
log(f"[v03] done {time.time()-t0:.0f}s")
