"""v04: isolated Arnaud clusters painted in REAL space (own LOS profile, own beams and SED) through the build's MMF.

(a) Noiseless response: one cluster (Y500 = 1) at a pixel centre, theta500 = 0.5..20'; channel maps
    D_i = B_i f_i (y conv) built explicitly; Y_hat read at the centre.  Expected 1 in the continuum; the grid
    normalisation predicts sigma_Hankel^2 / sigma_disc^2.
(b) Cross-response: cluster B painted at separation s (multiples of the pixel) from the filter centre A;
    Y_hat_A / Y_B vs the build's MMFModel.cross_response (Hankel) — the kernel of the build's P(D).
(c) Noisy recovery: grids of q = 10 clusters (spacing max(30', 10 theta500)) in our CMB + atmosphere + white noise
    skies; mean and std of (Y_hat - Y)/sigma_Y over all clusters (claim 'mmf_mc': unbiased, unit scatter).
"""
import os, sys, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import vlib as V                                    # noqa: E402
import numpy as np                                  # noqa: E402

log = V.Tee(os.path.join(V.HERE, "v04_injection.log"))
NSIM = int(os.environ.get("NSIM", 24))
THS = [0.5, 1.0, 2.0, 5.0, 10.0, 20.0]
t0 = time.time()
G = V.Grid()
P = V.Painter()
Gs = P.presmooth(G)
cmb = V.CMB()
sim = V.SkySim(G, cmb)
MODES = ("baseline", "goal", "white")
F = {m: V.BuildMMF(G, m) for m in MODES}
A = np.stack([V.beam(G.ell, V.FWHM[i]) * V.sed_uK(V.FREQ[i]) / Gs for i in range(6)])   # (6, ...) uK per unit y
Bw = V.beam(G.ell, 1.4)
log(f"[v04] set-up {time.time()-t0:.0f}s")


def channels_from_y(yk):
    """Six channel rffts [uK] of a painted (pre-smoothed) y map rfft."""
    return A * yk[None]


def yhat_map(mode, yk, th, noise=None):
    """Filtered Y_hat map [arcmin^2] of the pure-y sky yk (+ optional channel noise rffts)."""
    f = F[mode]
    if mode == "white":
        D = Bw / Gs * yk + (0 if noise is None else noise)             # single y map with the 1.4' beam
        Z = Bw / (4e-6 * V.ARCMIN) ** 2 * D                            # inv_Ny * w * D, w = 1/B
    else:
        D = channels_from_y(yk) + (0 if noise is None else noise)
        Z = f.ycomb(D)
    return f.filt(Z, th)


# ------------------------------------------------------------------ (a) noiseless response
c0 = 300.125                                    # pixel centre (pixel 1200)
ic = 1200
log("(a) noiseless response Y_hat/Y of a real-space painted cluster at a pixel centre")
for mode in MODES:
    f = F[mode]
    row, rowp = [], []
    for th in THS:
        m = P.paint(G, np.array([c0]), np.array([c0]), np.array([1.0]), np.array([th]))
        Yh = yhat_map(mode, G.rfft(m), th)
        row.append(Yh[ic, ic])
        rowp.append((f.sigma(th) / f.sigma_disc(th)) ** 2)
    log(f"   [{mode:8s}] theta500 = " + "  ".join(f"{t:6.1f}'" for t in THS))
    log(f"   [{mode:8s}] R - 1    = " + "  ".join(f"{r-1:+.1e}" for r in row))
    log(f"   [{mode:8s}] grid-normalisation prediction sigma_H^2/sigma_disc^2 - 1 = " + "  ".join(f"{r-1:+.1e}" for r in rowp))
    log(f"   [{mode:8s}] R / prediction - 1 = " + "  ".join(f"{r/p-1:+.1e}" for r, p in zip(row, rowp)))

# ------------------------------------------------------------------ (b) cross-response vs the build's Hankel rho
log("(b) cross-response: cluster B painted at separation s from the filter centre A, vs MMFModel.cross_response")
seps = np.array([0.0, 0.5, 1.0, 2.5, 5.0, 10.0, 20.0, 40.0, 80.0])
pairs = [(1.0, 1.0), (1.0, 0.3), (2.0, 4.0), (4.0, 2.0), (8.0, 8.0), (2.0, 16.0), (16.0, 2.0), (0.5, 40.0)]
worst = {}
for mode in ("baseline", "goal", "white"):
    f = F[mode]
    worst[mode] = 0.0
    for thA, thB in pairs:
        m = np.zeros((G.n, G.n))
        # B at (c0 + s, c0) for each s: separate maps (linear; avoids cross-talk between the B copies)
        out = []
        for s in seps:
            m[:] = 0.0
            P.paint(G, np.array([c0 + s]), np.array([c0]), np.array([1.0]), np.array([thB]), m=m)
            out.append(yhat_map(mode, G.rfft(m), thA)[ic, ic])
        out = np.array(out)
        lib = f.model.cross_response(thA, thB, seps)
        sc = max(1.0, np.abs(lib).max())
        d = np.abs(out - lib).max() / sc
        worst[mode] = max(worst[mode], d)
        log(f"   [{mode:8s}] thA={thA:5.1f} thB={thB:5.1f}: map {np.array2string(out, precision=4, max_line_width=200)}")
        log(f"   {'':10s}                  build {np.array2string(lib, precision=4, max_line_width=200)}  max|diff|/max(1,|rho|) = {d:.1e}")
    log(f"   [{mode}] worst |map - build| / max(1, |rho|max) = {worst[mode]:.1e}")
log(f"   [{time.time()-t0:.0f}s]")

# ------------------------------------------------------------------ (c) noisy recovery
log(f"(c) q = 10 clusters on grids in CMB + atmosphere + white noise skies, {NSIM} realisations")
grids = {}
for th in THS:
    S = max(30.0, 10 * th)
    k = int(round(G.L / S))
    xs = (np.arange(k) * (G.L / k) // G.pix + 0.5) * G.pix + 0.0          # pixel centres
    X, Y = np.meshgrid(xs, xs)
    m = P.paint(G, X.ravel(), Y.ravel(), np.ones(X.size), np.full(X.size, th))
    ij = (np.round(Y.ravel() / G.pix - 0.5).astype(int), np.round(X.ravel() / G.pix - 0.5).astype(int))
    grids[th] = (G.rfft(m), ij, X.size)
rng = np.random.default_rng(424242)
rec = {m: {th: [] for th in THS} for m in MODES}
for r in range(NSIM):
    Xn = sim.realise(rng)
    nw = G.rfft(rng.standard_normal((G.n, G.n)) * (4e-6 / G.pix))          # white control: y noise map
    for mode in MODES:
        noise = nw if mode == "white" else Xn[mode]
        for th in THS:
            yk, ij, nc = grids[th]
            sig = F[mode].sigma(th)
            Yinj = 10.0 * sig
            Yh = yhat_map(mode, yk * Yinj, th, noise=noise)
            rec[mode][th].append((Yh[ij] - Yinj) / sig)
    if r % 6 == 0 or r == NSIM - 1:
        log(f"   realisation {r+1}/{NSIM} [{time.time()-t0:.0f}s]")
summary = {}
for mode in MODES:
    log(f"   [{mode}]")
    for th in THS:
        v = np.concatenate(rec[mode][th])
        # errors: realisation-level scatter of the per-realisation means (accounts for correlations within a patch)
        per = np.array([x.mean() for x in rec[mode][th]])
        e_mean = per.std(ddof=1) / np.sqrt(NSIM)
        sd = v.std(ddof=1)
        e_sd = np.array([x.std() for x in rec[mode][th]]).std(ddof=1) / np.sqrt(NSIM) if grids[th][2] > 4 else sd / np.sqrt(2 * v.size)
        e_sd = max(e_sd, sd / np.sqrt(2 * v.size))
        resp = 1 + v.mean() / 10.0
        log(f"      theta500={th:5.1f}': N = {v.size:6d}  mean (Yhat-Y)/sigma = {v.mean():+.4f} +- {e_mean:.4f} ({v.mean()/e_mean:+.1f} sig)"
            f"   -> response {resp:.5f}   std = {sd:.4f} +- {e_sd:.4f} ({(sd-1)/e_sd:+.1f} sig)")
        summary[f"{mode}_{th}"] = (v.mean(), e_mean, sd, e_sd)
np.savez(os.path.join(V.OUT, "v04_injection.npz"), **{k: np.array(v) for k, v in summary.items()})
log(f"[v04] done {time.time()-t0:.0f}s")
