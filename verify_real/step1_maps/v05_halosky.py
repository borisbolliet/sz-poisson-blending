"""v05: Poisson halo skies (M500 > 1e13) painted on 10 x 10 deg periodic patches and filtered with the build's MMF.

For every patch (own sampler, own painter, own beams/SED; build filter):
  * delta(x) = Y_hat_A(x) / sigma_A of the halo-only sky at every pixel, for 12 filter sizes theta_A (every third
    primary node of the build's s06b grid, incl. 1.007', 4.025', 16.09'), for
      baseline / goal MMF (all neighbours), baseline_sub / goal_sub (only neighbours with Y_B/sigma_Y(theta_B) < 5,
      the build's s06c definition), and the old white filter (control; ensemble-mean y removed = c=1 convention).
    By Slivnyak's theorem the distribution of delta over random points is the distribution of the neighbour sum of
    any halo (= the build's untruncated P(D)).  Hybrid histograms (0.002 bins in [-20, 20], log bins beyond) are
    accumulated per block of patches.
  * explicit injection (first NINJ patches): a 20 x 20 grid (30' spacing, random offset) of clusters
    (theta_A, q_A) in {1.007', 4.025'} x {5, 7}, painted like the halos, added to the halo sky AND to a
    CMB + atmosphere + white noise realisation (baseline and goal), filtered; boost = (Y_hat(sky+inj+noise) -
    Y_hat(inj+noise)) / Y_A, detection Y_hat/sigma_A > 5.  Also the halo-only delta at the same pixels (linearity).
Halos with theta500 > 60' (5 theta500 > half the box) are not painted (counted).
"""
import os, sys, time, resource
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import vlib as V                                    # noqa: E402
import numpy as np                                  # noqa: E402

NPATCH = int(os.environ.get("NPATCH", 200))
BLOCK = int(os.environ.get("BLOCK", 10))
NINJ = int(os.environ.get("NINJ", 48))
P0 = int(os.environ.get("P0", 0))                   # first patch index (restart)
log = V.Tee(os.path.join(V.HERE, f"v05_halosky{os.environ.get('TAG', '')}.log"))
t0 = time.time()
G = V.Grid()
P = V.Painter()
Gs = P.presmooth(G)
S = V.PatchSampler(G.L)
th_tab = np.load(f"{V.ROOT}/data_real/rho_mmf_baseline.npz")["th"]
TA_all = th_tab[::2]
NODES = TA_all[6:40:3]
MODES = ("baseline", "goal", "white")
RUNS = ("baseline", "goal", "white", "baseline_sub", "goal_sub")
F = {m: V.BuildMMF(G, m) for m in MODES}
SIG = {m: np.array([F[m].sigma(t) for t in NODES]) for m in MODES}
T = {m: F[m].yresp_transfer(extra=Gs) for m in MODES}
from szsim import signal as sg                      # noqa: E402
# filter kernels K[m][j] (sr^2 g T): Y_hat[sr] = irfft(K * yk)
K = {m: [] for m in MODES}
for j, t in enumerate(NODES):
    gt = sg.template_hankel(G.ell * t * V.ARCMIN)                # the build's filter template, not cached
    for m in MODES:
        K[m].append(((SIG[m][j] * V.ARCMIN ** 2) ** 2 * gt * T[m]).astype(np.float32))
    del gt
lo, hi = 100.0, 20000.0
band = (G.ell >= lo) & (G.ell <= hi)
log(f"[v05] {NPATCH} patches (from {P0}), {G.n}^2 x {G.pix}'; nodes theta_A = {np.round(NODES, 4)}")
for m in ("baseline", "goal"):
    a = np.stack([V.beam(G.ell, V.FWHM[i]) * V.sed_uK(V.FREQ[i]) for i in range(6)])
    resp = np.einsum("ixy,ixy->xy", F[m].w2d, a)[band]
    log(f"   [{m}] build ILC weights x OUR beams/SED: sum_i w_i B_i f_i = 1 within max |.-1| = {np.abs(resp-1).max():.1e} (l in [100, 20000])")
    del a, resp
for m in MODES:
    log(f"   sigma_A[{m}] = {np.array2string(SIG[m], precision=4)}")
# expected mean y of the painted population (white control: ensemble mean removed)
from szsim import signal as sg                      # noqa: E402
zc = 0.5 * (S.z[1:] + S.z[:-1])
lMc = 0.5 * (S.lnM[1:] + S.lnM[:-1])
Ymed = np.exp(np.interp(zc, S.zt, S.ly0)[:, None] + S.alpha * (lMc[None, :] - np.log(1e14)))
YBAR = V.RATIO * float((S.lam * Ymed).sum()) * np.exp(0.5 * S.sig ** 2) / G.L ** 2
log(f"   expected halos per patch {S.N_expected:,.1f} (HMF table total x area: {S.N_table:,.1f}); "
    f"ensemble mean y = {YBAR:.4e} (build s05: ybar 3.5895e-07 x RATIO = {3.5895e-07*V.RATIO:.4e})")
model_sub = {m: F[m].model for m in ("baseline", "goal")}

# injection configuration
INJ_TH = [NODES[np.argmin(np.abs(NODES - 1.0))], NODES[np.argmin(np.abs(NODES - 4.0))]]
INJ_Q = [5.0, 7.0]
CFG = [(a, b) for a in range(2) for b in range(2)]            # (theta index, q index)
if NINJ > 0:
    cmb = V.CMB()
    sim = V.SkySim(G, cmb)
    A6 = np.stack([V.beam(G.ell, V.FWHM[i]) * V.sed_uK(V.FREQ[i]) / Gs for i in range(6)])
    log(f"   injection: theta_A = {np.round(INJ_TH, 4)}, q_A = {INJ_Q}, 400 per patch, {NINJ} patches")
log(f"   set-up {time.time()-t0:.0f}s, rss {resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1e9:.2f} GB")


def new_hist():
    return {r: [(np.zeros(V.H_NB), np.zeros(V.H_NB)) for _ in NODES] for r in RUNS}


H = new_hist()
mom = []                    # per patch: (run, node) -> [sum x, sum x^2, n, min, max]
pinfo = []
inj_rows = []
block_id = P0 // BLOCK
for p in range(P0, P0 + NPATCH):
    tp = time.time()
    rng = np.random.default_rng([2026, 11, 11, p])
    h = S.draw(rng)
    big = h["th"] > 60.0
    keep = ~big
    m_all = P.paint(G, h["x"][keep], h["y"][keep], h["Y"][keep], h["th"][keep])
    ymean = m_all.mean()
    yk = {"all": G.rfft(m_all)}
    del m_all
    nbright = {}
    for m in ("baseline", "goal"):
        qB = h["Y"] / model_sub[m].sigma_Y(h["th"])
        br = keep & (qB >= 5.0)
        nbright[m] = int(br.sum())
        mb = P.paint(G, h["x"][br], h["y"][br], h["Y"][br], h["th"][br])
        yk[m + "_sub"] = yk["all"] - G.rfft(mb)
        del mb
    ykw = yk["all"].copy()
    ykw[0, 0] -= YBAR * G.n * G.n                             # white control: remove the ensemble mean y
    row = {}
    dsky = {}
    for r in RUNS:
        m = r.replace("_sub", "")
        src = ykw if r == "white" else (yk[r] if r.endswith("_sub") else yk["all"])
        for j in range(len(NODES)):
            dmap = G.irfft(K[m][j] * src)                       # Y_hat [sr]
            scale = 1.0 / (V.ARCMIN ** 2 * SIG[m][j])
            s1, s2, mn, mx = V.hist_accum(dmap.ravel(), scale, H[r][j][0], H[r][j][1], V.NTHREADS)
            row[(r, j)] = (s1, s2, dmap.size, mn, mx)
            if p < P0 + NINJ and r in ("baseline", "goal") and NODES[j] in INJ_TH:
                dsky[(r, NODES[j])] = dmap * scale
            del dmap
    mom.append(row)
    pinfo.append((p, h["x"].size, int(big.sum()), ymean, nbright["baseline"], nbright["goal"]))

    # ---------------------------------------------------------------- explicit injection
    if p < P0 + NINJ:
        off = rng.random(2) * 30.0
        gx = (np.floor((off[0] + 30.0 * np.arange(20)) / G.pix) + 0.5) * G.pix
        gy = (np.floor((off[1] + 30.0 * np.arange(20)) / G.pix) + 0.5) * G.pix
        X, Y = np.meshgrid(gx, gy)
        X, Y = X.ravel(), Y.ravel()
        ii = np.arange(400)
        cfg = (ii // 20 + ii % 20) % 4                       # interleaved configurations
        thi = np.array([INJ_TH[CFG[c][0]] for c in cfg])
        qi = np.array([INJ_Q[CFG[c][1]] for c in cfg])
        iy = np.round(Y / G.pix - 0.5).astype(int)
        ix = np.round(X / G.pix - 0.5).astype(int)
        Xn = sim.realise(rng)
        Zn = {m: F[m].ycomb(Xn[m]) for m in ("baseline", "goal")}    # noise: the six channels through the build ILC
        del Xn
        for m in ("baseline", "goal"):
            f = F[m]
            Yi = qi * np.array([f.sigma(t) for t in thi])
            yinj = G.rfft(P.paint(G, X, Y, Yi, thi))
            # by linearity, Z of (channels of y + noise) = T_y * y + Z_noise, with T_y = inv_Ny sum_i w_i B_i f_i / G_s
            # (explicit channel construction validated in v04)
            Z = {"tot": T[m] * (yk["all"] + yinj) + Zn[m], "iso": T[m] * yinj + Zn[m], "nl": T[m] * yinj}
            for a, tA in enumerate(INJ_TH):
                sA = f.sigma(tA)
                sel = thi == tA
                maps = {k: f.filt(Z[k], tA) for k in Z}
                for k in np.nonzero(sel)[0]:
                    inj_rows.append((p, 0 if m == "baseline" else 1, a, qi[k], Yi[k], maps["tot"][iy[k], ix[k]],
                                     maps["iso"][iy[k], ix[k]], maps["nl"][iy[k], ix[k]],
                                     dsky[(m, tA)][iy[k], ix[k]], sA))
                del maps
            del Z
            f._gcache.clear()
        del Zn
    if (p + 1) % BLOCK == 0 or p == P0 + NPATCH - 1:
        np.savez_compressed(os.path.join(V.OUT, f"v05_block{block_id:03d}.npz"),
                            nodes=NODES, runs=np.array(RUNS),
                            **{f"cnt_{r}_{j}": H[r][j][0] for r in RUNS for j in range(len(NODES))},
                            **{f"s1_{r}_{j}": H[r][j][1] for r in RUNS for j in range(len(NODES))})
        block_id += 1
        H = new_hist()
        np.savez(os.path.join(V.OUT, f"v05_moments{os.environ.get('TAG', '')}.npz"),
                 mom=np.array([[mm[(r, j)] for r in RUNS for j in range(len(NODES))] for mm in mom]),
                 pinfo=np.array(pinfo), inj=np.array(inj_rows), nodes=NODES, runs=np.array(RUNS),
                 sig=np.array([SIG[m] for m in MODES]), ybar=YBAR)
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e9
    log(f"   patch {p}: {h['x'].size:,} halos ({int(big.sum())} with theta500 > 60' not painted; bright q_B>=5: "
        f"{nbright['baseline']}/{nbright['goal']}); mean y {ymean:.3e}; {time.time()-tp:.1f}s; total {time.time()-t0:.0f}s; "
        f"peak rss {rss:.2f} GB")
log(f"[v05] done {time.time()-t0:.0f}s")
