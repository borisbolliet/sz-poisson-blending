"""w03 (part i): the noise level an analyst would measure on the filtered maps (noise + all halos), vs theta.

Same skies as w02 (seed [2026, 11, 12, 2, p]; every STEP-th patch).  Per patch and filter (baseline, goal):
  * halo map: all painted halos M500 > 1e13 (vlib painter), filtered with the build's MMF y-response;
  * noise map: OWN 6-channel CMB + correlated atmosphere + white noise (vlib.SkySim; seed [2026, 11, 12, 3 or 4, p]),
    through the build's full 6-channel MMF (vlib.BuildMMF.ycomb) - an independent realisation per filter;
  * at the 12 build nodes theta_j = 0.25-40.5' (every third, = step 1's v05 nodes): f = (noise + halos) / sigma_Y(theta_j)
    per patch (one 100 deg^2 tile, 1' sub-lattice, 360,000 points):
      plain std;  robust half-width (84.13 - 15.87 percentile) / 2;  3-sigma-clipped std (iterate: centre = median,
      keep |f - centre| < 3 std, until the std changes by < 1e-7).
    Also the halo-only and noise-only maps.
  * global histograms of f and of the halo-only map (all patches, 0.5' sub-lattice) -> the same three estimators on the
    pooled pixels (the main result);
    the analytic route: halo-only histogram convolved with N(0, var_noise) -> robust / clipped widths.
Compared with the build's sigma_eff / sigma_Y: r_i = sqrt(1 + sub-threshold confusion variance), r_ii = robust
half-width of noise + all-halo P(D) (data_real/step2/inputs.npz, at the same nodes).
Output: out/w03_noise.npz.   Env: NPATCH (500), STEP (4), VNTHREADS (3)
"""
import os, sys, time, resource
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import vlib2 as W                                     # noqa: E402
from vlib2 import V, np                               # noqa: E402
from szsim import signal as sg                        # noqa: E402
from scipy.special import ndtr                        # noqa: E402

NPATCH = int(os.environ.get("NPATCH", 500))
STEP = int(os.environ.get("STEP", 4))
log = W.Tee(os.path.join(W.HERE, "w03_noise.log"))
t0 = time.time()
G = V.Grid()
P = V.Painter()
S = V.PatchSampler(G.L)
Gs = P.presmooth(G)
RUNS = ("baseline", "goal")
F = {m: V.BuildMMF(G, m) for m in RUNS}
TH = {m: W.nodes_of(m) for m in RUNS}
NODES = list(range(6, 40, 3))
SIG = {m: np.array([F[m].sigma(TH[m][j]) for j in NODES]) for m in RUNS}
T2 = {m: F[m].yresp_transfer(extra=Gs).astype(np.float32) for m in RUNS}
K2N = {}
for m in RUNS:
    K2N[m] = []
    for k, j in enumerate(NODES):
        gt = sg.template_hankel(G.ell * TH[m][j] * V.ARCMIN)
        K2N[m].append(((SIG[m][k] * V.ARCMIN ** 2) ** 2 * gt / (V.ARCMIN ** 2 * SIG[m][k])).astype(np.float32))
        del gt
    F[m]._gcache.clear()
SKY = V.SkySim(G, V.CMB())
inp = np.load(f"{W.ROOT}/data_real/step2/inputs.npz")
log(f"[w03] patches 0..{NPATCH-1} step {STEP} ({len(range(0, NPATCH, STEP))} patches, "
    f"{len(range(0, NPATCH, STEP)) * G.L**2 / 3600:.0f} deg^2), nodes {np.round(TH['baseline'][NODES], 3).tolist()}; "
    f"threads {W.NTHREADS}; set-up {time.time()-t0:.0f}s; rss {resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1e9:.2f} GB")


def clipped_std(v, k=3.0, maxit=50):
    keep = np.ones(v.size, bool)
    s_old = -1.0
    for _ in range(maxit):
        vv = v[keep]
        c = np.median(vv)
        s = vv.std()
        keep = np.abs(v - c) < k * s
        if abs(s - s_old) < 1e-6 * s:
            break
        s_old = s
    return s, keep.mean()


def robust_hw(v):
    a, b = np.percentile(v, [15.865525, 84.134475])
    return 0.5 * (b - a)


KEYS = [f"{m}_{kind}_{k}" for m in RUNS for kind in ("tot", "halo") for k in range(len(NODES))]
# tSZ power of the painted halo sky at l = 3000 (Poisson, no clustering): annulus 2900-3100, pre-smoothing removed
ANN = (G.ell >= 2900.0) & (G.ell <= 3100.0)
W_ANN = (G.mult * ANN)
L_ANN = float((G.ell * W_ANN).sum() / W_ANN.sum())
CL3000 = []
HC = {k: np.zeros(V.H_NB) for k in KEYS}
HS = {k: np.zeros(V.H_NB) for k in KEYS}
HM = {k: np.zeros(3) for k in KEYS}
NV = {m: np.zeros((len(NODES), 2)) for m in RUNS}                 # noise-only: sum n^2, count
STAT = {m: [] for m in RUNS}
plist = list(range(0, NPATCH, STEP))
for ip, p in enumerate(plist):
    tp = time.time()
    h = S.draw(np.random.default_rng([2026, 11, 12, 2, p]))         # the same sky as w02 patch p
    keep = h["th"] <= 60.0
    mp = P.paint(G, h["x"][keep], h["y"][keep], h["Y"][keep], h["th"][keep])
    yk = G.rfft(mp)
    del mp
    CL3000.append(float((W_ANN * np.abs(yk) ** 2 / Gs ** 2).sum() / W_ANN.sum() * G.omega / G.n ** 2))
    for im, m in enumerate(RUNS):
        X = SKY.realise(np.random.default_rng([2026, 11, 12, 3 + im, p]), modes=(m,))
        Zn = F[m].ycomb(X[m])
        del X
        Zs = T2[m] * yk
        row = []
        for k in range(len(NODES)):
            s2d = G.irfft(K2N[m][k] * Zs)
            n2d = G.irfft(K2N[m][k] * Zn)
            s_ = s2d[::2, ::2].ravel(); n_ = n2d[::2, ::2].ravel()          # 0.5' lattice: pooled histograms
            f_ = s_ + n_
            f4 = (s2d[::4, ::4] + n2d[::4, ::4]).ravel()                    # 1' lattice: per-patch (per-tile) estimates
            s4 = s2d[::4, ::4].ravel()
            del s2d, n2d
            cs, cf = clipped_std(f4)
            row.append([f4.std(), robust_hw(f4), cs, cf, s4.std(), robust_hw(s4), 0.0, n_.std(), f_.mean(), np.median(f4)])
            NV[m][k] += [np.sum(n_ * n_), n_.size]
            for key, arr in ((f"{m}_tot_{k}", f_), (f"{m}_halo_{k}", s_)):
                V.hist_accum(arr, 1.0, HC[key], HS[key], W.NTHREADS)
                HM[key] += np.array([arr.size, arr.sum(), (arr * arr).sum()])
        STAT[m].append(row)
        del Zn, Zs
    del yk
    if ip % 10 == 0 or ip == len(plist) - 1:
        r = np.array(STAT["baseline"][-1])
        log(f"   patch {p} ({ip+1}/{len(plist)}): baseline at 1.007': std {r[3,0]:.4f} robust {r[3,1]:.4f} clipped {r[3,2]:.4f} "
            f"noise-only std {r[3,7]:.4f}; {time.time()-tp:.1f}s [{time.time()-t0:.0f}s]; peak rss "
            f"{resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1e9:.2f} GB")

# ------------------------------------------------------------------ pooled estimates
edges = V.hist_edges()
cen_lin = 0.5 * (edges[1:] + edges[:-1])


def hist_stats(c, s1):
    """plain std (from bin sums), robust half-width and 3-sigma-clipped std from a pooled histogram (bin means)."""
    xm = np.where(c > 0, s1 / np.maximum(c, 1), cen_lin)
    n = c.sum()
    cdf = np.cumsum(c) / n
    q16, q84 = np.interp([0.15865525, 0.84134475], cdf, edges[1:])
    keep = np.ones(c.size, bool)
    s_old = -1
    for _ in range(100):
        cc = c * keep
        mu = (cc * xm).sum() / cc.sum()
        sd = np.sqrt((cc * (xm - mu) ** 2).sum() / cc.sum())
        cdfk = np.cumsum(cc) / cc.sum()
        med = np.interp(0.5, cdfk, edges[1:])
        keep = np.abs(xm - med) < 3 * sd
        if abs(sd - s_old) < 1e-7 * sd:
            break
        s_old = sd
    return 0.5 * (q84 - q16), sd


def convolve_noise(c, s1, var_n):
    """halo-only pooled histogram (+) N(0, var_n): returns the distribution on a fine grid (for widths)."""
    xm = np.where(c > 0, s1 / np.maximum(c, 1), cen_lin)
    sel = c > 0
    xs, ws = xm[sel], c[sel] / c.sum()
    grid = np.arange(-12.0, 60.0, 0.002)
    cdf = np.zeros_like(grid)
    sn = np.sqrt(var_n)
    for a in range(0, xs.size, 400):
        cdf += (ws[a:a + 400, None] * ndtr((grid[None, :] - xs[a:a + 400, None]) / sn)).sum(0)
    tail = 1.0 - cdf[-1]
    return grid, cdf, tail


log("\n=== part (i): noise level measured on the filtered maps (noise + all halos), sigma_eff / sigma_Y ===")
log("    columns: pooled over all patches [median over patches; 16-84% of patches]; build r_i, r_ii at the same node")
out = {}
for m in RUNS:
    S_ = np.array(STAT[m])                                         # (npatch, nnode, 10)
    ri = inp[f"{m}_r_i"][NODES]; rii = inp[f"{m}_r_ii"][NODES]
    log(f"\n  [{m}]  theta_A | plain std | robust half-width | 3-sigma-clipped std (kept frac) | noise-only std | "
        f"build r_i | build r_ii | analytic (halo hist (+) noise): robust, clipped | halo-only: std, robust, clipped")
    res = []
    for k in range(len(NODES)):
        kt, kh = f"{m}_tot_{k}", f"{m}_halo_{k}"
        n_, s1, s2 = HM[kt]
        std_pool = np.sqrt(s2 / n_ - (s1 / n_) ** 2)
        rob_pool, clip_pool = hist_stats(HC[kt], HS[kt])
        nh, h1, h2 = HM[kh]
        std_h = np.sqrt(h2 / nh - (h1 / nh) ** 2)
        rob_h, clip_h = hist_stats(HC[kh], HS[kh])
        var_n = NV[m][k, 0] / NV[m][k, 1]
        grid, cdf, tail = convolve_noise(HC[kh], HS[kh], var_n)
        a16, a84 = np.interp([0.15865525, 0.84134475], cdf, grid)
        # clipped std of the convolved distribution (iterative, on the fine grid)
        pdf = np.diff(np.concatenate([[0.0], cdf]))
        keepg = np.ones(grid.size, bool); s_old = -1
        for _ in range(100):
            pp = pdf * keepg
            mu = (pp * grid).sum() / pp.sum(); sd = np.sqrt((pp * (grid - mu) ** 2).sum() / pp.sum())
            med = np.interp(0.5, np.cumsum(pp) / pp.sum(), grid)
            keepg = np.abs(grid - med) < 3 * sd
            if abs(sd - s_old) < 1e-7 * sd:
                break
            s_old = sd
        pc = lambda a: f"{np.median(a):.4f}; {np.percentile(a, 16):.4f}-{np.percentile(a, 84):.4f}"
        log(f"   {TH[m][NODES[k]]:7.3f}' | {std_pool:.4f} [{pc(S_[:, k, 0])}] | {rob_pool:.4f} [{pc(S_[:, k, 1])}] | "
            f"{clip_pool:.4f} [{pc(S_[:, k, 2])}] ({np.median(S_[:, k, 3]):.5f}) | {np.sqrt(var_n):.4f} | {ri[k]:.4f} | "
            f"{rii[k]:.4f} | {0.5*(a84-a16):.4f}, {sd:.4f} | {std_h:.4f}, {rob_h:.4f}, {clip_h:.4f}")
        res.append([TH[m][NODES[k]], std_pool, rob_pool, clip_pool, np.sqrt(var_n), ri[k], rii[k], 0.5 * (a84 - a16), sd,
                    std_h, rob_h, clip_h, np.median(S_[:, k, 0]), np.median(S_[:, k, 1]), np.median(S_[:, k, 2])])
    res = np.array(res)
    out[f"{m}_res"] = res
    out[f"{m}_stat"] = S_
    # summary ratios vs the build
    log(f"   [{m}] over theta_A = 0.25-40': plain std / r_ii = {np.min(res[:,1]/res[:,6]):.3f}-{np.max(res[:,1]/res[:,6]):.3f}; "
        f"robust / r_ii - 1 = {np.min(res[:,2]/res[:,6]-1):+.4f} to {np.max(res[:,2]/res[:,6]-1):+.4f}; "
        f"clipped / r_i - 1 = {np.min(res[:,3]/res[:,5]-1):+.4f} to {np.max(res[:,3]/res[:,5]-1):+.4f}; "
        f"clipped / r_ii - 1 = {np.min(res[:,3]/res[:,6]-1):+.4f} to {np.max(res[:,3]/res[:,6]-1):+.4f}")
# ------------------------------------------------------------------ tSZ power (claim: D_3000(143) = 9.07 / 3.24 uK^2)
cl = np.array(CL3000)
f143 = V.sed_uK(143.0)
Dl = L_ANN * (L_ANN + 1) * cl / (2 * np.pi)
c_low = (0.6 / 0.8) ** 1.79
log(f"\n=== tSZ power of the painted Poisson halo sky at l = {L_ANN:.0f} (annulus 2900-3100), {cl.size} patches ===")
log(f"   1e12 l(l+1) C_l / 2pi = {1e12*Dl.mean():.4f} +- {1e12*Dl.std(ddof=1)/np.sqrt(cl.size):.4f} (patch scatter "
    f"{1e12*Dl.std(ddof=1):.3f}); own SED f(143 GHz) = {f143:.1f} uK per y -> D_3000(143) = {Dl.mean()*f143**2:.2f} +- "
    f"{Dl.std(ddof=1)/np.sqrt(cl.size)*f143**2:.2f} uK^2 = {Dl.mean()*f143**2/3.42:.2f} x SPT (3.42 +- 0.54)")
log(f"   low-pressure population (every Y x {c_low:.4f}, power x {c_low**2:.4f}): D_3000(143) = {Dl.mean()*f143**2*c_low**2:.2f} +- "
    f"{Dl.std(ddof=1)/np.sqrt(cl.size)*f143**2*c_low**2:.2f} uK^2 = {Dl.mean()*f143**2*c_low**2/3.42:.2f} x SPT "
    f"({(Dl.mean()*f143**2*c_low**2-3.42)/0.54:+.1f} sigma)")
np.savez(os.path.join(W.OUT, "w03_noise.npz"), nodes=np.array(NODES), th=TH["baseline"][NODES], **out, cl3000=cl,
         **{f"c_{k}": HC[k] for k in KEYS}, **{f"s_{k}": HS[k] for k in KEYS}, **{f"m_{k}": HM[k] for k in KEYS})
log(f"\n[w03] done {time.time()-t0:.0f}s; peak rss {resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1e9:.2f} GB")
