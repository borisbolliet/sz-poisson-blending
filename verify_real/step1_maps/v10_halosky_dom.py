"""v10: map-level P(D) WITH the build's dominance rule (each detection counted under its brightest cluster).

The build's headline fractions count a primary A only if every neighbour's individual contribution satisfies
x_B = rho_AB(s) Y_B / sigma_A < q_A (truncated compound Poisson x P_dom).  Without the rule the same P(D) gives
~2x larger fractions (faint halos sitting on bright clusters), so the headline can only be tested with it.

For the same skies as v05 (patches 0..NPATCH-1, same seeds): on a 2 x 2 sub-lattice of pixels (0.5' spacing,
1.44 M points per patch), for every run and theta_A node:
  * delta(x) from the filtered painted maps (as v05; map quantity, independent of the build's P(D));
  * x_max(x) = max_B x_B(x) over the catalogue halos (all M > 1e13, or q_B < 5 for the *_sub runs), by
    max-painting each halo whose peak x_B(0) >= 0.5 out to the radius where x_B drops below 0.5; rho from the
    build's rho tables via blendlib.rho_at (the same kernel as the build; validated against painted pairs in v04b);
  * histograms of delta in classes of x_max aligned with the build's q grid (class 0: x_max < 0.5, fine bins;
    class c >= 1: Qd[c-1] <= x_max < Qd[c], coarse bins), per block of patches.
The dominance-aware probability for (theta_A, q_A = Qd[i]) then uses classes 0..i (x_max < q_A).
"""
import os, sys, time, resource
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import vlib as V                                    # noqa: E402
import numpy as np                                  # noqa: E402
import numba as nb                                  # noqa: E402
sys.path.insert(0, f"{V.ROOT}/analysis/blending")
import blendlib as bl                               # noqa: E402
from szsim import signal as sg                      # noqa: E402

NPATCH = int(os.environ.get("NPATCH", 160))
BLOCK = int(os.environ.get("BLOCK", 16))
P0 = int(os.environ.get("P0", 0))
TAG = os.environ.get("TAG", "")
log = V.Tee(os.path.join(V.HERE, f"v10_halosky_dom{TAG}.log"))
t0 = time.time()
G = V.Grid()
P = V.Painter()
Gs = P.presmooth(G)
S = V.PatchSampler(G.L)
th_tab = np.load(f"{V.ROOT}/data_real/rho_mmf_baseline.npz")["th"]
NODES = th_tab[::2][6:40:3]
MODES = ("baseline", "goal", "white")
RUNS = ("baseline", "goal", "white", "baseline_sub", "goal_sub")
F = {m: V.BuildMMF(G, m) for m in MODES}
SIG = {m: np.array([F[m].sigma(t) for t in NODES]) for m in MODES}
T = {m: F[m].yresp_transfer(extra=Gs) for m in MODES}
K = {m: [] for m in MODES}
for j, t in enumerate(NODES):
    gt = sg.template_hankel(G.ell * t * V.ARCMIN)
    for m in MODES:
        K[m].append(((SIG[m][j] * V.ARCMIN ** 2) ** 2 * gt * T[m]).astype(np.float32))
    del gt
for m in MODES:
    F[m]._gcache.clear()
TAB = {m: bl.RhoTable(np.load(f"{V.ROOT}/data_real/rho_mmf_{m}.npz")) for m in ("baseline", "goal")}
TAB["white"] = bl.load_rho_table()
model = {m: F[m].model for m in MODES}
QN = np.concatenate([np.arange(0.1, 20.0, 0.1), np.geomspace(20.0, 3000.0, 60)])   # = pdlib1.QNODES
QD = QN[QN >= 0.4999]                                # class boundaries (q_A >= 0.5 primaries)
NCL = QD.size + 1
X0 = float(QD[0])
# coarse delta bins for the x_max >= 0.5 classes
CE = np.concatenate([-np.logspace(np.log10(4.0) + 4, np.log10(4.0), 101)[:-1], np.linspace(-4, 4, 401),
                     np.logspace(np.log10(4.0), np.log10(4.0) + 4, 101)[1:]])
NCB = CE.size - 1
SUB = 2                                              # sub-lattice step (0.5')
NL = G.n // SUB
log(f"[v10] {NPATCH} patches from {P0}, lattice {NL}^2 (0.5'), nodes {np.round(NODES, 4)}; {NCL} x_max classes, "
    f"{NCB} coarse delta bins; set-up {time.time()-t0:.0f}s")


@nb.njit(cache=True)
def x_peak(rho, lth0, dlth, RB, thA, ths, Ys, sigA, out):
    for k in range(ths.size):
        out[k] = bl.rho_at(rho, lth0, dlth, RB, thA, ths[k], 0.0) * Ys[k] / sigA


@nb.njit(parallel=True, cache=True)
def max_paint(xm_thr, xs, ys, Ys, ths, rho, lth0, dlth, RB, thA, sigA, x0, pixL, offL, nthr):
    """Per-thread max maps on the lattice (centres at offL + i*pixL): x_max = max_B rho(s) Y_B/sigma_A where >= x0."""
    n = xm_thr.shape[1]
    NU = 151
    nh = xs.size
    chunk = (nh + nthr - 1) // nthr
    for t in nb.prange(nthr):
        for h in range(t * chunk, min(nh, (t + 1) * chunk)):
            thB = ths[h]
            amp = Ys[h] / sigA
            S_ = 5.0 * thA + 5.0 * thB + RB
            # radius: last s on the table nodes with x >= x0
            R = 0.0
            for it in range(NU):
                s = S_ * (it / (NU - 1.0)) ** 2
                if bl.rho_at(rho, lth0, dlth, RB, thA, thB, s) * amp >= x0:
                    R = s
            if R <= 0.0:
                continue
            R = R + S_ * (2.0 / (NU - 1.0))                 # cover the next table interval
            x = xs[h]
            y = ys[h]
            i0 = int(np.floor((x - R - offL) / pixL))
            i1 = int(np.ceil((x + R - offL) / pixL))
            j0 = int(np.floor((y - R - offL) / pixL))
            j1 = int(np.ceil((y + R - offL) / pixL))
            for jj in range(j0, j1 + 1):
                dy = offL + jj * pixL - y
                for ii in range(i0, i1 + 1):
                    dx = offL + ii * pixL - x
                    r = np.sqrt(dx * dx + dy * dy)
                    if r <= R:
                        v = bl.rho_at(rho, lth0, dlth, RB, thA, thB, r) * amp
                        a = jj % n
                        b = ii % n
                        if v > xm_thr[t, a, b]:
                            xm_thr[t, a, b] = v


@nb.njit(cache=True)
def class_accum(d, xm, QD, CE, H0c, H0s, Hc, Hs):
    """class 0 (x_max < QD[0]) -> fine histogram (vlib bins); class c >= 1 -> coarse histogram row c."""
    ncb = CE.size - 1
    for k in range(d.size):
        x = d[k]
        c = np.searchsorted(QD, xm[k], side="right")
        if c == 0:
            if x >= -20.0 and x < 20.0:
                b = 300 + int((x + 20.0) / 0.002)
                if b > 300 + 19999:
                    b = 300 + 19999
            elif x >= 20.0:
                b = 300 + 20000 + int(np.log10(x / 20.0) * 100.0)
                if b > 2 * 300 + 20000 - 1:
                    b = 2 * 300 + 20000 - 1
            else:
                b = 299 - int(np.log10(-x / 20.0) * 100.0)
                if b < 0:
                    b = 0
            H0c[b] += 1.0
            H0s[b] += x
        else:
            b = np.searchsorted(CE, x, side="right") - 1
            if b < 0:
                b = 0
            if b > ncb - 1:
                b = ncb - 1
            Hc[c, b] += 1.0
            Hs[c, b] += x


def new_acc():
    return {r: [dict(H0c=np.zeros(V.H_NB), H0s=np.zeros(V.H_NB), Hc=np.zeros((NCL, NCB)), Hs=np.zeros((NCL, NCB)))
                for _ in NODES] for r in RUNS}


acc = new_acc()
block_id = P0 // BLOCK
offL = 0.5 * G.pix                                     # lattice point (i*SUB) has centre (i*SUB + 1/2) pix
pixL = SUB * G.pix
xm_thr = np.zeros((V.NTHREADS, NL, NL), np.float32)
YBAR = float(np.load(os.path.join(V.OUT, "v05_moments.npz"))["ybar"])
stats = []
for p in range(P0, P0 + NPATCH):
    tp = time.time()
    rng = np.random.default_rng([2026, 11, 11, p])
    h = S.draw(rng)
    keep = h["th"] <= 60.0
    m_all = P.paint(G, h["x"][keep], h["y"][keep], h["Y"][keep], h["th"][keep])
    yk = {"all": G.rfft(m_all)}
    del m_all
    sub_mask = {}
    for m in ("baseline", "goal"):
        qB = h["Y"] / model[m].sigma_Y(h["th"])
        br = keep & (qB >= 5.0)
        sub_mask[m] = keep & ~br
        mb = P.paint(G, h["x"][br], h["y"][br], h["Y"][br], h["th"][br])
        yk[m + "_sub"] = yk["all"] - G.rfft(mb)
        del mb
    ykw = yk["all"].copy()
    ykw[0, 0] -= YBAR * G.n * G.n                         # white control: ensemble mean removed (as v05)
    nsel_tot = 0
    for r in RUNS:
        m = r.replace("_sub", "")
        src = ykw if r == "white" else (yk[r] if r.endswith("_sub") else yk["all"])
        hm = sub_mask[m] if r.endswith("_sub") else keep
        xs, ys, Ys, ths = h["x"][hm], h["y"][hm], h["Y"][hm], h["th"][hm]
        tab = TAB[m]
        for j in range(len(NODES)):
            dmap = G.irfft(K[m][j] * src)[::SUB, ::SUB] / (V.ARCMIN ** 2 * SIG[m][j])
            xp = np.empty(ths.size)
            x_peak(tab.rho, tab.lth0, tab.dlth, tab.R_BEAM, float(NODES[j]), ths, Ys, float(SIG[m][j]), xp)
            sel = xp >= X0
            nsel_tot += int(sel.sum())
            xm_thr[:] = 0.0
            max_paint(xm_thr, xs[sel], ys[sel], Ys[sel], ths[sel], tab.rho, tab.lth0, tab.dlth, tab.R_BEAM,
                      float(NODES[j]), float(SIG[m][j]), X0, pixL, offL, V.NTHREADS)
            xm = xm_thr.max(0)
            a = acc[r][j]
            class_accum(np.ascontiguousarray(dmap.ravel()), np.ascontiguousarray(xm.ravel().astype(np.float64)), QD, CE,
                        a["H0c"], a["H0s"], a["Hc"], a["Hs"])
            if j in (3, 6, 9) and r in ("baseline", "goal"):
                stats.append((p, RUNS.index(r), j, float((xm >= 1.0).mean()), float((xm >= 5.0).mean())))
            del dmap, xm
    if (p + 1) % BLOCK == 0 or p == P0 + NPATCH - 1:
        np.savez_compressed(os.path.join(V.OUT, f"v10_block{block_id:03d}.npz"), nodes=NODES, runs=np.array(RUNS),
                            QD=QD, CE=CE,
                            **{f"H0c_{r}_{j}": acc[r][j]["H0c"] for r in RUNS for j in range(len(NODES))},
                            **{f"H0s_{r}_{j}": acc[r][j]["H0s"] for r in RUNS for j in range(len(NODES))},
                            **{f"Hc_{r}_{j}": acc[r][j]["Hc"].astype(np.float32) for r in RUNS for j in range(len(NODES))},
                            **{f"Hs_{r}_{j}": acc[r][j]["Hs"].astype(np.float32) for r in RUNS for j in range(len(NODES))})
        block_id += 1
        acc = new_acc()
        np.save(os.path.join(V.OUT, f"v10_stats{TAG}.npy"), np.array(stats))
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e9
    log(f"   patch {p}: {h['x'].size:,} halos; {nsel_tot:,} (run, node, halo) max-paintings; {time.time()-tp:.1f}s; "
        f"total {time.time()-t0:.0f}s; peak rss {rss:.2f} GB")
log(f"[v10] done {time.time()-t0:.0f}s")
