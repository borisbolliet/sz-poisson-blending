"""v12: explicit-noise re-test of the per-object fractions (follow-up of the ~2 sigma B3 excess in the v05 injections).

Fresh skies (patches P0.., not used for the v05 injections) and fresh CMB + atmosphere + white noise realisations.
At every pixel x and for theta_A = 1.007', 4.025' (baseline, goal): delta(x) = filtered halo sky / sigma_A and
n(x) = filtered noise / sigma_A (the six channels through the build's MMF).  A cluster of signal-to-noise q_A placed
at x would be measured at q_A + n(x) + delta(x) (linearity verified to 1e-8 in v05), so for q_A = 5, 6, 7 and
q_cut = 5 we count explicitly over all pixels:
   detected = q + n + delta > 5;   alone = q + n > 5;   B2_10 = detected & delta > 0.1 q;   B3 = detected & !alone;
   lost = alone & !detected;  mean boost = <delta/q | detected>,
and compare with the analytic noise marginalisation over the SAME delta pixels (Phi(q + delta - 5)), which is what
the build's P(D) does.  Errors: jackknife over patches.
"""
import os, sys, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import vlib as V                                    # noqa: E402
import numpy as np                                  # noqa: E402
from scipy.special import ndtr                      # noqa: E402
from szsim import signal as sg                      # noqa: E402

NPATCH = int(os.environ.get("NPATCH", 60))
P0 = int(os.environ.get("P0", 100))
log = V.Tee(os.path.join(V.HERE, "v12_noise_explicit.log"))
t0 = time.time()
G = V.Grid()
P = V.Painter()
Gs = P.presmooth(G)
S = V.PatchSampler(G.L)
cmb = V.CMB()
sim = V.SkySim(G, cmb)
th_tab = np.load(f"{V.ROOT}/data_real/rho_mmf_baseline.npz")["th"]
NODES = th_tab[::2][6:40:3]
TH = [NODES[3], NODES[6]]                                   # 1.0069', 4.025'
MODES = ("baseline", "goal")
F = {m: V.BuildMMF(G, m) for m in MODES}
T = {m: F[m].yresp_transfer(extra=Gs) for m in MODES}
QA = [5.0, 6.0, 7.0]
KEYS = ("det", "alone", "b2", "b3", "lost", "mb")
obs = np.zeros((NPATCH, 2, 2, 3, len(KEYS)))               # explicit counts (sums over pixels)
ana = np.zeros_like(obs)                                    # analytic marginalisation over the same delta pixels
nstat = np.zeros((NPATCH, 2, 2, 2))
log(f"[v12] {NPATCH} fresh patches from {P0}; theta_A = {np.round(TH, 4)}; q_A = {QA}; set-up {time.time()-t0:.0f}s")
for k, p in enumerate(range(P0, P0 + NPATCH)):
    tp = time.time()
    rng = np.random.default_rng([2026, 11, 11, p])
    h = S.draw(rng)
    keep = h["th"] <= 60.0
    yk = G.rfft(P.paint(G, h["x"][keep], h["y"][keep], h["Y"][keep], h["th"][keep]))
    nrng = np.random.default_rng([777, p])                  # fresh, independent noise stream
    X = sim.realise(nrng)
    Zn = {m: F[m].ycomb(X[m]) for m in MODES}
    del X
    for a, m in enumerate(MODES):
        f = F[m]
        for b, tA in enumerate(TH):
            sA = f.sigma(tA)
            d = (f.filt(T[m] * yk, tA) / sA).ravel()
            n = (f.filt(Zn[m], tA) / sA).ravel()
            nstat[k, a, b] = (n.mean(), n.var())
            for c, q in enumerate(QA):
                det = q + n + d > 5.0
                alone = q + n > 5.0
                obs[k, a, b, c] = (det.sum(), alone.sum(), (det & (d > 0.1 * q)).sum(), (det & ~alone).sum(),
                                   (alone & ~det).sum(), (d[det] / q).sum())
                Pd = ndtr(q + d - 5.0)
                P0_ = ndtr(q - 5.0)
                ana[k, a, b, c] = (Pd.sum(), P0_ * d.size, (Pd * (d > 0.1 * q)).sum(), ((Pd - P0_) * (d > 0)).sum(),
                                   ((P0_ - Pd) * (d < 0)).sum(), (Pd * d / q).sum())
            f._gcache.clear()
    if k % 10 == 0 or k == NPATCH - 1:
        log(f"   patch {p}: {time.time()-tp:.1f}s, total {time.time()-t0:.0f}s")


def fracs(s):
    return dict(det=s[..., 0] / (G.n * G.n), B2_10=s[..., 2] / s[..., 0], B3=s[..., 3] / s[..., 0], lost=s[..., 4] / s[..., 1],
                mb=s[..., 5] / s[..., 0])


def jk(v):
    v = np.asarray(v); n = len(v)
    return np.sqrt((n - 1) / n * ((v - v.mean(0)) ** 2).sum(0))


O, A = obs.sum(0), ana.sum(0)
fo, fa = fracs(O / NPATCH), fracs(A / NPATCH)
loo_o = [fracs((O - obs[i]) / (NPATCH - 1)) for i in range(NPATCH)]
loo_d = [fracs((O - obs[i]) / (NPATCH - 1)) for i in range(NPATCH)]
# error on the difference explicit - analytic (paired by patch: the same delta pixels)
diff_loo = [{k: fracs((O - obs[i]) / (NPATCH - 1))[k] - fracs((A - ana[i]) / (NPATCH - 1))[k] for k in fo} for i in range(NPATCH)]
v07 = np.load(os.path.join(V.OUT, "v07_nodeprobs.npz"))
B6 = np.load(os.path.join(V.OUT, "v06_pd_builder_nodes.npz"))
QN = B6["qnodes"]
log(f"\n   noise at all pixels: mean {nstat[...,0].mean():+.5f}, variance {nstat[...,1].mean():.5f} (expected 0, 1)")
for a, m in enumerate(MODES):
    for b, tA in enumerate(TH):
        j = int(np.argmin(np.abs(NODES - tA)))
        for c, q in enumerate(QA):
            kq = int(np.argmin(np.abs(QN - q)))
            dt = v07[f"{m}_det"][1, j, kq]
            pm = dict(det=dt, B2_10=v07[f"{m}_b2"][1, j, kq, 1] / dt, B3=v07[f"{m}_b3"][1, j, kq] / dt,
                      lost=v07[f"{m}_lost"][1, j, kq] / ndtr(q - 5), mb=v07[f"{m}_mean"][1, j, kq] / dt)
            line = []
            for kk in ("det", "B2_10", "B3", "lost", "mb"):
                e = jk([l[kk][a, b, c] for l in diff_loo])
                dd = fo[kk][a, b, c] - fa[kk][a, b, c]
                line.append(f"{kk} explicit {fo[kk][a,b,c]:.4f} analytic {fa[kk][a,b,c]:.4f} (diff {dd:+.4f}+-{e:.4f}, "
                            f"{dd/e if e > 0 else 0:+.1f}s; 400-patch map P(D) {pm[kk]:.4f})")
            log(f"   [{m}] theta_A={tA:.3f}' q_A={q}: " + "; ".join(line))
log(f"[v12] done {time.time()-t0:.0f}s")
