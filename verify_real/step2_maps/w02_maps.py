"""w02: map-level blended detections of every primary on Poisson halo skies (lens: maps).

Per 10 x 10 deg periodic patch (2400^2 x 0.25'), fresh seeds (step 1 used [2026, 11, 11, p]; here [2026, 11, 12, 2, p]):
  * halos M500 > 1e13, 0.01 < z < 3 from vlib's own Poisson sampler (continuous positions, Y with lognormal scatter);
    painted with vlib's own Arnaud painter (halos with theta500 > 60' are not painted: counted);
  * the y map is filtered with the build's MMF y-response (baseline, goal) and the old white filter (control; the
    ensemble mean y is removed, c = 1 convention), at every node theta_j of the build's grid that a primary needs;
  * the filtered map is read EXACTLY at each primary's continuous position by a type-2 NUFFT;
    delta_A,j = (Y_hat_j(x_A) - R_j(theta_A) Y_A) / sigma_j = neighbour sum in the node-j filter (self response R_j
    as the same discrete mode sum); x_max_A,j = brightest single-neighbour contribution rho_j(theta_B, s) Y_B / sigma_j
    (own Hankel rho; for the attribution rule only);
  * primaries: q_A = Y_A / sigma_Y(theta_A) >= 0.5 in that filter; the two nodes bracketing ln theta_A are stored
    (j, w) so that the detection probability can be interpolated linearly in ln theta as the build's kernel is.
  * noise is marginalised analytically later (w04): P(q_obs > e) = Phi(q_A + delta - e).
  Part (i) (noise estimators on the filtered maps) is w03, on the same skies.
Output: out/maps/patch_<p>.npz (per-primary arrays)
Env: NPATCH (default 400), P0 (0), VNTHREADS (3)
"""
import os, sys, time, resource
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import vlib2 as W                                     # noqa: E402
from vlib2 import V, np                               # noqa: E402
from szsim import signal as sg                        # noqa: E402
from scipy.interpolate import CubicSpline             # noqa: E402

NPATCH = int(os.environ.get("NPATCH", 400))
P0 = int(os.environ.get("P0", 0))
TAG = os.environ.get("TAG", "")
DMAP = os.path.join(W.OUT, "maps"); os.makedirs(DMAP, exist_ok=True)
log = W.Tee(os.path.join(W.HERE, f"w02_maps{TAG}.log"))
t0 = time.time()
G = V.Grid()
P = V.Painter()
S = V.PatchSampler(G.L)
blk = W.Block(G)
MF = {m: W.ModeFilter(G, P, m, blk, keep_g=False) for m in W.MODES}
# block kernels K_j = pref_j g_j T (real, float32), own rho tables, self-response tables
KJ = {m: MF[m].K32 for m in W.MODES}
TAB, ENV, SELF = {}, {}, {}
for m in W.MODES:
    d = np.load(os.path.join(W.OUT, f"rho_{m}.npz"))
    tab_, env_ = d["tab"], d["E"]                      # read each array ONCE (npz[key] re-reads the whole array)
    TAB[m] = [tab_[j] for j in range(len(MF[m].th))]
    ENV[m] = [env_[j] for j in range(len(MF[m].th))]
    assert np.allclose(d["th"], MF[m].th)
    del tab_, env_
    lt_, R_ = W.load_self(os.path.join(W.OUT, f"self_{m}.npz"))
    SELF[m] = (lt_, [CubicSpline(a_, b_) for a_, b_ in zip(lt_, R_)])
LTB = W.LTB0 + W.DLTB * np.arange(W.NTB)
YBAR = float(np.load(os.path.join(V.OUT, "v05_moments.npz"))["ybar"])   # step-1 simulation: ensemble mean y
log(f"[w02] {NPATCH} patches from {P0} ({NPATCH * G.L**2 / 3600:.0f} deg^2); threads {W.NTHREADS}; "
    f"set-up {time.time()-t0:.0f}s; rss {resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1e9:.2f} GB")
log(f"   nodes: " + "; ".join(f"{m} {len(MF[m].th)} ({MF[m].th[0]:.2f}-{MF[m].th[-1]:.1f}')" for m in W.MODES)
    )


# ------------------------------------------------------------------ unit test: one isolated halo -> delta = 0
def unit_test():
    rng = np.random.default_rng(5)
    out = []
    for m in W.MODES:
        mf = MF[m]
        lt, R = SELF[m]
        for thA in (0.18, 0.37, 1.3, 5.1, 45.0):
            nn = len(mf.th)
            f = np.clip((np.log(thA) - mf.lth[0]) / mf.dlth, 0.0, nn - 1 - 1e-9)
            j = min(int(f), nn - 2)
            w = f - j
            x0, y0 = rng.random(2) * G.L
            Y = 20.0 * mf.sigma_Y(thA)
            mp = P.paint(G, np.array([x0]), np.array([y0]), np.array([Y]), np.array([thA]))
            yb = blk.take(G.rfft(mp)) * blk.phase
            for jn in [jj for jj, ww in ((j, 1 - w), (j + 1, w)) if ww > 0]:
                v = blk.evaluate(KJ[m][jn] * yb, [x0], [y0])[0]
                Rv = R[jn](np.log(thA))
                out.append((m, thA, abs(v - Rv * Y) / mf.sig[jn]))
    worst = {}
    for m, t, e in out:
        worst[(m, t)] = max(worst.get((m, t), 0.0), e)
    return max(e for _, _, e in out), " ".join(f"{m[0]}{t}:{e:.0e}" for (m, t), e in worst.items())


_ut = unit_test()
log(f"   unit test (single painted halo, q = 20, three filters, 5 sizes, the bracketing nodes with weight > 0): max |delta| = "
    f"{_ut[0]:.1e} sigma (sizes 0.18-45', incl. outside the white node range); per mode/size: {_ut[1]}")
nb_stats = []
for p in range(P0, P0 + NPATCH):
    tp = time.time()
    rng = np.random.default_rng([2026, 11, 12, 2, p])
    h = S.draw(rng)
    keep = h["th"] <= 60.0
    nskip = int((~keep).sum())
    x, y, z, Y, th, M = (np.ascontiguousarray(h[k][keep]) for k in ("x", "y", "z", "Y", "th", "M"))
    ids = np.arange(x.size, dtype=np.int64)
    lth = np.log(th)
    mp = P.paint(G, x, y, Y, th)
    yk = G.rfft(mp)
    del mp
    ybp = blk.take(yk) * blk.phase
    tpaint = time.time() - tp
    out = dict(nhalo=x.size, nskip=nskip, area=G.L ** 2)
    tnu = tdom = 0.0
    nbright = 0
    Rmax = 0.0
    for m in W.MODES:
        mf = MF[m]
        nn = len(mf.th)
        src = ybp
        if m == "white":
            src = ybp.copy()
            src[W.KB, W.KB] -= YBAR * G.n * G.n              # ensemble-mean y removed (white control only)
        sigA = mf.sigma_Y(th)
        q = Y / sigA
        prim = np.nonzero(q >= 0.5)[0]
        f = np.clip((lth[prim] - mf.lth[0]) / mf.dlth, 0.0, nn - 1 - 1e-9)
        jA = f.astype(np.int64)
        jA = np.minimum(jA, nn - 2)
        wA = np.clip(f - jA, 0.0, 1.0)
        npr = prim.size
        d0 = np.zeros(npr); d1 = np.zeros(npr); x0 = np.zeros(npr); x1 = np.zeros(npr)
        lt, R = SELF[m]
        for jn in range(nn):
            lo = (jA == jn) & (wA < 1.0)                     # nodes with zero weight are skipped
            hi = ((jA + 1) == jn) & (wA > 0.0)
            sel = np.nonzero(lo | hi)[0]
            if sel.size == 0:
                continue
            ia = prim[sel]
            ta = time.time()
            v = blk.evaluate(KJ[m][jn] * src, x[ia], y[ia])
            tnu += time.time() - ta
            Rv = R[jn](lth[ia])
            if (lth[ia].min() < lt[jn][0] - 1e-9) or (lth[ia].max() > lt[jn][-1] + 1e-9):
                raise RuntimeError(f"self-response table range exceeded: {m} node {jn}")
            dl = (v - Rv * Y[ia]) / mf.sig[jn]
            # attribution: brightest single neighbour at node jn
            ta = time.time()
            amp = Y / mf.sig[jn]
            xpk = np.interp(lth, LTB, ENV[m][jn][:, 0]) * amp
            br = np.nonzero(xpk >= W.X0_DOM)[0]
            Rb = W.search_radius(ENV[m][jn], lth[br], W.X0_DOM / amp[br])
            Rmax = max(Rmax, float(Rb.max()) if Rb.size else 0.0)
            xm = np.zeros(sel.size)
            W.xmax_pairs(x[ia], y[ia], ids[ia], x[br], y[br], ids[br], lth[br], amp[br], Rb, TAB[m][jn], G.L, xm)
            tdom += time.time() - ta
            nbright += br.size
            ml = lo[sel]
            mh = hi[sel]
            d0[sel[ml]] = dl[ml]; x0[sel[ml]] = xm[ml]
            d1[sel[mh]] = dl[mh]; x1[sel[mh]] = xm[mh]
        out.update({f"{m}_z": z[prim].astype(np.float32), f"{m}_th": th[prim].astype(np.float32), f"{m}_q": q[prim],
                    f"{m}_M": M[prim].astype(np.float32), f"{m}_j": jA.astype(np.int16), f"{m}_w": wA.astype(np.float32),
                    f"{m}_d0": d0.astype(np.float32), f"{m}_d1": d1.astype(np.float32), f"{m}_x0": x0.astype(np.float32),
                    f"{m}_x1": x1.astype(np.float32)})
    np.savez(os.path.join(DMAP, f"patch_{p:04d}.npz"), **out)
    tn = 0.0
    del yk, ybp
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e9
    log(f"   patch {p}: {x.size:,} halos ({nskip} > 60'); primaries q>=0.5: " +
        " ".join(f"{m} {out[m + '_q'].size:,}" for m in W.MODES) +
        f"; bright {nbright:,}; max search radius {Rmax:.1f}'; paint {tpaint:.1f}s nufft {tnu:.1f}s dom {tdom:.1f}s "
        f"total {time.time()-tp:.1f}s [{time.time()-t0:.0f}s]; peak rss {rss:.2f} GB")
log(f"[w02] done {time.time()-t0:.0f}s")
