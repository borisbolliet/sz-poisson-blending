"""x07 (indep lens): the build's stored per-tile arrays (data_real/step3/tiles, truth) run through OWN code for the
definitions of RESULTS.md 'Terms' (written from the text, not from t05_analysis.py), to test the build's analysis logic
separately from its sky statistics.  Plus small own computations on the build's tiles:
  T0  halos per tile, theta500 range, halos with q_true > 5 per tile (own sigma_Y call);
  R1, R2, R6, F1-F5, T1-T3, N1-N3, N5-N7 recomputed from the stored arrays (x_max at the true positions with the own
      step-1 rho table; N_exp with the build's selection fit and the build's r_inj, as the build does);
  D3  closure (d_S_tsz - d_tab_sum) / sigma from the stored arrays;
  W1  isolated-search window maximum within 1' of the halo (stored offsets);
  A2  number of q > 4 peaks of tile 0 (baseline, all halos) with the analytic covariance, own maps (build t01: 508).
Usage: python x07_build_arrays.py [NTILES]
"""
import os, sys, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
from scipy.special import ndtr
import vlib3 as V

NT = int(sys.argv[1]) if len(sys.argv) > 1 else 324
log = V.Tee(os.path.join(V.HERE, "x07_build_arrays.log"))
t00 = time.time()
TH = V.THETAS
ZB = np.array([0.01, 0.1, 0.5, 1.0, 1.5, 2.0, 3.0])
SELB = dict(np.load(os.path.join(V.BUILD_DATA, "selection.npz")))
TH_INJ = SELB["theta_inj"]
rng = np.random.default_rng(777)


def p_sel(fits, q, th):
    P = np.array([ndtr((np.sqrt(np.maximum(q * q + b, 0.0)) - 5.0 - mu) / np.exp(ls)) for b, mu, ls in fits])
    x = np.clip(np.log(th), np.log(TH_INJ[0]), np.log(TH_INJ[-1]))
    i = np.clip(np.searchsorted(np.log(TH_INJ), x) - 1, 0, len(TH_INJ) - 2)
    w = (x - np.log(TH_INJ[i])) / np.log(TH_INJ[i + 1] / TH_INJ[i])
    n = np.arange(q.size)
    return (1 - w) * P[i, n] + w * P[i + 1, n]


def rinj_build(mode):
    d = os.path.join(V.BUILD_DATA, "injections")
    a = [np.load(os.path.join(d, f))["sig"] for f in sorted(os.listdir(d)) if f.startswith(f"inj_{mode}_")]
    return np.mean(a, 0) / V.sigma_Y(mode, TH)


def stats(pid, mode, rinj, rho):
    r = V.build_tile(pid); h = V.build_truth(pid)
    p = mode + "_"
    Y, th, z, hy, hx = h["Y500"], h["theta500"], h["z"], h["y"], h["x"]
    qt = Y / V.sigma_Y(mode, th)
    sY = V.sigma_Y(mode, TH)
    sig = r[p + "it1_sig"]
    q, j, iy, ix = r[p + "it1_q"], r[p + "it1_j"], r[p + "it1_iy"], r[p + "it1_ix"]
    d5 = np.flatnonzero(q > 5)
    ti, ex = r[p + "d_top_idx"][d5], r[p + "d_top_ex"][d5]
    prim, cp = ti[:, 0], ex[:, 0]
    c2 = np.where(np.isfinite(ex[:, 1]), ex[:, 1], 0.0); i2 = ti[:, 1]
    sd = sig[j[d5]]
    delta = r[p + "d_S_tsz"][d5] - cp
    nz = r[p + "d_S_cmb"][d5] + r[p + "d_S_noise"][d5]
    real = cp / sd >= 2
    seen = set(); main = np.zeros(d5.size, bool)
    for a in np.argsort(-q[d5]):
        if real[a] and prim[a] not in seen:
            main[a] = True; seen.add(int(prim[a]))
    credited = np.array(sorted(seen), np.int64)
    fidx, fqa = r[p + "f_idx"], r[p + "f_q_alone"]
    qa = fqa[np.searchsorted(fidx, prim)]
    rth = TH[j[d5]] / th[prim]; wrong = (rth > 2) | (rth < 0.5)
    o = dict(r0=r[p + "it0_sig"] / sY, r1=sig / sY, n5=d5.size, n5_it0=int((r[p + "it0_q"] > 5).sum()),
             none=int((~real).sum()), frag=int((real & ~main).sum()), main=int(main.sum()),
             b10=int((main & (delta > 0.1 * cp)).sum()), b20=int((main & (delta > 0.2 * cp)).sum()),
             needs=int((main & (qa < 5)).sum()), merge=int((main & (i2 >= 0) & (qt[np.maximum(i2, 0)] > 5) & ~np.isin(i2, credited) & (c2 > 0.1 * cp)).sum()),
             wrong=int((main & wrong).sum()))
    alone5 = fidx[fqa > 5]
    o.update(n_alone5=alone5.size, lost=int((~np.isin(alone5, credited)).sum()))
    fs = sig[r[p + "f_j"]]
    own = r[p + "f_own"] / fs
    fn_ = (r[p + "f_S_cmb"] + r[p + "f_S_noise"]) / fs
    fdl = (r[p + "f_S_tsz"] - r[p + "f_own"]) / fs
    single = own + fn_ > 5
    blend = (own + fn_ + fdl > 5) & (r[p + "f_nb_val"][:, 0] / fs < own)
    o.update(fp_single=int(single.sum()), fp_blend=int(blend.sum()), fp_b10=int((blend & (fdl > 0.1 * own)).sum()),
             fp_boost=int((blend & ~single).sum()), fp_lost=int((single & ~blend).sum()))
    # attribution with the own step-1 rho table
    att = main & (c2 >= 0.1 * cp)
    A, B = prim[att], i2[att]
    hq3 = np.flatnonzero(qt > 3)
    xi = np.unique(np.concatenate([hq3, A, B])).astype(np.int64)
    xm = np.zeros(xi.size); xa = np.zeros(xi.size, np.int64)
    V.xmax_true(xi, hy, hx, Y, np.log(th), V.sigma_Y(mode, th[xi]), rho.tab, np.log(V.Rho1.TH0), V.Rho1.D,
                V.Rho1.S[1] - V.Rho1.S[0], 60.0, V.LBOX, xm, xa)
    xof = lambda hh: xm[np.searchsorted(xi, hh)]
    o["att"] = (c2[att] / cp[att], qt[A], qt[B], xof(A), xof(B))
    o["dom"] = (xof(hq3) / qt[hq3], np.isin(hq3, credited), fqa[np.searchsorted(fidx, hq3)] > 5)
    # counts with the build's selection
    o["Ndet"] = np.histogram(z[credited], ZB)[0].astype(float)
    o["Nalone"] = np.histogram(z[alone5], ZB)[0].astype(float)
    sigfin = np.interp(np.log(th), np.log(TH), o["r1"]) * V.sigma_Y(mode, th)
    qp = Y / sigfin * np.interp(np.log(th), np.log(TH), rinj)
    o["Nexp"] = np.histogram(z, ZB, weights=p_sel(SELB[mode + "_fit"], qp, th))[0]
    for mn in V.MERGES:
        k = V.apply_merge(iy * V.PIX, ix * V.PIX, q, mn, 5.0)
        pos = np.searchsorted(d5, k)
        o["Ndet_" + mn] = np.histogram(z[np.unique(prim[pos[real[pos]]])], ZB)[0].astype(float)
    # D3, W1
    o["closure"] = (r[p + "d_S_tsz"] - r[p + "d_tab_sum"])[d5] / sd
    dpx = np.hypot(r[p + "f_q_alone_dy"], r[p + "f_q_alone_dx"]) * V.PIX
    o["w1"] = (dpx[fqa > 5] <= 1.0)
    o["T0"] = (h["y"].size, th.min(), th.max(), int((qt > 5).sum()))
    return o


def boot(num, den):
    T = num.sum() / den.sum() - 1
    b = rng.integers(0, num.shape[0], (2000, num.shape[0]))
    return T, (num[b].sum(1).sum(1) / den[b].sum(1).sum(1) - 1).std(), num.sum(0) / den.sum(0) - 1


log(f"x07: build's stored arrays, {NT} tiles, through own code for the definitions")
for mode in V.MODES:
    rho = V.Rho1(mode)
    rinj = rinj_build(mode)
    S = [stats(p, mode, rinj, rho) for p in range(NT)]
    A = {k: np.array([s[k] for s in S]) for k in S[0] if k not in ("att", "dom", "closure", "w1", "T0")}
    log(f"\n==================== {mode}")
    T0 = np.array([s["T0"] for s in S])
    log(f"  T0: halos per tile {T0[:, 0].min():.0f}-{T0[:, 0].max():.0f}; theta500 {T0[:, 1].min():.3f}'-{T0[:, 2].max():.1f}'; q_true > 5 per tile {T0[:, 3].mean():.1f}")
    r0, r1 = A["r0"].mean(0), A["r1"].mean(0)
    log(f"  R1/R2: r before {r0.min():.4f}-{r0.max():.4f}, after {r1.min():.4f}-{r1.max():.4f}; R6: q > 5 before {A['n5_it0'].sum()} -> after {A['n5'].sum()}")
    n5, nm = A["n5"].sum(), A["main"].sum()
    pc = lambda a, b: f"{100 * a / b:.2f}%"
    log(f"  F: main {nm}; none {pc(A['none'].sum(), n5)}, fragments {pc(A['frag'].sum(), n5)}, b10 {pc(A['b10'].sum(), nm)}, b20 {pc(A['b20'].sum(), nm)}, "
        f"only thanks to neighbours {pc(A['needs'].sum(), nm)}, merges {pc(A['merge'].sum(), nm)}, wrong size {pc(A['wrong'].sum(), nm)}, "
        f"lost {pc(A['lost'].sum(), A['n_alone5'].sum())} (of {A['n_alone5'].sum()})")
    nb_, ns_ = A["fp_blend"].sum(), A["fp_single"].sum()
    log(f"  F5: forced Y > 10% {pc(A['fp_b10'].sum(), nb_)}, boosted {pc(A['fp_boost'].sum(), nb_)}, lost {pc(A['fp_lost'].sum(), ns_)}, change {100 * (nb_ / ns_ - 1):+.2f}%")
    att = [np.concatenate([s["att"][k] for s in S]) for k in range(5)]
    rat, qA, qB, xA, xB = att
    log(f"  T1: contested {rat.size} ({100 * rat.size / nm:.1f}% of main)")
    for lo, hi in ((0.1, 0.25), (0.25, 0.5), (0.5, 1.01)):
        m = (rat >= lo) & (rat < hi)
        line = f"     c_B/c_A {lo}-{min(hi, 1)}: N {m.sum()}, A larger true q {100 * np.mean(qA[m] > qB[m]):.0f}%"
        for fd in (0.5, 1.0, 2.0):
            ca = xA[m] < fd * qA[m]; cb = xB[m] < fd * qB[m]
            line += f"; fdom {fd}: A {100 * np.mean(ca & ~cb):.0f} B {100 * np.mean(~ca & cb):.0f} both {100 * np.mean(ca & cb):.0f} neither {100 * np.mean(~ca & ~cb):.0f}"
        log(line)
    dr = np.concatenate([s["dom"][0] for s in S]); dc = np.concatenate([s["dom"][1] for s in S]); da = np.concatenate([s["dom"][2] for s in S])
    line = "  T3: credited fraction by x_max/q:"
    for lo, hi in ((0, 0.1), (0.1, 0.25), (0.25, 0.5), (0.5, 0.75), (0.75, 1.0), (1.0, 1.5), (1.5, 1e9)):
        m = da & (dr >= lo) & (dr < hi)
        line += f" [{lo},{hi if hi < 1e8 else 'inf'}) {m.sum()} {100 * dc[m].mean():.1f}%;"
    log(line)
    T, e, zb = boot(A["Ndet"], A["Nexp"])
    log(f"  N1: {100 * T:+.2f} +- {100 * e:.2f}%; z bins " + ", ".join(f"{100 * v:+.2f}" for v in zb))
    T1, e1, _ = boot(A["Ndet"], A["Nalone"]); T2, e2, _ = boot(A["Nalone"], A["Nexp"])
    log(f"  N2: split 1 {100 * T1:+.2f} +- {100 * e1:.2f}%, split 2 {100 * T2:+.2f} +- {100 * e2:.2f}%")
    log("  N5/N6: credited with merging vs none: " + ", ".join(f"{mn} {100 * (A['Ndet_' + mn].sum() / A['Ndet'].sum() - 1):+.2f}% (Delta/N {100 * (A['Ndet_' + mn].sum() / A['Nexp'].sum() - 1):+.2f}%)" for mn in V.MERGES))
    cl = np.abs(np.concatenate([s["closure"] for s in S]))
    log(f"  D3: closure median {np.median(cl):.1e} sigma, 95% {np.percentile(cl, 95):.1e}, max {cl.max():.1e}")
    w1 = np.concatenate([s["w1"] for s in S])
    log(f"  W1: window maximum within 1' of the halo's pixel for {100 * w1.mean():.2f}% of the q_alone > 5 halos")

# A2-type: analytic-covariance S/N maps of tile 0 (baseline, all halos): number of q > 4 peaks
G = V.grid()
tr = V.build_truth(0)
P2 = V.Painter(G, eps=1e-9, scheme="mix2")
X = V.tsz_channels(P2.T(tr["y"], tr["x"], tr["Y500"], tr["theta500"]), G) + V.cmb_build_rng(300000, G) + V.noise_build_rng(400000, "baseline", G)
Fa = V.MMF(G, mode="baseline", analytic=True)
det = V.detect(Fa, Fa.combine(X))
log(f"\nA2: tile 0 baseline, analytic covariance, own S/N maps: {det['q'].size} peaks q > 4 (build t01 (d2): 508 detections)")
log(f"done [{time.time() - t00:.0f} s], peak RSS {V.peak_rss_gb():.2f} GB")
