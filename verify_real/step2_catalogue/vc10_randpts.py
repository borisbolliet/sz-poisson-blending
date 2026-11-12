"""vc10: random points on one fresh sky (seed 5555): the map-sigma ratios of the build (claim C11).
   r_ii(theta_A) = robust (84th - 16th)/2 half-width of n + delta - mu at random positions, all halos M500 > 1e13
   r_i(theta_A)  = sqrt(1 + var(delta_sub)),  delta_sub = sum over neighbours with own q_B = Y_B / sigma_Y(theta_B) < 5
delta from my neighbour sums (vclib.point_sums: own index, derivation's rho tables, M0 = 30'); the noise n ~ N(0, 1) is
added analytically (CDF(t) = mean Phi(t - delta_i)).  mu = my expected truncated-kernel mean (vc01).  NR points per
theta_A; errors of the quantiles by 20 independent sub-samples.  Compared with the build's inputs.npz r_i, r_ii.
"""
import os, sys, time
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import vclib as V
import numpy as np
from scipy.special import ndtr
from scipy.optimize import brentq

t0 = time.time()
NR = 100_000
THA = np.array([0.5, 1.0, 2.0, 4.0, 8.0, 16.0, 32.0])
R3, TH, S, sigtab = V.load_rho3()
lth0, dlth = np.log(TH[0]), np.log(TH[1] / TH[0])
MU = np.load(f"{HERE}/out/expected_mean.npz")["num"]
from szsim import sampler
cat = sampler.draw_catalog(5555, fields=("ra", "dec", "Y500", "theta500"))
N = len(cat["Y500"])
perm, tier_off, band_off, band_base, nband, thmax = V.build_index(cat["ra"], cat["dec"], cat["theta500"])
ra = cat.pop("ra")[perm]; dec = cat.pop("dec")[perm]; th = cat.pop("theta500")[perm]; Y = cat.pop("Y500")[perm]
del cat, perm
X = np.empty(N, np.float32); Yv = np.empty(N, np.float32); Zv = np.empty(N, np.float32)
IB = np.empty(N, np.int32); WB = np.empty(N, np.float32)
SUBb = np.empty(N, np.int8); SUBg = np.empty(N, np.int8)
for a in range(0, N, 4_000_000):
    sl = slice(a, a + 4_000_000)
    rad = np.radians(ra[sl].astype(np.float64)); dd = np.radians(dec[sl].astype(np.float64))
    X[sl] = np.cos(dd) * np.cos(rad); Yv[sl] = np.cos(dd) * np.sin(rad); Zv[sl] = np.sin(dd)
    t64 = th[sl].astype(np.float64)
    fb = np.clip((np.log(t64) - lth0) / dlth, 0, len(TH) - 1.000001)
    IB[sl] = fb.astype(np.int32); WB[sl] = fb - IB[sl]
    SUBb[sl] = Y[sl] / V.sigma_Y("baseline", t64) < 5.0
    SUBg[sl] = Y[sl] / V.sigma_Y("goal", t64) < 5.0
print(f"sky seed 5555: {N:,} halos; sub-threshold fraction baseline {SUBb.mean():.4f}, goal {SUBg.mean():.4f} "
      f"[{time.time()-t0:.0f}s]", flush=True)
rng = np.random.default_rng(5556)
I = np.load(f"{V.ROOT}/data_real/step2/inputs.npz")
QL = [0.15865525, 0.84134475]


def quant(d, p):
    f = lambda t: ndtr(t - d).mean() - p
    return brentq(f, -10, 10, xtol=1e-7)


for run, fi, SUB in (("baseline", 1, SUBb), ("goal", 2, SUBg)):
    print(f"\n[{run}] theta_A   r_ii mine (+- err)   build   | r_i mine (+- err)  build  | var_all / sigma^2 mine  build (Parseval)")
    for tA in THA:
        u = rng.random(NR); v = rng.random(NR)
        pra = 360.0 * u; pdec = np.degrees(np.arcsin(2 * v - 1))
        r_ = np.radians(pra); d_ = np.radians(pdec)
        px = np.cos(d_) * np.cos(r_); py = np.cos(d_) * np.sin(r_); pz = np.sin(d_)
        pth = np.full(NR, tA)
        fa = np.clip((np.log(pth) - lth0) / dlth, 0, len(TH) - 1.000001); IA = fa.astype(np.int32); WA = fa - IA
        sig3 = np.tile([float(V.sigma_Y(f, tA)) for f in V.FILTERS], (NR, 1))
        o = V.point_sums(pra, pdec, px, py, pz, pth, ra, dec, X, Yv, Zv, Y, th, IB, WB, SUB, band_off, band_base, nband,
                         V.TIER_H, thmax, R3, IA, WA, sig3, V.M0)
        mu = np.interp(np.log(tA), np.log(TH), MU[fi]) / sig3[0, fi]
        d = o[:, fi] - mu
        q16, q84 = quant(d, QL[0]), quant(d, QL[1])
        rii = 0.5 * (q84 - q16)
        sub = np.array_split(rng.permutation(NR), 20)
        rs = [0.5 * (quant(d[s], QL[1]) - quant(d[s], QL[0])) for s in sub]
        e_rii = np.std(rs, ddof=1) / np.sqrt(len(sub))
        ds = o[:, 3 + fi]
        vs = ds.var(ddof=1); e_vs = np.std([ds[s].var(ddof=1) for s in sub], ddof=1) / np.sqrt(len(sub))
        ri = np.sqrt(1 + vs); e_ri = 0.5 * e_vs / ri
        rb_ii = np.interp(np.log(tA), np.log(I[f"{run}_th"]), I[f"{run}_r_ii"])
        rb_i = np.interp(np.log(tA), np.log(I[f"{run}_th"]), I[f"{run}_r_i"])
        va = d.var(ddof=1); vb = np.interp(np.log(tA), np.log(I[f"{run}_th"]), I[f"{run}_vpars_all"])
        print(f"   {tA:5.1f}'   {rii:.4f} +- {e_rii:.4f}   {rb_ii:.4f}   | {ri:.4f} +- {e_ri:.4f}  {rb_i:.4f} | {va:.3f}  {vb:.3f}"
              f"   [{time.time()-t0:.0f}s]", flush=True)

# ------------------------------------------------------------------ (2) the build's kernel at single theta_A nodes
print("\n(2) the build's blended kernel K(theta_A, q_A; e) = P(dominant and q_A + n + delta - mu > e) at single theta_A nodes "
      "vs my random-point sums (same quantity: mean over points of 1(xmax < fdom q_A) Phi(q_A + delta - mu - e))")
D2 = f"{V.ROOT}/data_real/step2"
KB = {n: np.load(f"{D2}/kernel_{n}.npz") for n in ("baseline_f1", "baseline_f0.5", "baseline_f2", "goal_f1")}
thn = KB["baseline_f1"]["th"]; QA = KB["baseline_f1"]["QA"]; E = KB["baseline_f1"]["EDGES"]
QSEL = [int(np.argmin(np.abs(QA - q))) for q in (1.0, 2.0, 3.0, 3.5, 4.0, 4.5, 5.0, 6.0, 7.0, 8.0, 10.0, 15.0, 20.0)]
for inode, npts in ((7, 1_000_000), (8, 1_000_000), (15, 200_000), (24, 200_000)):
    tA = float(thn[inode])
    u = rng.random(npts); v = rng.random(npts)
    pra = 360.0 * u; pdec = np.degrees(np.arcsin(2 * v - 1))
    r_ = np.radians(pra); d_ = np.radians(pdec)
    px = np.cos(d_) * np.cos(r_); py = np.cos(d_) * np.sin(r_); pz = np.sin(d_)
    pth = np.full(npts, tA)
    fa = np.clip((np.log(pth) - lth0) / dlth, 0, len(TH) - 1.000001); IA = fa.astype(np.int32); WA = fa - IA
    sig3 = np.tile([float(V.sigma_Y(f, tA)) for f in V.FILTERS], (npts, 1))
    o = V.point_sums(pra, pdec, px, py, pz, pth, ra, dec, X, Yv, Zv, Y, th, IB, WB, SUBb, band_off, band_base, nband,
                     V.TIER_H, thmax, R3, IA, WA, sig3, V.M0)
    grp = rng.integers(0, 20, npts)
    print(f"  theta_A = {tA:.4f}' (build node {inode}), {npts:,} random points  [{time.time()-t0:.0f}s]", flush=True)
    for name, fi, fdom in (("baseline_f1", 1, 1.0), ("baseline_f0.5", 1, 0.5), ("baseline_f2", 1, 2.0), ("goal_f1", 2, 1.0)):
        K = KB[name]
        assert np.allclose(K["th"], thn) and np.allclose(K["QA"], QA)
        mu = np.interp(np.log(tA), np.log(TH), MU[fi]) / sig3[0, fi]
        d = o[:, fi] - mu; xm = o[:, 6 + fi]
        worst = (0.0, None); rows = []
        for k in QSEL:
            q = QA[k]
            dom = xm < fdom * q
            P = dom[:, None] * ndtr(q + d[:, None] - E[None, :])            # (npts, 19)
            m = P.mean(0)
            gm = np.array([P[grp == g].mean(0) for g in range(20)])
            err = gm.std(0, ddof=1) / np.sqrt(20)
            kb = K["Kb"][inode, k]
            ok = (kb > 1e-4) & (kb < 0.9999) & (err > 0)
            pull = np.where(ok, (m - kb) / np.where(ok, err, 1), 0.0)
            j = int(np.argmax(np.abs(pull)))
            if abs(pull[j]) > abs(worst[0]):
                worst = (pull[j], (q, E[j], m[j], kb[j], err[j]))
            rows.append((q, dom.mean(), K["Pdom"][inode, k], m, kb, err))
        print(f"    [{name}] max |pull| over q_A in {{1..20}}, 19 edges: {worst[0]:+.2f} at q_A = {worst[1][0]:.2f}, e = {worst[1][1]:.2f} "
              f"(mine {worst[1][2]:.5f} +- {worst[1][4]:.5f}, build {worst[1][3]:.5f})", flush=True)
        for q, pd_m, pd_b, m, kb, err in rows:
            if q in (QA[QSEL[2]], QA[QSEL[4]], QA[QSEL[6]]):
                je = [1, 2]                                                  # e = 5, 6
                print(f"       q_A = {q:.2f}: P_dom mine {pd_m:.5f} build {pd_b:.5f}; K(e=5) mine {m[1]:.5f} +- {err[1]:.5f} build "
                      f"{kb[1]:.5f}; K(e=6) mine {m[2]:.5f} +- {err[2]:.5f} build {kb[2]:.5f}; naive Phi(q - 5) {ndtr(q-5):.5f}")
print(f"total {time.time()-t0:.0f}s; peak rss {V.rss_gb():.2f} GB")
