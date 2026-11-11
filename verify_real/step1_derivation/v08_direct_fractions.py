"""v08: blending fractions counted DIRECTLY on a freshly sampled full sky with MY kernels (v07 tables, my sigma_Y),
my own neighbour search (3D cells on the unit sphere, numba) and my own fraction bookkeeping.
No rate histogram, FFT, HMF integral, builder table or builder code.

Per primary A (q_A = Y500/sigma_Y(theta500) > 0.5 for that instrument; Y500 includes the sampled scatter):
   delta_A = sum_{B != A, s < 5 thA + 5 thB + M0} rho(thA, thB, s) Y_B / sigma_A - mu(thA)
   (mu = catalogue-weighted mean of the truncated kernel: ~0 for the compensated MMF, ybar*DC for the white control)
   x_max = brightest single contribution; dominant if x_max < q_A.
Noise marginalised exactly per object (q_obs = q_A + n + delta, n ~ N(0,1)):
   det_b = dom Phi(q_A+delta-q_c); single = Phi(q_A-q_c); boosted = dom (det_b - single)+ 1(delta>0);
   lost = dom (single - det_b)+ 1(delta<=0) / N_single; satellites = (1-dom) single / N_single;
   Y boosted > eps = det_b 1(delta > eps q_A) / N_det_b; mean Y bias = sum det_b delta/q_A / N_det_b.
Neighbour sets: all (M500 > 1e13), M500 > 1e14 only, sub-threshold (own q_B < 5) only.
Also: random points (delta distribution, variance vs Campbell, robust width of n + delta), redshift tilt,
Gaussian-equivalent counts.  Usage: python v08_direct_fractions.py <seed> <M0> [nsub]
"""
import os, sys, time, resource
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import vlib as V
V.set_threads(4)
import numpy as np
import numba as nb
from scipy.special import ndtr
from scipy.interpolate import CubicSpline

SEED = int(sys.argv[1]) if len(sys.argv) > 1 else 20261111
M0 = float(sys.argv[2]) if len(sys.argv) > 2 else 60.0
NSUB = int(sys.argv[3]) if len(sys.argv) > 3 else 0          # >0: random subsample of primaries (timing / tests)
AM = V.AM
QC = [4.0, 5.0, 6.0, 7.0]
ZB = np.array([0.01, 0.1, 0.25, 0.5, 1.0, 1.5, 2.0, 3.0])
TIER_EDGES = np.array([0.0, 1.5, 5.0, 15.0, 1e9])
TIER_H = np.array([1.0, 1.5, 2.5, 6.0]) * np.pi / 180.0       # 3D cell sizes [rad] per tier
t0 = time.time()


def rss():
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e9


@nb.njit(inline="always")
def s_index(s, NS):
    if s < 8.0:
        f = s / 0.04; k = int(f); return k, f - k
    elif s < 40.0:
        f = (s - 8.0) / 0.2; k = int(f); return 200 + k, f - k
    elif s < 200.0:
        f = (s - 40.0); k = int(f); return 360 + k, f - k
    else:
        f = (s - 200.0) / 4.0; k = int(f)
        if 520 + k > NS - 2:
            return NS - 2, 1.0
        return 520 + k, f - k


@nb.njit(parallel=True)
def sums(px, py, pz, pth, psig, pidx, X, Y_, Z, YB, TB, CB, SB, toff, cst, coff, ncell, hcell, thmax,
         R, lth0, dlth, M0):
    NP = px.shape[0]; NT = R.shape[0]; NS = R.shape[2]
    out = np.zeros((NP, 6))          # D_all, xmax_all, D_14, xmax_14, D_sub, xmax_sub
    npair = np.zeros(NP, np.int64)
    for a in nb.prange(NP):
        xa = px[a]; ya = py[a]; za = pz[a]; tA = pth[a]; sA = psig[a]; ia_self = pidx[a]
        fa = (np.log(tA) - lth0) / dlth
        if fa < 0.0:
            fa = 0.0
        if fa > NT - 1.000001:
            fa = NT - 1.000001
        ia = int(fa); wa = fa - ia
        Dall = 0.0; Mall = -1e300; D14 = 0.0; M14 = -1e300; Dsb = 0.0; Msb = -1e300; npr = 0
        for t in range(toff.shape[0] - 1):
            Rmax = 5.0 * tA + 5.0 * thmax[t] + M0                                   # arcmin
            if Rmax > 10000.0:
                Rmax = 10000.0
            cR = 2.0 * np.sin(0.5 * Rmax * 2.908882086657216e-4)
            cR2 = cR * cR
            n = ncell[t]; h = hcell[t]
            k = int(np.ceil(cR / h))
            ix0 = int((xa + 1.0) / h); iy0 = int((ya + 1.0) / h); iz0 = int((za + 1.0) / h)
            for ix in range(max(ix0 - k, 0), min(ix0 + k, n - 1) + 1):
                for iy in range(max(iy0 - k, 0), min(iy0 + k, n - 1) + 1):
                    base = coff[t] + (ix * n + iy) * n
                    izlo = max(iz0 - k, 0); izhi = min(iz0 + k, n - 1)
                    p0 = cst[base + izlo]; p1 = cst[base + izhi + 1]
                    for p in range(p0, p1):
                        dx = np.float64(X[p]) - xa; dy = np.float64(Y_[p]) - ya; dz = np.float64(Z[p]) - za
                        c2 = dx * dx + dy * dy + dz * dz
                        if c2 > cR2:
                            continue
                        if p == ia_self:
                            continue
                        s = 2.0 * np.arcsin(0.5 * np.sqrt(c2)) / 2.908882086657216e-4
                        tB = np.float64(TB[p])
                        if s >= 5.0 * tA + 5.0 * tB + M0:
                            continue
                        fb = (np.log(tB) - lth0) / dlth
                        if fb < 0.0:
                            fb = 0.0
                        if fb > NT - 1.000001:
                            fb = NT - 1.000001
                        ib = int(fb); wb = fb - ib
                        ks, ws = s_index(s, NS)
                        r = 0.0
                        for da in range(2):
                            fA = wa if da == 1 else 1.0 - wa
                            for db in range(2):
                                fB = wb if db == 1 else 1.0 - wb
                                r += fA * fB * ((1.0 - ws) * R[ia + da, ib + db, ks] + ws * R[ia + da, ib + db, ks + 1])
                        x = r * np.float64(YB[p]) / sA
                        npr += 1
                        Dall += x
                        if x > Mall:
                            Mall = x
                        if CB[p] == 2:
                            D14 += x
                            if x > M14:
                                M14 = x
                        if SB[p]:
                            Dsb += x
                            if x > Msb:
                                Msb = x
        out[a, 0] = Dall; out[a, 1] = Mall; out[a, 2] = D14; out[a, 3] = M14; out[a, 4] = Dsb; out[a, 5] = Msb
        npair[a] = npr
    return out, npair


def global_order(x, y, z, th):
    """Permutation sorting halos by (tier, 3D cell of that tier) + the cell-start tables (positions in the sorted order)."""
    tier = (np.searchsorted(TIER_EDGES, th, side="left") - 1).astype(np.int8)
    key = np.empty(len(th), np.int64)
    off_key = 0; meta = []
    for t in range(len(TIER_EDGES) - 1):
        m = tier == t
        h = TIER_H[t]; n = int(np.ceil(2.0 / h)) + 1
        k = np.minimum(((x[m] + 1.0) / h).astype(np.int64), n - 1)
        k *= n; k += np.minimum(((y[m] + 1.0) / h).astype(np.int64), n - 1)
        k *= n; k += np.minimum(((z[m] + 1.0) / h).astype(np.int64), n - 1)
        key[m] = k + off_key
        meta.append((n, off_key)); off_key += n ** 3 + 1
        del k, m
    perm = np.argsort(key, kind="stable")
    key = key[perm]
    toff = [0]; cst = []; coff = []; ncell = []; thmax = []
    cof = 0
    for t, (n, ok) in enumerate(meta):
        cs = np.searchsorted(key, ok + np.arange(n ** 3 + 1))
        cst.append(cs); coff.append(cof); cof += n ** 3 + 1; ncell.append(n)
        toff.append(int(cs[-1]))
    del key
    tsort = th[perm]
    for t in range(len(meta)):
        seg = tsort[toff[t]:toff[t + 1]]
        thmax.append(float(min(seg.max(), 1e4)) if len(seg) else 0.0)
    del tsort
    return perm, np.array(toff), np.concatenate(cst).astype(np.int64), np.array(coff, np.int64), np.array(ncell, np.int64), \
        np.array(TIER_H), np.array(thmax)


def fractions(qA, delta, xmax, z=None):
    dom = xmax < qA
    out = {}
    for qc in QC:
        ps = ndtr(qA - qc); pb = ndtr(qA + delta - qc); det = dom * pb
        Ns, Nd = ps.sum(), det.sum()
        r = dict(N_single=Ns, N_det_b=Nd, net=Nd / Ns - 1,
                 B2_05=(det * (delta > 0.05 * qA)).sum() / Nd, B2_10=(det * (delta > 0.1 * qA)).sum() / Nd,
                 B2_20=(det * (delta > 0.2 * qA)).sum() / Nd, B2neg_10=(det * (delta < -0.1 * qA)).sum() / Nd,
                 boosted=(dom * np.clip(pb - ps, 0, None) * (delta > 0)).sum() / Nd,
                 lost=(dom * np.clip(ps - pb, 0, None) * (delta <= 0)).sum() / Ns,
                 sat=((~dom) * ps).sum() / Ns, meanbias=(det * delta / qA).sum() / Nd)
        if z is not None and qc == 5.0:
            iz = np.clip(np.searchsorted(ZB, z) - 1, 0, len(ZB) - 2)
            r["zNs"] = np.bincount(iz, ps, len(ZB) - 1); r["zNd"] = np.bincount(iz, det, len(ZB) - 1)
        out[qc] = r
    return out


# ------------------------------------------------------------------ catalogue
from szsim import sampler
cat = sampler.draw_catalog(SEED, fields=("ra", "dec", "z", "M500", "Y500", "theta500"))
ra = np.radians(cat.pop("ra").astype(np.float64)); dec = np.radians(cat.pop("dec").astype(np.float64))
cd = np.cos(dec)
x = (cd * np.cos(ra)).astype(np.float32); y = (cd * np.sin(ra)).astype(np.float32); z3 = np.sin(dec).astype(np.float32)
del ra, dec, cd
Y = cat.pop("Y500"); th = cat.pop("theta500")                       # float32
cls = ((cat.pop("M500") >= 1e14) * 2).astype(np.int8)
zz = cat.pop("z"); del cat
N = len(Y)
A_SKY = 4 * np.pi / AM ** 2
perm, toff, cst, coff, ncell, hcell, thmax = global_order(x, y, z3, th)
x = x[perm]; y = y[perm]; z3 = z3[perm]; Y = Y[perm]; th = th[perm]; cls = cls[perm]; zz = zz[perm]
del perm
print(f"seed {SEED}: N = {N:,}; M0 = {M0}'; tiers (theta edges {TIER_EDGES[1:-1]}): {np.diff(toff).tolist()}; "
      f"cells per axis {ncell.tolist()}; rss {rss():.2f} GB [{time.time()-t0:.0f}s]", flush=True)
SW = np.load(os.path.join(HERE, "sigmaY_mine_wide.npz"))
CF = np.load(os.path.join(HERE, "confusion_mine.npz"))
results = {}
rng = np.random.default_rng(SEED + 7)
for mode in ("baseline", "goal", "white"):
    ts = time.time()
    sp = CubicSpline(np.log(SW["theta500"]), np.log(SW[mode]))
    sig = np.exp(sp(np.log(th.astype(np.float64)))).astype(np.float32)
    q = Y / sig                                                               # float32
    sub = q < 5.0
    T = np.load(os.path.join(HERE, f"rho_table_{mode}.npz"))
    R = np.ascontiguousarray(T["rho"]); TH = T["th"]; S = T["s"]
    lth0, dlth = np.log(TH[0]), np.log(TH[1] / TH[0])
    # catalogue-weighted mean of the truncated kernel, mu(theta_A) [sigma units], on the table nodes
    ring = 2 * np.pi * S * V.trap_w(S)
    ib = np.clip(np.rint((np.log(th) - lth0) / dlth).astype(np.int32), 0, len(TH) - 1)
    WY = np.bincount(ib, Y, len(TH)); WY14 = np.bincount(ib[cls == 2], Y[cls == 2], len(TH)); WYs = np.bincount(ib[sub], Y[sub], len(TH))
    del ib
    mu = {}
    for key, W in (("all", WY), ("14", WY14), ("sub", WYs)):
        mm = np.empty(len(TH))
        for i in range(len(TH)):
            inside = S[None, :] < (5 * TH[i] + 5 * TH + M0)[:, None]
            mm[i] = (W * ((R[i].astype(np.float64) * ring[None, :] * inside).sum(1))).sum() / A_SKY / T["sigY"][i]
        mu[key] = mm
    prim = np.nonzero(q > 0.5)[0]                    # positions in the cell-sorted order (already spatially local)
    if NSUB:
        prim = np.sort(rng.choice(prim, NSUB, replace=False))
    print(f"[{mode}] primaries q>0.5: {len(prim):,}; index built; mu(all) at 1/4/16': "
          f"{np.interp(np.log([1, 4, 16]), np.log(TH), mu['all'])}; rss {rss():.2f} GB [{time.time()-ts:.0f}s]", flush=True)
    tk = time.time()
    out, npair = sums(x[prim].astype(np.float64), y[prim].astype(np.float64), z3[prim].astype(np.float64),
                      th[prim].astype(np.float64), sig[prim].astype(np.float64), prim.astype(np.int64), x, y, z3, Y, th, cls, sub,
                      toff, cst, coff, ncell, hcell, thmax, R, lth0, dlth, M0)
    print(f"[{mode}] neighbour sums: {time.time()-tk:.0f}s, mean pairs per primary {npair.mean():,.0f}; rss {rss():.2f} GB", flush=True)
    qA = q[prim].astype(np.float64); tA = th[prim].astype(np.float64)
    m_all = np.interp(np.log(tA), np.log(TH), mu["all"]); m14 = np.interp(np.log(tA), np.log(TH), mu["14"])
    msb = np.interp(np.log(tA), np.log(TH), mu["sub"])
    res = {"all": fractions(qA, out[:, 0] - m_all, out[:, 1], zz[prim]),
           "1e14": fractions(qA, out[:, 2] - m14, out[:, 3], zz[prim]),
           "sub": fractions(qA, out[:, 4] - msb, out[:, 5], zz[prim])}
    # Gaussian-equivalent selection with my catalogue confusion variances (v06)
    gauss = {}
    for key, ck in (("all", f"{mode}_all"), ("sub", f"{mode}_subthr")):
        v = np.interp(np.log(tA), np.log(CF["thA"]), CF[ck])
        gauss[key] = [ndtr((qA - qc) / np.sqrt(1 + v)).sum() for qc in QC]
    results[mode] = dict(res=res, gauss=gauss)
    for key in ("all", "1e14", "sub"):
        print(f"\n=== [{mode}] neighbours: {key}; seed {SEED}; M0 = {M0}' ===")
        for qc in QC:
            r = res[key][qc]
            print(f"  q>{qc:.0f}: N_single {r['N_single']:,.0f}  N_det_b {r['N_det_b']:,.0f} ({r['net']:+.4f})  Y>5% {r['B2_05']:.4f}  "
                  f"Y>10% {r['B2_10']:.4f}  Y>20% {r['B2_20']:.4f}  Y<-10% {r['B2neg_10']:.4f}  boosted {r['boosted']:.4f}  "
                  f"lost {r['lost']:.4f}  sat {r['sat']:.4f}  meanbias {r['meanbias']:+.4f}")
            if key in gauss:
                Ng = gauss[key][QC.index(qc)]
                print(f"        Gaussian-equivalent N_gauss {Ng:,.0f}: (N_det_b - N_gauss)/N_gauss = {r['N_det_b']/Ng-1:+.4f}")
        r5 = res[key][5.0]
        print(f"  q>5 Delta N/N in z bins {ZB.tolist()}: " + " ".join(f"{a/b-1:+.4f}" for a, b in zip(r5['zNd'], r5['zNs'])))
    # random points: delta distribution at the builder's theta_A values
    if not NSUB:
        NR = 20000
        for tAr in (1.0069, 4.025, 16.0897):
            rr = rng.random(NR); dd = np.arcsin(2 * rng.random(NR) - 1); aa = 2 * np.pi * rr
            rx = np.cos(dd) * np.cos(aa); ry = np.cos(dd) * np.sin(aa); rz = np.sin(dd)
            sA = float(np.exp(sp(np.log(tAr))))
            o2, _ = sums(rx, ry, rz, np.full(NR, tAr), np.full(NR, sA), np.full(NR, -1, np.int64), x, y, z3, Y, th, cls, sub,
                         toff, cst, coff, ncell, hcell, thmax, R, lth0, dlth, M0)
            for key, col, mk in (("all", 0, "all"), ("sub", 4, "sub")):
                d = o2[:, col] - np.interp(np.log(tAr), np.log(TH), mu[mk])
                # robust width of n + delta: CDF(t) = mean Phi(t - delta)
                tg = np.linspace(-4, 4, 4001)
                cdf = np.array([ndtr(t - d).mean() for t in tg])
                q16, q84 = np.interp([0.15866, 0.84134], cdf, tg)
                vC = np.interp(np.log(tAr), np.log(CF["thA"]), CF[f"{mode}_{'all' if key == 'all' else 'subthr'}"])
                qd = np.quantile(d, [0.02275, 0.15866, 0.5, 0.84134, 0.97725])
                print(f"  [{mode}] random points thA={tAr:.3f}' ({key}): mean {d.mean():+.4f}  var {d.var():.4f} (Campbell, v06 sky: {vC:.4f})  "
                      f"quantiles {np.round(qd, 3)}  robust half-width of n+delta {(q84-q16)/2:.4f}  "
                      f"P(d>0.5) {np.mean(d>0.5):.4f} P(d<-0.5) {np.mean(d<-0.5):.4f}", flush=True)
    del out, sig, q, sub, R
    print(f"[{mode}] done {time.time()-ts:.0f}s; peak rss {rss():.2f} GB", flush=True)
np.save(os.path.join(HERE, f"v08_results_seed{SEED}_M0{int(M0)}{'_sub'+str(NSUB) if NSUB else ''}.npy"), results, allow_pickle=True)
print(f"total {time.time()-t0:.0f}s; peak rss {rss():.2f} GB")
