"""vc01: one fresh full-sky Poisson halo realisation -> direct per-object blended and single-cluster (z, q_obs) histograms.

Usage: vc01_sky.py <seed> [test]
  * sky: szsim.sampler.draw_catalog(seed), M500 > 1e13, 0.01 < z < 3 (29.3 M halos);
  * primaries: every halo with max_f q_med,f >= QLO (weight 1) plus a random fraction FSUB of the others (weight 1/FSUB);
    q_med = Y_med(M, z) / sigma_f(theta500) (the primary's own scatter is integrated per object, K-point shifted lattice);
  * neighbour sums with my own index (dec bands sorted in RA, tiers in theta_B), the derivation's rho tables, M0 = 30';
  * per-object noise marginalisation (normal CDFs), dominance x_B < fdom q_A, expected-mean subtraction (c = 1).
Output: out/sky_<seed>.npz with HB (48 blocks, 13 blended variants, 16 z, 17 edges) and HS (48, 9 single variants, ...).
'test': 3000 random primaries only, plus a brute-force check of the neighbour sums for 40 primaries.
"""
import os, sys, time
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import vclib as V                                   # sets the thread env vars (VC_THREADS, default 4) before numba
import numpy as np
import numba as nb

SEED = int(sys.argv[1])
TEST = len(sys.argv) > 2 and sys.argv[2] == "test"
S8T = float(sys.argv[3]) if len(sys.argv) > 3 else None      # optional: sigma8 of the truth (Om fixed), e.g. 0.7365
QLO, FSUB, K = 0.4, 0.03, 12
OUT = f"{HERE}/out"
os.makedirs(OUT, exist_ok=True)
t0 = time.time()

R3, TH, S, sigtab = V.load_rho3()
V.s_nodes_check(S)
lth0, dlth = np.log(TH[0]), np.log(TH[1] / TH[0])
TAB = None if S8T is None else V.table_at_sigma8(S8T)
fn_mu = f"{OUT}/expected_mean.npz" if S8T is None else f"{OUT}/expected_mean_s8{S8T}.npz"
if not os.path.exists(fn_mu):
    num, WY = V.expected_mean_numerators(R3, TH, S, TAB)
    np.savez(fn_mu, num=num, WY=WY, TH=TH, M0=V.M0)
MU = np.load(fn_mu)["num"]                                   # (3, NT) numerators, / sigma_f(thA) per primary
rth, RT = V.r_tables()
_I = np.load(f"{V.ROOT}/data_real/step2/inputs.npz")                # read as data, for the diagnostic variant only
MUB_TH, MUB = _I["white_th"], _I["white_mu"]

from szsim import sampler, signal as sg
cat = sampler.draw_catalog(SEED, fields=("ra", "dec", "z", "M500", "Y500", "theta500"), table=TAB)
N = int(cat["meta"]["N"]); Nexp = float(cat["meta"]["N_expected"])
perm, tier_off, band_off, band_base, nband, thmax = V.build_index(cat["ra"], cat["dec"], cat["theta500"])
ra = cat.pop("ra")[perm]; dec = cat.pop("dec")[perm]; th = cat.pop("theta500")[perm]; Y = cat.pop("Y500")[perm]
zz = cat.pop("z")[perm]; MM = cat.pop("M500")[perm]
del cat, perm
X = np.empty(N, np.float32); Yv = np.empty(N, np.float32); Zv = np.empty(N, np.float32)
IB = np.empty(N, np.int32); WB = np.empty(N, np.float32)
for a in range(0, N, 4_000_000):                                   # chunked (memory)
    sl = slice(a, a + 4_000_000)
    rad = np.radians(ra[sl].astype(np.float64)); dd = np.radians(dec[sl].astype(np.float64))
    X[sl] = np.cos(dd) * np.cos(rad); Yv[sl] = np.cos(dd) * np.sin(rad); Zv[sl] = np.sin(dd)
    fb = np.clip((np.log(th[sl].astype(np.float64)) - lth0) / dlth, 0, len(TH) - 1.000001)
    IB[sl] = fb.astype(np.int32); WB[sl] = fb - IB[sl]
del rad, dd, fb
print(f"seed {SEED}: N = {N:,} (expected {Nexp:,.0f}); tiers (theta edges {V.TIER_TH[1:-1].tolist()}): "
      f"{np.diff(tier_off).tolist()}, theta max per tier {np.round(thmax, 2).tolist()}; rss {V.rss_gb():.2f} GB "
      f"[{time.time()-t0:.0f}s]", flush=True)

# ------------------------------------------------------------------ primaries
m = sg.SZModel()
qmax = np.zeros(N, np.float32)
CH = 4_000_000
for a in range(0, N, CH):
    sl = slice(a, a + CH)
    ym = m.Y500_med(MM[sl].astype(np.float64), zz[sl].astype(np.float64))
    for f in V.FILTERS:
        qmax[sl] = np.maximum(qmax[sl], ym / V.sigma_Y(f, th[sl].astype(np.float64)))
rng = np.random.default_rng(SEED + 12345)
if TEST:
    pidx = np.sort(rng.choice(np.nonzero(qmax >= 0.2)[0], 3000, replace=False)); wgt = np.ones(len(pidx))
else:
    full = qmax >= QLO
    sub = (~full) & (rng.random(N) < FSUB)
    pidx = np.nonzero(full | sub)[0]
    wgt = np.where(full[pidx], 1.0, 1.0 / FSUB)
    del full, sub
NP = len(pidx)
thA = th[pidx].astype(np.float64)
ymA = m.Y500_med(MM[pidx].astype(np.float64), zz[pidx].astype(np.float64))
zA = zz[pidx].astype(np.float64)
del qmax, MM, zz                                                           # memory: not needed any more
sig3 = np.stack([V.sigma_Y(f, thA) for f in V.FILTERS], 1)                 # (NP, 3)
qmed = (ymA[:, None] / sig3).T.copy()                                      # (3, NP)
fa = np.clip((np.log(thA) - lth0) / dlth, 0, len(TH) - 1.000001); IA = fa.astype(np.int32); WA = fa - IA
mu = np.stack([np.interp(np.log(thA), np.log(TH), MU[f]) / sig3[:, f] for f in range(3)]
              + [np.interp(np.log(thA), np.log(MUB_TH), MUB)])                                # (4, NP); row 3: build's white mu
print(f"primaries: {NP:,} (q_med >= {QLO} in some filter: {int((wgt == 1).sum()):,}; subsampled {int((wgt > 1).sum()):,} "
      f"x {1/FSUB:.0f}); mu (expected, white) at thA = 1, 4, 16': "
      + " ".join(f"{np.interp(np.log(t), np.log(TH), MU[0]) / V.sigma_Y('white', t):.4f}" for t in (1.0, 4.0, 16.0))
      + f"; rss {V.rss_gb():.2f} GB [{time.time()-t0:.0f}s]", flush=True)

# ------------------------------------------------------------------ neighbour sums
order = rng.permutation(NP).astype(np.int64)
ts = time.time()
res, npair = V.neighbour_sums(order, pidx.astype(np.int64), ra, dec, X, Yv, Zv, Y, th, IB, WB, band_off, band_base,
                              nband, V.TIER_H, thmax, R3, IA, WA, sig3, V.M0)
tsum = time.time() - ts
print(f"neighbour sums: {tsum:.0f}s, {npair.sum():,} pairs ({npair.mean():,.0f} per primary); rss {V.rss_gb():.2f} GB",
      flush=True)
delta = res[:, :3].T.copy(); xmax = res[:, 3:].T.copy()
del res

if TEST:
    # brute force for 40 primaries over ALL halos (vectorised numpy, independent of the index)
    am = V.AM
    worst = 0.0
    for k in rng.choice(NP, 40, replace=False):
        a = pidx[k]
        sel_l = []; ss_l = []
        for c0 in range(0, N, 2_000_000):                     # chunked over all halos (memory)
            sl = slice(c0, c0 + 2_000_000)
            c2 = ((X[sl].astype(np.float64) - X[a]) ** 2 + (Yv[sl].astype(np.float64) - Yv[a]) ** 2
                  + (Zv[sl].astype(np.float64) - Zv[a]) ** 2)
            s_ = 2 * np.arcsin(0.5 * np.sqrt(c2)) / am
            w_ = np.nonzero(s_ < 5 * thA[k] + 5 * th[sl].astype(np.float64) + V.M0)[0]
            sel_l.append(w_ + c0); ss_l.append(s_[w_])
        sel = np.concatenate(sel_l); ss = np.concatenate(ss_l)
        keep = sel != a
        sel = sel[keep]; ss = ss[keep]
        tb = th[sel].astype(np.float64)
        fbb = np.clip((np.log(tb) - lth0) / dlth, 0, len(TH) - 1.000001); ib = fbb.astype(int); wb = fbb - ib
        ia, wa = IA[k], WA[k]
        # s index (piecewise uniform) in numpy
        ks = np.where(ss < 8, ss / 0.04, np.where(ss < 40, 200 + (ss - 8) / 0.2, np.where(ss < 200, 360 + (ss - 40), 520 + (ss - 200) / 4)))
        kk = np.minimum(ks.astype(int), 669); ws = np.where(ks >= 670, 1.0, ks - kk)
        dB = np.zeros(3); mB = np.full(3, -1e300)
        for f in range(3):
            r = 0.0
            for da, fA in ((0, 1 - wa), (1, wa)):
                for db in (0, 1):
                    fB = wb if db else 1 - wb
                    r = r + fA * fB * ((1 - ws) * R3[ia + da, ib + db, kk, f] + ws * R3[ia + da, ib + db, kk + 1, f])
            x = r * Y[sel].astype(np.float64) / sig3[k, f]
            dB[f] = x.sum(); mB[f] = x.max() if len(x) else -1e300
        err = max(np.abs(dB - delta[:, k]).max(), np.abs(mB - xmax[:, k]).max())
        worst = max(worst, err)
    print(f"TEST brute force (40 primaries, all {N:,} halos): max |delta_index - delta_brute|, |xmax diff| = {worst:.2e}",
          flush=True)

# ------------------------------------------------------------------ histograms
zbin = np.searchsorted(V.ZED, zA, side="right") - 1
zbin[(zbin < 0) | (zbin >= len(V.ZED) - 1)] = -1
blk = V.block_of(ra[pidx], dec[pidx])
rr = np.ones((6, NP), np.float32)
for kind, rn in RT.items():
    rr[kind] = V.r_of(thA, rth, rn)
u0 = np.clip(rng.random(NP), 1e-12, 1 - 1e-12)
bf = np.array([b[1] for b in V.BVARS], np.int64); bfd = np.array([b[2] for b in V.BVARS]); brk = np.array([b[3] for b in V.BVARS], np.int64)
bc = np.array([b[4] for b in V.BVARS]); bmu = np.array([b[5] for b in V.BVARS], np.int64)
sf = np.array([s[1] for s in V.SVARS], np.int64); srk = np.array([s[2] for s in V.SVARS], np.int64); sc = np.array([s[3] for s in V.SVARS])
ts = time.time()
HB, HS = V.accumulate(qmed, delta, xmax, mu, rr, zbin.astype(np.int64), blk, wgt, u0, 0.173, K, V.EDGES, bf, bfd, brk, bc, bmu,
                      sf, srk, sc, V.NBLK, len(V.ZED) - 1)
HB = HB.sum(0); HS = HS.sum(0)
tacc = time.time() - ts
# realised (catalogue-weighted) mean of the truncated kernel, for the record
num_real = V.realised_mean_numerators(R3, TH, S, Y.astype(np.float64), th)
ratio = num_real / MU
print(f"histograms: {tacc:.0f}s; realised / expected kernel mean at thA = 1, 4, 16' (white): "
      + " ".join(f"{np.interp(np.log(t), np.log(TH), ratio[0]):.4f}" for t in (1.0, 4.0, 16.0)), flush=True)

for name, (vb, vs) in (("baseline_f1", (2, 1)), ("goal_f1", (7, 4)), ("white_f1", (0, 0))):
    Cb = HB[:, vb].sum(0); Cs = HS[:, vs].sum(0)
    j5 = int(np.argmin(np.abs(V.EDGES - 5.0)))
    print(f"  [{name}] q>5: N_single {Cs[:, j5].sum():,.1f}  N_blended {Cb[:, j5].sum():,.1f}  Delta/N {Cb[:, j5].sum()/Cs[:, j5].sum()-1:+.4%}")
np.savez(f"{OUT}/sky_{SEED}{'_test' if TEST else ''}.npz" if S8T is None else f"{OUT}/skyS8_{S8T}_{SEED}.npz", HB=HB, HS=HS, EDGES=V.EDGES, ZED=V.ZED,
         bnames=np.array([b[0] for b in V.BVARS]), snames=np.array([s[0] for s in V.SVARS]), N=N, Nexp=Nexp, NP=NP,
         npairs=npair.sum(), num_real=num_real, tsum=tsum, tacc=tacc, QLO=QLO, FSUB=FSUB, K=K, M0=V.M0,
         dsum_w=np.array([np.sum(wgt * delta[f]) for f in range(3)]))
print(f"saved; total {time.time()-t0:.0f}s; peak rss {V.rss_gb():.2f} GB")
