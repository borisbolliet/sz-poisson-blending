"""V4: independent flat periodic-torus Monte Carlo of catalogue blending (own sampler, own matched-filter
cross-response table, own neighbour sums). Reproduces the run_pass definitions:
  primary q_A > 0.5 (per config), n_A ~ N(0,1), delta_A = sum_B rho_AB Y_B / sigma_A,
  mu_A = sigma_A RATIO ybar_sky / N_w,  c=1: q_b = q+n+delta-mu, c=0: q_b = q+n+delta,
  dominant: max_B rho_AB Y_B/sigma_A < q_A;  det_b = q_b > qc & dominant
  B2(eps) = frac(det_b with (delta - c mu)/q > eps);  B3 = frac(det_b with q+n < qc)
Usage: python v4_torus_blend.py BEAM NOISE L_deg NTORI SEED
"""
import os, sys, time, resource
os.environ["OMP_NUM_THREADS"] = "2"
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
from scipy.special import j0
from scipy.spatial import cKDTree
from scipy.interpolate import RegularGridInterpolator
import own_model as om
from own_sampler import Sampler

BEAM = float(sys.argv[1]); NOISE = float(sys.argv[2]); Ldeg = float(sys.argv[3]); NT = int(sys.argv[4]); SEED = int(sys.argv[5])
L = Ldeg * 60.0  # arcmin
t0 = time.time()
rss = lambda: resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e9

# ---------- own rho table
sb = BEAM / np.sqrt(8 * np.log(2))
THN = np.geomspace(0.05, 150.0, 90); LTH = np.log(THN)
SN = np.concatenate([np.linspace(0, 20, 801), np.geomspace(20, 1500, 401)[1:]])
nl = 30000; l = np.linspace(0, 12 / sb, nl); wl = np.full(nl, l[1] - l[0]); wl[0] *= .5; wl[-1] *= .5
B2 = np.exp(-(l * sb) ** 2)
G = om.g_of(np.outer(THN, l))
J = j0(np.outer(SN, l))
Ct = np.empty((len(THN), len(THN), len(SN)), np.float32)
for i in range(len(THN)):
    Wt = (l * wl * B2 * G[i])[None, :] * G
    Ct[i] = (Wt @ J.T) / (2 * np.pi)
del J
diag = np.array([Ct[i, i, 0] for i in range(len(THN))])
RHO = Ct / diag[:, None, None]
TA, TB, SS = np.meshgrid(THN, THN, SN, indexing="ij")
RHO[SS > 5 * (TA + TB) + 4 * BEAM] = 0.0
del TA, TB, SS, Ct
rgi = RegularGridInterpolator((LTH, LTH, SN), RHO, bounds_error=False, fill_value=0.0)
def rho(tA, tB, s):
    return rgi(np.column_stack([np.log(np.clip(tA, THN[0], THN[-1])), np.log(np.clip(tB, THN[0], THN[-1])), s]))
# accuracy test vs direct Hankel
rng0 = np.random.default_rng(5)
tA_t = np.exp(rng0.uniform(np.log(0.2), np.log(20), 150)); tB_t = np.exp(rng0.uniform(np.log(0.2), np.log(20), 150))
s_t = rng0.uniform(0, 1, 150) * (2 * (tA_t + tB_t) + 2 * BEAM)
ex = []
for a, b_, s in zip(tA_t, tB_t, s_t):
    CAA = np.sum(wl * l * B2 * om.g_of(l * a) ** 2)
    ex.append(np.sum(wl * l * B2 * om.g_of(l * a) * om.g_of(l * b_) * j0(l * s)) / CAA)
ex = np.array(ex); ap = rho(tA_t, tB_t, s_t); big = ex > 0.02
print(f"[beam {BEAM}] rho table: max |rel err| where rho>0.02: {np.max(np.abs(ap[big]/ex[big]-1)):.2e}; max abs err {np.max(np.abs(ap-ex)):.2e}; built {time.time()-t0:.0f}s", flush=True)

thg = np.geomspace(0.01, 300, 800); sYg = om.sigmaY(thg, BEAM, NOISE)
sigY = lambda th: np.exp(np.interp(np.log(th), np.log(thg), np.log(sYg)))
S = Sampler(Mmin=1e13, nz=900, nM=600)
fsky = (L * om.ARC) ** 2 / (4 * np.pi)
ME = [1e13, 3e13, 1e14]; ZE = [1.0, 2.0]
QC = [4.0, 5.0, 6.0]
CFG = {"base": (1.0, 1e13, 3.0), "noise_x0.5": (0.5, 1e13, 3.0), "noise_x2": (2.0, 1e13, 3.0),
       "floor_3e13": (1.0, 3e13, 3.0), "floor_1e14": (1.0, 1e14, 3.0), "zmax_1": (1.0, 1e13, 1.0)}
if BEAM != 1.4:
    CFG = {"this_beam": (1.0, 1e13, 3.0)}
NB = 8  # jackknife blocks per side
acc = {}
for it in range(NT):
    rng = np.random.default_rng(SEED + it)
    z, M = S.draw(rng, fsky); N = len(z)
    x = rng.random(N) * L; y = rng.random(N) * L
    Y = om.Ymed(M, z) * np.exp(0.173 * rng.standard_normal(N))
    th = om.theta500(M, z); sA = sigY(th); q = Y / sA; nA = rng.standard_normal(N)
    grp = 3 * np.digitize(M, ME[1:]) + np.digitize(z, ZE)
    ybar_g = om.RATIO * np.bincount(grp, weights=Y, minlength=9) / L ** 2
    fmin = min(c[0] for c in CFG.values())
    prim = np.nonzero(q > 0.5 * fmin)[0]
    print(f"torus {it}: N={N} (expected {S.Ntot*fsky:.0f}), primaries {len(prim)}, ybar_sky={ybar_g.sum():.4e}", flush=True)
    Ssum = np.zeros((len(prim), 9)); Smax = np.zeros((len(prim), 9))
    edges = [0, 1, 2, 4, 8, 16, 32, 64, np.inf]
    pos = np.column_stack([x, y])
    for bi in range(len(edges) - 1):
        nb = np.nonzero((th >= edges[bi]) & (th < edges[bi + 1]))[0]
        if len(nb) == 0: continue
        tbmax = th[nb].max()
        tree = cKDTree(pos[nb], boxsize=L)
        for a in range(0, len(prim), 4000):
            pa = prim[a:a + 4000]
            r = np.minimum(5 * (th[pa] + tbmax) + 4 * BEAM, L / 2 - 1)
            lists = tree.query_ball_point(pos[pa], r)
            cnt = np.array([len(v) for v in lists])
            if cnt.sum() == 0: continue
            ia = np.repeat(np.arange(len(pa)), cnt); jb = nb[np.concatenate([np.asarray(v, int) for v in lists if len(v)])]
            A = pa[ia]; keep = A != jb; ia, A, jb = ia[keep], A[keep], jb[keep]
            d = np.abs(pos[A] - pos[jb]); d = np.minimum(d, L - d); s = np.hypot(d[:, 0], d[:, 1])
            t = rho(th[A], th[jb], s) * Y[jb]
            g = grp[jb]; k = (a + ia) * 9 + g
            Ssum.ravel()[:] += np.bincount(k, weights=t, minlength=Ssum.size)
            np.maximum.at(Smax.ravel(), k, t)
    print(f"   sums done {time.time()-t0:.0f}s rss {rss():.2f} GB", flush=True)
    sp = sA[prim]; qp = q[prim]; np_ = nA[prim]; zp = z[prim]
    blk = (np.floor(x[prim] / L * NB) * NB + np.floor(y[prim] / L * NB)).astype(int) + it * NB * NB
    for cn, (f, floor, zmax) in CFG.items():
        gl = [3 * m + zc for m in range(3) for zc in range(3) if ME[m] >= floor * 0.999 and [0, 1, 2][zc] < zmax - 1e-9]
        Q = qp / f; keep = (Q > 0.5) & (zp < zmax)
        D = Ssum[:, gl].sum(1) / (sp * f); X = Smax[:, gl].max(1) / (sp * f)
        mu = sp * om.RATIO * ybar_g[gl].sum() / NOISE ** 2 / f
        dom = X < Q
        for qc in QC:
            single = keep & (Q + np_ > qc)
            rec = acc.setdefault((cn, qc), {})
            def add(key, m):
                rec.setdefault(key, []).append(np.bincount(blk[m] - it * NB * NB, minlength=NB * NB))
            add("single", single)
            add("sum_delta_single", single)  # placeholder count
            rec.setdefault("dsum", []).append(np.bincount(blk[single] - it*NB*NB, weights=D[single], minlength=NB*NB))
            rec.setdefault("musum", []).append(np.bincount(blk[single] - it*NB*NB, weights=mu[single], minlength=NB*NB))
            for c in (0, 1):
                dp = D - c * mu; det = keep & (Q + np_ + dp > qc) & dom
                add(f"det{c}", det); add(f"b3_{c}", det & (Q + np_ < qc)); add(f"b2_{c}", det & (dp / Q > 0.1))
                if c == 1 and qc == 5.0:
                    idx = np.nonzero(det)[0]
                    tr = cKDTree(np.column_stack([x[prim][idx], y[prim][idx]]), boxsize=L)
                    for tt in (1.0, 2.0, 5.0):
                        nn = tr.query_ball_point(np.column_stack([x[prim][idx], y[prim][idx]]), tt, return_length=True) - 1
                        m = np.zeros(len(prim), bool); m[idx[nn >= 1]] = True
                        add(f"pb_{tt:g}", m)
    del Ssum, Smax
print(f"total {time.time()-t0:.0f}s peak rss {rss():.2f} GB")
area_tot = NT * L ** 2

def jk(num, den):
    num = np.concatenate(num); den = np.concatenate(den); Nn, Nd = num.sum(), den.sum(); n = len(num)
    rj = (Nn - num) / (Nd - den); return Nn / Nd, np.sqrt((n - 1) / n * np.sum((rj - rj.mean()) ** 2))

base5 = {}
for (cn, qc), rec in acc.items():
    r2, e2 = jk(rec["b2_1"], rec["det1"]); r3, e3 = jk(rec["b3_1"], rec["det1"])
    r20, e20 = jk(rec["b2_0"], rec["det0"]); r30, e30 = jk(rec["b3_0"], rec["det0"])
    dm, edm = jk(rec["dsum"], rec["single"]); mm, emm = jk(rec["musum"], rec["single"])
    nd = np.concatenate(rec["det1"]).sum(); ns = np.concatenate(rec["single"]).sum()
    print(f"{cn:11s} q>{qc:g}: N_single={ns} Sigma_single={ns/area_tot*3600:.4f}/deg2; c=1: B2(0.1)={r2:.4f}+-{e2:.4f} B3={r3:.4f}+-{e3:.4f} | "
          f"c=0: B2(0.1)={r20:.4f}+-{e20:.4f} B3={r30:.4f}+-{e30:.4f} | <delta>={dm:.4f}+-{edm:.4f} <mu>={mm:.4f} ({(dm-mm)/edm:+.2f} sig)")
    if qc == 5.0:
        base5[cn] = (r2, e2, r3, e3)
        if "pb_1" in rec:
            Sb = nd / area_tot
            for tt in (1.0, 2.0, 5.0):
                r, e = jk(rec[f"pb_{tt:g}"], rec["det1"])
                print(f"      blended-catalogue P(>=1 within {tt:g}') = {r:.4e} +- {e:.1e}; Poisson(Sigma_b) {1-np.exp(-Sb*np.pi*tt**2):.4e}; ratio {r/(1-np.exp(-Sb*np.pi*tt**2)):.2f}")
if "base" in base5:
    b = base5["base"]
    for cn, v in base5.items():
        print(f"ratio to base (q=5,c=1) {cn:11s}: B2 x{v[0]/b[0]:.3f}  B3 x{v[2]/b[2]:.3f}")
