"""V02: independent map-free but EXACT Fourier-space Monte Carlo on periodic flat patches (torus).
For each patch (L x L arcmin, N^2 modes), halos (M>1e13, my own sampler from colossus Tinker08) are
summed in Fourier space: H_c(l) = sum_{B in class c} Y_B g(l theta_B) e^{i l.x_B} (type-1 NUFFT, CIC in
ln theta over a template bank). A's matched-filter output (filter at theta_A) at x:
  Yhat(x) = sigma_A^2 / (N_w L^2) sum_l g(l theta_A) B_l^2 H(l) e^{-i l.x}       (type-2 NUFFT)
which equals sum_B rho(thA,thB,s) Y_B exactly (Poisson summation; periodic images), incl. the l=0 (DC) mode
(= 'raw' convention).  Self term removed exactly (same discretisation).  No rho table, no neighbour search.
Dominance: x_B <= q_B (Cauchy-Schwarz), so only B with q_B >= q_A can dominate; those pairs are evaluated
with my own direct Hankel quadrature.
Usage: python v02_torus_mc.py first_seed n_patches
"""
import os, sys, time
os.environ["OMP_NUM_THREADS"] = "2"
import numpy as np
import finufft
from scipy.special import j0
import indep_model as im

L = 600.0; N = 1536  # l_max = 8.04/arcmin, B^2(l_max) = 1e-10
OUT = os.path.join(im.HERE, "torus")
os.makedirs(OUT, exist_ok=True)
LN0, DLN = np.log(0.2), 0.1
NB = int(np.ceil((np.log(80) - LN0) / DLN)) + 2
THN = np.exp(LN0 + DLN * np.arange(NB))
SIGN = im.sigmaY_fast(THN)
RAND_TH = [0.5237, 0.9463, 1.0975, 1.9826, 4.1533]
X1_TH = (1.0975, 4.1533)  # x>1 neighbour counts only here
NX1 = 5000
NRAND = 20000
EPS = 1e-8

k1 = np.fft.fftfreq(N, 1.0 / N)
KX, KY = np.meshgrid(k1, k1, indexing="ij")
K2 = (KX ** 2 + KY ** 2).astype(np.int64)
LMOD = 2 * np.pi / L * np.sqrt(K2.astype(float))
del KX, KY
B2 = im.B2(LMOD)
# fine radial table of g(u) evaluated through np.interp in l for each node
LR = np.linspace(0, LMOD.max() * 1.0001, 400001)
def g_l(theta):
    return np.interp(LMOD, LR, im.g_of_u(LR * theta))
# self-response matrix S_ij = sigma_i^2/(N_w L^2) sum_k g_i g_j B^2   (radial bincount)
cnt = np.bincount(K2.ravel())
k2u = np.nonzero(cnt)[0]
lu = 2 * np.pi / L * np.sqrt(k2u.astype(float))
GU = np.array([im.g_of_u(lu * t) for t in THN])            # (NB, nu)
S = (GU * (cnt[k2u] * im.B2(lu))[None, :]) @ GU.T / (im.NW * L * L) * (SIGN ** 2)[:, None]
del GU


def cic(theta):
    f = (np.log(theta) - LN0) / DLN
    i = np.floor(f).astype(int); w = f - i
    return i, w


_LP = np.linspace(0, 7.5, 3001)
_GP = np.array([im.g_of_u(_LP * t) for t in THN])


def _g_cic(theta):
    f = (np.log(theta) - LN0) / DLN
    i = np.floor(f).astype(int); w = (f - i)[:, None]
    return (1 - w) * _GP[i] + w * _GP[i + 1]


def rho_pairs(thA, thB, s):
    l = _LP
    wl = l * im.B2(l)
    out = np.empty(len(s))
    for a in range(0, len(s), 4000):
        b = min(len(s), a + 4000)
        ga = _g_cic(thA[a:b]) if np.ndim(thA) else None
        gb = _g_cic(thB[a:b])
        I = np.trapezoid(wl[None, :] * ga * gb * j0(np.outer(s[a:b], l)), l, axis=1) / (2 * np.pi) / im.NW
        out[a:b] = I * im.sigmaY_fast(thA[a:b]) ** 2
    return out


def pdist(xa, ya, xb, yb):
    dx = np.abs(xa - xb); dx = np.minimum(dx, L - dx)
    dy = np.abs(ya - yb); dy = np.minimum(dy, L - dy)
    return np.hypot(dx, dy)


def run(seed, grid):
    t0 = time.time()
    rng = np.random.default_rng(seed)
    z, M, Y, th = im.sample_halos(rng, L * L, *grid)
    n = len(z)
    X = rng.random(n) * L; Yp = rng.random(n) * L
    cls = (np.searchsorted([1e13, 3e13, 1e14], M, side="right") - 1).astype(np.int8)
    xs = 2 * np.pi * X / L; ys = 2 * np.pi * Yp / L
    ib, wb = cic(th)
    assert ib.min() >= 0 and ib.max() + 1 < NB
    H = np.zeros((3, N, N), complex)
    for j in np.unique(np.concatenate([ib, ib + 1])):
        sel0 = ib == j; sel1 = ib + 1 == j
        sel = sel0 | sel1
        wts = np.where(sel0[sel], 1 - wb[sel], wb[sel]) * Y[sel]
        c = np.zeros((3, sel.sum()), complex)
        cc = cls[sel]
        for k in range(3):
            c[k, cc == k] = wts[cc == k]
        F = finufft.nufft2d1(xs[sel], ys[sel], c, (N, N), eps=EPS, isign=1, modeord=1, nthreads=2)
        H += g_l(THN[j])[None] * F
    t1 = time.time()
    sigA = im.sigmaY_fast(th); q = Y / sigA
    prim = np.nonzero(q > 0.5)[0]
    ia, wa = ib[prim], wb[prim]
    Dp = np.zeros((len(prim), 3))
    for i in np.unique(np.concatenate([ia, ia + 1])):
        sel0 = ia == i; sel1 = ia + 1 == i
        sel = sel0 | sel1
        Phi = (SIGN[i] ** 2 / (im.NW * L * L)) * g_l(THN[i]) * B2
        vals = finufft.nufft2d2(xs[prim][sel], ys[prim][sel], Phi[None] * H, eps=EPS, isign=-1, modeord=1, nthreads=2).real
        w = np.where(sel0[sel], 1 - wa[sel], wa[sel])
        Dp[sel] += (w[:, None] * vals.T) / SIGN[i]
    # exact self term (same discretisation): sum_{a in {i,i+1}} sum_{b in {i,i+1}} wa wb S_ab / sig_a
    wA = np.column_stack([1 - wa, wa]); idx = np.column_stack([ia, ia + 1])
    selfx = np.zeros(len(prim))
    for a in range(2):
        for b in range(2):
            selfx += wA[:, a] * wA[:, b] * S[idx[:, a], idx[:, b]] / SIGN[idx[:, a]]
    selfx *= Y[prim]
    Dp[np.arange(len(prim)), cls[prim]] -= selfx
    t2 = time.time()
    if BRUTE:
        pick = np.random.default_rng(1).choice(len(prim), 40, replace=False)
        pick = np.concatenate([pick, np.argsort(th[prim])[-10:]])
        for a in pick:
            A_ = prim[a]
            dd = pdist(X[A_], Yp[A_], X, Yp); ok = (dd < 5 * th[A_] + 5 * th + 6.0); ok[A_] = False
            jj = np.nonzero(ok)[0]
            rr = np.array([im.rho(th[A_], th[j], [dd[j]])[0] for j in jj]) if len(jj) < 3000 else rho_pairs(np.full(len(jj), th[A_]), th[jj], dd[jj])
            Db = np.bincount(cls[jj], weights=rr * Y[jj], minlength=3) / sigA[A_]
            print(f"   brute thA={th[A_]:.2f}' q={q[A_]:.1f}: torus {np.round(Dp[a],5)} brute(exact rho) {np.round(Db,5)}", flush=True)
        # check rho_pairs (CIC templates) vs exact
        tA = np.exp(np.random.default_rng(2).uniform(np.log(0.3), np.log(30), 50)); tB = np.exp(np.random.default_rng(3).uniform(np.log(0.3), np.log(30), 50))
        ss = np.random.default_rng(4).uniform(0, 1, 50) * (2 * tA + 2 * tB + 2)
        ex = np.array([im.rho(a, b, [c])[0] for a, b, c in zip(tA, tB, ss)]); ap = rho_pairs(tA, tB, ss)
        r0 = np.array([im.rho(a, b, [0.0])[0] for a, b in zip(tA, tB)])
        print(f"   rho_pairs vs exact: max |diff|/rho0 = {np.max(np.abs(ap-ex)/r0):.2e}", flush=True)
    # dominance: candidates B (primaries) with q_B >= q_A
    qp = q[prim]; thp = th[prim]; Xp, Ypp = X[prim], Yp[prim]
    o = np.argsort(-qp)
    xmax = np.zeros((len(prim), 3))
    PA, PB, PS = [], [], []
    for r, a in enumerate(o):
        if r == 0:
            continue
        bs = o[:r]
        d = pdist(Xp[a], Ypp[a], Xp[bs], Ypp[bs])
        ok = d < 5 * thp[a] + 5 * thp[bs] + 6.0
        if ok.any():
            PA.append(np.full(ok.sum(), a)); PB.append(bs[ok]); PS.append(d[ok])
    if PA:
        PA = np.concatenate(PA); PB = np.concatenate(PB); PS = np.concatenate(PS)
        r_ = rho_pairs(thp[PA], thp[PB], PS)
        x = r_ * Y[prim][PB] / im.sigmaY_fast(thp[PA])
        cB = cls[prim][PB]
        for k in range(3):
            s_ = cB == k
            np.maximum.at(xmax[:, k], PA[s_], x[s_])
        npairs = len(PA)
    else:
        npairs = 0
    t3 = time.time()
    # random points (exact filter at theta_A, no CIC) + count of neighbours with x>1 (candidates q_B>=1)
    RD = {}; RX1 = {}
    qall_big = np.nonzero(q >= 1.0)[0]
    for t in RAND_TH:
        rx = rng.random(NRAND) * L; ry = rng.random(NRAND) * L
        sA = float(im.sigmaY_fast(t))
        Phi = (sA ** 2 / (im.NW * L * L)) * g_l(t) * B2
        vals = finufft.nufft2d2(2 * np.pi * rx / L, 2 * np.pi * ry / L, Phi[None] * H, eps=EPS, isign=-1, modeord=1, nthreads=2).real
        RD[t] = (vals.T / sA)
        # x>1 neighbours (first NX1 points, two filter scales only)
        nx1 = np.zeros((NX1, 3))
        for a0 in (range(0, NX1, 500) if t in X1_TH else []):
            d = pdist(rx[a0:a0 + 500, None], ry[a0:a0 + 500, None], X[qall_big][None], Yp[qall_big][None])
            ii, jj = np.nonzero(d < 5 * t + 5 * th[qall_big][None] + 6.0)
            if len(ii):
                bsel = qall_big[jj]
                rr = rho_pairs(np.full(len(ii), t), th[bsel], d[ii, jj])
                x = rr * Y[bsel] / sA
                for k in range(3):
                    s_ = (x > 1) & (cls[bsel] == k)
                    np.add.at(nx1[:, k], a0 + ii[s_], 1)
        RX1[t] = nx1 if t in X1_TH else np.zeros((0, 3))
    t4 = time.time()
    noise = rng.standard_normal(len(prim))
    ybar = np.array([Y[cls == k].sum() for k in range(3)]) / (L * L)
    np.savez(os.path.join(OUT, f"patch_{seed}.npz"), q=qp, n=noise, th=thp, cls=cls[prim], D=Dp, xmax=xmax,
             X=Xp, Yp=Ypp, ybar=ybar, nhalo=n, rand_th=np.array(RAND_TH),
             **{f"rD_{k}": RD[t] for k, t in enumerate(RAND_TH)}, **{f"rX1_{k}": RX1[t] for k, t in enumerate(RAND_TH)})
    print(f"patch {seed}: n={n} prim={len(prim)} pairs={npairs} | H {t1-t0:.1f}s prim {t2-t1:.1f}s dom {t3-t2:.1f}s rand {t4-t3:.1f}s", flush=True)


BRUTE = False
if __name__ == "__main__":
    s0, npch = int(sys.argv[1]), int(sys.argv[2])
    BRUTE = len(sys.argv) > 3
    grid = im.hmf_grid()
    print(f"NB={NB} nodes, S diag range {S.diagonal().min()/SIGN.min():.4f}; diag S_ii/sig_i^0 min/max {np.min(np.diag(S)):.5f} {np.max(np.diag(S)):.5f}", flush=True)
    for s in range(s0, s0 + npch):
        run(s, grid)
