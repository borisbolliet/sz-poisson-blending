"""Independent minimal map pipeline (no szsim / analysis imports).

* Own Poisson draw of halos on a 600'x600' torus from our own colossus Tinker08 table (M500 in [1e13,1e16] Msun,
  0.01<z<3), own Y-M relation with lognormal scatter, own sigma_Y.
* Painting done in Fourier space: TSC deposit of Y per log-theta bin (linear interpolation in ln theta between bins),
  times g(l theta_b) (Hankel transform of the unit-Y500 Arnaud template) / TSC window, times beam.
* 12 matched filters theta_k = geomspace(0.5,10,12), l=0 mode removed; maxQ over scales; 3x3 local maxima; greedy
  2' merge.  Per-candidate primary from direct Hankel cross-responses c_i = Y_i R_k(r_i; theta_i).
Usage: python v03_mappipe.py <first_patch> <last_patch_excl>
"""
import os, sys, time
os.environ["OMP_NUM_THREADS"] = "1"; os.environ["NUMBA_NUM_THREADS"] = "1"
sys.path.insert(0, os.path.dirname(__file__))
import numpy as np
import numba as nb
from scipy import fft as sfft
from scipy.special import j0
from scipy.spatial import cKDTree
import own_model as om

SRC = os.environ.get("SRC", "own")
MMIN = float(os.environ.get("MMIN", "0"))
OUT = os.path.join(os.path.dirname(__file__), ("maps_out" if SRC == "own" else "maps_out_cat") + (f"_M{MMIN:.0e}" if MMIN > 0 else ""))
N = 2400; PIX = 0.25; L = N * PIX; AREA = L * L
TK = np.geomspace(0.5, 10.0, 12); K = 12
SIGB = om.SIGB; NW = om.NW

# ---------------- halo population table (own)
_tab = None


def table():
    global _tab
    if _tab is None:
        zc, lc, d2, dz, dl = om.hmf_table(nz=900, nM=900)
        cell = (d2 * dz * dl).ravel()
        _tab = (zc, lc, dz, dl, cell)
    return _tab


THG = np.geomspace(1e-3, 300, 500)
SYG = om.sigmaY(THG)


def sigY(th):
    return np.exp(np.interp(np.log(th), np.log(THG), np.log(SYG)))


def draw(seed):
    zc, lc, dz, dl, cell = table()
    rng = np.random.default_rng(seed)
    lam = cell.sum() * AREA
    n = rng.poisson(lam)
    ic = rng.choice(cell.size, n, p=cell / cell.sum())
    iz, iM = np.divmod(ic, lc.size)
    z = zc[iz] + (rng.random(n) - 0.5) * dz
    M = np.exp(lc[iM] + (rng.random(n) - 0.5) * dl)
    Y = om.Y500_med(M, z) * np.exp(om.SIG_LNY * rng.standard_normal(n))
    th = om.theta500(M, z)
    x = rng.random(n) * L; y = rng.random(n) * L
    q = Y / sigY(th)
    return dict(x=x, y=y, z=z, M=M, Y=Y, th=th, q=q, lam=lam)


# ---------------- Fourier grids and filters
fx = 2 * np.pi * np.fft.fftfreq(N, d=PIX)
fy = 2 * np.pi * np.fft.rfftfreq(N, d=PIX)
LL = np.sqrt(fx[:, None] ** 2 + fy[None, :] ** 2).astype(np.float64)
BEAM = np.exp(-0.5 * (LL * SIGB) ** 2)


def sinc(a):
    return np.sinc(a / np.pi)


WTSC = (sinc(fx * PIX / 2) ** 3)[:, None] * (sinc(fy * PIX / 2) ** 3)[None, :]
MULT = np.full(fy.size, 2.0); MULT[0] = 1.0; MULT[-1] = 1.0


def filters():
    psis, sig = [], []
    for t in TK:
        T = BEAM * om.gfun(LL * t)
        s2 = 1.0 / ((T ** 2 * MULT[None, :]).sum() / NW / AREA)
        p = (s2 * T / NW).astype(np.float32); p[0, 0] = 0.0
        psis.append(p); sig.append(np.sqrt(s2))
    return psis, np.array(sig)


@nb.njit(cache=True)
def tsc_deposit(m, x, y, w, pix):
    n = m.shape[0]
    for h in range(x.size):
        gx = x[h] / pix; gy = y[h] / pix
        ix = int(np.floor(gx + 0.5)); iy = int(np.floor(gy + 0.5))
        dx = gx - ix; dy = gy - iy
        wx = (0.5 * (0.5 - dx) ** 2, 0.75 - dx * dx, 0.5 * (0.5 + dx) ** 2)
        wy = (0.5 * (0.5 - dy) ** 2, 0.75 - dy * dy, 0.5 * (0.5 + dy) ** 2)
        for a in range(3):
            for b in range(3):
                m[(iy + a - 1) % n, (ix + b - 1) % n] += w[h] * wy[a] * wx[b]


@nb.njit(cache=True)
def local_max(Q, thr):
    n0, n1 = Q.shape
    iy = []; ix = []
    for i in range(n0):
        for j in range(n1):
            v = Q[i, j]
            if v < thr:
                continue
            ok = True
            for di in (-1, 0, 1):
                for dj in (-1, 0, 1):
                    if di == 0 and dj == 0:
                        continue
                    w = Q[(i + di) % n0, (j + dj) % n1]
                    if w > v or (w == v and (di < 0 or (di == 0 and dj < 0))):
                        ok = False
            if ok:
                iy.append(i); ix.append(j)
    return np.array(iy, dtype=np.int64), np.array(ix, dtype=np.int64)


LH = np.linspace(0, 12.0, 2401)
BH2 = np.exp(-(LH * SIGB) ** 2)
GK = np.array([om.gfun(LH * t) for t in TK])


def resp(k, th, r, sig):
    """R_k(r; th) for arrays th, r (direct Hankel quadrature)."""
    out = np.empty(th.size)
    for a in range(0, th.size, 1500):
        tt = th[a:a + 1500]; rr = r[a:a + 1500]
        g = om.gfun(LH[None, :] * tt[:, None])
        ig = LH[None, :] * BH2[None, :] * GK[k][None, :] * g * j0(LH[None, :] * rr[:, None])
        out[a:a + 1500] = np.trapezoid(ig, LH, axis=1) / (2 * np.pi) * sig[k] ** 2 / NW
    return out


def pdist(ax, ay, bx, by):
    dx = (ax - bx + L / 2) % L - L / 2; dy = (ay - by + L / 2) % L - L / 2
    return np.hypot(dx, dy)


_BAND = None


def from_cat(pid):
    """Same halos as the analysis patches (stored szsim catalogues, own equal-area cut); own q from own sigma_Y."""
    global _BAND
    if _BAND is None:
        _BAND = np.load(os.path.join(os.path.dirname(__file__), "band_patches.npz"))
    f = _BAND
    h = dict(x=f[f"p{pid}_x"] % L, y=f[f"p{pid}_y"] % L, z=f[f"p{pid}_z"].astype(float), M=f[f"p{pid}_M500"].astype(float),
             Y=f[f"p{pid}_Y500"].astype(float), th=f[f"p{pid}_theta500"].astype(float))
    h["q"] = h["Y"] / sigY(h["th"]); h["lam"] = 71014.26
    if MMIN > 0:
        keep = h["M"] >= MMIN
        h = {k: (v[keep] if isinstance(v, np.ndarray) else v) for k, v in h.items()}
    return h


def run_patch(pid, psis, sig):
    t0 = time.time()
    h = draw(1000 + pid) if SRC == "own" else from_cat(pid)
    nh = h["x"].size
    # ---- painting by theta bins
    LNT0, DLT = np.log(0.03), 0.08
    t = (np.log(h["th"]) - LNT0) / DLT
    i0 = np.floor(t).astype(int); w = t - i0
    bins = np.concatenate([i0, i0 + 1]); ww = np.concatenate([h["Y"] * (1 - w), h["Y"] * w])
    xx = np.concatenate([h["x"], h["x"]]); yy = np.concatenate([h["y"], h["y"]])
    G = np.zeros((N, N // 2 + 1), np.complex128)
    m = np.zeros((N, N))
    for b in np.unique(bins):
        s = bins == b
        m[:] = 0.0
        tsc_deposit(m, xx[s], yy[s], ww[s], PIX)
        G += sfft.rfft2(m) * om.gfun(LL * np.exp(LNT0 + DLT * b))
    del m
    sky = G / WTSC / PIX ** 2            # rfft2 of the y-pixel map (beam-free)
    ybar = sky[0, 0].real / N ** 2
    D = (sky * BEAM).astype(np.complex64)
    del G, sky
    rng = np.random.default_rng((50000 if SRC == 'own' else 90000) + pid)
    Nf = sfft.rfft2((rng.standard_normal((N, N)) * (np.sqrt(NW) / PIX)).astype(np.float32))
    t_paint = time.time() - t0
    # ---- filtering
    hs = np.flatnonzero(h["q"] >= 1.0)
    hiy = np.round(h["y"][hs] / PIX).astype(int) % N; hix = np.round(h["x"][hs] / PIX).astype(int) % N
    hS = np.zeros((hs.size, K)); hN = np.zeros((hs.size, K))
    Qb = np.full((N, N), -np.inf, np.float32); kb = np.zeros((N, N), np.int8)
    Sb = np.zeros((N, N), np.float32); Nb = np.zeros((N, N), np.float32)
    Qn = np.full((N, N), -np.inf, np.float32)
    var_noise = np.zeros(K)
    for k in range(K):
        S = sfft.irfft2(D * psis[k], s=(N, N)); Nn = sfft.irfft2(Nf * psis[k], s=(N, N))
        var_noise[k] = Nn.var(dtype=np.float64)
        hS[:, k] = S[hiy, hix]; hN[:, k] = Nn[hiy, hix]
        q = (S + Nn) / sig[k]
        up = q > Qb
        Qb[up] = q[up]; kb[up] = k; Sb[up] = S[up]; Nb[up] = Nn[up]
        np.maximum(Qn, Nn / sig[k], out=Qn)
        del S, Nn, q, up
    t_filt = time.time() - t0
    iy, ix = local_max(Qb, np.float32(4.0))
    cand = dict(iy=iy, ix=ix, k=kb[iy, ix].astype(int), Q=Qb[iy, ix].astype(float), S=Sb[iy, ix].astype(float),
                Nn=Nb[iy, ix].astype(float))
    niy, nix = local_max(Qn, np.float32(4.0))
    noise_cand = np.stack([nix * PIX, niy * PIX, Qn[niy, nix]], 1)
    del Qb, kb, Sb, Nb, Qn
    # ---- decomposition of candidates
    tree = cKDTree(np.stack([h["x"], h["y"]], 1), boxsize=L)
    big = np.flatnonzero(h["th"] > 2.0)
    px = cand["ix"] * PIX; py = cand["iy"] * PIX
    nc = px.size
    ip = np.full(nc, -1); cp = np.zeros(nc); dp = np.full(nc, 99.0); nc10 = np.zeros(nc, int); c2 = np.zeros(nc)
    for j in range(nc):
        k = cand["k"][j]
        idx = np.array(tree.query_ball_point([px[j], py[j]], 12.0), int)
        rb = pdist(h["x"][big], h["y"][big], px[j], py[j])
        extra = big[rb < 5 * h["th"][big] + 5 * TK[k] + 5.0]
        idx = np.unique(np.concatenate([idx, extra]))
        if idx.size == 0:
            continue
        r = pdist(h["x"][idx], h["y"][idx], px[j], py[j])
        c = h["Y"][idx] * resp(k, h["th"][idx], r, sig)
        o = np.argsort(-c)
        ip[j] = idx[o[0]]; cp[j] = c[o[0]]; dp[j] = r[o[0]]
        c2[j] = c[o[1]] if c.size > 1 else 0.0
        nc10[j] = (c[o[1:]] > 0.1 * cp[j]).sum()
    # neighbour counts of q_true > qc within r of each candidate
    qtrees = {qc: cKDTree(np.stack([h["x"][h["q"] > qc], h["y"][h["q"] > qc]], 1), boxsize=L) for qc in (4, 5, 6, 7)}
    pts = np.stack([px % L, py % L], 1)
    nq = np.zeros((nc, 4, 3), int)
    for a, qc in enumerate((4, 5, 6, 7)):
        for b_, rr in enumerate((1.0, 2.0, 3.0)):
            nq[:, a, b_] = qtrees[qc].query_ball_point(pts, rr, return_length=True)
    # ---- halo side: q_iso and q_pos
    dlt = pdist(h["x"][hs], h["y"][hs], hix * PIX, hiy * PIX)
    own = np.stack([h["Y"][hs] * resp(k, h["th"][hs], dlt, sig) for k in range(K)], 1)
    q_iso = ((own + hN) / sig[None, :]).max(1)
    q_pos = ((hS + hN) / sig[None, :]).max(1)
    t_dec = time.time() - t0
    # ---- sanity: map S vs truth sum for brightest 30 candidates (full neighbourhood up to 60')
    san = []
    for j in np.argsort(-cand["Q"])[:30]:
        k = cand["k"][j]
        idx = np.array(tree.query_ball_point([px[j], py[j]], 60.0), int)
        r = pdist(h["x"][idx], h["y"][idx], px[j], py[j])
        rmax = 5 * h["th"][idx] + 5 * TK[k] + 8 * SIGB
        far = np.setdiff1d(np.flatnonzero(pdist(h["x"], h["y"], px[j], py[j]) < 5 * h["th"] + 5 * TK[k] + 8 * SIGB), idx)
        idx = np.concatenate([idx[r < rmax], far]); r = pdist(h["x"][idx], h["y"][idx], px[j], py[j])
        Ssum = (h["Y"][idx] * resp(k, h["th"][idx], r, sig)).sum() - ybar * sig[k] ** 2 * om.RATIO / NW
        san.append([cand["S"][j] / sig[k], Ssum / sig[k]])
    np.savez(f"{OUT}/patch_{pid:03d}.npz", sig=sig, ybar=ybar, lam=h["lam"], nh=nh,
             q=h["q"], M=h["M"], hs=hs, q_iso=q_iso, q_pos=q_pos, var_noise=var_noise,
             c_iy=cand["iy"], c_ix=cand["ix"], c_k=cand["k"], c_Q=cand["Q"], c_S=cand["S"], c_N=cand["Nn"],
             c_ip=ip, c_cp=cp, c_dp=dp, c_nc10=nc10, c_c2=c2, c_nq=nq, noise_cand=noise_cand, sanity=np.array(san))
    return t_paint, t_filt, t_dec, time.time() - t0, np.array(san)


if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True)
    a, b = int(sys.argv[1]), int(sys.argv[2])
    psis, sig = filters()
    print("discrete sigma_k / Hankel - 1:", np.round(sig / om.sigmaY(TK) - 1, 6), flush=True)
    for pid in range(a, b):
        if os.path.exists(f"{OUT}/patch_{pid:03d}.npz"):
            continue
        tp, tf, td, tt, san = run_patch(pid, psis, sig)
        d = san[:, 0] - san[:, 1]
        print(f"patch {pid}: paint {tp:.0f}s filt {tf:.0f}s dec {td:.0f}s total {tt:.0f}s | map-truth (sigma): mean {d.mean():+.2e} rms {d.std():.2e} max {np.abs(d).max():.2e}", flush=True)
