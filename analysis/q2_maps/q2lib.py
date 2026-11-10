"""Map-level blending pipeline (Q2): painting, multi-scale matched filter, peak finding, truth matching.

Geometry: flat periodic (torus) patch, L = 600' (10 deg), pixel 0.25' (2400^2), pixel j has centre (j+0.5)*PIX.
Units: arcmin for angles, Y in arcmin^2, y dimensionless, l in arcmin^-1 inside the FFT code.

Tables (Hankel transforms of the library template g(u) = szsim.signal.template_hankel):
  * paint profile  P(r; theta) = (1/2pi) int dl l g(l theta) B_l J0(l r)   [arcmin^-2 per unit Y500]
    (beam-convolved unit-Y500 Arnaud template, integrates to RATIO_5R500 = 1.796)
  * response       R_k(r; theta) = sigma_k^2/N_w (1/2pi) int dl l B_l^2 g(l theta_k) g(l theta) J0(l r)
    (Y_hat of the scale-k matched filter at distance r from a unit-Y500 halo of size theta; R_k(0;theta_k)=1;
     identical to SZModel.cross_response(theta_k, theta, r))
Both are tabulated on a log-theta grid (dln theta = 0.04) and a linear r grid per theta node and are interpolated
bilinearly (linear in ln theta at fixed r).
"""
import os
import numpy as np
import numba as nb
from scipy.special import j0
from szsim import signal as sg, cosmo_hmf as ch

ROOT = "/Users/boris/GitHub/sz-poisson-blending"
DATA = f"{ROOT}/data/q2_maps"
ARC = ch.ARCMIN
PIX = 0.25
NPIX = 2400
LBOX = PIX * NPIX            # 600 arcmin
THETA_K = np.geomspace(0.5, 10.0, 12)
MODEL = sg.SZModel()
SIGB = MODEL.beam_fwhm_arcmin / np.sqrt(8 * np.log(2))   # arcmin
NW = MODEL.noise_y_arcmin ** 2                               # arcmin^2
LNTH0, DLNTH, NTH = np.log(0.02), 0.04, 200                  # theta nodes 0.02' .. ~57' ... extend below
TH_NODES = np.exp(LNTH0 + DLNTH * np.arange(NTH))
NR = 2048


def _rmax_paint(th):
    return 5.0 * th + 6.0 * SIGB


def _rmax_resp(th):
    return 5.0 * th + 5.0 * THETA_K[-1] + 8.0 * SIGB


def build_tables(path=f"{DATA}/tables.npz", force=False):
    if os.path.exists(path) and not force:
        return dict(np.load(path))
    sb = SIGB * ARC
    sigk = MODEL.sigma_Y_exact(THETA_K) * ARC ** 2           # sr
    Nw = NW * ARC ** 2                                       # sr
    lmax = 9.0 / sb
    paint = np.zeros((NTH, NR)); resp = np.zeros((len(THETA_K), NTH, NR))
    rP = np.array([_rmax_paint(t) for t in TH_NODES]); rR = np.array([_rmax_resp(t) for t in TH_NODES])
    for i, th in enumerate(TH_NODES):
        dl = 0.25 / (rR[i] * ARC)
        l = np.arange(0.0, lmax + dl, dl)
        w = np.full_like(l, dl); w[0] *= 0.5; w[-1] *= 0.5
        B = np.exp(-0.5 * (l * sb) ** 2)
        gi = sg.template_hankel(l * th * ARC)
        rp = np.linspace(0, rP[i], NR) * ARC
        rr = np.linspace(0, rR[i], NR) * ARC
        vp = w * l * gi * B / (2 * np.pi)
        WR = np.stack([w * l * B * B * gi * sg.template_hankel(l * tk * ARC) / (2 * np.pi) * s * s / Nw
                       for tk, s in zip(THETA_K, sigk)], axis=1)       # (nl, K)
        for a in range(0, NR, 256):
            J = j0(np.outer(rp[a:a + 256], l)); paint[i, a:a + 256] = J @ vp * ARC ** 2
            J = j0(np.outer(rr[a:a + 256], l)); resp[:, i, a:a + 256] = (J @ WR).T
    np.savez(path, paint=paint, resp=resp, rP=rP, rR=rR, th_nodes=TH_NODES, theta_k=THETA_K,
             sigk_arcmin2=sigk / ARC ** 2, lnth0=LNTH0, dlnth=DLNTH)
    return dict(np.load(path))


# ---------------------------------------------------------------- numba kernels
@nb.njit(cache=True, inline="always")
def _tab_eval(tab, rmax, lnth0, dlnth, nth, lnth, r):
    t = (lnth - lnth0) / dlnth
    if t < 0.0:
        t = 0.0
    if t > nth - 1.000001:
        t = nth - 1.000001
    i0 = int(t); w = t - i0
    nr = tab.shape[1]
    v = 0.0
    for a, wa in ((i0, 1.0 - w), (i0 + 1, w)):
        u = r / rmax[a] * (nr - 1)
        if u < nr - 1:
            j = int(u); f = u - j
            v += wa * ((1 - f) * tab[a, j] + f * tab[a, j + 1])
    return v


@nb.njit(cache=True)
def paint_map(m, xs, ys, Ys, ths, paint, rP, lnth0, dlnth, pix):
    n = m.shape[0]; nth = paint.shape[0]
    for h in range(xs.size):
        lnth = np.log(ths[h])
        t = (lnth - lnth0) / dlnth
        t = min(max(t, 0.0), nth - 1.000001)
        i0 = int(t)
        R = max(rP[i0], rP[i0 + 1])
        x, y, Y = xs[h], ys[h], Ys[h]
        jx0 = int(np.floor((x - R) / pix - 0.5)); jx1 = int(np.ceil((x + R) / pix - 0.5))
        jy0 = int(np.floor((y - R) / pix - 0.5)); jy1 = int(np.ceil((y + R) / pix - 0.5))
        for jy in range(jy0, jy1 + 1):
            dy = (jy + 0.5) * pix - y
            iy = jy % n
            for jx in range(jx0, jx1 + 1):
                dx = (jx + 0.5) * pix - x
                r = np.sqrt(dx * dx + dy * dy)
                if r < R:
                    m[iy, jx % n] += Y * _tab_eval(paint, rP, lnth0, dlnth, nth, lnth, r)


@nb.njit(cache=True)
def resp_eval(resp_k, rR, lnth0, dlnth, lnth, r):
    out = np.empty(r.size)
    nth = resp_k.shape[0]
    for i in range(r.size):
        out[i] = _tab_eval(resp_k, rR, lnth0, dlnth, nth, lnth[i], r[i])
    return out


@nb.njit(cache=True)
def local_max(Q, thr):
    """Pixels that are >= all 8 periodic neighbours and >= thr (strict > for half of the neighbours to break ties)."""
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


@nb.njit(cache=True)
def update_best(Qb, kb, Sb, Nb, S, Nn, sig, k):
    n0, n1 = S.shape
    for i in range(n0):
        for j in range(n1):
            q = (S[i, j] + Nn[i, j]) / sig
            if q > Qb[i, j]:
                Qb[i, j] = q; kb[i, j] = k; Sb[i, j] = S[i, j]; Nb[i, j] = Nn[i, j]


# ---------------------------------------------------------------- filter bank
def lgrid():
    f = np.fft.fftfreq(NPIX, d=PIX) * 2 * np.pi
    fr = np.fft.rfftfreq(NPIX, d=PIX) * 2 * np.pi
    return f, fr


def filter_bank():
    """Fourier filters psi_k (rfft layout) with psi_k(0)=0, and the discrete-grid sigma_k [arcmin^2]."""
    f, fr = lgrid()
    l = np.sqrt(f[:, None] ** 2 + fr[None, :] ** 2)
    B = np.exp(-0.5 * (l * SIGB) ** 2)
    # rfft half-plane multiplicity for the Parseval sum
    mult = np.full(fr.size, 2.0); mult[0] = 1.0
    if NPIX % 2 == 0:
        mult[-1] = 1.0
    psis, sig = [], []
    for tk in THETA_K:
        T = sg.template_hankel(l * tk) * B          # u = l[arcmin^-1] * theta[arcmin]
        T2sum = (T ** 2 * mult[None, :]).sum()
        s2 = 1.0 / (T2sum / NW / (NPIX ** 2 * PIX ** 2))
        psi = s2 * T / NW
        psi[0, 0] = 0.0
        psis.append(psi.astype(np.float64)); sig.append(np.sqrt(s2))
    return psis, np.array(sig)


# ---------------------------------------------------------------- patches and truth
MCUTS = (1e13, 3e13, 1e14)      # components A=[1e13,3e13), B=[3e13,1e14), C=[1e14, inf)
VARIANTS = {"V13": (1, 1, 1), "V3e13": (0, 1, 1), "V14": (0, 0, 1), "V0": (0, 0, 0)}   # which of A,B,C are on
_PATCH_FILE = f"{DATA}/patches_10deg_M1e13.npz"


_PATCH_FILE2 = f"{DATA}/patches_10deg_M1e13_set2.npz"     # REPAIR: pids 108..323 from new skies (seeds 4..9)


def load_patch(pid):
    with np.load(_PATCH_FILE if pid < 108 else _PATCH_FILE2) as f:
        d = {k: f[f"p{pid}_{k}"].astype(np.float64) for k in ("x", "y", "z", "M500", "Y500", "theta500", "q")}
    d["x"] *= 60.0; d["y"] *= 60.0            # deg -> arcmin, in [0, 600)
    d["x"] %= LBOX; d["y"] %= LBOX
    return d


def kappa_mean(sig_k):
    """Y_hat response of the full (l=0 included) filter to a uniform y=1: removed by psi(0)=0."""
    return sig_k ** 2 * sg.RATIO_5R500 / NW


class Truth:
    """Periodic KD-trees over the halos of a patch, split in theta tiers, for exact signal decomposition."""
    EDGES = (0.0, 0.5, 1.0, 2.0, 4.0, 8.0, 16.0, 1e3)

    def __init__(self, d, tables, mask=None):
        from scipy.spatial import cKDTree
        self.d = d
        idx_all = np.arange(d["x"].size) if mask is None else np.flatnonzero(mask)
        self.tiers = []
        for lo, hi in zip(self.EDGES[:-1], self.EDGES[1:]):
            idx = idx_all[(d["theta500"][idx_all] >= lo) & (d["theta500"][idx_all] < hi)]
            if idx.size == 0:
                continue
            tmax = d["theta500"][idx].max()
            pts = np.stack([d["x"][idx], d["y"][idx]], 1) % LBOX
            self.tiers.append((idx, tmax, cKDTree(pts, boxsize=LBOX)))
        self.resp, self.rR = tables["resp"], tables["rR"]
        self.lnth = np.log(d["theta500"])

    def contributions(self, x0, y0, k):
        """Indices and contributions c_i = Y_i R_k(|x_i-x0|; theta_i) [arcmin^2] of every halo with c_i != 0."""
        I, C = [], []
        for idx, tmax, tree in self.tiers:
            rq = 5 * tmax + 5 * THETA_K[k] + 8 * SIGB
            loc = tree.query_ball_point([x0 % LBOX, y0 % LBOX], rq)
            if not loc:
                continue
            ii = idx[np.asarray(loc)]
            dx = (self.d["x"][ii] - x0 + LBOX / 2) % LBOX - LBOX / 2
            dy = (self.d["y"][ii] - y0 + LBOX / 2) % LBOX - LBOX / 2
            r = np.hypot(dx, dy)
            c = self.d["Y500"][ii] * resp_eval(self.resp[k], self.rR, LNTH0, DLNTH, self.lnth[ii], r)
            nz = c != 0
            I.append(ii[nz]); C.append(c[nz])
        if not I:
            return np.zeros(0, int), np.zeros(0)
        return np.concatenate(I), np.concatenate(C)


# ---------------------------------------------------------------- one patch
def process_patch(pid, outdir=f"{DATA}/patch_out", qmin_cand=3.5, qmin_halo=1.0, noise_seed_offset=10_000):
    import time
    from scipy import fft as sfft
    t0 = time.time()
    tab = build_tables()
    d = load_patch(pid)
    psis, sig = filter_bank()
    psis = [p.astype(np.float32) for p in psis]
    comp_masks = [(d["M500"] >= MCUTS[0]) & (d["M500"] < MCUTS[1]),
                  (d["M500"] >= MCUTS[1]) & (d["M500"] < MCUTS[2]),
                  d["M500"] >= MCUTS[2]]
    ft, means = [], []
    for msk in comp_masks:
        m = np.zeros((NPIX, NPIX))
        paint_map(m, d["x"][msk], d["y"][msk], d["Y500"][msk], d["theta500"][msk], tab["paint"], tab["rP"],
                  LNTH0, DLNTH, PIX)
        means.append(m.mean()); ft.append(sfft.rfft2(m.astype(np.float32))); del m
    t_paint = time.time() - t0
    rng = np.random.default_rng(noise_seed_offset + pid)
    nmap = rng.standard_normal((NPIX, NPIX)) * (MODEL.noise_y_arcmin / PIX)
    means.append(nmap.mean()); ft.append(sfft.rfft2(nmap.astype(np.float32))); del nmap
    K = len(THETA_K)
    best = {v: (np.full((NPIX, NPIX), -np.inf, np.float32), np.zeros((NPIX, NPIX), np.int8),
                np.zeros((NPIX, NPIX), np.float32), np.zeros((NPIX, NPIX), np.float32)) for v in VARIANTS}
    hsel = np.flatnonzero(d["q"] >= qmin_halo)
    hiy = np.floor(d["y"][hsel] / PIX).astype(int) % NPIX; hix = np.floor(d["x"][hsel] / PIX).astype(int) % NPIX
    hval = np.zeros((hsel.size, K, 4), np.float32)
    pct_q = np.array([0.1, 1, 2.5, 16, 50, 84, 97.5, 99, 99.9])
    st = {k: [] for k in ("var_A", "var_B", "var_C", "var_N", "var_AB", "var_ABN", "cov_ABN",
                          "mad_N", "mad_ABN", "mad_ABCN", "mad_BCN", "mad_CN", "pct_AB", "pct_A")}
    npk = []
    ps = {v: [] for v in VARIANTS}
    for k in range(K):
        F = [sfft.irfft2(f * psis[k], s=(NPIX, NPIX)) for f in ft]
        A, B, C, N = F
        hval[:, k, :] = np.stack([A[hiy, hix], B[hiy, hix], C[hiy, hix], N[hiy, hix]], 1)
        AB = A + B
        st["var_A"].append(A.var(dtype=np.float64)); st["var_B"].append(B.var(dtype=np.float64)); st["var_C"].append(C.var(dtype=np.float64))
        st["var_N"].append(N.var(dtype=np.float64)); st["var_AB"].append(AB.var(dtype=np.float64)); st["var_ABN"].append((AB + N).var(dtype=np.float64))
        st["cov_ABN"].append(np.mean((AB - AB.mean()) * (N - N.mean()), dtype=np.float64))
        # robust widths / percentiles on the (2x2-decimated, 1.44e6 pixel) map to limit memory
        mad = lambda a: 1.482602 * np.median(np.abs(a - np.median(a)))
        a_, b_, c_, n_ = A[::2, ::2], B[::2, ::2], C[::2, ::2], N[::2, ::2]
        st["mad_N"].append(mad(n_)); st["mad_ABN"].append(mad(a_ + b_ + n_)); st["mad_ABCN"].append(mad(a_ + b_ + c_ + n_))
        st["mad_BCN"].append(mad(b_ + c_ + n_)); st["mad_CN"].append(mad(c_ + n_))
        st["pct_AB"].append(np.percentile(a_ + b_, pct_q)); st["pct_A"].append(np.percentile(a_, pct_q))
        # single-scale noise-only peaks (for the Gaussian-field peak-count check)
        iy, ix = local_max((N / sig[k]).astype(np.float32), np.float32(3.0))
        npk.append((N[iy, ix] / sig[k]).astype(np.float32))
        for v, on in VARIANTS.items():
            S = on[0] * A + on[1] * B + on[2] * C
            S = S if isinstance(S, np.ndarray) else np.zeros_like(N)
            Qb, kb, Sb, Nb = best[v]
            update_best(Qb, kb, Sb, Nb, S, N, sig[k], k)
            # per-scale detection mode: local maxima of each q_k map separately
            iy, ix = local_max(((S + N) / sig[k]).astype(np.float32), np.float32(qmin_cand))
            ps[v].append(np.stack([iy, ix, np.full(iy.size, k), (S[iy, ix] + N[iy, ix]) / sig[k], S[iy, ix], N[iy, ix]], 1))
        del F, A, B, C, N, AB
    out = dict(pid=pid, sig=sig, means=np.array(means), hsel=hsel, hval=hval, pct_q=pct_q,
               t_paint=t_paint)
    for kk, v in st.items():
        out["st_" + kk] = np.array(v)
    for k in range(K):
        out[f"npk_{k}"] = npk[k]
    for v in VARIANTS:
        Qb, kb, Sb, Nb = best[v]
        iy, ix = local_max(Qb, np.float32(qmin_cand))
        out[f"cand_{v}"] = np.stack([iy, ix, kb[iy, ix]], 1).astype(np.int32)
        out[f"candv_{v}"] = np.stack([Qb[iy, ix], Sb[iy, ix], Nb[iy, ix]], 1).astype(np.float64)
    for v in VARIANTS:
        a = np.concatenate(ps[v], 0)
        out[f"candps_{v}"] = a[:, :3].astype(np.int32)
        out[f"candvps_{v}"] = a[:, 3:].astype(np.float64)
    out["t_total"] = time.time() - t0
    os.makedirs(outdir, exist_ok=True)
    np.savez(f"{outdir}/patch_{pid:03d}.npz", **out)
    return pid, out["t_total"], t_paint


@nb.njit(cache=True)
def merge_greedy(x, y, th, Q, r0, a, box):
    """Greedy merging: visit candidates by decreasing Q; reject one closer than r_m = max(r0, a*theta_k(accepted))
    to an already accepted (brighter) candidate.  Periodic distances.  Returns accepted indices (by decreasing Q)."""
    order = np.argsort(-Q)
    acc = np.empty(Q.size, np.int64); na = 0
    for j in order:
        ok = True
        for t in range(na):
            i = acc[t]
            dx = (x[j] - x[i] + box / 2) % box - box / 2
            dy = (y[j] - y[i] + box / 2) % box - box / 2
            rm = max(r0, a * th[i])
            if dx * dx + dy * dy < rm * rm:
                ok = False
                break
        if ok:
            acc[na] = j; na += 1
    return acc[:na]


def pids_from_env(default="0:108"):
    """REPAIR: patch-id list from env PIDS='a:b' (default = the original 108 patches)."""
    a, b = map(int, os.environ.get("PIDS", default).split(":"))
    return list(range(a, b))


TAG = os.environ.get("TAG", "")      # REPAIR: suffix for output files of runs on other patch sets
