"""Shared code for the catalogue-level blending analysis (Q2).

Contents
  * rho table I/O and trilinear interpolation (python + numba)
  * zone-algorithm neighbour search on the sphere (numba, parallel): for each query point A
    with filter scale theta_A, D_A[c] = sum_{B != A, class c} rho(thA, thB, s_AB) Y_B and
    xmax_A[c] = max_B rho Y_B   (Y in arcmin^2)
  * analytic compound-Poisson ("P(D)") machinery for the neighbour sum
Units: angles in arcmin unless stated, Y500 in arcmin^2, masses Msun (h-free).
"""
import os
import numpy as np
import numba as nb
from szsim import cosmo_hmf as ch, signal as sg

DATA = os.path.join(ch.DATA_DIR, "blending")
RHO_FILE = os.path.join(DATA, "rho_table.npz")
MASS_EDGES = np.array([1e13, 3e13, 1e14, 1e17])  # neighbour mass classes c0, c1, c2
FLOORS = {"1e13": (0, 1, 2), "3e13": (1, 2), "1e14": (2,)}
ARCMIN2_FULLSKY = 4 * np.pi / ch.ARCMIN ** 2


# ----------------------------------------------------------------------------- rho table
class RhoTable:
    def __init__(self, d):
        self.th = d["th"]; self.t = d["t"]; self.rho = np.ascontiguousarray(d["rho"])
        self.sigY = d["sigY"]; self.R_BEAM = float(d["R_BEAM"])
        self.lth0 = float(np.log(self.th[0])); self.dlth = float(np.log(self.th[1] / self.th[0]))
        self.NT = len(self.th); self.NU = len(self.t)

    def args(self):
        return (self.rho, self.lth0, self.dlth, self.R_BEAM)


def load_rho_table():
    return RhoTable(np.load(RHO_FILE))


@nb.njit(cache=True, inline="always")
def _idx(f, n):
    if f < 0.0:
        f = 0.0
    if f > n - 1.000001:
        f = n - 1.000001
    i = int(f)
    return i, f - i


@nb.njit(cache=True)
def rho_at(rho, lth0, dlth, R_BEAM, thA, thB, s):
    S = 5.0 * thA + 5.0 * thB + R_BEAM
    if s >= S:
        return 0.0
    NT = rho.shape[0]; NU = rho.shape[2]
    # [fix after verification] theta_B beyond the table (> 80', expected 0.21 halos per M>1e13 sky): the
    # lookup clamps thB to the last node but the support S uses the true thB, which stretched the 80'
    # kernel without rescaling it (DC response and variance overestimated ~(thB/80')^2 and ~(thB/80')^4).
    # Now the stretched kernel is rescaled by (S_tab/S)^2 so that its DC response (independent of thB) is exact.
    amp = 1.0
    thmax = np.exp(lth0 + dlth * (NT - 1))
    if thB > thmax:
        St = 5.0 * thA + 5.0 * thmax + R_BEAM
        amp = (St / S) ** 2
    t = np.sqrt(s / S) * (NU - 1)
    it, wt = _idx(t, NU)
    ia, wa = _idx((np.log(thA) - lth0) / dlth, NT)
    ib, wb = _idx((np.log(thB) - lth0) / dlth, NT)
    r = 0.0
    for da in range(2):
        fa = wa if da else 1.0 - wa
        for db in range(2):
            fb = wb if db else 1.0 - wb
            r += fa * fb * ((1.0 - wt) * rho[ia + da, ib + db, it] + wt * rho[ia + da, ib + db, it + 1])
    return amp * r


def rho_interp_py(tab, thA, thB, s):
    return rho_at(tab.rho, tab.lth0, tab.dlth, tab.R_BEAM, thA, thB, s)


# ----------------------------------------------------------------------------- zone search
class ZoneSet:
    """Halos sorted by (dec zone, ra) for fast disc queries."""

    def __init__(self, ra, dec, Y, th, cls, orig, hz_deg=0.2):
        self.hz = hz_deg
        nzone = int(np.ceil(180.0 / hz_deg))
        iz = np.minimum(((dec.astype(np.float64) + 90.0) / hz_deg).astype(np.int32), nzone - 1)
        key = iz * 400.0 + ra.astype(np.float64)
        o = np.argsort(key, kind="stable")
        del key
        self.ra = ra[o].astype(np.float32)
        self.dec = dec[o].astype(np.float32)
        self.cosd = np.cos(np.radians(self.dec.astype(np.float64))).astype(np.float32)
        self.Y = Y[o].astype(np.float32)
        self.th = th[o].astype(np.float32)
        self.cls = cls[o].astype(np.int8)
        self.orig = orig[o].astype(np.int32)
        izs = iz[o]; del iz
        self.zstart = np.searchsorted(izs, np.arange(nzone + 1)).astype(np.int64)
        del izs, o
        self.thmax = float(th.max()) if len(th) else 0.0
        self.n = len(self.ra)


@nb.njit(cache=True)
def _lower(a, lo, hi, x):
    while lo < hi:
        mid = (lo + hi) // 2
        if a[mid] < x:
            lo = mid + 1
        else:
            hi = mid
    return lo


@nb.njit(parallel=True, cache=True)
def _neigh_kernel(qra, qdec, qth, qself, ra, dec, cosd, Y, th, cls, orig, zstart, hz, r_extra,
                  rho, lth0, dlth, R_BEAM, D, xmax, npair):
    nq = qra.shape[0]
    nzone = zstart.shape[0] - 1
    d2r = np.pi / 180.0
    for i in nb.prange(nq):
        ra0 = qra[i]; dec0 = qdec[i]; thA = qth[i]
        r_deg = (5.0 * thA + r_extra) / 60.0
        cd0 = np.cos(dec0 * d2r)
        if r_deg >= 90.0 - abs(dec0) - 1e-9:
            dra = 181.0
        else:
            dra = np.degrees(np.arcsin(min(1.0, np.sin(r_deg * d2r) / cd0)))
        z0 = int((dec0 - r_deg + 90.0) / hz)
        z1 = int((dec0 + r_deg + 90.0) / hz)
        if z0 < 0:
            z0 = 0
        if z1 > nzone - 1:
            z1 = nzone - 1
        for z in range(z0, z1 + 1):
            a = zstart[z]; b = zstart[z + 1]
            if b <= a:
                continue
            # up to two RA intervals
            for piece in range(2):
                if dra > 180.0:
                    if piece == 1:
                        break
                    lo, hi = a, b
                else:
                    rlo = ra0 - dra; rhi = ra0 + dra
                    if piece == 0:
                        lo_v = max(rlo, 0.0); hi_v = min(rhi, 360.0)
                    else:
                        if rlo < 0.0:
                            lo_v = rlo + 360.0; hi_v = 360.0
                        elif rhi > 360.0:
                            lo_v = 0.0; hi_v = rhi - 360.0
                        else:
                            break
                    lo = _lower(ra, a, b, lo_v)
                    hi = _lower(ra, lo, b, hi_v)
                for j in range(lo, hi):
                    if orig[j] == qself[i]:
                        continue
                    sdd = np.sin(0.5 * (np.float64(dec[j]) - dec0) * d2r)
                    sda = np.sin(0.5 * (np.float64(ra[j]) - ra0) * d2r)
                    hv = sdd * sdd + cd0 * np.float64(cosd[j]) * sda * sda
                    s = 2.0 * np.arcsin(min(1.0, np.sqrt(hv))) / d2r * 60.0  # arcmin
                    rr = rho_at(rho, lth0, dlth, R_BEAM, thA, np.float64(th[j]), s)
                    if rr != 0.0:
                        c = cls[j]
                        x = rr * np.float64(Y[j])
                        D[i, c] += x
                        if x > xmax[i, c]:
                            xmax[i, c] = x
                        npair[i] += 1


def neighbour_sums(zsets, tab, qra, qdec, qth, qself, ncls=3):
    """Sum over all zone sets (tiers).  Returns D (nq, ncls), xmax (nq, ncls), npair (nq)."""
    nq = len(qra)
    D = np.zeros((nq, ncls)); xmax = np.zeros((nq, ncls)); npair = np.zeros(nq, np.int64)
    qra = np.ascontiguousarray(qra, np.float64); qdec = np.ascontiguousarray(qdec, np.float64)
    qth = np.ascontiguousarray(qth, np.float64); qself = np.ascontiguousarray(qself, np.int64)
    for zs in zsets:
        if zs.n == 0:
            continue
        r_extra = 5.0 * zs.thmax + tab.R_BEAM + 1e-6
        _neigh_kernel(qra, qdec, qth, qself, zs.ra, zs.dec, zs.cosd, zs.Y, zs.th, zs.cls, zs.orig,
                      zs.zstart, zs.hz, r_extra, *tab.args(), D, xmax, npair)
    return D, xmax, npair


def mass_class(M):
    return (np.searchsorted(MASS_EDGES, M, side="right") - 1).clip(0, len(MASS_EDGES) - 2).astype(np.int8)


def dc_response(sigY_arcmin2, model=None):
    """int d^2s rho_AB(s) = sigma_A^2 RATIO^2 / N_w  [arcmin^2] (independent of theta_B)."""
    m = model or sg.SZModel()
    return (np.asarray(sigY_arcmin2) * ch.ARCMIN ** 2) ** 2 * sg.RATIO_5R500 ** 2 / m.N_white_sr / ch.ARCMIN ** 2


# ----------------------------------------------------------------------------- analytic P(D)
def neighbour_population(zsub=6, msub=6, ngh=16, model=None):
    """Discretised neighbour population: theta_B, Y_B (with scatter), class, w = expected number per
    arcmin^2 (full sky / 4 pi).  Grid = HMF table subsampled (trapezoid weights) x Gauss-Hermite."""
    m = model or sg.SZModel()
    T = ch.default_table()
    z = T.z[::zsub]; lnM = T.lnM[::msub]
    d2N = T.d2N[::zsub, ::msub]
    wz = np.full(len(z), z[1] - z[0]); wz[0] *= 0.5; wz[-1] *= 0.5
    x, wg = np.polynomial.hermite_e.hermegauss(ngh); wg = wg / wg.sum()
    Z, LM = np.meshgrid(z, lnM, indexing="ij")
    M = np.exp(LM)
    th = m.theta500(M, Z)
    Ym = m.Y500_med(M, Z)
    out = [[], [], [], []]
    for c in range(len(MASS_EDGES) - 1):
        # exact class boundaries: trapezoid weights with partial-cell clipping (HMFTable._weights)
        wM = T._weights(lnM, np.log(MASS_EDGES[c]), np.log(MASS_EDGES[c + 1]))
        W = d2N * wz[:, None] * wM[None, :] / ARCMIN2_FULLSKY
        sel = (W > 0).ravel()
        out[0].append(np.repeat(th.ravel()[sel], ngh))
        out[1].append((Ym.ravel()[sel][:, None] * np.exp(m.sigma_lnY * x)[None, :]).ravel())
        out[2].append(np.full(sel.sum() * ngh, c, np.int8))
        out[3].append((W.ravel()[sel][:, None] * wg[None, :]).ravel())
    thB, YB, cB, wB = [np.concatenate(o) for o in out]
    return thB, YB, cB, wB


XLOG_EDGES = np.concatenate([[0.0], np.logspace(-8, np.log10(500.0), 2601)])  # bin 0 = x < 1e-8
X_CAP = 500.0


@nb.njit(parallel=True, cache=True)
def _rate_hist_kernel(thA, sigA, thB, YB, cB, wB, tnodes, rho, lth0, dlth, R_BEAM, ledges_lo, dlog, nb_,
                      W, WX, WX2, cap, capx):
    # W[c, b] etc. accumulated per thread chunk then reduced: use per-point loop with atomic-free trick:
    n = thB.shape[0]
    NU = tnodes.shape[0]
    nthr = nb.get_num_threads()
    Wt = np.zeros((nthr, W.shape[0], W.shape[1])); WXt = np.zeros_like(Wt); WX2t = np.zeros_like(Wt)
    capt = np.zeros((nthr, W.shape[0])); capxt = np.zeros((nthr, W.shape[0]))
    chunk = (n + nthr - 1) // nthr
    for tid in nb.prange(nthr):
        for k in range(tid * chunk, min(n, (tid + 1) * chunk)):
            S = 5.0 * thA + 5.0 * thB[k] + R_BEAM
            c = cB[k]
            for it in range(NU):
                t = tnodes[it]
                # trapezoid weight in t of dA = 4 pi S^2 t^3 dt
                if it == 0 or it == NU - 1:
                    dt = 0.5 * (tnodes[1] - tnodes[0])
                else:
                    dt = tnodes[1] - tnodes[0]
                dA = 4.0 * np.pi * S * S * t * t * t * dt
                if dA == 0.0:
                    continue
                s = S * t * t
                rr = rho_at(rho, lth0, dlth, R_BEAM, thA, thB[k], s)
                x = rr * YB[k] / sigA
                w = wB[k] * dA
                if x >= 500.0:
                    capt[tid, c] += w
                    capxt[tid, c] += w * x
                    continue
                if x < 1e-8:
                    b = 0
                else:
                    b = 1 + int((np.log10(x) - ledges_lo) / dlog)
                    if b > nb_ - 1:
                        b = nb_ - 1
                Wt[tid, c, b] += w; WXt[tid, c, b] += w * x; WX2t[tid, c, b] += w * x * x
    for tid in range(nthr):
        W += Wt[tid]; WX += WXt[tid]; WX2 += WX2t[tid]; cap += capt[tid]; capx += capxt[tid]


def rate_histogram(thA, sigA, pop, tab):
    """R(x | thA) binned in log x for each neighbour class: (W, WX, WX2, cap, capx) with
    W[c,b] = expected number of neighbours (per filter) with x in bin b, x = rho Y / sigma_A (SNR units)."""
    thB, YB, cB, wB = pop
    nbins = len(XLOG_EDGES) - 1
    W = np.zeros((3, nbins)); WX = np.zeros_like(W); WX2 = np.zeros_like(W); cap = np.zeros(3); capx = np.zeros(3)
    ledges_lo = np.log10(XLOG_EDGES[1]); dlog = np.log10(XLOG_EDGES[2] / XLOG_EDGES[1])
    _rate_hist_kernel(float(thA), float(sigA), thB, YB, cB.astype(np.int64), wB, tab.t, *tab.args(),
                      ledges_lo, dlog, nbins, W, WX, WX2, cap, capx)
    return W, WX, WX2, cap, capx


class PD:
    """FFT compound-Poisson distribution of delta = sum_i x_i given a binned rate."""

    def __init__(self, N=2 ** 18, dd=0.005, x_small=0.02, off=1.0):
        self.N, self.dd, self.xs, self.off = N, dd, x_small, off
        self.k = 2 * np.pi * np.fft.rfftfreq(N, d=dd)
        self.grid = np.arange(N) * dd - off

    def pdf(self, W, WX, WX2, xcut=np.inf):
        """W, WX, WX2: 1D binned rates (already summed over classes).  Contributions in bins with
        mean x >= xcut are excluded (truncated compound Poisson).  Returns p_j (prob per cell)."""
        with np.errstate(invalid="ignore", divide="ignore"):
            xb = np.where(W > 0, WX / W, 0.0)
        keep = (W > 0) & (xb < xcut)
        small = keep & (xb < self.xs)
        big = keep & ~small
        mu_s = WX[small].sum(); v_s = WX2[small].sum()
        # CIC of the large contributions onto the grid (origin at delta = 0 -> index 0 of R)
        R = np.zeros(self.N)
        f = xb[big] / self.dd
        i0 = np.floor(f).astype(np.int64); fr = f - i0
        np.add.at(R, i0, W[big] * (1 - fr))
        np.add.at(R, i0 + 1, W[big] * fr)
        lam = W[big].sum()
        # + Gaussian smoothing of width 2*dd (variance 1e-4): suppresses Gibbs ringing of the
        #   off-grid shift / near-delta distributions (without it a 1e-4 ringing floor appears)
        lnphi_c = (np.fft.rfft(R) - lam - 1j * self.k * (mu_s + self.off)
                   - 0.5 * self.k ** 2 * (v_s + (2 * self.dd) ** 2))
        p = np.fft.irfft(np.exp(lnphi_c), n=self.N)
        return np.clip(p, 0, None)

    @staticmethod
    def lam_above(W, WX, xcut):
        with np.errstate(invalid="ignore", divide="ignore"):
            xb = np.where(W > 0, WX / W, 0.0)
        return W[(W > 0) & (xb >= xcut)].sum()


# ----------------------------------------------------------------------------- [repair] fine-grid CDF
def pd_cdf_fine(W, WX, WX2, lam_cap, dd, N=2 ** 22):
    """[added after verification] High-resolution CDF of the untruncated neighbour sum for small delta.

    The production PD grid (dd = 1e-3, plus a Gaussian smoothing of width 2 dd, and a CDF read at the cell
    centres instead of the cell edges) cannot resolve medians of ~0.006 (neighbour floor 1e14, small theta_A).
    Exact identity used here: since every x >= 0 (|rho| < 1e-5 rho(0) beyond the support), for t < x_c
        P(delta <= t) = exp(-Lambda(x >= x_c)) * P_trunc(delta <= t),
    where P_trunc is the compound Poisson with the rate restricted to x < x_c.  P_trunc is computed on a grid
    of spacing dd (range N dd, x_c = range / 5 so the truncated sum stays well inside the FFT box).
    Returns (t_edges, cdf, x_c): cdf[j] = P(delta <= t_edges[j]) valid for t_edges < x_c."""
    off = 50 * dd
    pd_ = PD(N=N, dd=dd, x_small=2 * dd, off=off)
    xc = min(N * dd / 5.0, X_CAP)
    lam_above = PD.lam_above(W, WX, xc) + lam_cap
    p = pd_.pdf(W, WX, WX2, xcut=xc)
    p /= p.sum()
    cdf = np.exp(-lam_above) * np.cumsum(p)
    return pd_.grid + 0.5 * dd, cdf, xc
