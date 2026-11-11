"""Fast, cosmology-dependent version of the catalogue-level compound-Poisson blending calculation
(analysis/blending/b03_analytic.py), for the grid in sigma8, Omega_m, (1-b), sigma_lnY.

What depends on the parameters
  * neighbour rate R(x | theta_A): HMF (sigma8, Om), theta500(M,z) (Om via rho_c(z), D_A), Y500 (Om via E, D_A;
    (1-b)), Y scatter (sigma_lnY)
  * primary population H(z-bin, theta_A node, q node)
What does not: rho(thA, thB, s) table (angles only), sigma_Y(theta) (beam, noise, template).

Speed-up relative to b03 (validated in c01_validate.py against data/blending/analytic.npz):
  rate histogram  W(x|thA) = sum_B w_B int dA delta(x - rho Y_B / sigma_A)
  is computed as a convolution in ln:  ln x = ln rho + ln Y - ln sigma_A.
  Neighbours are binned on (fine ln theta_B grid, ln Y grid); for each theta_A node the kernel area histogram
  K(theta_B; ln rho) is built once from the same rho table (blendlib.rho_at), and W, WX, WX2 follow from FFT
  convolutions of (n, nY, nY^2) with (K, K rho, K rho^2): totals of the first two moments are exact.
Everything h-free, angles arcmin, Y500 arcmin^2, masses Msun.  Nothing is written to data/.
"""
import os, sys
import numpy as np
import numba as nb
from scipy.special import ndtr

ROOT = "/Users/boris/GitHub/sz-poisson-blending"
sys.path.insert(0, os.path.join(ROOT, "analysis", "blending"))
import blendlib as bl                          # noqa: E402  (rho table, rho_at)
from szsim import signal as sg, cosmo_hmf as ch  # noqa: E402

from colossus.cosmology import cosmology        # noqa: E402
from colossus.lss import mass_function          # noqa: E402

OUT = os.path.join(ROOT, "data_cosmo")
ARCMIN = ch.ARCMIN
ARCMIN2_FULLSKY = 4 * np.pi / ARCMIN ** 2
C_KMS = 299792.458
MASS_EDGES = bl.MASS_EDGES
P18 = dict(cosmology.cosmologies["planck18"])     # flat, H0 67.66, Om0 .3111, Ob0 .049, s8 .8102, ns .9665
FID = dict(sigma8=0.81, Om=0.31, omb=0.8, slnY=0.17)   # grid centre
QNODES = np.concatenate([np.arange(0.1, 20.0, 0.1), np.geomspace(20.0, 3000.0, 60)])
Q_PRIM = 0.5
QCUT = 5.0
QEDGES = np.array([5.0, 6.0, 7.0, 8.0, 10.0, 12.5, 15.0, 20.0, 30.0, 50.0, np.inf])
ZEDGES = np.array([0.01, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 2.0, 3.0])

_RHO = None
_SIGA = None
_MODEL0 = None


def rho_tab():
    global _RHO, _SIGA, _MODEL0
    if _RHO is None:
        _RHO = bl.load_rho_table()
        _MODEL0 = sg.SZModel()
        _SIGA = _MODEL0.sigma_Y(_RHO.th)
    return _RHO, _SIGA, _MODEL0


# ----------------------------------------------------------------------------- cosmology
class Cosmo:
    """Background + HMF for flat LCDM = Planck18 with sigma8, Om0 replaced (h, Ob0, ns fixed)."""
    _n = 0

    def __init__(self, sigma8, Om):
        Cosmo._n += 1
        p = dict(P18); p["sigma8"] = float(sigma8); p["Om0"] = float(Om)
        self.c = cosmology.setCosmology(f"grid{os.getpid()}_{Cosmo._n}", **p, persistence="")
        self.h = self.c.h
        zg = np.linspace(0.0, 4.0, 40001)
        self._chi = (zg, self.c.comovingDistance(0.0, zg) / self.h)

    def Ez(self, z):
        return self.c.Ez(np.asarray(z, float))

    def DA(self, z):
        z = np.asarray(z, float)
        return np.interp(z, *self._chi) / (1 + z)

    def rho_c(self, z):
        return self.c.rho_c(np.asarray(z, float)) * 1e9 * self.h ** 2

    def dVdzdO(self, z):
        return (C_KMS / self.c.H0) / self.Ez(z) * np.interp(z, *self._chi) ** 2

    def d2N(self, z, lnM):
        """full-sky d2N/dz/dlnM on (z, lnM) grid."""
        cosmology.setCurrent(self.c)
        M = np.exp(lnM) * self.h
        out = np.empty((len(z), len(lnM)))
        for i, zz in enumerate(z):
            out[i] = mass_function.massFunction(M, float(zz), mdef="500c", model="tinker08",
                                                q_out="dndlnM") * self.h ** 3
        return out * (self.dVdzdO(z) * 4 * np.pi)[:, None]


class SZ:
    """Planck13 Y-M with (1-b), sigma_lnY; angles from the Cosmo object."""

    def __init__(self, cosmo, omb=0.8, slnY=0.173, logA=-0.19, alpha=1.79, beta_E=2 / 3, Mp=6e14):
        self.co, self.omb, self.slnY = cosmo, omb, slnY
        self.logA, self.alpha, self.beta_E, self.Mp = logA, alpha, beta_E, Mp

    def Y500_med(self, M, z):
        A = 1e-4 * 10 ** self.logA
        Y = A * self.co.Ez(z) ** self.beta_E * (self.omb * M / self.Mp) ** self.alpha
        return Y / self.co.DA(z) ** 2 / ARCMIN ** 2

    def theta500(self, M, z):
        R = (3 * M / (4 * np.pi * 500 * self.co.rho_c(z))) ** (1 / 3)
        return R / self.co.DA(z) / ARCMIN


def trap_w(x):
    w = np.full(len(x), x[1] - x[0]); w[0] *= .5; w[-1] *= .5
    return w


# ----------------------------------------------------------------------------- grids
NZ_NB, NM_NB, NGH_NB = 1201, 201, 16       # neighbour population (b03: zsub=1, msub=6, ngh=16)
NZ_PR, NM_PR, NGH_PR = 601, 601, 24        # primaries (b03: zsub=2, msub=2, ngh=24)
LNTB = np.linspace(np.log(0.03), np.log(600.0), 241)   # fine theta_B grid for the kernel
DLN = np.log(10.0) * np.log10(500.0 / 1e-8) / 2600     # = b03 log-bin width (0.00946)
LNR0, NR = np.log(1e-10), int(np.ceil((np.log(20.0) - np.log(1e-10)) / DLN)) + 1
LNY0 = np.log(1e-10)
NY = int(np.ceil((np.log(1e4) - LNY0) / DLN)) + 1
NFFT = 1 << int(np.ceil(np.log2(NR + NY + 2)))


def hmf_grid(cosmo, nz, nM, zmin=0.01, zmax=3.0, Mmin=1e13, Mmax=1e16):
    z = np.linspace(zmin, zmax, nz)
    lnM = np.linspace(np.log(Mmin), np.log(Mmax), nM)
    return z, lnM, cosmo.d2N(z, lnM)


def neighbour_binned(cosmo, sz):
    """n_c(jB, kY) for moments 0,1,2 of Y: arrays (3 classes, 3 moments, nB, NY) [per arcmin^2],
    plus ybar_c = sum w Y and counts."""
    z, lnM, d2N = hmf_grid(cosmo, NZ_NB, NM_NB)
    wz = trap_w(z)
    x, wg = np.polynomial.hermite_e.hermegauss(NGH_NB); wg = wg / wg.sum()
    Z, LM = np.meshgrid(z, lnM, indexing="ij")
    M = np.exp(LM)
    th = sz.theta500(M, Z).ravel()
    Ym = sz.Y500_med(M, Z).ravel()
    nB = len(LNTB)
    out = np.zeros((3, 3, nB, NY))
    ybar = np.zeros(3); cnt = np.zeros(3); ytrue = np.zeros(3)
    fb = np.clip((np.log(th) - LNTB[0]) / (LNTB[1] - LNTB[0]), 0, nB - 1.000001)
    ib = fb.astype(int); wb = fb - ib
    for c in range(3):
        wM = ch.HMFTable._weights(lnM, np.log(MASS_EDGES[c]), np.log(MASS_EDGES[c + 1]))
        W = (d2N * wz[:, None] * wM[None, :] / ARCMIN2_FULLSKY).ravel()
        sel = W > 0
        for g in range(NGH_NB):
            Y = Ym[sel] * np.exp(sz.slnY * x[g])
            w = W[sel] * wg[g]
            ky = np.clip(np.floor((np.log(Y) - LNY0) / DLN).astype(int), 0, NY - 1)
            for d, f in ((0, 1 - wb[sel]), (1, wb[sel])):
                idx = (ib[sel] + d) * NY + ky
                for mom in range(3):
                    out[c, mom] += np.bincount(idx, w * f * Y ** mom, minlength=nB * NY).reshape(nB, NY)
            ybar[c] += (w * Y).sum(); cnt[c] += w.sum() * ARCMIN2_FULLSKY
    return out, ybar, cnt


@nb.njit(cache=True)
def _kernel_hist(thA, lntb, tnodes, rho, lth0, dlth, R_BEAM, lnr0, dln, nr, K0, K1, K2, T1, T2):
    NU = tnodes.shape[0]
    dt = tnodes[1] - tnodes[0]
    rmin = np.exp(lnr0)
    for jb in range(lntb.shape[0]):
        thB = np.exp(lntb[jb])
        S = 5.0 * thA + 5.0 * thB + R_BEAM
        for it in range(NU):
            t = tnodes[it]
            w = dt if (it > 0 and it < NU - 1) else 0.5 * dt
            dA = 4.0 * np.pi * S * S * t * t * t * w
            if dA == 0.0:
                continue
            rr = bl.rho_at(rho, lth0, dlth, R_BEAM, thA, thB, S * t * t)
            if rr > rmin:
                k = int((np.log(rr) - lnr0) / dln)
                if k > nr - 1:
                    k = nr - 1
                K0[jb, k] += dA; K1[jb, k] += dA * rr; K2[jb, k] += dA * rr * rr
            else:
                T1[jb] += dA * rr; T2[jb] += dA * rr * rr


def kernel_fft(i):
    tab, sigA, _ = rho_tab()
    nB = len(LNTB)
    K = np.zeros((3, nB, NR)); T1 = np.zeros(nB); T2 = np.zeros(nB)
    _kernel_hist(float(tab.th[i]), LNTB, tab.t, *tab.args(), LNR0, DLN, NR, K[0], K[1], K[2], T1, T2)
    return np.fft.rfft(K, n=NFFT, axis=-1), T1, T2


def rates(nbin):
    """For each theta_A node: binned rate per class (W, WX, WX2) on the ln(rho Y) grid, x-bin centres,
    plus Gaussian 'tiny' part (rho < 1e-10 incl. negative ring): mean and variance per class."""
    tab, sigA, _ = rho_tab()
    NT = tab.NT
    nF = np.fft.rfft(nbin, n=NFFT, axis=-1)           # (3 cls, 3 mom, nB, nf)
    sums = nbin.sum(-1)                                  # (3, 3, nB)
    nv = NR + NY - 1
    W = np.zeros((NT, 3, nv)); WX = np.zeros_like(W); WX2 = np.zeros_like(W)
    mut = np.zeros((NT, 3)); vart = np.zeros((NT, 3))
    for i in range(NT):
        KF, T1, T2 = kernel_fft(i)
        for mom, arr in ((0, W), (1, WX), (2, WX2)):
            prod = (nF[:, mom] * KF[mom][None]).sum(1)  # (3, nf)
            arr[i] = np.fft.irfft(prod, n=NFFT, axis=-1)[:, :nv] / sigA[i] ** mom
        mut[i] = (sums[:, 1] * T1[None]).sum(-1) / sigA[i]
        vart[i] = (sums[:, 2] * T2[None]).sum(-1) / sigA[i] ** 2
    # FFT round-off: clean tiny negatives
    tol = 1e-12 * np.abs(W).max()
    bad = W < tol
    W[bad] = 0; WX[bad] = 0; WX2[bad] = 0
    return W, WX, WX2, mut, vart


def primary_binned(cosmo, sz, zedges=ZEDGES):
    """H[zb, i, k]: expected full-sky count of halos (M>1e13) with q_A>Q_PRIM, CIC in ln theta (rho-table
    nodes) and quadratic Lagrange in q (QNODES), split in z bins."""
    tab, _, m0 = rho_tab()
    z, lnM, d2N = hmf_grid(cosmo, NZ_PR, NM_PR)
    wz = trap_w(z); wM = trap_w(lnM)
    x, wg = np.polynomial.hermite_e.hermegauss(NGH_PR); wg /= wg.sum()
    M = np.exp(lnM)
    nzb = len(zedges) - 1
    nq = len(QNODES)
    H = np.zeros(nzb * tab.NT * nq)
    zb_of = np.clip(np.searchsorted(zedges, z, side="right") - 1, 0, nzb - 1)
    for iz, zz in enumerate(z):
        th = sz.theta500(M, zz)
        q0 = sz.Y500_med(M, zz) / m0.sigma_Y(th)
        q = (q0[:, None] * np.exp(sz.slnY * x)[None, :]).ravel()
        w = ((d2N[iz] * wz[iz] * wM)[:, None] * wg[None, :]).ravel()
        tt = np.repeat(th, NGH_PR)
        s = q > Q_PRIM
        q, w, tt = q[s], w[s], tt[s]
        fa = np.clip((np.log(tt) - tab.lth0) / tab.dlth, 0, tab.NT - 1.000001)
        ia = fa.astype(int); wa = fa - ia
        k = np.clip(np.searchsorted(QNODES, q) - 1, 0, nq - 2)
        j0 = np.clip(np.where(q - QNODES[k] < QNODES[k + 1] - q, k - 1, k), 0, nq - 3)
        x0, x1, x2 = QNODES[j0], QNODES[j0 + 1], QNODES[j0 + 2]
        L = [(q - x1) * (q - x2) / ((x0 - x1) * (x0 - x2)), (q - x0) * (q - x2) / ((x1 - x0) * (x1 - x2)),
             (q - x0) * (q - x1) / ((x2 - x0) * (x2 - x1))]
        base = zb_of[iz] * tab.NT * nq
        for da, fa_ in ((0, 1 - wa), (1, wa)):
            for db in range(3):
                H += np.bincount(base + (ia + da) * nq + j0 + db, w * fa_ * L[db], minlength=H.size)
    return H.reshape(nzb, tab.NT, nq)


# ----------------------------------------------------------------------------- P(D)
class PDg:
    def __init__(self, N, dd, xs, off=1.0):
        self.N, self.dd, self.xs, self.off = N, dd, xs, off
        self.k = 2 * np.pi * np.fft.rfftfreq(N, d=dd)
        self.grid = np.arange(N) * dd - off

    def pdf(self, xb, W, WX2, xcut, mu_g, var_g):
        """truncated compound Poisson; xb bin means (W>0 only), Gaussian part (mu_g, var_g) added."""
        keep = xb < xcut
        small = keep & (xb < self.xs)
        big = keep & ~small
        mu_s = (W[small] * xb[small]).sum() + mu_g
        v_s = WX2[small].sum() + var_g
        f = xb[big] / self.dd
        i0 = np.floor(f).astype(np.int64); fr = f - i0
        R = np.bincount(i0, W[big] * (1 - fr), minlength=self.N + 1)[:self.N] + \
            np.bincount(i0 + 1, W[big] * fr, minlength=self.N + 1)[:self.N]
        lam = W[big].sum()
        lnphi = (np.fft.rfft(R) - lam - 1j * self.k * (mu_s + self.off) - 0.5 * self.k ** 2 * (v_s + (2 * self.dd) ** 2))
        p = np.clip(np.fft.irfft(np.exp(lnphi), n=self.N), 0, None)
        return p / p.sum()


PD_FAST = dict(small=(1 << 16, 0.002, 0.004), big=(1 << 17, 0.008, 0.016))
PD_B03 = dict(small=(1 << 18, 0.001, 0.002), big=(1 << 19, 0.002, 0.004))


def blend_probs(W, WX, WX2, mut, vart, H, pdmode=PD_FAST, qcut=QCUT, qedges=QEDGES):
    """Per (theta_A node, q node): P(dominant & q_b in q-bin), and at q_cut: Pdet, B2(0.1), B2(0.2), B3, lost,
    sum of fractional Y bias.  Map mean removed (c=1)."""
    tab = rho_tab()[0]
    NT, nq = tab.NT, len(QNODES)
    pds = PDg(*pdmode["small"]), PDg(*pdmode["big"])
    nE = len(qedges) - 1
    Pbin = np.zeros((NT, nq, nE))
    keys = ["det", "b2_10", "b2_20", "b3", "lost", "ybias"]
    P = {k: np.zeros((NT, nq)) for k in keys}
    Hany = np.abs(H).sum(0)
    for i in range(NT):
        Wt, WXt, WX2t = W[i].sum(0), WX[i].sum(0), WX2[i].sum(0)
        nz = Wt > 0
        xb = WXt[nz] / Wt[nz]; Wn = Wt[nz]; WX2n = WX2t[nz]
        mu_g, var_g = mut[i].sum(), vart[i].sum()
        sh = WXt.sum() + mu_g                      # monopole shift = exact mean of delta under this rate
        order = np.argsort(xb); xs_sorted = xb[order]; Wcum = np.concatenate([[0], np.cumsum(Wn[order])])
        for k, qa in enumerate(QNODES):
            if Hany[i, k] == 0:
                continue
            xcut = min(qa, bl.X_CAP)
            lam_above = Wcum[-1] - Wcum[np.searchsorted(xs_sorted, xcut, side="left")]
            pdom = np.exp(-lam_above)
            pd_ = pds[0] if xcut < 20 else pds[1]
            p = pd_.pdf(xb, Wn, WX2n, xcut, mu_g, var_g)
            g = pd_.grid; dd = pd_.dd; jmax = len(g)
            tail = np.concatenate([np.cumsum(p[::-1])[::-1], [0.0]])
            pgt = np.concatenate([np.cumsum((p * (g - sh))[::-1])[::-1], [0.0]])

            def jidx(d):
                return int(min(max(np.floor((d + pd_.off) / dd) + 1, 0), jmax))

            def make_I(a, extra=False):
                jlo, jhi = jidx(a - 8.5), jidx(a + 8.5)
                ph = ndtr(g[jlo:jhi] - a)
                pg = p[jlo:jhi] * ph
                part = np.concatenate([np.cumsum(pg[::-1])[::-1], [0.0]])

                def I(jt):
                    jt = min(max(jt, 0), jmax)
                    if jt >= jhi:
                        return tail[jt]
                    return part[max(jt - jlo, 0)] + tail[jhi]
                if extra:
                    yb = (pg * (g[jlo:jhi] - sh)).sum() + pgt[jhi]
                    return I, yb
                return I

            # q-bins
            Ie = [make_I(e - qa + sh)(0) if np.isfinite(e) else 0.0 for e in qedges]
            Pbin[i, k] = pdom * (np.array(Ie[:-1]) - np.array(Ie[1:]))
            I5, yb = make_I(qcut - qa + sh, extra=True)
            Iall = I5(0)
            j0 = jidx(sh)
            Ipos = I5(j0)
            phi0 = ndtr(qa - qcut)
            P["det"][i, k] = pdom * Iall
            P["b3"][i, k] = pdom * max(Ipos - phi0 * tail[j0], 0.0)
            P["lost"][i, k] = pdom * max(phi0 * (1 - tail[j0]) - (Iall - Ipos), 0.0)
            P["b2_10"][i, k] = pdom * I5(jidx(sh + 0.1 * qa))
            P["b2_20"][i, k] = pdom * I5(jidx(sh + 0.2 * qa))
            P["ybias"][i, k] = pdom * yb / qa
    return Pbin, P


def summarize(H, Pbin, P, qcut=QCUT, qedges=QEDGES):
    """Full-sky numbers.  H (nzb, NT, nq)."""
    Phi = ndtr(QNODES - qcut)
    Hs = H.sum(0)
    Ns = (Hs * Phi[None]).sum()
    Nd = (Hs * P["det"]).sum()
    r = dict(N_single=Ns, N_det_b=Nd, ratio=Nd / Ns,
             B2_10=(Hs * P["b2_10"]).sum() / Nd, B2_20=(Hs * P["b2_20"]).sum() / Nd,
             B3=(Hs * P["b3"]).sum() / Nd, lost=(Hs * P["lost"]).sum() / Ns,
             ybias=(Hs * P["ybias"]).sum() / Nd)
    # (z, q) binned counts, blended and naive
    PhiE = np.array([ndtr(QNODES - e) if np.isfinite(e) else np.zeros_like(QNODES) for e in qedges])
    Pn = (PhiE[:-1] - PhiE[1:]).T                      # (nq, nE)
    r["Nzq_b"] = np.einsum("zik,ike->ze", H, Pbin)
    r["Nzq_n"] = np.einsum("zik,ke->ze", H, Pn)
    return r


def naive_counts(sigma8, Om, omb, slnY, zedges=ZEDGES, qedges=QEDGES):
    co = Cosmo(sigma8, Om); sz = SZ(co, omb, slnY)
    H = primary_binned(co, sz, zedges)
    PhiE = np.array([ndtr(QNODES - e) if np.isfinite(e) else np.zeros_like(QNODES) for e in qedges])
    Pn = (PhiE[:-1] - PhiE[1:]).T
    return np.einsum("zik,ke->ze", H, Pn)


def run_point(sigma8, Om, omb, slnY, pdmode=PD_FAST, verbose=False, keep_rates=False):
    import time
    t0 = time.time()
    tab, sigA, m0 = rho_tab()
    co = Cosmo(sigma8, Om); sz = SZ(co, omb, slnY)
    nbin, ybar_c, cnt_c = neighbour_binned(co, sz)
    t1 = time.time()
    W, WX, WX2, mut, vart = rates(nbin)
    del nbin
    t2 = time.time()
    H = primary_binned(co, sz)
    t3 = time.time()
    Pbin, P = blend_probs(W, WX, WX2, mut, vart, H, pdmode)
    t4 = time.time()
    r = summarize(H, Pbin, P)
    # confusion statistics per theta_A node
    nv = W.shape[-1]
    xc = np.exp(LNR0 + LNY0 + (np.arange(nv) + 1.0) * DLN)[None, :] / sigA[:, None]   # nominal bin centre
    with np.errstate(invalid="ignore", divide="ignore"):
        xbar = np.where(W.sum(1) > 0, WX.sum(1) / W.sum(1), xc)
    r["mu_c"] = WX.sum(-1) + mut                         # (NT, 3) mean of delta per class
    r["var500_c"] = np.array([[(WX2[i, c] * (xbar[i] < 500)).sum() + vart[i, c] for c in range(3)]
                              for i in range(tab.NT)])
    r["var5_c"] = np.array([[(WX2[i, c] * (xbar[i] < 5)).sum() + vart[i, c] for c in range(3)]
                            for i in range(tab.NT)])
    r["lam1"] = np.array([(W[i].sum(0) * (xbar[i] >= 1)).sum() for i in range(tab.NT)])
    r["ybar_c"] = ybar_c; r["ytot"] = ybar_c.sum() * sg.RATIO_5R500; r["cnt_c"] = cnt_c
    r["mu_formula"] = ybar_c.sum() * bl.dc_response(sigA, m0) / sigA
    # detection-weighted mean boost and variance (weights = blended detections per theta_A node)
    wdet = (H.sum(0) * P["det"]).sum(1)
    r["wdet"] = wdet
    r["mu_det"] = (wdet * r["mu_c"].sum(1)).sum() / wdet.sum()
    r["var5_det"] = (wdet * r["var5_c"].sum(1)).sum() / wdet.sum()
    r["var500_det"] = (wdet * r["var500_c"].sum(1)).sum() / wdet.sum()
    r["H"] = H; r["Pdet"] = P["det"]; r["params"] = np.array([sigma8, Om, omb, slnY])
    if keep_rates:
        r["W"] = W; r["WX"] = WX; r["WX2"] = WX2; r["mut"] = mut; r["vart"] = vart
    r["timing"] = np.array([t1 - t0, t2 - t1, t3 - t2, t4 - t3])
    if verbose:
        print(f"  timing: neighbours {t1-t0:.1f}s rates {t2-t1:.1f}s primaries {t3-t2:.1f}s PD {t4-t3:.1f}s", flush=True)
    return r
