"""Step-2 MAP-LEVEL verification (lens: maps): shared code.  Extends the step-1 simulation check's vlib
(verify_real/step1_maps/vlib.py: own SO LAT noise, own CAMB CMB, own Arnaud painter, own Poisson patch sampler,
the build's filter N_y(l), w_i(l), sigma_Y(theta)).  NOTHING from analysis/realistic/step2 (the build's step-2 code)
is imported here.

New in step 2:
  * exact read-out of the filtered map at arbitrary (continuous) halo positions by a type-2 NUFFT (finufft) of the
    filtered Fourier coefficients (no pixel interpolation): Y_hat_j(x_A) for the filter at node theta_j;
  * self response R_j(theta) of the node-j filter to a painted halo of size theta at its own position, as the same
    discrete mode sum the map uses (so the neighbour sum delta = (Y_hat - R Y_A) / sigma_j is exact on the grid);
  * own Hankel tables rho_j(theta_B, s) of the filter response (trapezoid on the build's l grid, the build's filter
    template on the filter side, OUR projection-slice template on the cluster side), used ONLY to decide the
    attribution (dominance) rule x_B = rho_AB(s) Y_B / sigma_A < fdom q_A;
  * dominance search (numba): brightest single-neighbour contribution x_max at every primary, own pair search.
"""
import os, sys
NTHREADS = int(os.environ.get("VNTHREADS", "3"))
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "NUMBA_NUM_THREADS"):
    os.environ[_v] = str(NTHREADS)
import numpy as np                                     # noqa: E402  (BLAS threads fixed here, before vlib)
import numba as nb                                     # noqa: E402
import scipy.fft as sfft                               # noqa: E402
from scipy.special import j0, ndtr                     # noqa: E402

ROOT = "/Users/boris/GitHub/sz-poisson-blending"
HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "out")
os.makedirs(OUT, exist_ok=True)
sys.path.insert(0, f"{ROOT}/verify_real/step1_maps")
sys.path.insert(0, ROOT)
import vlib as V                                       # noqa: E402   (vlib resets these env vars to 4)
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "NUMBA_NUM_THREADS"):
    os.environ[_v] = str(NTHREADS)
from numba.core import config as _nbc                  # noqa: E402
_nbc.reload_config()
V.NTHREADS = NTHREADS                                  # vlib's FFT workers / histogram threads
nb.set_num_threads(NTHREADS)
import finufft                                         # noqa: E402

ARCMIN = V.ARCMIN
KB = 600                       # NUFFT coefficient block |kx|, |ky| <= KB (l <= 21,600 on the axes; MMF band ends at 20,000)
MODES = ("baseline", "goal", "white")
X0_DOM = 0.25                  # smallest fdom * q_A tested (fdom 0.5, q_A >= 0.5)


class Tee(V.Tee):
    pass


def nodes_of(mode):
    """The build's / step 1's theta_A nodes (s09 tables): 48 MMF nodes 0.1-139', 40 white nodes 0.25-80'."""
    return np.load(f"{ROOT}/data_real/pd_tables_{mode}.npz")["th"].astype(float)


# ------------------------------------------------------------------------------------------- block <-> rfft layout
class Block:
    """Full-plane coefficient block k in [-KB, KB]^2 built from an rfft2 array (n, n/2+1) by Hermitian symmetry."""

    def __init__(self, grid, kb=KB):
        n = grid.n
        self.n, self.kb = n, kb
        k = np.arange(-kb, kb + 1)
        self.rows_pos = k % n
        self.rows_neg = (-k) % n
        self.N = 2 * kb + 1
        dl = 2 * np.pi / (grid.L * ARCMIN)
        self.ell = dl * np.hypot(k[:, None], k[None, :])                  # |l| [rad^-1] on the block
        self.phase = np.exp(-1j * np.pi * (k[:, None] + k[None, :]) / n)    # pixel centres at (j + 1/2) pix
        self.L = grid.L

    def take(self, F):
        """rfft-layout array (complex or real) -> block [ky, kx] (complex)."""
        kb = self.kb
        B = np.empty((self.N, self.N), complex)
        B[:, kb:] = F[self.rows_pos][:, :kb + 1]
        B[:, :kb] = np.conj(F[self.rows_neg][:, kb:0:-1])
        return B

    def take_real(self, F):
        kb = self.kb
        B = np.empty((self.N, self.N), F.dtype)
        B[:, kb:] = F[self.rows_pos][:, :kb + 1]
        B[:, :kb] = F[self.rows_neg][:, kb:0:-1]
        return B

    def evaluate(self, coef, x, y, eps=1e-10):
        """Re sum_k coef[ky, kx] exp(i (kx X + ky Y)) at positions x, y [arcmin] (X = 2 pi x / L)."""
        if len(x) == 0:
            return np.zeros(0)
        X = (2 * np.pi / self.L) * np.asarray(x, float)
        Y = (2 * np.pi / self.L) * np.asarray(y, float)
        f = finufft.nufft2d2(Y, X, np.ascontiguousarray(coef), isign=1, eps=eps, nthreads=NTHREADS)
        return f.real


# ------------------------------------------------------------------------------------------- filters on the block
class ModeFilter:
    """The build's filter for one mode on the NUFFT block: Y_hat_j(x) [arcmin^2] = Re sum_k c_jk e^{ik.X} with
    c_jk = (sigma_j arcmin^2)^2 g(l theta_j) T(l) y_k phase_k / n^2 / arcmin^2, T = y-response transfer of
    vlib.BuildMMF (inv_Ny sum_i w_i B_i f_i with OUR beams/SED, divided by the painting pre-smoothing)."""

    def __init__(self, grid, painter, mode, blk, keep_g=True):
        from szsim import signal as sg
        self.mode, self.g, self.blk = mode, grid, blk
        self.F = V.BuildMMF(grid, mode)
        self.th = nodes_of(mode)
        self.lth = np.log(self.th)
        self.dlth = float(self.lth[1] - self.lth[0])
        assert np.allclose(np.diff(self.lth), self.dlth)
        self.sig = np.array([self.F.sigma(t) for t in self.th])          # Hankel sigma_Y at the nodes [arcmin^2]
        Gs = painter.presmooth(grid)
        T2 = self.F.yresp_transfer(extra=Gs)                              # rfft layout, real
        self.Tb = blk.take_real(T2)                                       # block
        self.Gsb = blk.take_real(Gs)
        self.pref = [((s * ARCMIN ** 2) ** 2) / grid.n ** 2 / ARCMIN ** 2 for s in self.sig]
        if keep_g:
            self.gb = [sg.template_hankel(blk.ell * t * ARCMIN) for t in self.th]   # build's filter template
            self.K32 = None
        else:                                          # only the float32 block kernels K_j = pref_j g_j T
            self.gb = None
            self.K32 = [(self.pref[j] * sg.template_hankel(blk.ell * t * ARCMIN) * self.Tb).astype(np.float32)
                        for j, t in enumerate(self.th)]
        self.model = self.F.model
        del T2

    def coef(self, j, yblock):
        return (self.pref[j] * self.gb[j] * self.Tb) * yblock * self.blk.phase

    def sigma_Y(self, theta):
        """sigma_Y(theta500) [arcmin^2] of the build (spline of SZModel / MMFModel), used for q_A = Y / sigma."""
        return self.model.sigma_Y(theta)

    def self_response_table(self, gmine, nsub=16, lo_end=np.log(0.03), hi_end=np.log(60.0)):
        """R_j(theta) on [ln theta_j - dlth, ln theta_j + dlth] (step dlth/nsub; the first node down to 0.03', the last
        up to 60' for primaries outside the node range): the node-j filter's response to a painted unit-Y halo of size
        theta at its own position, as the discrete mode sum over the block (the painted halo's DFT =
        Y g_ours(l theta) G_s(l) / pix^2 to 2e-5, v02 (4))."""
        lt = []
        R = []
        h = self.dlth / nsub
        for j in range(len(self.th)):
            a = self.lth[j] - self.dlth * 1.0001
            b = self.lth[j] + self.dlth * 1.0001
            if j == 0:
                a = min(a, lo_end)
            if j == len(self.th) - 1:
                b = max(b, hi_end)
            ls = np.linspace(a, b, int(np.ceil((b - a) / h)) + 1)        # exact end points
            Rj = np.empty(ls.size)
            base = self.pref[j] * self.gb[j] * self.Tb * self.Gsb / self.g.pix ** 2
            for k, l_ in enumerate(ls):
                Rj[k] = float(np.sum(base * gmine(self.blk.ell * np.exp(l_) * ARCMIN)).real)
            lt.append(ls)
            R.append(Rj)
        return lt, R


def save_self(path, lt, R, th):
    np.savez(path, th=th, n=len(lt), **{f"lt_{j}": lt[j] for j in range(len(lt))}, **{f"R_{j}": R[j] for j in range(len(R))})


def load_self(path):
    d = np.load(path)
    n = int(d["n"])
    return [d[f"lt_{j}"] for j in range(n)], [d[f"R_{j}"] for j in range(n)]


def gmine_from_paint_tables(path=os.path.join(V.OUT, "paint_tables.npz")):
    """OUR Fourier template g(u) (projection-slice theorem; vlib v02), tabulated on u in [0, 1600]; 0 beyond."""
    d = np.load(path)
    uf, gf = d["u_fine"], d["g_fine"]

    def g(u):
        return np.interp(u, uf, gf, right=0.0)
    return g


# ------------------------------------------------------------------------------------------- own Hankel rho tables
S_MAX = 400.0
NS = 1001
LTB0, DLTB, NTB = np.log(0.03), 0.05, 186          # theta_B grid 0.03'..~320'


def s_grid():
    return S_MAX * (np.arange(NS) / (NS - 1.0)) ** 2


def build_rho_table(mf, gmine, lmax_white=30000.0):
    """rho_j(theta_B, s) for every node j of the mode: sigma_j^2 (1/2pi) int dl l g_build(l th_j) g_ours(l th_B)
    inv_Ny(l) S(l) J0(l s) (trapezoid, dl = 1; MMF: the build's l grid 0..20000 with its sharp cut; white: 0..30000).
    S(l) = sum_i w_i B_i f_i with OUR beams/SED (MMF), 1 (white).  Returns (nnode, NTB, NS) float32."""
    from szsim import signal as sg
    inst = mf.F.inst
    if mf.mode == "white":
        l = np.arange(0.0, lmax_white + 0.5, 1.0)
        inv = np.interp(l, inst.ell, inst.inv_Ny)
        S = np.ones_like(l)
    else:
        l = inst.ell.copy()
        inv = inst.inv_Ny.copy()
        a = np.stack([V.beam(l, V.FWHM[i]) * V.sed_uK(V.FREQ[i]) for i in range(6)])
        S = np.einsum("li,il->l", inst.weights, a)
        S[inv == 0] = 0.0
    w = np.full(l.size, l[1] - l[0]); w[0] *= 0.5; w[-1] *= 0.5
    s = s_grid() * ARCMIN
    thB = np.exp(LTB0 + DLTB * np.arange(NTB))
    keep = inv > 0
    lk = l[keep]
    base = (w * l * inv * S)[keep] / (2 * np.pi)
    J = j0(np.outer(lk, s))                                              # (nl, NS)
    GB = np.stack([gmine(lk * t * ARCMIN) for t in thB])                # (NTB, nl)
    out = np.empty((len(mf.th), NTB, NS), np.float32)
    for jn, t in enumerate(mf.th):
        sA = mf.sig[jn] * ARCMIN ** 2
        A = GB * (base * sg.template_hankel(lk * t * ARCMIN))[None, :]
        out[jn] = (sA ** 2 * (A @ J)).astype(np.float32)
    S_dev = float(np.abs(S[keep] - 1).max())
    return out, thB, S_dev


def envelope(tab):
    """E[..., k] = max_{k' >= k} max(rho[..., k'], 0): non-increasing in s (for the search radius)."""
    E = np.maximum(tab, 0.0)[..., ::-1]
    E = np.maximum.accumulate(E, axis=-1)[..., ::-1]
    return np.ascontiguousarray(E)


@nb.njit(cache=True, inline="always")
def _rho_interp(tab, lth, s):
    """bilinear in (ln theta_B on the LTB grid, sqrt(s/S_MAX) * (NS-1)); tab (NTB, NS)."""
    t = (lth - LTB0) / DLTB
    if t < 0.0:
        t = 0.0
    if t > NTB - 1.000001:
        t = NTB - 1.000001
    i = int(t)
    a = t - i
    u = np.sqrt(s / S_MAX) * (NS - 1)
    if u >= NS - 1:
        return 0.0
    k = int(u)
    b = u - k
    return ((1 - a) * ((1 - b) * tab[i, k] + b * tab[i, k + 1])
            + a * ((1 - b) * tab[i + 1, k] + b * tab[i + 1, k + 1]))


@nb.njit(cache=True)
def search_radius(E, lthB, thr):
    """largest s with envelope(theta_B) >= thr (max over the two bracketing theta_B rows) + one grid step; 0 if none."""
    nb_ = lthB.size
    R = np.zeros(nb_)
    for b in range(nb_):
        t = (lthB[b] - LTB0) / DLTB
        if t < 0.0:
            t = 0.0
        if t > NTB - 1.000001:
            t = NTB - 1.000001
        i = int(t)
        best = -1
        for row in (i, i + 1):
            if E[row, 0] < thr[b]:
                continue
            lo = 0
            hi = NS - 1
            if E[row, hi] >= thr[b]:
                best = NS - 1
                continue
            while hi - lo > 1:                       # E non-increasing: last index with E >= thr
                mid = (lo + hi) // 2
                if E[row, mid] >= thr[b]:
                    lo = mid
                else:
                    hi = mid
            if lo > best:
                best = lo
        if best >= 0:
            k2 = min(best + 2, NS - 1)
            R[b] = S_MAX * (k2 / (NS - 1.0)) ** 2
    return R


@nb.njit(parallel=True, cache=True)
def xmax_pairs(ax, ay, aid, bx, by, bid, blth, bamp, bR, tab, L, out):
    """out[a] = max over bright b != a, |x_a - x_b| < R_b (periodic min image) of rho(theta_b, s) * amp_b."""
    na = ax.size
    nbr = bx.size
    half = 0.5 * L
    for a in nb.prange(na):
        m = out[a]
        for b in range(nbr):
            if bid[b] == aid[a]:
                continue
            dx = ax[a] - bx[b]
            if dx > half:
                dx -= L
            elif dx < -half:
                dx += L
            R = bR[b]
            if dx > R or dx < -R:
                continue
            dy = ay[a] - by[b]
            if dy > half:
                dy -= L
            elif dy < -half:
                dy += L
            d2 = dx * dx + dy * dy
            if d2 > R * R:
                continue
            x = _rho_interp(tab, blth[b], np.sqrt(d2)) * bamp[b]
            if x > m:
                m = x
        out[a] = m


# ------------------------------------------------------------------------------------------- binning
ZED = np.array([0.01, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.25, 1.5, 1.75, 2.0, 2.5, 3.0])  # cosmolib bins


def qedges(qcut):
    return np.concatenate([np.geomspace(qcut, 100.0, 9), [np.inf]])


EDGES = np.unique(np.concatenate([qedges(5.0)[:-1], qedges(6.0)[:-1], [4.0, 7.0]]))   # = the build's 19 edges
FULLSKY_ARCMIN2 = 4 * np.pi / ARCMIN ** 2
