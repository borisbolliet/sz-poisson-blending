"""Step-3 INDEPENDENT verification (lens: indep): shared code.

NOTHING is imported from szsim/finder.py or analysis/realistic/step3/.  Used from the project: szsim.instrument (SO LAT
numbers, so_lat_noise, cmb_cl_TT, Instrument, MMFModel), szsim.signal (the Arnaud Hankel table template_hankel and the
SZModel scaling relation), szsim.sampler (HMF cell grid and the exact bilinear in-cell draw), szsim.cosmo_hmf.

Own code (different numerical routes from the build where possible):
  * Fourier layout: full complex 2D FFT (numpy order), in-band modes kept as a compact list (the build: rfft half plane);
    covariances in physical units [uK^2 sr] (the build: 'rfft units');
  * tSZ painter: exact Fourier synthesis by type-1 NUFFT (finufft) per ln(theta) node, 4-point Lagrange interpolation
    of the template in ln(theta) (the build: real-space painting at 0.25' with a 0.8' beam, 2-node linear mixture);
  * CMB + noise simulator for fresh skies: own construction (atmosphere = common mode + independent part per tube);
    for the build's tiles the build's random draws are re-created (same seeds, same draw order) to compare maps;
  * MMF: per-mode C^-1 a by explicit solve per bin; sigma from the lattice sum; S/N maps by ifft2 (or per channel);
  * peak finder (SZiFi maxima_lomem rules), disc masks, whitened / literal-binary masked covariance, merges;
  * decomposition: exact lattice sums (own numba kernel, 1D phase factorisation) for the largest contributors, own
    continuous Hankel tables (Gauss-Legendre per covariance bin) for ranking, isolated search and forced photometry;
  * step 1's analytic MMF response rho(theta_A, theta_B, s) by own Hankel quadrature on Instrument.so_lat's 1/N_y.
Geometry as the build (needed to compare on its tiles): tile 600' x 600', N = 1200 pixels of 0.5'; sample (iy, ix) at
(y, x) = (0.5 iy, 0.5 ix) arcmin; axis 0 = y.
"""
import os, sys, time, resource
NTH = int(os.environ.get("VNTHREADS", "1"))
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "NUMBA_NUM_THREADS"):
    os.environ[_v] = str(NTH)
import numpy as np                                   # noqa: E402
import numba as nb                                   # noqa: E402
import scipy.fft as sfft                             # noqa: E402
from scipy.special import j0, ndtr                   # noqa: E402
from scipy.interpolate import CubicSpline            # noqa: E402
import finufft                                       # noqa: E402

ROOT = "/Users/boris/GitHub/sz-poisson-blending"
HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "out")
os.makedirs(OUT, exist_ok=True)
sys.path.insert(0, ROOT)
from szsim import instrument as ins, signal as sg, sampler as smp, cosmo_hmf as ch   # noqa: E402

AR = np.pi / 10800.0
N, PIX, LBOX = 1200, 0.5, 600.0
NPX = N * N
OMP = (PIX * AR) ** 2              # pixel solid angle [sr]
OMT = NPX * OMP                    # tile solid angle [sr]
OMP_AM = PIX ** 2                  # [arcmin^2]
LMIN, LMAX = 100.0, 20000.0
DL0 = 2 * np.pi / (LBOX * AR)      # 36 exactly
BW = 4 * np.pi / (N * PIX * AR)    # SZiFi bin width bin_fac pi / (n dx), bin_fac 4 -> 72
THETAS = np.geomspace(0.5, 15.0, 15)
FREQ = np.asarray(ins.SO_LAT["freqs"], float)
FWHM = np.asarray(ins.SO_LAT["fwhm"], float)
F_UK = ins.tsz_f_uK(FREQ)
SIGB = FWHM * AR / np.sqrt(8 * np.log(2))
QTH, QTH_NOISE, MASK_R = 4.0, 4.0, 3.0
MODES = ("baseline", "goal")
BUILD_DATA = os.path.join(ROOT, "data_real", "step3")


def peak_rss_gb():
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e9


class Tee:
    def __init__(self, path):
        self.f = open(path, "w")

    def __call__(self, *a):
        s = " ".join(str(x) for x in a)
        print(s, flush=True)
        self.f.write(s + "\n"); self.f.flush()


def beams(ell):
    return np.exp(-0.5 * (np.asarray(ell, float)[..., None] * SIGB) ** 2)        # (..., 6)


# ============================================================================ templates
UG, GG = sg.template_hankel()
_CS = CubicSpline(np.log(UG[1:]), GG[1:])


def g_lin(u):
    """Step 1's MMF template (linear interpolation of the Arnaud Hankel table)."""
    return sg.template_hankel(np.asarray(u, float))


def g_cub(u):
    """Smooth version of the same table (cubic spline in ln u): used for painting."""
    u = np.asarray(u, float)
    out = np.zeros(u.shape)
    lo = u < UG[1]
    mid = (~lo) & (u <= UG[-1])
    out[lo] = GG[0] + (GG[1] - GG[0]) * (u[lo] / UG[1]) ** 2
    out[mid] = _CS(np.log(u[mid]))
    return out


# ============================================================================ grid
class Grid:
    """Full-plane Fourier grid (numpy fft order) with the in-band mode list and SZiFi bins (width 72, from l = 0)."""

    def __init__(self):
        k = np.fft.fftfreq(N, 1.0 / N).round().astype(np.int64)
        self.k = k
        n2 = k[:, None] ** 2 + k[None, :] ** 2
        isq = np.floor(np.sqrt(n2.astype(float))).astype(np.int64)
        isq -= (isq * isq > n2)
        isq += ((isq + 1) * (isq + 1) <= n2)
        b = isq // 2                                 # floor(|l| / 72), |l| = 36 sqrt(n2)
        ell = DL0 * np.sqrt(n2.astype(float))
        inb = (n2 >= 8) & (n2 <= 308641)             # 100 <= |l| <= 20000 exactly
        idx = np.flatnonzero(inb.ravel())
        self.idx = idx
        self.mrow = (idx // N).astype(np.int64)
        self.mcol = (idx % N).astype(np.int64)
        self.mi0 = k[self.mrow]
        self.mi1 = k[self.mcol]
        self.mell = ell.ravel()[idx]
        self.mbin = b.ravel()[idx]
        sh, self.mshell = np.unique(n2.ravel()[idx], return_inverse=True)
        self.mshell = self.mshell.astype(np.int64)
        self.shell_n2 = sh
        self.shell_ell = DL0 * np.sqrt(sh.astype(float))
        self.nbin = int(self.mbin.max()) + 1
        self.bins_used = np.unique(self.mbin)
        self.cnt = np.bincount(self.mbin, minlength=self.nbin).astype(float)
        self.nm = idx.size
        self.a_shell = beams(self.shell_ell) * F_UK[None, :]        # (nshell, 6)
        self.k0 = self.k * DL0                                       # rad^-1 for integer index
        del n2, isq, b, ell, inb

    def a_modes(self):
        return self.a_shell[self.mshell]

    def to_full(self, v):
        F = np.zeros(NPX, complex)
        F[self.idx] = v
        return F.reshape(N, N)

    def analytic_cov_shell(self, mode):
        """Analytic C_ij(l) = B_i B_j C_l^CMB + N_ij(l) [uK^2 sr] on the in-band shells, (nshell, 6, 6)."""
        L = self.shell_ell
        B = beams(L)
        Nn = np.moveaxis(ins.so_lat_noise(L, mode), 2, 0)
        return Nn + B[:, :, None] * B[:, None, :] * ins.cmb_cl_TT(L)[:, None, None]

    def analytic_cov_binned(self, mode):
        """Analytic covariance averaged over the in-band modes of each bin [uK^2 sr]."""
        Cs = self.analytic_cov_shell(mode)
        w = np.bincount(self.mshell, minlength=self.shell_ell.size).astype(float)       # modes per shell
        sb = np.zeros(self.shell_ell.size, np.int64)
        sb[self.mshell] = self.mbin
        C = np.zeros((self.nbin, 6, 6))
        for i in range(6):
            for j in range(6):
                C[:, i, j] = np.bincount(sb, weights=w * Cs[:, i, j], minlength=self.nbin)
        C[self.bins_used] /= self.cnt[self.bins_used][:, None, None]
        return C


_G = None


def grid():
    global _G
    if _G is None:
        _G = Grid()
    return _G


# ============================================================================ simulation
def _cl_cmb_modes(G):
    return ins.cmb_cl_TT(G.shell_ell)[G.mshell]


def cmb_build_rng(seed, G):
    """The build's lensed CMB draw re-created (one standard normal map, same order), compact in-band, (nm, 6) [uK DFT]."""
    rng = np.random.default_rng(seed)
    w = sfft.fft2(rng.standard_normal((N, N)), workers=NTH).ravel()[G.idx]
    t = w * np.sqrt(_cl_cmb_modes(G) / OMP)
    return beams(G.mell) * t[:, None]


def noise_build_rng(seed, mode, G):
    """The build's white + atmospheric draws re-created (6 white maps, then per tube two maps; Cholesky form)."""
    rng = np.random.default_rng(seed)
    wl = ins.SO_LAT["white"][mode]
    X = np.empty((G.nm, 6), complex)
    for i in range(6):
        X[:, i] = sfft.fft2(rng.standard_normal((N, N)) * (wl[i] / PIX), workers=NTH).ravel()[G.idx]
    shp = (G.mell / ins.SO_LAT["ell_knee"]) ** ins.SO_LAT["alpha_knee"]
    red = ins.SO_LAT["N_red"] * 4 * np.pi * ins.SO_LAT["fsky"] / ins.so_t_eff()
    r = ins.SO_LAT["atm_corr"]
    for (i, j) in ins.SO_LAT["tubes"]:
        xa = sfft.fft2(rng.standard_normal((N, N)), workers=NTH).ravel()[G.idx] / np.sqrt(OMP)
        xb = sfft.fft2(rng.standard_normal((N, N)), workers=NTH).ravel()[G.idx] / np.sqrt(OMP)
        X[:, i] += np.sqrt(red[i] * shp) * xa
        X[:, j] += np.sqrt(red[j] * shp) * (r * xa + np.sqrt(1 - r * r) * xb)
    return X


def cmb_mine(rng, G):
    """Own CMB draw: complex Gaussian modes with Hermitian symmetry from a real white map."""
    return cmb_build_rng_from_map(rng.standard_normal((N, N)), G)


def cmb_build_rng_from_map(wmap, G):
    t = sfft.fft2(wmap, workers=NTH).ravel()[G.idx] * np.sqrt(_cl_cmb_modes(G) / OMP)
    return beams(G.mell) * t[:, None]


def noise_mine(rng, mode, G):
    """Own noise draw from szsim.instrument.so_lat_noise: white part in pixel space; atmosphere per tube as a common
    mode c plus independent parts e_a:  n_a = sqrt(r N_a) c + sqrt((1 - r) N_a) e_a  (N_a = atmospheric power)."""
    L = G.shell_ell
    Nfull = ins.so_lat_noise(L, mode)                     # (6, 6, nshell) uK^2 sr
    Nw = ins.so_lat_noise(L, mode, atm=False)
    Natm = np.array([Nfull[i, i] - Nw[i, i] for i in range(6)])          # (6, nshell)
    wl = ins.SO_LAT["white"][mode]
    X = np.empty((G.nm, 6), complex)
    for i in range(6):
        X[:, i] = sfft.fft2(rng.standard_normal((N, N)) * (wl[i] / PIX), workers=NTH).ravel()[G.idx]
    for (a, b) in ins.SO_LAT["tubes"]:
        rr = Nfull[a, b] / np.sqrt(Natm[a] * Natm[b])                    # = 0.9 at every l
        rr = rr[G.mshell]
        c = sfft.fft2(rng.standard_normal((N, N)), workers=NTH).ravel()[G.idx] / np.sqrt(OMP)
        for ch_ in (a, b):
            e = sfft.fft2(rng.standard_normal((N, N)), workers=NTH).ravel()[G.idx] / np.sqrt(OMP)
            na = Natm[ch_][G.mshell]
            X[:, ch_] += np.sqrt(rr * na) * c + np.sqrt((1 - rr) * na) * e
    return X


# ============================================================================ painter (type-1 NUFFT per ln theta node)
TH0_NODE, DNODE = 0.2, 0.04


def lagrange4(lnth, th0=TH0_NODE, d=DNODE):
    """Cubic Lagrange weights in ln theta: nodes i0-1 .. i0+2 (node n at ln th0 + n d).  Returns (i0, w (n, 4))."""
    t = (np.asarray(lnth, float) - np.log(th0)) / d
    i0 = np.floor(t).astype(np.int64)
    f = t - i0
    w = np.stack([-f * (f - 1) * (f - 2) / 6, (f + 1) * (f - 1) * (f - 2) / 2,
                  -(f + 1) * f * (f - 2) / 2, (f + 1) * f * (f - 1) / 6], 1)
    return i0, w


class Painter:
    """tSZ Fourier coefficients of a set of halos on the in-band modes:
         T(k) = (1/Omega_pix) sum_h Y_h g(|k| theta_h) exp(-i k.x_h)     [y DFT units; Y in arcmin^2]
    g = g_cub interpolated in ln theta (4-point Lagrange, node spacing 0.04); one NUFFT per node."""

    def __init__(self, G, eps=1e-10, scheme="lag4"):
        """scheme 'lag4': own 4-point Lagrange in ln theta on nodes 0.2' e^(0.04 n);  'mix2': a 2-node linear mixture
        in ln theta on nodes 0.24' e^(0.04 n) (the build's painted template, to compare maps at the 1e-8 level)."""
        self.G = G
        self.eps = eps
        self.scheme = scheme
        self.th0 = TH0_NODE if scheme == "lag4" else 0.24
        self._gn = {}

    def g_node(self, n):
        if n not in self._gn:
            th = self.th0 * np.exp(DNODE * n)
            self._gn[n] = g_cub(self.G.shell_ell * th * AR)
        return self._gn[n]

    def stencil(self, lnth):
        if self.scheme == "lag4":
            i0, w = lagrange4(lnth)
            return i0 - 1, w
        t = (np.asarray(lnth, float) - np.log(self.th0)) / DNODE
        t = np.clip(t, 0.0, 141 - 1.000001)
        i0 = t.astype(np.int64)
        f = t - i0
        return i0, np.stack([1 - f, f], 1)

    def T(self, ys, xs, Ys, ths, exact=False):
        G = self.G
        ys = np.asarray(ys, float); xs = np.asarray(xs, float); Ys = np.asarray(Ys, float); ths = np.asarray(ths, float)
        out = np.zeros(G.nm, complex)
        if exact:                                   # direct template of each halo's own size (few halos only)
            for h in range(ys.size):
                F = finufft.nufft2d1(np.array([2 * np.pi * ys[h] / LBOX]), np.array([2 * np.pi * xs[h] / LBOX]),
                                     np.array([Ys[h] + 0j]), (N, N), eps=self.eps, isign=-1, modeord=1, nthreads=NTH,
                                     upsampfac=2.0)
                out += g_cub(G.shell_ell * ths[h] * AR)[G.mshell] * F.ravel()[G.idx]
            return out / OMP_AM
        first, w = self.stencil(np.log(ths))
        ns = w.shape[1]
        nodes = np.concatenate([first + m for m in range(ns)])
        wts = np.concatenate([w[:, m] for m in range(ns)]) * np.tile(Ys, ns)
        hh = np.tile(np.arange(ys.size), ns)
        order = np.argsort(nodes, kind="stable")
        nodes, wts, hh = nodes[order], wts[order], hh[order]
        starts = np.flatnonzero(np.r_[True, nodes[1:] != nodes[:-1]])
        ends = np.r_[starts[1:], nodes.size]
        py = 2 * np.pi * ys / LBOX
        px = 2 * np.pi * xs / LBOX
        for s, e in zip(starts, ends):
            n = int(nodes[s])
            sel = hh[s:e]
            F = finufft.nufft2d1(py[sel], px[sel], wts[s:e] + 0j, (N, N), eps=self.eps, isign=-1, modeord=1,
                                 nthreads=NTH, upsampfac=1.25 if self.eps >= 1e-9 else 2.0)
            out += self.g_node(n)[G.mshell] * F.ravel()[G.idx]
        return out / OMP_AM


def tsz_channels(T, G):
    """(nm,) y DFT -> (nm, 6) uK DFT: beams x SED."""
    return beams(G.mell) * (F_UK[None, :] * T[:, None])


# ============================================================================ covariance
def binned_cov(Xc, G):
    """Binned 6 x 6 spectra of the compact in-band modes [uK^2 sr]: mean over the in-band modes of each bin of
    Re(X_i X_j^*), times Omega_pix / N_pix."""
    C = np.zeros((G.nbin, 6, 6))
    nch = Xc.shape[1]
    for i in range(nch):
        for j in range(i, nch):
            v = np.bincount(G.mbin, weights=(Xc[:, i] * np.conj(Xc[:, j])).real, minlength=G.nbin)
            C[:, i, j] = v
            C[:, j, i] = v
    C[G.bins_used] /= G.cnt[G.bins_used][:, None, None]
    return C * (OMP / NPX)


@nb.njit(cache=True)
def _bin_matvec(Mb, mbin, x, out):
    for m in range(x.shape[0]):
        b = mbin[m]
        for i in range(x.shape[1]):
            s = 0.0 + 0.0j
            for j in range(x.shape[1]):
                s += Mb[b, i, j] * x[m, j]
            out[m, i] = s


def masked_cov(Xc, mask, G, Cref=None, method="whitened"):
    """Covariance with the binary cluster mask (1 = kept).  'whitened': whiten each in-band mode with Cref of its bin
    (symmetric inverse square root), mask the whitened maps, binned pseudo-spectra / f, un-whiten.  'binary': the
    band-limited channel maps are masked as they are (literal f_sky).  Returns (C, f)."""
    f = float(mask.mean())
    if method == "binary":
        Xm = np.empty_like(Xc)
        for i in range(Xc.shape[1]):
            m = sfft.ifft2(G.to_full(Xc[:, i]), workers=NTH).real
            Xm[:, i] = sfft.fft2(m * mask, workers=NTH).ravel()[G.idx]
        return binned_cov(Xm, G) / f, f
    bu = G.bins_used
    ev, U = np.linalg.eigh(Cref[bu])
    Wm = np.zeros_like(Cref); Wi = np.zeros_like(Cref)
    Wm[bu] = np.einsum("bij,bj,bkj->bik", U, ev ** -0.5, U)
    Wi[bu] = np.einsum("bij,bj,bkj->bik", U, ev ** 0.5, U)
    Z = np.empty_like(Xc)
    _bin_matvec(Wm, G.mbin, Xc, Z)
    for i in range(Xc.shape[1]):
        z = sfft.ifft2(G.to_full(Z[:, i]), workers=NTH).real
        Z[:, i] = sfft.fft2(z * mask, workers=NTH).ravel()[G.idx]
    Cw = binned_cov(Z, G) / f
    C = np.zeros_like(Cw)
    C[bu] = np.einsum("bij,bjk,bkl->bil", Wi[bu], Cw[bu], Wi[bu])
    return C, f


# ============================================================================ matched filter
@nb.njit(cache=True)
def _psi_modes(Cinv, mbin, a, out):
    for m in range(a.shape[0]):
        b = mbin[m]
        for i in range(a.shape[1]):
            s = 0.0
            for j in range(a.shape[1]):
                s += Cinv[b, i, j] * a[m, j]
            out[m, i] = s


class MMF:
    """Six-channel MMF on the in-band modes.  C: binned covariance [uK^2 sr] (each mode takes its own bin's value) or
    mode='baseline'/'goal' with analytic=True (exact analytic C(|k|) per mode).
      psi(k) = C^-1 a,  aCa = a^T C^-1 a,   sigma_j^-2 = (1/Omega_tile) sum_k g_j(k)^2 aCa(k)   [sr^-2]
      Y_hat_j(x) = sigma_j^2 ifft2[g_j(k) sum_i psi_i(k) X_i(k)]   [sr]."""

    def __init__(self, G, C=None, mode=None, analytic=False, template="lin", thetas=THETAS):
        self.G = G
        self.thetas = np.asarray(thetas, float)
        a = G.a_modes()
        if analytic:
            Cs = G.analytic_cov_shell(mode)
            ps = np.linalg.solve(Cs, G.a_shell[..., None])[..., 0]          # (nshell, 6)
            self.psi = ps[G.mshell]
            self.C = None
        else:
            self.C = C
            Ci = np.zeros_like(C)
            bu = G.bins_used
            # explicit solve per bin (C^-1 = solve(C, I)), not np.linalg.inv
            Ci[bu] = np.linalg.solve(C[bu], np.broadcast_to(np.eye(6), (bu.size, 6, 6)))
            self.Cinv = Ci
            self.psi = np.empty((G.nm, 6))
            _psi_modes(Ci, G.mbin, a, self.psi)
        self.aCa = np.einsum("mi,mi->m", a, self.psi)
        self.gfun = g_lin if template == "lin" else g_cub
        gs = np.array([self.gfun(G.shell_ell * t * AR) for t in self.thetas])      # (nth, nshell)
        self.gs = gs
        aCa_s = np.bincount(G.mshell, weights=self.aCa, minlength=G.shell_ell.size)  # sum over the modes of a shell
        self.norm = (gs ** 2 * aCa_s[None, :]).sum(1) / OMT
        self.sigma = self.norm ** -0.5                     # sr
        self.sigma_am = self.sigma / AR ** 2               # arcmin^2

    def combine(self, Xc):
        return np.einsum("mi,mi->m", self.psi, Xc)

    def ymap(self, D, j):
        """Y_hat map [arcmin^2] at scale j from D = combine(X)."""
        G = self.G
        F = G.to_full(self.gs[j][G.mshell] * D)
        return sfft.ifft2(F, workers=NTH).real * (self.sigma[j] ** 2 / AR ** 2)

    def qmap(self, D, j):
        return self.ymap(D, j) / self.sigma_am[j]

    def ymap_perchannel(self, Xc, j):
        """Same quantity by a different route: one inverse FFT per channel of Psi_i X_i, summed in real space."""
        G = self.G
        out = np.zeros((N, N))
        gj = self.gs[j][G.mshell]
        for i in range(6):
            out += sfft.ifft2(G.to_full(gj * self.psi[:, i] * Xc[:, i]), workers=NTH).real
        return out * (self.sigma[j] ** 2 / AR ** 2)


# ============================================================================ peaks, masks, finder
@nb.njit(cache=True)
def peaks_one_scale(qm, q0, qp, j, nth, qth):
    """SZiFi maxima_lomem: q > qth; strictly larger than the 4 axis neighbours (periodic); along scale q_j > q_{j-1}
    and q_j >= q_{j+1}, q_{-1} = 0, last scale only q_{n-1} > q_{n-2}."""
    n0, n1 = q0.shape
    oy = []; ox = []
    for iy in range(n0):
        ym = (iy - 1) % n0; yp = (iy + 1) % n0
        for ix in range(n1):
            v = q0[iy, ix]
            if v <= qth:
                continue
            if not (v > q0[yp, ix] and v > q0[ym, ix]):
                continue
            xm = (ix - 1) % n1; xp = (ix + 1) % n1
            if not (v > q0[iy, xp] and v > q0[iy, xm]):
                continue
            if j == 0:
                ok = v > 0.0 and (nth == 1 or v >= qp[iy, ix])
            elif j == nth - 1:
                ok = v > qm[iy, ix]
            else:
                ok = v > qm[iy, ix] and v >= qp[iy, ix]
            if ok:
                oy.append(iy); ox.append(ix)
    return np.array(oy, dtype=np.int64), np.array(ox, dtype=np.int64)


@nb.njit(cache=True)
def disc_mask(iy, ix, rad_pix, n):
    m = np.ones((n, n))
    for k in range(iy.size):
        R = rad_pix[k]
        Ri = int(np.ceil(R))
        for dy in range(-Ri, Ri + 1):
            for dx in range(-Ri, Ri + 1):
                if dx * dx + dy * dy <= R * R:
                    m[(iy[k] + dy) % n, (ix[k] + dx) % n] = 0.0
    return m


def detect(filt, D, qth=QTH, keep=None):
    """Peaks of the (theta, y, x) S/N cube, rolling window of three maps.  keep: optional dict to which the S/N maps
    are written (scale -> map)."""
    nth = len(filt.thetas)
    J, IY, IX, Q = [], [], [], []
    prev, cur = None, filt.qmap(D, 0)
    for j in range(nth):
        nxt = filt.qmap(D, j + 1) if j + 1 < nth else cur
        if keep is not None:
            keep[j] = cur
        a, b = peaks_one_scale(prev if prev is not None else cur, cur, nxt, j, nth, qth)
        J.append(np.full(a.size, j, np.int64)); IY.append(a); IX.append(b); Q.append(cur[a, b])
        prev, cur = cur, nxt
    j, iy, ix, q = (np.concatenate(J), np.concatenate(IY), np.concatenate(IX), np.concatenate(Q))
    return dict(j=j, iy=iy, ix=ix, q=q)


def cluster_mask(det, thetas):
    sel = det["q"] > QTH_NOISE
    rad = MASK_R * thetas[det["j"][sel]] / PIX
    return disc_mask(det["iy"][sel], det["ix"][sel], rad, N), int(sel.sum())


def run_finder(Xc, G, thetas=THETAS, mask_method="whitened", C0=None, C1=None, keep_last=None):
    """SZiFi-like finder: iteration 0 with the covariance of the full tile, mask q > 4 detections with 3 theta discs,
    re-estimate the covariance, rerun once.  C0 / C1 may be given (e.g. the build's stored covariances)."""
    if C0 is None:
        C0 = binned_cov(Xc, G)
    F0 = MMF(G, C=C0, thetas=thetas)
    det0 = detect(F0, F0.combine(Xc))
    mask, nmask = cluster_mask(det0, thetas)
    if C1 is None:
        C1, fsky = masked_cov(Xc, mask, G, Cref=C0, method=mask_method)
    else:
        fsky = float(mask.mean())
    F1 = MMF(G, C=C1, thetas=thetas)
    det1 = detect(F1, F1.combine(Xc), keep=keep_last)
    return dict(C0=C0, F0=F0, det0=det0, mask=mask, fsky=fsky, C1=C1, F1=F1, det1=det1)


# ============================================================================ merges (SZiFi cat.merge_detections)
def pdist(y, x, y0, x0):
    dy = (y - y0 + LBOX / 2) % LBOX - LBOX / 2
    dx = (x - x0 + LBOX / 2) % LBOX - LBOX / 2
    return np.sqrt(dy * dy + dx * dx)


def merge_closest(y, x, q, radius):
    keep = []
    rem = np.arange(q.size)
    while rem.size:
        d = pdist(y[rem], x[rem], y[rem[0]], x[rem[0]])
        g = d < radius
        keep.append(rem[g][np.argmax(q[rem[g]])])
        rem = rem[~g]
    return np.array(keep, np.int64)


def merge_fof(y, x, q, radius):
    n = q.size
    parent = np.arange(n)

    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a
    for a in range(n):
        d = pdist(y[a + 1:], x[a + 1:], y[a], x[a])
        for b in np.flatnonzero(d < radius) + a + 1:
            ra, rb = find(a), find(b)
            if ra != rb:
                parent[rb] = ra
    roots = np.array([find(a) for a in range(n)])
    keep = [np.flatnonzero(roots == r)[np.argmax(q[roots == r])] for r in np.unique(roots)]
    return np.array(sorted(keep), np.int64)


MERGES = ("none", "closest1", "closest2", "closest10", "fof10")


def apply_merge(y, x, q, name, qcut=5.0):
    idx = np.flatnonzero(q > qcut)
    if name == "none" or idx.size == 0:
        return idx
    r = float(name.replace("closest", "").replace("fof", ""))
    fn = merge_closest if name.startswith("closest") else merge_fof
    return idx[fn(y[idx], x[idx], q[idx], r)]


# ============================================================================ responses
GL_PER_BIN = 6


class HankelNodes:
    """Quadrature nodes per covariance bin (the binned covariance is constant within a bin): Gauss-Legendre nodes; the
    weights W reproduce either the continuous measure  int d^2l/(2 pi)^2 = (1/2 pi) int l dl  (discrete=False) or the
    DISCRETE lattice measure  (1/Omega_tile) sum_{in-band modes k of the bin}  (discrete=True: weights solve the moment
    equations of the bin's |k| values up to degree npb - 1)."""

    def __init__(self, G, npb=GL_PER_BIN, discrete=True):
        xg, wg = np.polynomial.legendre.leggauss(npb)
        ls, ws, bs = [], [], []
        if discrete:
            lo_b = np.maximum(G.bins_used * BW, LMIN); hi_b = np.minimum((G.bins_used + 1) * BW, LMAX)
            c_b = np.zeros(G.nbin); h_b = np.ones(G.nbin)
            c_b[G.bins_used] = 0.5 * (hi_b + lo_b); h_b[G.bins_used] = 0.5 * (hi_b - lo_b)
            tm = (G.mell - c_b[G.mbin]) / h_b[G.mbin]
            mom = np.array([np.bincount(G.mbin, weights=tm ** p, minlength=G.nbin) for p in range(npb)])   # (npb, nbin)
        for b in G.bins_used:
            lo, hi = max(b * BW, LMIN), min((b + 1) * BW, LMAX)
            x = 0.5 * (hi - lo) * xg + 0.5 * (hi + lo)
            ls.append(x); bs.append(np.full(npb, b))
            if discrete:
                Vm = np.vstack([xg ** p for p in range(npb)])
                ws.append(np.linalg.solve(Vm, mom[:, b]) / OMT)
            else:
                ws.append(0.5 * (hi - lo) * wg * x / (2 * np.pi))
        self.l = np.concatenate(ls); self.w = np.concatenate(ws); self.b = np.concatenate(bs)
        self.a = beams(self.l) * F_UK[None, :]


R_NEAR = np.arange(0.0, 60.0 + 1e-9, 0.02)       # arcmin


class ResponseTables:
    """Angle-averaged response of a tile filter to a unit-Y500 halo (quadrature of HankelNodes):
         R_j(theta_n, r) = sigma_j^2 sum_m W_m g_j(l_m) aCa(l_m) g_cub(l_m theta_n) J0(l_m r)
    on the painter's theta nodes (so a halo's response = the same 4-point Lagrange mixture as its painting)."""

    def __init__(self, filt, hn, nodes, J0mat=None):
        self.nodes = np.asarray(nodes, np.int64)
        if filt.C is not None:
            aCa = np.einsum("li,lij,lj->l", hn.a, filt.Cinv[hn.b], hn.a)
        else:
            raise ValueError("analytic filter: use the build-independent rho table instead")
        th_n = TH0_NODE * np.exp(DNODE * self.nodes)
        Gn = g_cub(np.outer(th_n * AR, hn.l))                           # (nn, nl)
        if J0mat is None:
            J0mat = j0(np.outer(hn.l, R_NEAR * AR))
        nth = len(filt.thetas)
        self.R = np.empty((nth, self.nodes.size, R_NEAR.size))
        for j in range(nth):
            wj = hn.w * filt.gfun(hn.l * filt.thetas[j] * AR) * aCa * filt.sigma[j] ** 2
            self.R[j] = (Gn * wj[None, :]) @ J0mat
        self.n0 = int(self.nodes[0])
        self.dr = R_NEAR[1] - R_NEAR[0]


@nb.njit(cache=True)
def _lag4(t):
    return (-t * (t - 1) * (t - 2) / 6, (t + 1) * (t - 1) * (t - 2) / 2, -(t + 1) * t * (t - 2) / 2, (t + 1) * t * (t - 1) / 6)


@nb.njit(cache=True)
def resp_eval(R, j, n0, lnth, r, dr, lnth0, dnode):
    """Response of scale j to a halo of ln theta at distance r: 4-point Lagrange in ln theta (painter's nodes) x
    4-point Lagrange in r.  Zero beyond the table."""
    nr = R.shape[2]
    u = r / dr
    k = int(u)
    if k + 2 >= nr:
        return 0.0
    tr = u - k
    wr = _lag4(tr)
    t = (lnth - lnth0) / dnode
    i0 = int(np.floor(t))
    f = t - i0
    wt = _lag4(f)
    s = 0.0
    for a in range(4):
        n = i0 - 1 + a - n0
        if n < 0 or n >= R.shape[1]:
            continue
        v = 0.0
        for b in range(4):
            kk = k - 1 + b
            if kk < 0:
                kk = -kk                       # even in r
            v += wr[b] * R[j, n, kk]
        s += wt[a] * v
    return s


@nb.njit(cache=True)
def lattice_contrib(mi0, mi1, mshell, A, gsh_rows, wts, dy_rad, dx_rad, k0):
    """Exact lattice sum  sum_k A(k) g_h(|k|) cos(k.D)  with g_h = sum_a wts[a] gsh_rows[a, shell] and the phase
    factorised in 1D: cos(k0 dy + k1 dx) = Re(e^{i k0 dy} e^{i k1 dx})."""
    n = k0.size
    c0 = np.empty(n); s0 = np.empty(n); c1 = np.empty(n); s1 = np.empty(n)
    for i in range(n):
        c0[i] = np.cos(k0[i] * dy_rad); s0[i] = np.sin(k0[i] * dy_rad)
        c1[i] = np.cos(k0[i] * dx_rad); s1[i] = np.sin(k0[i] * dx_rad)
    tot = 0.0
    na = gsh_rows.shape[0]
    for m in range(A.size):
        sh = mshell[m]
        g = 0.0
        for a in range(na):
            g += wts[a] * gsh_rows[a, sh]
        r0 = mi0[m] % n; r1 = mi1[m] % n
        tot += A[m] * g * (c0[r0] * c1[r1] - s0[r0] * s1[r1])
    return tot


class Exact:
    """Exact lattice contribution of a halo (Y [arcmin^2], theta) to the filtered value of scale j at offset
    D = detection - halo [arcmin]:  Y sigma_j^2 (1/Omega_tile) sum_k g_j aCa g_h cos(k.D)   [arcmin^2].
    template 'nodes' = the painter's 4-point Lagrange mixture; 'own' = g_cub of the halo's own size; 'mix2' = a two-node
    linear mixture on another node set (to reproduce a mixture painter)."""

    def __init__(self, filt, painter):
        self.f = filt
        self.P = painter
        G = filt.G
        self.G = G
        self.A = {}

    def Aj(self, j):
        if j not in self.A:
            f = self.f
            self.A[j] = f.gs[j][self.G.mshell] * f.aCa * f.sigma[j] ** 2 / OMT
        return self.A[j]

    def contrib(self, j, Y, theta, dy, dx, template="nodes", mix=None):
        G = self.G
        if template == "nodes":
            first, w = self.P.stencil(np.array([np.log(theta)]))
            rows = np.stack([self.P.g_node(int(first[0]) + a) for a in range(w.shape[1])])
            wts = w[0]
        elif template == "own":
            rows = g_cub(G.shell_ell * theta * AR)[None, :]
            wts = np.ones(1)
        else:                                   # mix: (th_lo, th_hi, w_hi)
            tl, th_, wh = mix
            rows = np.stack([g_cub(G.shell_ell * tl * AR), g_cub(G.shell_ell * th_ * AR)])
            wts = np.array([1 - wh, wh])
        s = lattice_contrib(G.mi0, G.mi1, G.mshell, self.Aj(j), rows, wts, dy * AR, dx * AR, G.k0)
        return Y * s


# ============================================================================ step 1's analytic MMF response
class Rho1:
    """rho(theta_A, theta_B, s) of step 1's analytic MMF (Instrument.so_lat, l = 100-20000, g_lin both sides) by own
    Gauss-Legendre quadrature; table on ln theta nodes (spacing 0.08) x s (0-60', 0.05')."""
    D = 0.08
    TH0 = 0.2
    S = np.arange(0.0, 60.0 + 1e-9, 0.05)

    def __init__(self, mode, path=None):
        path = path or os.path.join(OUT, f"rho1_{mode}.npz")
        if os.path.exists(path):
            d = np.load(path)
            self.tab, self.sig_nodes = d["tab"], d["sig_nodes"]
        else:
            inst = ins.Instrument.so_lat(mode, lmin=LMIN, lmax=LMAX)
            npan = 400
            xg, wg = np.polynomial.legendre.leggauss(8)
            e = np.linspace(LMIN, LMAX, npan + 1)
            l = (0.5 * np.diff(e)[:, None] * xg[None, :] + 0.5 * (e[1:] + e[:-1])[:, None]).ravel()
            w = (0.5 * np.diff(e)[:, None] * wg[None, :]).ravel()
            invN = np.interp(l, inst.ell, inst.inv_Ny)
            nn = int(np.ceil(np.log(80.0 / self.TH0) / self.D)) + 1
            th = self.TH0 * np.exp(self.D * np.arange(nn))
            Gm = g_lin(np.outer(th * AR, l))                                   # (nn, nl)
            J = j0(np.outer(l, self.S * AR))
            wl = w * l * invN / (2 * np.pi)
            norm = (Gm ** 2 * wl[None, :]).sum(1)                              # sigma^-2 [sr^-2]
            self.sig_nodes = norm ** -0.5 / AR ** 2                            # arcmin^2
            tab = np.empty((nn, nn, self.S.size))
            for a in range(nn):
                tab[a] = (Gm * (Gm[a] * wl)[None, :]) @ J / norm[a]
            self.tab = tab
            np.savez(path, tab=tab, sig_nodes=self.sig_nodes)
        self.tab = self.tab.astype(np.float32)


@nb.njit(cache=True)
def rho1_eval(tab, lthA, lthB, s, th0l, d, ds):
    nn = tab.shape[0]; ns = tab.shape[2]
    u = s / ds
    k = int(u)
    if k + 2 >= ns:
        return 0.0
    wr = _lag4(u - k)
    ta = (lthA - th0l) / d; ia = int(np.floor(ta)); wa = _lag4(ta - ia)
    tb = (lthB - th0l) / d; ib = int(np.floor(tb)); wb = _lag4(tb - ib)
    s_ = 0.0
    for a in range(4):
        na = ia - 1 + a
        if na < 0 or na >= nn:
            continue
        for b in range(4):
            nb_ = ib - 1 + b
            if nb_ < 0 or nb_ >= nn:
                continue
            v = 0.0
            for c in range(4):
                kk = k - 1 + c
                if kk < 0:
                    kk = -kk
                v += wr[c] * tab[na, nb_, kk]
            s_ += wa[a] * wb[b] * v
    return s_


@nb.njit(cache=True)
def xmax_true(idx, hy, hx, hY, hlth, sigA, tab, th0l, d, ds, rcut, box, out_max, out_arg):
    """Largest single-neighbour contribution x = Y_B rho(A, B, s) / sigma_Y(A) at each halo idx[a] (true position,
    step 1's filter), brute force over all halos within rcut."""
    for a in range(idx.size):
        h = idx[a]
        best = -1e300; barg = -1
        for b in range(hY.size):
            if b == h:
                continue
            dy = (hy[b] - hy[h] + 0.5 * box) % box - 0.5 * box
            if abs(dy) > rcut:
                continue
            dx = (hx[b] - hx[h] + 0.5 * box) % box - 0.5 * box
            if abs(dx) > rcut:
                continue
            r = np.sqrt(dx * dx + dy * dy)
            if r > rcut:
                continue
            x = hY[b] * rho1_eval(tab, hlth[h], hlth[b], r, th0l, d, ds) / sigA[a]
            if x > best:
                best = x; barg = b
        out_max[a] = best; out_arg[a] = barg


# ============================================================================ halo population (szsim.sampler machinery)
FSKY_TILE = 100.0 / (4 * np.pi * (180 / np.pi) ** 2)


def draw_tile_halos(seed, Mmin=1e13, Mmax=1e16, zmin=0.01, zmax=3.0):
    """Poisson halos of one 10 x 10 deg periodic tile: the sampler's cell intensities x (100 deg^2 / full sky),
    the sampler's exact bilinear in-cell draw, uniform positions, Y500 with the model's log-normal scatter."""
    model = sg.SZModel()
    table = ch.default_table()
    rng = np.random.default_rng(seed)
    zz, lm, d = smp._cell_grid(table, Mmin, Mmax, zmin, zmax)
    dz = np.diff(zz); dl = np.diff(lm)
    lam = 0.25 * (d[:-1, :-1] + d[1:, :-1] + d[:-1, 1:] + d[1:, 1:]) * dz[:, None] * dl[None, :] * FSKY_TILE
    cnt = rng.poisson(lam).ravel()
    cidx = np.repeat(np.arange(cnt.size), cnt)
    nmc = lam.shape[1]
    i = cidx // nmc; j = cidx % nmc
    n = cidx.size
    f00 = d[i, j]; f10 = d[i + 1, j]; f01 = d[i, j + 1]; f11 = d[i + 1, j + 1]
    x = smp._lin_inv(rng.random(n), f00 + f01, f10 + f11)
    y = smp._lin_inv(rng.random(n), f00 + (f10 - f00) * x, f01 + (f11 - f01) * x)
    z = zz[i] + x * dz[i]
    M = np.exp(lm[j] + y * dl[j])
    th = model.theta500(M, z)
    Y = model.Y500_med(M, z) * np.exp(model.sigma_lnY * rng.standard_normal(n))
    py = rng.uniform(0, LBOX, n); px = rng.uniform(0, LBOX, n)
    return dict(y=py, x=px, z=z, M500=M, theta500=th, Y500=Y, lam=float(lam.sum()))


_MMF1 = {}


def sigma_Y(mode, theta):
    """Step 1's MMF sigma_Y [arcmin^2] (szsim.instrument.MMFModel spline, as steps 1-2)."""
    if mode not in _MMF1:
        _MMF1[mode] = ins.MMFModel(ins.Instrument.so_lat(mode, lmin=LMIN, lmax=LMAX))
    return _MMF1[mode].sigma_Y(theta)


def sigma_Y_exact(mode, theta):
    if mode not in _MMF1:
        _MMF1[mode] = ins.MMFModel(ins.Instrument.so_lat(mode, lmin=LMIN, lmax=LMAX))
    return _MMF1[mode].sigma_Y_exact(theta)


# ============================================================================ build tiles
def build_truth(pid):
    d = dict(np.load(os.path.join(BUILD_DATA, "truth", f"truth_{pid:03d}.npz")))
    return {k: v.astype(np.float64) for k, v in d.items()}


def build_tile(pid):
    return dict(np.load(os.path.join(BUILD_DATA, "tiles", f"tile_{pid:03d}.npz")))


RFFT_TO_PHYS = OMP / NPX       # build covariance ('rfft units', E|X|^2 = Npx C / Omega_pix) -> uK^2 sr


# ============================================================================ batch kernels for the fresh-sky pipeline
@nb.njit(cache=True)
def table_topk(qy, qx, qj, own, hy, hx, hY, hlth, R, n0, dr, lnth0, dnode, rcut, box, K, out_idx, out_val, out_sum):
    """For each query point (qy, qx) [arcmin] at scale qj: table contributions Y_h R_j(theta_h, r) of all halos within
    rcut except own[d] (-1: none); the K largest (indices, values) and the sum of all of them."""
    for d in range(qj.size):
        j = qj[d]
        tv = np.full(K, -1e300)
        ti = np.full(K, -1, np.int64)
        s = 0.0
        for h in range(hY.size):
            if h == own[d]:
                continue
            ddy = (hy[h] - qy[d] + 0.5 * box) % box - 0.5 * box
            if abs(ddy) > rcut:
                continue
            ddx = (hx[h] - qx[d] + 0.5 * box) % box - 0.5 * box
            if abs(ddx) > rcut:
                continue
            r = np.sqrt(ddy * ddy + ddx * ddx)
            if r > rcut:
                continue
            c = hY[h] * resp_eval(R, j, n0, hlth[h], r, dr, lnth0, dnode)
            s += c
            if c > tv[K - 1]:
                k = K - 1
                while k > 0 and c > tv[k - 1]:
                    tv[k] = tv[k - 1]; ti[k] = ti[k - 1]
                    k -= 1
                tv[k] = c; ti[k] = h
        out_sum[d] = s
        for k in range(K):
            out_idx[d, k] = ti[k]; out_val[d, k] = tv[k]


@nb.njit(cache=True)
def own_resp(qy, qx, qj, h_idx, hy, hx, hY, hlth, R, n0, dr, lnth0, dnode, box, out):
    for d in range(qj.size):
        h = h_idx[d]
        ddy = (hy[h] - qy[d] + 0.5 * box) % box - 0.5 * box
        ddx = (hx[h] - qx[d] + 0.5 * box) % box - 0.5 * box
        out[d] = hY[h] * resp_eval(R, qj[d], n0, hlth[h], np.sqrt(ddy * ddy + ddx * ddx), dr, lnth0, dnode)


@nb.njit(cache=True)
def alone_search(cy, cx, h_idx, hy, hx, hY, hlth, R, n0, dr, lnth0, dnode, box, pix, win, nmap, sig, out_v, out_j):
    """Isolated search: max over a (2 win + 1)^2 pixel window around pixel (cy, cx) and all scales of
    [Y_h R_j(|p - x_h|) + CN_j(p)] / sigma_j."""
    nsc = R.shape[0]
    for d in range(cy.size):
        h = h_idx[d]
        best = -1e300; bj = -1
        for j in range(nsc):
            for a in range(-win, win + 1):
                py = (cy[d] + a) * pix
                ddy = (py - hy[h] + 0.5 * box) % box - 0.5 * box
                for b in range(-win, win + 1):
                    px = (cx[d] + b) * pix
                    ddx = (px - hx[h] + 0.5 * box) % box - 0.5 * box
                    v = (hY[h] * resp_eval(R, j, n0, hlth[h], np.sqrt(ddy * ddy + ddx * ddx), dr, lnth0, dnode)
                         + nmap[d, j, a + win, b + win]) / sig[j]
                    if v > best:
                        best = v; bj = j
        out_v[d] = best; out_j[d] = bj
