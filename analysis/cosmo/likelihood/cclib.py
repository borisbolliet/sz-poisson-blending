"""Cluster-count likelihood with and without blending (compound-Poisson confusion) -- shared library.

Observable: counts in bins of (z of the primary, q_obs).  Model (full sky):

  N[zb, qb] = int dz dlnM d2N/dzdlnM  int dlnY p(lnY|M,z)  K(qb | theta500(M,z), q(M,z,Y))

  naive  : K_n(qb | q)        = Phi(b - q) - Phi(a - q)                    (q_obs = q + n, n ~ N(0,1))
  blended: K_b(qb | theta, q) = P_dom(theta, q) * [F(b - q) - F(a - q)],   F = CDF of y = n + delta_tr - mu(theta)
           delta_tr = compound-Poisson neighbour sum with the rate truncated at x < q (dominance: A is the
           brightest contributor), P_dom = exp(-Lambda(x >= q)), mu = <delta> (untruncated) = map-mean removal.

Implementation:  K_b is tabulated as G[i, k, t] = P_dom(th_i, q_k) * F_ik(t)   (t = q_obs - q_A), on the 40 theta
nodes of the rho table and the 259 q nodes of b03 (data/blending/analytic.npz), and interpolated
(linear in ln theta, linear in q, linear in t) at every population point.

Units: M500c Msun (h-free), angles arcmin, Y arcmin^2.  Threads: capped by the caller (NUMBA_NUM_THREADS).
"""
import os, sys, time, math
import numpy as np
import numba as nb
from scipy.special import ndtr
from colossus.cosmology import cosmology
from colossus.lss import mass_function

ROOT = "/Users/boris/GitHub/sz-poisson-blending"
sys.path.insert(0, f"{ROOT}/analysis/blending")
import blendlib as bl                                    # noqa: E402
from szsim import signal as sg, cosmo_hmf as ch          # noqa: E402

OUT = f"{ROOT}/data_cosmo"
FIG = f"{ROOT}/figures/cosmo"
C_KMS = 299792.458
ARCMIN = ch.ARCMIN
FULLSKY_DEG2 = 4 * np.pi * (180 / np.pi) ** 2           # 41252.96
FID = dict(Om0=0.3111, sigma8=0.8102, h=0.6766, Ob0=0.049, ns=0.9665)       # colossus planck18
SR_FID = dict(logA=-0.19, alpha=1.79, beta_E=2.0 / 3.0, one_minus_b=0.8, sigma_lnY=0.173)
M_PIVOT = 6e14
_SZ = sg.SZModel()                                       # instrument only: sigma_Y(theta500) [arcmin^2]


def sigma_Y(th):
    return _SZ.sigma_Y(th)


# ============================================================================ cosmology-dependent population
_CACHE = {}


def get_cosmo(Om0, sigma8, h=FID["h"], Ob0=FID["Ob0"], ns=FID["ns"]):
    key = (round(Om0, 10), round(sigma8, 10), h, Ob0, ns)
    if key not in _CACHE:
        if len(_CACHE) > 40:
            _CACHE.clear()
        _CACHE[key] = cosmology.Cosmology(name=f"c{len(_CACHE)}", flat=True, Om0=Om0, Ob0=Ob0, H0=100 * h,
                                          sigma8=sigma8, ns=ns, persistence="", print_warnings=False)
    c = _CACHE[key]
    cosmology.setCurrent(c)
    return c


class Grid:
    """(z, lnM) midpoint grid; z cells of width dz starting at zmin, lnM cells between Mmin and Mmax."""

    def __init__(self, zmin=0.01, zmax=3.0, dz=0.01, Mmin=1e13, Mmax=1e16, nM=180):
        self.zE = np.round(np.arange(zmin, zmax + 0.5 * dz, dz), 10)
        self.z = 0.5 * (self.zE[1:] + self.zE[:-1]); self.dz = dz
        self.lnME = np.linspace(np.log(Mmin), np.log(Mmax), nM + 1)
        self.lnM = 0.5 * (self.lnME[1:] + self.lnME[:-1]); self.dlnM = self.lnME[1] - self.lnME[0]
        self.M = np.exp(self.lnM)


GRID = Grid()


def hmf_grid(cosmo_pars, grid=GRID, model="tinker08"):
    """Full-sky d2N/dz/dlnM on the grid (nz, nM), plus E(z), D_A [Mpc], rho_c [Msun/Mpc^3]."""
    c = get_cosmo(**cosmo_pars)
    h = c.h
    z = grid.z
    chi = c.comovingDistance(0.0, z) / h
    Ez = c.Ez(z)
    DA = chi / (1 + z)
    rhoc = c.rho_c(z) * 1e9 * h ** 2
    dV = (C_KMS / c.H0) / Ez * chi ** 2 * 4 * np.pi
    Mh = grid.M * h
    d2N = np.empty((len(z), len(grid.M)))
    for iz, zz in enumerate(z):
        d2N[iz] = mass_function.massFunction(Mh, float(zz), mdef="500c", model=model, q_out="dndlnM") * h ** 3 * dV[iz]
    return d2N, Ez, DA, rhoc


def population(cosmo_pars, sr=SR_FID, grid=GRID, ngh=20, qmin=0.1, zedges=None, hmf=None):
    """Primary population points: lnth, q (incl. lognormal scatter, Gauss-Hermite), full-sky weight, z-bin index.
    Points with q < qmin are dropped (they contribute < 1e-4 of the q>4 counts, see c02 log)."""
    d2N, Ez, DA, rhoc = hmf if hmf is not None else hmf_grid(cosmo_pars, grid)
    Z = grid.z[:, None]; M = grid.M[None, :]
    R500 = (3 * M / (4 * np.pi * 500 * rhoc[:, None])) ** (1 / 3)
    th = R500 / DA[:, None] / ARCMIN
    Ymed = 1e-4 * 10 ** sr["logA"] * Ez[:, None] ** sr["beta_E"] * (sr["one_minus_b"] * M / M_PIVOT) ** sr["alpha"] \
        / DA[:, None] ** 2 / ARCMIN ** 2
    q0 = Ymed / sigma_Y(th)
    x, wg = np.polynomial.hermite_e.hermegauss(ngh); wg = wg / wg.sum()
    w0 = d2N * grid.dz * grid.dlnM
    q = (q0[..., None] * np.exp(sr["sigma_lnY"] * x)).ravel()
    w = (w0[..., None] * wg).ravel()
    lth = np.repeat(np.log(th).ravel(), ngh)
    zi = np.repeat(np.repeat(np.arange(len(grid.z)), len(grid.M)), ngh)
    zb = np.full(zi.size, -1, np.int64) if zedges is None else np.searchsorted(zedges, grid.z, side="right")[zi] - 1
    if zedges is not None:
        zb[(grid.z[zi] >= zedges[-1]) | (grid.z[zi] < zedges[0])] = -1
    s = q > qmin
    return dict(lth=lth[s], q=q[s], w=w[s], zb=zb[s] if zedges is not None else np.zeros(s.sum(), np.int64),
                z=grid.z[zi][s], th=np.exp(lth[s]))


def neighbour_pop(cosmo_pars, sr=SR_FID, grid=GRID, ngh=16, hmf=None):
    """Neighbour population for the P(D) rate (blendlib format): thB, YB, class, expected number per arcmin^2."""
    d2N, Ez, DA, rhoc = hmf if hmf is not None else hmf_grid(cosmo_pars, grid)
    Z = grid.z[:, None]; M = grid.M[None, :]
    R500 = (3 * M / (4 * np.pi * 500 * rhoc[:, None])) ** (1 / 3)
    th = (R500 / DA[:, None] / ARCMIN).ravel()
    Ymed = (1e-4 * 10 ** sr["logA"] * Ez[:, None] ** sr["beta_E"] * (sr["one_minus_b"] * M / M_PIVOT) ** sr["alpha"]
            / DA[:, None] ** 2 / ARCMIN ** 2).ravel()
    W = (d2N * grid.dz * grid.dlnM).ravel() / bl.ARCMIN2_FULLSKY
    cls = np.broadcast_to(bl.mass_class(grid.M)[None, :], d2N.shape).ravel()
    x, wg = np.polynomial.hermite_e.hermegauss(ngh); wg = wg / wg.sum()
    thB = np.repeat(th, ngh); YB = (Ymed[:, None] * np.exp(sr["sigma_lnY"] * x)).ravel()
    cB = np.repeat(cls, ngh).astype(np.int8); wB = (W[:, None] * wg).ravel()
    return thB, YB, cB, wB


def rates(pop, tab=None):
    """Rate histograms R(x | theta_A node) for all 40 rho-table nodes (as in b03)."""
    tab = tab or bl.load_rho_table()
    sigA = sigma_Y(tab.th)
    nb_ = len(bl.XLOG_EDGES) - 1
    W = np.zeros((tab.NT, 3, nb_)); WX = np.zeros_like(W); WX2 = np.zeros_like(W)
    CAP = np.zeros((tab.NT, 3)); CAPX = np.zeros((tab.NT, 3))
    for i in range(tab.NT):
        W[i], WX[i], WX2[i], CAP[i], CAPX[i] = bl.rate_histogram(tab.th[i], sigA[i], pop, tab)
    return dict(W=W, WX=WX, WX2=WX2, CAP=CAP, CAPX=CAPX, mu=WX.sum((1, 2)) + CAPX.sum(1), th=tab.th)


# ============================================================================ blended kernel table
QNODES = np.concatenate([np.arange(0.1, 20.0, 0.1), np.geomspace(20.0, 3000.0, 60)])        # = b03
TGRID = np.concatenate([np.arange(-40.0, 20.0, 0.05), np.geomspace(20.0, 1500.0, 160)])


def build_G(R, qnodes=QNODES, tgrid=TGRID, N=2 ** 17, dd=0.01, off=10.0, noise_var=1.0, verbose=True):
    """G[i,k,t] = P_dom(th_i, q_k) * P(n + delta_tr - mu_i <= t), map mean removed.
    Also returns P_dom and the untruncated mean mu."""
    W, WX, WX2, CAP = R["W"].sum(1), R["WX"].sum(1), R["WX2"].sum(1), R["CAP"].sum(1)
    mu = R["mu"]
    NT = W.shape[0]
    k = 2 * np.pi * np.fft.rfftfreq(N, d=dd)
    g_edge = np.arange(N) * dd - off + 0.5 * dd          # cdf[j] = P(y <= g_edge[j])
    G = np.zeros((NT, len(qnodes), len(tgrid)), np.float32)
    Pdom = np.zeros((NT, len(qnodes)))
    t0 = time.time()
    for i in range(NT):
        with np.errstate(invalid="ignore", divide="ignore"):
            xb = np.where(W[i] > 0, WX[i] / W[i], 0.0)
        for kk, qa in enumerate(qnodes):
            xcut = min(qa, bl.X_CAP)
            keep = (W[i] > 0) & (xb < xcut)
            small = keep & (xb < 2 * dd)
            big = keep & ~small
            mu_s = WX[i][small].sum(); v_s = WX2[i][small].sum()
            Rg = np.zeros(N)
            f = xb[big] / dd; i0 = np.floor(f).astype(np.int64); fr = f - i0
            np.add.at(Rg, i0, W[i][big] * (1 - fr)); np.add.at(Rg, i0 + 1, W[i][big] * fr)
            lam = W[i][big].sum()
            lnphi = np.fft.rfft(Rg) - lam - 1j * k * (mu_s + off) - 0.5 * k ** 2 * (v_s + noise_var)
            p = np.clip(np.fft.irfft(np.exp(lnphi), n=N), 0, None)
            p /= p.sum()
            cdf = np.cumsum(p)
            Pdom[i, kk] = np.exp(-(W[i][(W[i] > 0) & (xb >= xcut)].sum() + CAP[i]))
            G[i, kk] = Pdom[i, kk] * np.interp(tgrid + mu[i], g_edge, cdf, left=0.0, right=1.0)
        if verbose and (i % 10 == 9 or i == NT - 1):
            print(f"   build_G: theta node {i+1}/{NT} done, {time.time()-t0:.1f}s", flush=True)
    return G, Pdom


@nb.njit(parallel=True, cache=True)
def _counts_G(G, lth0, dlth, qnodes, tgrid, lth, q, w, zb, edges, nzb):
    NT, NQ, Nt = G.shape
    ne = edges.shape[0]
    nthr = nb.get_num_threads()
    acc = np.zeros((nthr, nzb, ne - 1))
    n = q.shape[0]
    chunk = (n + nthr - 1) // nthr
    for tid in nb.prange(nthr):
        F = np.zeros(ne)
        for p in range(tid * chunk, min(n, (tid + 1) * chunk)):
            if zb[p] < 0:
                continue
            fi = (lth[p] - lth0) / dlth
            if fi < 0.0:
                fi = 0.0
            if fi > NT - 1.000001:
                fi = NT - 1.000001
            i0 = int(fi); wi = fi - i0
            qq = q[p]
            if qq <= qnodes[0]:
                k0 = 0; wk = 0.0
            elif qq >= qnodes[NQ - 1]:
                k0 = NQ - 2; wk = 1.0
            else:
                k0 = np.searchsorted(qnodes, qq) - 1
                wk = (qq - qnodes[k0]) / (qnodes[k0 + 1] - qnodes[k0])
            for j in range(ne):
                t = edges[j] - qq
                if t <= tgrid[0]:
                    jt = 0; wt = 0.0; below = True
                else:
                    below = False
                    if t >= tgrid[Nt - 1]:
                        jt = Nt - 2; wt = 1.0
                    else:
                        jt = np.searchsorted(tgrid, t) - 1
                        wt = (t - tgrid[jt]) / (tgrid[jt + 1] - tgrid[jt])
                v = 0.0
                if not below:
                    for da in range(2):
                        fa = wi if da else 1.0 - wi
                        for dk in range(2):
                            fk = wk if dk else 1.0 - wk
                            g0 = G[i0 + da, k0 + dk, jt]; g1 = G[i0 + da, k0 + dk, jt + 1]
                            v += fa * fk * ((1.0 - wt) * g0 + wt * g1)
                F[j] = v
            for j in range(ne - 1):
                acc[tid, zb[p], j] += w[p] * (F[j + 1] - F[j])
    out = np.zeros((nzb, ne - 1))
    for tid in range(nthr):
        out += acc[tid]
    return out


LTH0 = float(np.log(bl.load_rho_table().th[0])); DLTH = float(np.log(bl.load_rho_table().th[1] / bl.load_rho_table().th[0]))


def counts_blended(G, pop, edges, nzb, qnodes=QNODES, tgrid=TGRID):
    e = np.asarray(edges, np.float64).copy(); e[~np.isfinite(e)] = 1e6
    return _counts_G(G, LTH0, DLTH, qnodes, tgrid, pop["lth"], pop["q"], pop["w"], pop["zb"], e, nzb)


@nb.njit(parallel=True, cache=False)
def _counts_cdf(q, w, zb, shift, scale, edges, nzb, tx, tF):
    """sum_p w_p [F((b - q_p - shift_p)/scale_p) - F((a - ...)/...)], F tabulated on (tx, tF) (or Phi if tx empty)."""
    ne = edges.shape[0]
    nthr = nb.get_num_threads()
    acc = np.zeros((nthr, nzb, ne - 1))
    n = q.shape[0]
    chunk = (n + nthr - 1) // nthr
    use_tab = tx.shape[0] > 0
    for tid in nb.prange(nthr):
        F = np.zeros(ne)
        for p in range(tid * chunk, min(n, (tid + 1) * chunk)):
            if zb[p] < 0:
                continue
            for j in range(ne):
                x = (edges[j] - q[p] - shift[p]) / scale[p]
                if use_tab:
                    if x <= tx[0]:
                        F[j] = 0.0
                    elif x >= tx[-1]:
                        F[j] = 1.0
                    else:
                        k = np.searchsorted(tx, x) - 1
                        f = (x - tx[k]) / (tx[k + 1] - tx[k])
                        F[j] = (1 - f) * tF[k] + f * tF[k + 1]
                else:
                    F[j] = 0.5 * math.erfc(-x / math.sqrt(2.0))
            for j in range(ne - 1):
                acc[tid, zb[p], j] += w[p] * (F[j + 1] - F[j])
    out = np.zeros((nzb, ne - 1))
    for tid in range(nthr):
        out += acc[tid]
    return out


def counts_gauss(pop, edges, nzb, shift=0.0, var=1.0, skew_alpha=None, omega=None):
    """Naive (shift=0, var=1) or effective kernel: q_obs = q + shift + N(0, var) [shift, var scalars or per-point].
    With skew_alpha/omega: q_obs = q + shift + n + SN(0, omega, alpha); n + SN is again skew-normal (scale
    sqrt(1+omega^2), shape alpha' = d'/sqrt(1-d'^2), d' = d omega/sqrt(1+omega^2), d = alpha/sqrt(1+alpha^2));
    its CDF is tabulated once per call (scipy) and interpolated."""
    e = np.asarray(edges, np.float64).copy(); e[~np.isfinite(e)] = 1e6
    n = pop["q"].size
    sh = np.broadcast_to(np.asarray(shift, np.float64), (n,)).copy()
    if skew_alpha is None:
        sc = np.broadcast_to(np.sqrt(np.asarray(var, np.float64)), (n,)).copy()
        tx = np.zeros(0); tF = np.zeros(0)
    else:
        from scipy.stats import skewnorm
        dl = skew_alpha / np.sqrt(1 + skew_alpha ** 2)
        om2 = omega ** 2 + 1.0
        dl2 = dl * omega / np.sqrt(om2)
        a2 = dl2 / np.sqrt(1 - dl2 ** 2)
        sc = np.full(n, np.sqrt(om2))
        tx = np.linspace(-12, 40, 5201); tF = skewnorm.cdf(tx, a2)
    zb = np.asarray(pop["zb"], np.int64)
    return _counts_cdf(pop["q"], pop["w"], zb, sh, sc, e, nzb, tx, tF)


# ============================================================================ Poisson likelihood helpers
def cash(D, mu):
    """-2 ln L (Poisson deviance) for data D (may be non-integer Asimov) and model mu."""
    D = np.asarray(D); mu = np.asarray(mu)
    t = np.where(D > 0, D * np.log(np.where(D > 0, D, 1) / mu), 0.0)
    return 2 * np.sum(mu - D + t)


def pull_sigma(Nb, Nn, fsky):
    """(Nb - Nn)/sqrt(Nn) for a survey of sky fraction fsky (full-sky counts in)."""
    return (Nb - Nn) * fsky / np.sqrt(np.maximum(Nn * fsky, 1e-300))
