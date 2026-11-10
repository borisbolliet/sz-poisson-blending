"""INDEPENDENT re-implementation (verifier) of the map-level SZ blending pipeline.

Nothing is imported from szsim or q2lib.  Different methods are used on purpose:
  * HMF: colossus called directly; own (z, lnM) midpoint grid; Poisson draw directly on a flat 10x10 deg torus
    (no full-sky catalogue, no equal-area cut).
  * profile: Abel-form projection with scipy quad weight='alg' (handles the 1/sqrt edge exactly).
  * beam convolution of the projected profile in REAL space with the exact radial Gaussian kernel (I0),
    not by Hankel transform.
  * matched filter: DISCRETE real-space template (point-sampled beam-convolved profile) correlated with the map
    by FFT; sigma_k = sigma_pix / sqrt(sum T_k^2); l=0 mode removed by subtracting the map mean.
  * response tables for the truth decomposition: own Hankel transform g(u) (Gauss-Legendre panels).
  * peaks: scipy.ndimage.maximum_filter(mode='wrap'); merging with a periodic cKDTree.
Units: arcmin, Y in arcmin^2, masses in Msun (h-free).
"""
import os
import numpy as np
import numba as nb
from scipy.special import j0, i0e, ndtr
from scipy.integrate import quad
from colossus.cosmology import cosmology
from colossus.lss import mass_function

OUT = "/Users/boris/GitHub/sz-poisson-blending/verify/map-blending/reproduction/out"
os.makedirs(OUT, exist_ok=True)
COS = cosmology.setCosmology("planck18")
H = COS.h
ARC = np.pi / 180 / 60
CKMS = 299792.458
PIX = 0.25
NPIX = 2400
LBOX = PIX * NPIX
FWHM = 1.4
SB = FWHM / np.sqrt(8 * np.log(2))          # arcmin
WNOISE = 4e-6                                 # y arcmin
NW = WNOISE ** 2                              # arcmin^2
SIGPIX = WNOISE / PIX
THK = np.geomspace(0.5, 10.0, 12)
K = THK.size
GN = dict(c=1.177, g=0.3081, a=1.0510, b=5.4905)
SIGLNY = 0.173


# ------------------------------------------------------------------ cosmology / HMF
def DA(z):
    return COS.angularDiameterDistance(np.asarray(z, float)) / H


def Ez(z):
    return COS.Ez(np.asarray(z, float))


def rhoc(z):
    return COS.rho_c(np.asarray(z, float)) * 1e9 * H ** 2


def dVdzdO(z):
    chi = COS.comovingDistance(0.0, np.asarray(z, float)) / H
    return CKMS / (100 * H) * chi ** 2 / Ez(z)


def dndlnM(M, z):
    return mass_function.massFunction(np.asarray(M) * H, float(z), mdef="500c", model="tinker08", q_out="dndlnM") * H ** 3


def Ymed(M, z):
    """median Y500 [arcmin^2]: E^{-2/3} D_A^2 Y / 1e-4 Mpc^2 = 10^-0.19 (0.8 M / 6e14)^1.79"""
    Y_Mpc2 = 1e-4 * 10 ** (-0.19) * Ez(z) ** (2.0 / 3.0) * (0.8 * np.asarray(M) / 6e14) ** 1.79
    return Y_Mpc2 / DA(z) ** 2 / ARC ** 2


def theta500(M, z):
    R = (3 * np.asarray(M) / (4 * np.pi * 500 * rhoc(z))) ** (1 / 3)
    return R / DA(z) / ARC


def hmf_grid(Mmin=1e13, Mmax=1e16, zmin=0.01, zmax=3.0, nz=600, nM=900, cache=True):
    fn = f"{OUT}/hmf_grid_{Mmin:.0e}_{Mmax:.0e}_{nz}_{nM}.npz"
    if cache and os.path.exists(fn):
        return dict(np.load(fn))
    ze = np.linspace(zmin, zmax, nz + 1); le = np.linspace(np.log(Mmin), np.log(Mmax), nM + 1)
    zc = 0.5 * (ze[1:] + ze[:-1]); lc = 0.5 * (le[1:] + le[:-1])
    dens = np.array([dndlnM(np.exp(lc), z) for z in zc]) * dVdzdO(zc)[:, None]   # per sr per dz per dlnM
    out = dict(ze=ze, le=le, zc=zc, lc=lc, dens=dens)
    np.savez(fn, **out)
    return out


# ------------------------------------------------------------------ profile
def gnfw(x):
    cx = GN["c"] * x
    return 1.0 / (cx ** GN["g"] * (1 + cx ** GN["a"]) ** ((GN["b"] - GN["g"]) / GN["a"]))


Y1 = quad(lambda x: 4 * np.pi * x * x * gnfw(x), 0, 1, limit=500, epsrel=1e-12)[0]
Y5 = quad(lambda x: 4 * np.pi * x * x * gnfw(x), 0, 5, limit=500, epsrel=1e-12)[0]
RATIO = Y5 / Y1


def proj_shape(s):
    """f(s) = 2 int_s^5 P(r) r dr / sqrt(r^2 - s^2) / Y1   (unit spherical Y500)."""
    out = np.zeros_like(s, dtype=float)
    for i, si in enumerate(s):
        if si >= 5:
            continue
        if si == 0:
            out[i] = 2 * quad(gnfw, 0, 5, limit=500, epsrel=1e-11)[0]
        else:
            # weight (r-s)^(-1/2) handled by 'alg' weight on [s,5]: w = (r-s)^a (5-r)^b with a=-1/2, b=0
            out[i] = 2 * quad(lambda r: gnfw(r) * r / np.sqrt(r + si), si, 5, weight="alg", wvar=(-0.5, 0.0),
                              limit=500, epsrel=1e-10)[0]
    return out / Y1


def profile_table(cache=True):
    fn = f"{OUT}/profile_table.npz"
    if cache and os.path.exists(fn):
        d = np.load(fn); return d["s"], d["f"]
    s = np.concatenate([[0.0], np.geomspace(1e-6, 5 * (1 - 1e-10), 4000), [5.0]])
    f = proj_shape(s)
    np.savez(fn, s=s, f=f)
    return s, f


S_TAB, F_TAB = profile_table()


def f_of(s):
    return np.interp(s, S_TAB, F_TAB, right=0.0)


def beam_conv_profile(theta, r):
    """Beam-convolved, unit-Y500 projected profile [arcmin^-2] at radii r (arcmin), real-space I0 kernel:
       F(r) = int_0^{5 th} r' dr' f(r'/th)/th^2 exp(-(r-r')^2/2s^2) i0e(r r'/s^2) / s^2"""
    rmax = 5 * theta
    n = int(max(3000, rmax / (SB / 25), 400 * 5))
    # nodes clustered near the sqrt edge at 5 theta and fine everywhere
    t = (np.arange(n) + 0.5) / n
    rp = rmax * (1 - (1 - t) ** 2); w = rmax * 2 * (1 - t) / n          # midpoint rule in t
    fp = f_of(rp / theta) / theta ** 2 * rp * w
    out = np.empty(r.size)
    for a in range(0, r.size, 256):
        rr = r[a:a + 256, None]
        Kmat = np.exp(-(rr - rp[None, :]) ** 2 / (2 * SB * SB)) * i0e(rr * rp[None, :] / SB ** 2) / SB ** 2
        out[a:a + 256] = Kmat @ fp
    return out


LNT0, DLNT, NT = np.log(0.02), 0.04, 205     # theta nodes 0.02' .. ~70'
TNODES = np.exp(LNT0 + DLNT * np.arange(NT))
NRP = 1500


def paint_table(cache=True):
    fn = f"{OUT}/paint_table.npz"
    if cache and os.path.exists(fn):
        d = np.load(fn); return d["tab"], d["rmax"]
    rmax = 5 * TNODES + 6 * SB
    tab = np.zeros((NT, NRP))
    for i, th in enumerate(TNODES):
        tab[i] = beam_conv_profile(th, np.linspace(0, rmax[i], NRP))
    np.savez(fn, tab=tab, rmax=rmax)
    return tab, rmax


# ------------------------------------------------------------------ Hankel of the unit template (for responses)
def g_hankel_table(cache=True):
    fn = f"{OUT}/g_hankel.npz"
    if cache and os.path.exists(fn):
        d = np.load(fn); return d["u"], d["g"]
    # Gauss-Legendre panels on [0,5], panels refined towards the edge
    xg, wg = np.polynomial.legendre.leggauss(10)
    edges = np.unique(np.concatenate([np.linspace(0, 4.9, 4000), 5 - np.geomspace(1e-7, 0.1, 400)[::-1], [5.0]]))
    a, b = edges[:-1], edges[1:]
    s = (0.5 * (b - a)[:, None] * xg[None, :] + 0.5 * (a + b)[:, None]).ravel()
    w = (0.5 * (b - a)[:, None] * wg[None, :]).ravel()
    fs = 2 * np.pi * s * f_of(s) * w
    u = np.concatenate([[0.0], np.geomspace(1e-3, 4e3, 3000)])
    g = np.array([fs @ j0(uu * s) for uu in u])
    np.savez(fn, u=u, g=g)
    return u, g


UG, GG = g_hankel_table()


def g_of(u):
    return np.interp(u, UG, GG, right=0.0)


# ------------------------------------------------------------------ discrete templates / sigma_k
def grid_r():
    j = np.arange(NPIX); d = np.minimum(j, NPIX - j) * PIX
    return np.sqrt(d[:, None] ** 2 + d[None, :] ** 2)


def templates(cache=True):
    """rfft2 of discrete templates T_k (centred at pixel (0,0), periodic) and sigma_k [arcmin^2]."""
    fn = f"{OUT}/templates.npz"
    if cache and os.path.exists(fn):
        d = np.load(fn); return d["Tft"], d["sig"], d["Tsum"]
    from scipy import fft as sfft
    r = grid_r()
    Tft, sig, Tsum = [], [], []
    for th in THK:
        rr = np.linspace(0, 5 * th + 7 * SB, 6000)
        T = np.interp(r, rr, beam_conv_profile(th, rr), right=0.0)
        S2 = (T ** 2).sum()
        sig.append(SIGPIX / np.sqrt(S2)); Tsum.append(T.sum() * PIX ** 2)
        Tft.append((sfft.rfft2(T) / S2).astype(np.complex64))
    Tft = np.array(Tft); sig = np.array(sig); Tsum = np.array(Tsum)
    np.savez(fn, Tft=Tft, sig=sig, Tsum=Tsum)
    return Tft, sig, Tsum


def sigma_Y_hankel(theta, nl=40000):
    """continuous matched-filter sigma_Y(theta) [arcmin^2] (own Hankel)."""
    th = np.atleast_1d(theta).astype(float)
    l = np.linspace(0, 8.0 / SB, nl)
    B2 = np.exp(-(l * SB) ** 2)
    out = np.array([np.trapezoid(l * B2 * g_of(l * t) ** 2, l) / (2 * np.pi) / NW for t in th])
    return 1 / np.sqrt(out)


def sigY_interp():
    fn = f"{OUT}/sigY.npz"
    if os.path.exists(fn):
        d = np.load(fn); tg, sg = d["t"], d["s"]
    else:
        tg = np.geomspace(0.005, 200, 400); sg = sigma_Y_hankel(tg); np.savez(fn, t=tg, s=sg)
    return lambda th: np.exp(np.interp(np.log(th), np.log(tg), np.log(sg)))


# ------------------------------------------------------------------ response tables R_k(r; theta)
NRR = 1536


def resp_table(sig, cache=True):
    """R_k(r;theta) = sig_k^2/N_w (1/2pi) int dl l B^2 g(l th_k) g(l th) J0(l r); r in [0, rmaxR(theta)]."""
    fn = f"{OUT}/resp_table.npz"
    if cache and os.path.exists(fn):
        d = np.load(fn); return d["R"], d["rmax"]
    rmax = 5 * TNODES + 5 * THK[-1] + 8 * SB
    R = np.zeros((K, NT, NRR))
    lmax = 7.5 / SB
    for i, th in enumerate(TNODES):
        dl = 0.3 / rmax[i]
        l = np.arange(0, lmax + dl, dl)
        w = np.full(l.size, dl); w[0] *= 0.5; w[-1] *= 0.5
        base = w * l * np.exp(-(l * SB) ** 2) * g_of(l * th) / (2 * np.pi) / NW
        Wm = np.stack([base * g_of(l * tk) * s * s for tk, s in zip(THK, sig)], 1)
        r = np.linspace(0, rmax[i], NRR)
        for a in range(0, NRR, 256):
            R[:, i, a:a + 256] = (j0(np.outer(r[a:a + 256], l)) @ Wm).T
    np.savez(fn, R=R, rmax=rmax)
    return R, rmax


@nb.njit(cache=True)
def tab_eval(tab, rmax, lnth, r):
    """bilinear in (ln theta, r) with per-node r range; zero outside."""
    nt, nr = tab.shape
    t = (lnth - LNT0) / DLNT
    if t < 0:
        t = 0.0
    if t > nt - 1.0000001:
        t = nt - 1.0000001
    i0 = int(t); w = t - i0
    v = 0.0
    for a in range(2):
        ia = i0 + a; wa = (1 - w) if a == 0 else w
        u = r / rmax[ia] * (nr - 1)
        if u < nr - 1:
            j = int(u); f = u - j
            v += wa * ((1 - f) * tab[ia, j] + f * tab[ia, j + 1])
    return v


@nb.njit(cache=True)
def paint(m, xs, ys, Ys, ths, tab, rmax):
    n = m.shape[0]
    for h in range(xs.size):
        lnth = np.log(ths[h])
        t = min(max((lnth - LNT0) / DLNT, 0.0), tab.shape[0] - 1.0000001)
        i0 = int(t)
        R = max(rmax[i0], rmax[i0 + 1])
        jx0 = int(np.floor((xs[h] - R) / PIX)); jx1 = int(np.ceil((xs[h] + R) / PIX))
        jy0 = int(np.floor((ys[h] - R) / PIX)); jy1 = int(np.ceil((ys[h] + R) / PIX))
        for jy in range(jy0, jy1 + 1):
            dy = (jy + 0.5) * PIX - ys[h]
            for jx in range(jx0, jx1 + 1):
                dx = (jx + 0.5) * PIX - xs[h]
                r = np.sqrt(dx * dx + dy * dy)
                if r < R:
                    m[jy % n, jx % n] += Ys[h] * tab_eval(tab, rmax, lnth, r)


@nb.njit(cache=True)
def contrib(R, rmax, k, x0, y0, xs, ys, Ys, lnths, idx):
    out = np.empty(idx.size)
    for t in range(idx.size):
        i = idx[t]
        dx = (xs[i] - x0 + LBOX / 2) % LBOX - LBOX / 2
        dy = (ys[i] - y0 + LBOX / 2) % LBOX - LBOX / 2
        out[t] = Ys[i] * tab_eval(R[k], rmax, lnths[i], np.sqrt(dx * dx + dy * dy))
    return out


# ------------------------------------------------------------------ patch draw
def draw_patch(seed, grid, area_deg2=LBOX ** 2 / 3600.0):
    rng = np.random.default_rng(seed)
    A_sr = area_deg2 * (np.pi / 180) ** 2
    dz = np.diff(grid["ze"]); dl = np.diff(grid["le"])
    lam = grid["dens"] * dz[:, None] * dl[None, :] * A_sr
    cnt = rng.poisson(lam)
    iz, im = np.nonzero(cnt)
    rep = cnt[iz, im]
    iz = np.repeat(iz, rep); im = np.repeat(im, rep)
    n = iz.size
    z = grid["ze"][iz] + rng.random(n) * dz[iz]
    M = np.exp(grid["le"][im] + rng.random(n) * dl[im])
    x = rng.random(n) * LBOX; y = rng.random(n) * LBOX
    Y = Ymed(M, z) * np.exp(SIGLNY * rng.standard_normal(n))
    th = theta500(M, z)
    return dict(x=x, y=y, z=z, M=M, Y=Y, th=th, lam=lam.sum())
