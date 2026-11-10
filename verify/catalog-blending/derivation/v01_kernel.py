"""Independent matched-filter kernel (verifier).  Everything in arcmin units.
Template transform by the Fourier-slice theorem (2D FT of the projection of a truncated sphere =
3D FT of the sphere):  g(u) = int_0^5 4 pi x^2 P(x) sin(ux)/(ux) dx / int_0^1 4 pi x^2 P(x) dx
(unit spherical Y500; g(0) = Y5R500/Y500).  No szsim import here.
sigma_Y^-2 = (1/2pi) int k dk B^2 g(k thA)^2 / Nw,  B = exp(-k^2 sb^2/2), Nw = (4e-6)^2 arcmin^2
rho(thA,thB,s) = sigma_A^2 (1/2pi) int k dk B^2 g(k thA) g(k thB) J0(k s) / Nw
DC = int d^2s rho = sigma_A^2 g(0)^2 / Nw (k -> 0 limit of 2D FT).
"""
import os, time
os.environ.setdefault("NUMBA_NUM_THREADS", "2")
import numpy as np, numba as nb
from scipy.integrate import quad
from scipy.special import j0 as sj0

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "v01_kernel.npz")
c500, gam, alp, bet = 1.177, 0.3081, 1.0510, 5.4905
P = lambda x: 1.0 / ((c500 * x) ** gam * (1 + (c500 * x) ** alp) ** ((bet - gam) / alp))
FWHM, NOISE = 1.4, 4e-6
SB = FWHM / np.sqrt(8 * np.log(2)); NW = NOISE ** 2

Y1 = quad(lambda x: 4 * np.pi * x * x * P(x), 0, 1, epsrel=1e-12, limit=500)[0]
Y5 = quad(lambda x: 4 * np.pi * x * x * P(x), 0, 5, epsrel=1e-12, limit=500)[0]
RATIO = Y5 / Y1
# Gauss-Legendre composite on x in [0,5] with x = 5 t^2 (cusp-friendly)
nodes, wts = np.polynomial.legendre.leggauss(40)
NPAN = 4000
edges = np.linspace(0, 1, NPAN + 1)
tt = (0.5 * (edges[1:] - edges[:-1])[:, None] * (nodes[None, :] + 1) + edges[:-1][:, None]).ravel()
wt = (0.5 * (edges[1:] - edges[:-1])[:, None] * wts[None, :]).ravel()
XX = 5 * tt ** 2; WX = wt * 10 * tt * 4 * np.pi * XX ** 2 * P(XX) / Y1
UG = np.concatenate([[0.0], np.geomspace(1e-4, 5000, 6000)])
GG = np.array([np.sum(WX * np.sinc(u * XX / np.pi)) for u in UG])


@nb.njit(cache=True)
def J0(x):  # Numerical Recipes bessj0 (|err| ~1e-8)
    ax = abs(x)
    if ax < 8.0:
        y = x * x
        a1 = 57568490574.0 + y * (-13362590354.0 + y * (651619640.7 + y * (-11214424.18 + y * (77392.33017 + y * (-184.9052456)))))
        a2 = 57568490411.0 + y * (1029532985.0 + y * (9494680.718 + y * (59272.64853 + y * (267.8532712 + y))))
        return a1 / a2
    z = 8.0 / ax; y = z * z; xx = ax - 0.785398164
    a1 = 1.0 + y * (-0.1098628627e-2 + y * (0.2734510407e-4 + y * (-0.2073370639e-5 + y * 0.2093887211e-6)))
    a2 = -0.1562499995e-1 + y * (0.1430488765e-3 + y * (-0.6911147651e-5 + y * (0.7621095161e-6 - y * 0.934935152e-7)))
    return np.sqrt(0.636619772 / ax) * (np.cos(xx) * a1 - z * np.sin(xx) * a2)


def g(u):
    return np.interp(u, UG, GG, right=0.0)


def sigY(th, nk=200000):
    k = np.linspace(0, 7.0 / SB, nk)
    return np.array([1 / np.sqrt(np.trapezoid(k * np.exp(-(k * SB) ** 2) * g(k * t) ** 2, k) / (2 * np.pi) / NW) for t in np.atleast_1d(th)])


@nb.njit(parallel=True, cache=True)
def _hankel(k, w, s, out):
    for i in nb.prange(s.shape[0]):
        acc = 0.0
        for j in range(k.shape[0]):
            acc += w[j] * J0(k[j] * s[i])
        out[i] = acc


def rho_exact(thA, thB, s, sigA):
    S = max(np.max(s), 1.0)
    dk = min(np.pi / (12 * S), np.pi / (12 * 5 * max(thA, thB)), 0.002)
    kmax = 6.5 / SB
    k = np.arange(0, kmax + dk, dk)
    w = k * np.exp(-(k * SB) ** 2) * g(k * thA) * g(k * thB) * dk
    w[0] *= 0.5
    out = np.zeros(len(s)); _hankel(k, w, np.ascontiguousarray(s, dtype=np.float64), out)
    return out / (2 * np.pi) / NW * sigA ** 2


def main():
    t0 = time.time()
    print(f"RATIO = Y5R500/Y500 = {RATIO:.5f}; g(0) quadrature = {GG[0]:.5f}")
    # independent check of Fourier slice: direct 2D Hankel of the Abel projection at a few u
    def proj(sv):
        if sv >= 5: return 0.0
        L = np.sqrt(25 - sv * sv)
        return 2 * quad(lambda l: P(np.sqrt(sv * sv + l * l)), 0, L, limit=400, epsrel=1e-10)[0] / Y1
    sg_ = np.concatenate([np.geomspace(1e-6, 0.5, 400), np.linspace(0.5, 5, 1201)[1:]])
    fg = np.array([proj(v) for v in sg_])
    for u in (0.0, 1.0, 5.0, 20.0):
        gh = np.trapezoid(2 * np.pi * sg_ * fg * sj0(u * sg_), sg_)
        print(f"   g({u}) slice {np.interp(u, UG, GG):.6f}  vs  Abel+Hankel {gh:.6f}")
    sp = np.sqrt(4 * np.pi * SB ** 2 * NW) / RATIO
    print(f"sigma_Y point-source = {sp:.4e} arcmin^2 (claim 4.693e-6)")
    for t in (1.0, 3.0, 10.0):
        print(f"sigma_Y({t}') = {sigY(t)[0]:.4e}")
    for t in (0.25, 1, 3, 10, 30):
        s = sigY(t)[0]
        print(f"DC response({t}') = sigma^2 RATIO^2/Nw = {s*s*RATIO**2/NW:.3f} arcmin^2")
    # spot rho values
    s3, s05, s2 = sigY(3.0)[0], sigY(0.5)[0], sigY(2.0)[0]
    print(f"rho(3,0.5,0) = {rho_exact(3, 0.5, np.array([0.0, 30.0]), s3)[0]:.4f} (claim 2.19); "
          f"rho(2,2,2) = {rho_exact(2, 2, np.array([2.0, 30.0]), s2)[0]:.4f} (claim 0.47); "
          f"rho(2,2,0) = {rho_exact(2, 2, np.array([0.0, 30.0]), s2)[0]:.5f} (must be 1)")
    # inflection radius of rho(th,th,s)
    for t in (0.3, 1.0, 3.0, 10.0):
        st = sigY(t)[0]
        s = np.linspace(0, 3 * t + 4, 3001)
        r = rho_exact(t, t, s, st)
        d2 = np.gradient(np.gradient(r, s), s)
        i = np.nonzero(np.diff(np.sign(d2[5:])))[0][0] + 5
        print(f"s_infl(theta500={t}') = {s[i]:.3f}'")
    # ---- table
    NT, NU = 48, 401
    TH = np.geomspace(0.2, 90.0, NT); TT = np.linspace(0, 1, NU)
    RB = 5 * np.sqrt(2) * SB
    SIG = sigY(TH)
    RHO = np.zeros((NT, NT, NU)); worst = 0
    for i in range(NT):
        for j in range(i, NT):
            S = 5 * TH[i] + 5 * TH[j] + RB
            s = S * TT ** 2
            r = rho_exact(TH[i], TH[j], s, SIG[i])
            RHO[i, j] = r
            RHO[j, i] = r * SIG[j] ** 2 / SIG[i] ** 2
        print(f"   row {i} {time.time()-t0:.0f}s", flush=True) if i % 8 == 0 else None
    for i in range(NT):
        for j in range(NT):
            S = 5 * TH[i] + 5 * TH[j] + RB; s = S * TT ** 2
            I = np.trapezoid(2 * np.pi * s * RHO[i, j], s)
            worst = max(worst, abs(I / (SIG[i] ** 2 * RATIO ** 2 / NW) - 1))
    print(f"table DC check: max |int rho d2s / (sigma^2 RATIO^2/Nw) - 1| over {NT*NT} pairs = {worst:.2e}")
    np.savez(OUT, TH=TH, TT=TT, RHO=RHO, SIG=SIG, RB=RB, RATIO=RATIO, UG=UG, GG=GG, SB=SB, NW=NW)
    print(f"done {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
