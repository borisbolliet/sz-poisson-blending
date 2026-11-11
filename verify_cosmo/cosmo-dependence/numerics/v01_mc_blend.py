"""Independent minimal re-implementation (verifier) of the catalogue-level blending counts vs cosmology.

Independent pieces: background (astropy FlatLambdaCDM, not colossus), HMF unit handling (colossus Tinker08
called directly), Y-M / theta500, population sampling, neighbour field drawn by DIRECT Monte Carlo around each
primary (Slivnyak: neighbours of a halo = Poisson field around a random point), dominance, mean removal from the
direct integral of rho, analytic noise average.  NO FFT/P(D)/rate-histogram machinery from cosmolib.
Shared (cosmology-independent) pieces: rho table + rho_at (blendlib), sigma_Y(theta) (szsim.SZModel).

usage: v01_mc_blend.py sigma8 Om omb slnY nprim seed
"""
import os, sys, time
os.environ["NUMBA_NUM_THREADS"] = "2"; os.environ["OMP_NUM_THREADS"] = "2"
import numpy as np
import numba as nb
from scipy.special import ndtr, ndtri
ROOT = "/Users/boris/GitHub/sz-poisson-blending"
sys.path.insert(0, os.path.join(ROOT, "analysis", "blending"))
import blendlib as bl
from szsim import signal as sg
from astropy.cosmology import FlatLambdaCDM
import astropy.units as u
from colossus.cosmology import cosmology
from colossus.lss import mass_function

ARC = np.pi / 180 / 60
FULL = 4 * np.pi / ARC ** 2
ZE = np.array([0.01, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 2.0, 3.0])
QE = np.array([5.0, 6.0, 7.0, 8.0, 10.0, 12.5, 15.0, 20.0, 30.0, 50.0, np.inf])


def population(s8, Om, omb, nz=500, nm=300):
    H0, Ob = 67.66, 0.049
    ap = FlatLambdaCDM(H0=H0, Om0=Om, Ob0=Ob, Tcmb0=2.7255, Neff=3.046, m_nu=0 * u.eV)
    h = H0 / 100
    cosmology.setCosmology("verif", flat=True, H0=H0, Om0=Om, Ob0=Ob, sigma8=s8, ns=0.9665, persistence="")
    ze = np.linspace(0.01, 3.0, nz + 1); z = 0.5 * (ze[1:] + ze[:-1]); dz = ze[1] - ze[0]
    le = np.linspace(np.log(1e13), np.log(1e16), nm + 1); lm = 0.5 * (le[1:] + le[:-1]); dl = le[1] - le[0]
    M = np.exp(lm)
    dndlnM = np.array([mass_function.massFunction(M * h, zz, mdef="500c", model="tinker08", q_out="dndlnM")
                       for zz in z]) * h ** 3                                  # Mpc^-3
    dV = ap.differential_comoving_volume(z).to(u.Mpc ** 3 / u.sr).value      # Mpc^3/sr
    n = dndlnM * dV[:, None] * 4 * np.pi * dz * dl                             # full-sky counts per cell
    DA = ap.angular_diameter_distance(z).to(u.Mpc).value
    rc = ap.critical_density(z).to(u.Msun / u.Mpc ** 3).value
    Ez = ap.efunc(z)
    Z, MM = np.meshgrid(z, M, indexing="ij")
    Y = 1e-4 * 10 ** -0.19 * Ez[:, None] ** (2 / 3) * (omb * MM / 6e14) ** 1.79 / DA[:, None] ** 2 / ARC ** 2
    th = (3 * MM / (4 * np.pi * 500 * rc[:, None])) ** (1 / 3) / DA[:, None] / ARC
    return dict(n=n.ravel(), Y=Y.ravel(), th=th.ravel(), z=Z.ravel(), M=MM.ravel())


@nb.njit(cache=True)
def nb_sums(seed, thA, sigA, a_arr, Lam, pk, cdf0, cdf1, cdf2, Ymed, thB, slnY, rho, lth0, dlth, RB, D, xmax):
    np.random.seed(seed)
    for i in range(thA.shape[0]):
        nn = np.random.poisson(Lam[i])
        a = a_arr[i]
        d = 0.0; mx = 0.0
        for _ in range(nn):
            r = np.random.random()
            if r < pk[i, 0]:
                c = cdf0
            elif r < pk[i, 0] + pk[i, 1]:
                c = cdf1
            else:
                c = cdf2
            t = np.searchsorted(c, np.random.random() * c[-1])
            if t >= Ymed.shape[0]:
                t = Ymed.shape[0] - 1
            YB = Ymed[t] * np.exp(slnY * np.random.standard_normal())
            S = a + 5.0 * thB[t]
            s = S * np.sqrt(np.random.random())
            rr = bl.rho_at(rho, lth0, dlth, RB, thA[i], thB[t], s)
            x = rr * YB / sigA[i]
            d += x
            if x > mx:
                mx = x
        D[i] = d; xmax[i] = mx


@nb.njit(cache=True)
def dc_int(thA, thBg, rho, lth0, dlth, RB, nt):
    """int d^2 s rho(thA, thB, s) on a thB grid."""
    out = np.zeros(thBg.shape[0])
    for j in range(thBg.shape[0]):
        S = 5.0 * thA + 5.0 * thBg[j] + RB
        acc = 0.0
        for k in range(nt):
            t = (k + 0.5) / nt
            s = S * t * t
            acc += bl.rho_at(rho, lth0, dlth, RB, thA, thBg[j], s) * 2 * np.pi * s * 2 * S * t / nt
        out[j] = acc
    return out


@nb.njit(cache=True)
def var_int(thA, sigA, Ymed, thB, wpop, xg, wg, slnY, xcut, rho, lth0, dlth, RB, nt):
    """sum_B n_B int d^2s E_Y[(rho Y/sigA)^2 1(x<xcut)]  (per arcmin^2 weights wpop)."""
    tot = 0.0
    for b in range(Ymed.shape[0]):
        S = 5.0 * thA + 5.0 * thB[b] + RB
        acc = 0.0
        for k in range(nt):
            t = (k + 0.5) / nt
            s = S * t * t
            rr = bl.rho_at(rho, lth0, dlth, RB, thA, thB[b], s)
            dA = 2 * np.pi * s * 2 * S * t / nt
            for g in range(xg.shape[0]):
                x = rr * Ymed[b] * np.exp(slnY * xg[g]) / sigA
                if x < xcut:
                    acc += wg[g] * x * x * dA
        tot += wpop[b] * acc
    return tot


def main():
    s8, Om, omb, slnY = map(float, sys.argv[1:5]); nprim = int(sys.argv[5]); seed = int(sys.argv[6])
    t0 = time.time()
    tab = bl.load_rho_table(); m0 = sg.SZModel(); args = tab.args(); RB = tab.R_BEAM
    P = population(s8, Om, omb)
    dens = P["n"] / FULL                                         # per arcmin^2
    ybar = (dens * P["Y"]).sum() * np.exp(slnY ** 2 / 2)
    print(f"params {s8,Om,omb,slnY}; N(M>1e13) full sky {P['n'].sum():,.0f}; "
          f"ytot = ybar*RATIO = {ybar*sg.RATIO_5R500:.4e}  ({time.time()-t0:.1f}s)", flush=True)
    xg, wg = np.polynomial.hermite_e.hermegauss(40); wg = wg / wg.sum()
    sigP = m0.sigma_Y(P["th"]); q0 = P["Y"] / sigP
    q = q0[:, None] * np.exp(slnY * xg)[None]
    Nsingle = (P["n"][:, None] * wg[None] * ndtr(q - 5.0)).sum()
    print(f"N_single(q_obs>5, full sky, naive) = {Nsingle:,.1f}")
    # naive (z,q) table
    zb = np.clip(np.searchsorted(ZE, P["z"], side="right") - 1, 0, 7)
    PhiE = np.stack([ndtr(q - e) if np.isfinite(e) else np.zeros_like(q) for e in QE])
    Nn = np.zeros((8, 10))
    for e in range(10):
        Nn[:, e] = np.bincount(zb, ((PhiE[e] - PhiE[e + 1]) * wg[None]).sum(1) * P["n"], minlength=8)
    # Var at thA = 2.30' node
    i23 = int(np.searchsorted(tab.th, 2.0)); thA0 = float(tab.th[i23]); sA0 = float(m0.sigma_Y(thA0))
    keep = dens > dens.max() * 1e-14
    xg16, wg16 = np.polynomial.hermite_e.hermegauss(16); wg16 /= wg16.sum()
    for xc in (500.0, 5.0):
        v = var_int(thA0, sA0, P["Y"][keep], P["th"][keep], dens[keep], xg16, wg16, slnY, xc, *args, 64)
        print(f"Var(delta | x<{xc:g}) at thA={thA0:.2f}' = {v:.4f}")
    hi = keep & (P["M"] >= 1e14)
    v_hi = var_int(thA0, sA0, P["Y"][hi], P["th"][hi], dens[hi], xg16, wg16, slnY, 500.0, *args, 64)
    print(f"  of which M>=1e14: {v_hi:.4f}")
    # mean shift sh(thA) from direct dc integral, on a thA grid
    thBg = np.geomspace(P["th"].min() * 0.99, P["th"].max() * 1.01, 120)
    thAg = np.geomspace(0.3, 60.0, 40)
    Ybw = dens * P["Y"] * np.exp(slnY ** 2 / 2)
    sh_g = np.array([(np.interp(np.log(P["th"]), np.log(thBg), dc_int(ta, thBg, *args, 400)) * Ybw).sum()
                     / m0.sigma_Y(ta) for ta in thAg])
    mu_formula = ybar * bl.dc_response(m0.sigma_Y(thAg), m0) / m0.sigma_Y(thAg)
    print(f"mean boost mu(thA): direct/dc_response-formula - 1: max {np.abs(sh_g/mu_formula-1).max():.2e}; "
          f"mu(2.30') = {np.interp(np.log(thA0), np.log(thAg), sh_g):.4f}", flush=True)
    # ---------------- primaries: importance-sampled (q_A > 0.5)
    rng = np.random.default_rng(seed)
    Pt = ndtr(np.log(q0 / 0.5) / slnY)
    emph = 0.03 + ndtr(q0 - 3.0)
    ut = P["n"] * Pt * emph
    cdf = np.cumsum(ut)
    t = np.minimum(np.searchsorted(cdf, rng.random(nprim) * cdf[-1]), len(cdf) - 1)
    xmin = np.log(0.5 / q0[t]) / slnY
    uu = ndtr(xmin) + rng.random(nprim) * (1 - ndtr(xmin))
    x = ndtri(np.clip(uu, 1e-300, 1 - 1e-16))
    qA = q0[t] * np.exp(slnY * x); thA = P["th"][t]; sigA = sigP[t]
    w = cdf[-1] / nprim / emph[t]                                 # full-sky expected count per sample
    zA = P["z"][t]
    print(f"primaries: sum w = {w.sum():,.0f} (expected N(q>0.5) {(P['n']*Pt).sum():,.0f})", flush=True)
    # neighbour sampling mixture
    nB = dens; thB = P["th"]
    N0, N1, N2 = nB.sum(), (nB * thB).sum(), (nB * thB ** 2).sum()
    cdf0 = np.cumsum(nB); cdf1 = np.cumsum(nB * thB); cdf2 = np.cumsum(nB * thB ** 2)
    a = 5 * thA + RB
    comp = np.stack([a ** 2 * N0, 10 * a * N1, np.full_like(a, 25 * N2)], 1)
    Lam = np.pi * comp.sum(1); pk = comp / comp.sum(1)[:, None]
    print(f"mean neighbours per primary {np.mean(Lam):.0f}", flush=True)
    D = np.zeros(nprim); xmax = np.zeros(nprim)
    t1 = time.time()
    nb_sums(seed, thA, sigA, a, Lam, pk, cdf0, cdf1, cdf2, P["Y"], thB, slnY, *args, D, xmax)
    print(f"neighbour sums {time.time()-t1:.0f}s", flush=True)
    sh = np.interp(np.log(thA), np.log(thAg), sh_g)
    delta = D - sh
    dom = xmax < np.minimum(qA, bl.X_CAP)
    pb = dom * ndtr(qA + delta - 5.0); ps = ndtr(qA - 5.0)
    nbat = 20; bi = np.arange(nprim) % nbat

    def est(fn):
        vals = np.array([fn(bi == k) for k in range(nbat)])
        return fn(np.ones(nprim, bool)), vals.std(ddof=1) / np.sqrt(nbat)
    res = {}
    res["ratio"] = est(lambda s: (w * pb)[s].sum() / (w * ps)[s].sum())
    res["B2_10"] = est(lambda s: (w * pb * (delta > 0.1 * qA))[s].sum() / (w * pb)[s].sum())
    res["B3"] = est(lambda s: (w * dom * (delta >= 0) * (ndtr(qA + delta - 5) - ps))[s].sum() / (w * pb)[s].sum())
    res["lost"] = est(lambda s: (w * dom * (delta < 0) * (ps - ndtr(qA + delta - 5)))[s].sum() / (w * ps)[s].sum())
    res["ybias"] = est(lambda s: (w * pb * delta / qA)[s].sum() / (w * pb)[s].sum())
    res["N_det_b"] = est(lambda s: (w * pb)[s].sum() * nbat if s.sum() < nprim else (w * pb).sum())
    res["N_single_mc"] = est(lambda s: (w * ps)[s].sum() * nbat if s.sum() < nprim else (w * ps).sum())
    for k, (v, e) in res.items():
        print(f"  {k:12s} {v:12.5f} +- {e:.5f}")
    zbA = np.clip(np.searchsorted(ZE, zA, side="right") - 1, 0, 7)
    Cb = np.zeros((nbat, 8, 10)); Cs = np.zeros((nbat, 8, 10))
    for e in range(10):
        hi_ = QE[e + 1]
        pbin = dom * (ndtr(qA + delta - QE[e]) - (ndtr(qA + delta - hi_) if np.isfinite(hi_) else 0))
        psin = ndtr(qA - QE[e]) - (ndtr(qA - hi_) if np.isfinite(hi_) else 0)
        for k in range(nbat):
            s = bi == k
            Cb[k, :, e] = np.bincount(zbA[s], (w * pbin)[s], minlength=8)
            Cs[k, :, e] = np.bincount(zbA[s], (w * psin)[s], minlength=8)
    C = Cb.sum(0) / Cs.sum(0)
    Cq = Cb.sum((0, 1)) / Cs.sum((0, 1))
    Cq_b = Cb.sum(1) / Cs.sum(1)
    print("C(z=0.01-0.25,q=5-6) = %.4f +- %.4f" % (C[0, 0], (Cb[:, 0, 0] / Cs[:, 0, 0]).std(ddof=1) / np.sqrt(nbat)))
    print("C(z=2-3, q=7-8)      = %.4f +- %.4f" % (C[7, 2], (Cb[:, 7, 2] / Cs[:, 7, 2]).std(ddof=1) / np.sqrt(nbat)))
    print("C(q) summed over z:", np.round(Cq, 4), "\n  err:", np.round(Cq_b.std(0, ddof=1) / np.sqrt(nbat), 4))
    print("C(z) summed over q:", np.round(Cb.sum((0, 2)) / Cs.sum((0, 2)), 4))
    print(f"done {time.time()-t0:.0f}s")
    np.savez(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                          f"v01_{s8}_{Om}_{omb}_{slnY}_s{seed}.npz"), C=C, Cb=Cb, Cs=Cs, Nn=Nn,
             Nsingle=Nsingle, ytot=ybar * sg.RATIO_5R500, **{k: np.array(v) for k, v in res.items()})


if __name__ == "__main__":
    main()
