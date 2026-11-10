"""V3: fully independent full-sky Monte Carlo (own sampler + brute-force counting).
Sampler: own midpoint-rule (z, lnM) grid from colossus Tinker08 500c (h converted), total N ~ Poisson(Lambda),
cells by inverse-CDF on the global cumulative (multinomial), uniform within a cell; positions = normalised 3D
Gaussians; own Y500/theta500; sigma_Y(theta500) from szsim (signal model, out of scope); lognormal scatter 0.173;
q_obs = q + N(0,1).  Seeds 900000+i (disjoint from all analysis seeds).  usage: v3 i0 i1"""
import sys, os, time, resource
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from bf_common import SortedPts, counts, triangles
from colossus.cosmology import cosmology
from colossus.lss import mass_function
from szsim import signal as sg

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "indep_mc"); os.makedirs(OUT, exist_ok=True)
cosmo = cosmology.setCosmology("planck18"); h = cosmo.h; AM = np.pi / 180 / 60; C = 299792.458
TH = np.array([0.5, 1, 2, 5, 10.0])
NZ, NM = 1500, 1500
ze = np.linspace(0.01, 3.0, NZ + 1); le = np.linspace(np.log(1e13), np.log(1e16), NM + 1)
zm = 0.5 * (ze[1:] + ze[:-1]); lm = 0.5 * (le[1:] + le[:-1])
chi = cosmo.comovingDistance(0.0, zm) / h
dV = C / (100 * h) / cosmo.Ez(zm) * chi ** 2 * 4 * np.pi
G = np.array([mass_function.massFunction(np.exp(lm) * h, z, mdef="500c", model="tinker08", q_out="dndlnM") * h ** 3
              for z in zm]) * dV[:, None]
lam = (G * np.diff(ze)[:, None] * np.diff(le)[None, :]).ravel()
LAM = lam.sum(); cum = np.cumsum(lam) / LAM
# background quantities on a fine z grid for marks
zf = np.linspace(0.0, 3.01, 30101)
Ezf = cosmo.Ez(zf); DAf = cosmo.comovingDistance(0.0, zf) / h / (1 + zf); rcf = cosmo.rho_c(zf) * 1e9 * h ** 2
m = sg.SZModel()


def draw(seed, chunk=3_000_000):
    rng = np.random.default_rng(seed)
    N = rng.poisson(LAM)
    keep = {k: [] for k in ["d4", "d5", "d6", "m14", "m3", "d5z"]}
    done = 0
    while done < N:
        n = min(chunk, N - done)
        c = np.searchsorted(cum, rng.random(n), side="right"); c = np.minimum(c, len(cum) - 1)
        iz, im = c // NM, c % NM
        z = ze[iz] + rng.random(n) * (ze[iz + 1] - ze[iz])
        M = np.exp(le[im] + rng.random(n) * (le[im + 1] - le[im]))
        g = rng.standard_normal((n, 3)); v = g / np.linalg.norm(g, axis=1)[:, None]
        E = np.interp(z, zf, Ezf); DA = np.interp(z, zf, DAf); rc = np.interp(z, zf, rcf)
        Y = 1e-4 * 10 ** -0.19 * E ** (2 / 3) * (0.8 * M / 6e14) ** 1.79 / DA ** 2 / AM ** 2
        th = (3 * M / (4 * np.pi * 500 * rc)) ** (1 / 3) / DA / AM
        Y *= np.exp(0.173 * rng.standard_normal(n))
        qo = Y / m.sigma_Y(th) + rng.standard_normal(n)
        ids = np.arange(done, done + n)
        for k, s in [("d4", qo > 4), ("d5", qo > 5), ("d6", qo > 6), ("m14", M >= 1e14), ("m3", M >= 3e13),
                     ("d5z", (qo > 5) & (z >= 1.0))]:
            keep[k].append((v[s], ids[s]))
        done += n
    return N, {k: (np.concatenate([a for a, _ in L]), np.concatenate([b for _, b in L])) for k, L in keep.items()}


def summ(Nc):
    return np.array([(Nc >= k).mean(0) for k in (1, 2, 3)])     # (3, T) P(N>=k)


for i in range(int(sys.argv[1]), int(sys.argv[2])):
    t0 = time.time(); seed = 900000 + i
    Ntot, P = draw(seed)
    S = {k: SortedPts(*P[k]) for k in P}
    rng = np.random.default_rng(seed + 12345)
    g = rng.standard_normal((100_000, 3)); rv = g / np.linalg.norm(g, axis=1)[:, None]
    noid = -np.ones(len(rv), np.int64)
    res = {"seed": seed, "Ntot": Ntot, "npop": {k: len(P[k][1]) for k in P}}
    for c, p in [("rand", "d5"), ("rand", "m14"), ("rand", "m3"), ("d5", "d5"), ("d4", "d4"), ("d6", "d6"),
                 ("d5", "m14"), ("d5", "m3"), ("d5", "d5z")]:
        cv, cid = (rv, noid) if c == "rand" else P[c]
        Nc = counts(cv, cid, S[p], TH)
        res[f"{c}|{p}"] = summ(Nc); res[f"n_{c}|{p}"] = len(cv)
        if c == p:
            res[f"pairs_{p}"] = Nc.sum(0) / 2
    res["tri_d5"] = np.array([triangles(S["d5"], t) for t in (2, 5, 10.0)])
    res["t"] = time.time() - t0; res["rss"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e9
    np.save(os.path.join(OUT, f"sky_{seed}.npy"), res, allow_pickle=True)
    print(f"seed {seed}: Ntot={Ntot} det5={res['npop']['d5']} t={res['t']:.1f}s rss={res['rss']:.2f}GB", flush=True)
