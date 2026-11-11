"""Cost and cosmology dependence of the P(D) blending kernel.
 (1) Cost: neighbour population -> rate histograms R(x|theta_A) on 40 theta nodes -> kernel table G -> counts.
     Resolution check: coarse neighbour population (dz=0.01, 180 lnM cells, 16 GH nodes) vs the b03 rates
     (1201 z nodes, 201 lnM nodes, 16 GH nodes).
 (2) Cosmology dependence: G recomputed at sigma8 + 0.03 and at Omega_m + 0.02 (h, Omega_b, n_s fixed).
     (a) change of the blended counts at the true cosmology when G is held at the fiducial
     (b) bias on (Omega_m, sigma8) [scaling relation fixed] when the data are blended at the shifted cosmology
         and the model uses the population at the trial cosmology but the fiducial G."""
import os, sys, time
os.environ.setdefault("NUMBA_NUM_THREADS", "3"); os.environ.setdefault("OMP_NUM_THREADS", "3")
import numpy as np
import cclib as cc

C = np.load(f"{cc.OUT}/counts_fid.npz")
ZE, QE = C["ZE"], C["QE"]
nz = len(ZE) - 1
G_b03 = np.load(f"{cc.OUT}/G_fid_b03rates.npz")["G"]

cosmos = {"fid": dict(cc.FID), "s8+0.03": dict(cc.FID, sigma8=cc.FID["sigma8"] + 0.03),
          "Om+0.02": dict(cc.FID, Om0=cc.FID["Om0"] + 0.02)}
Gs, T = {}, {}
for name, cp in cosmos.items():
    t0 = time.time()
    hmf = cc.hmf_grid(cp)
    npop = cc.neighbour_pop(cp, hmf=hmf)
    t1 = time.time()
    R = cc.rates(npop)
    t2 = time.time()
    G, Pdom = cc.build_G(R, verbose=False)
    t3 = time.time()
    pop = cc.population(cp, zedges=ZE, hmf=hmf)
    Nb = cc.counts_blended(G, pop, QE, nz)
    t4 = time.time()
    Gs[name] = G
    T[name] = (t1 - t0, t2 - t1, t3 - t2, t4 - t3)
    print(f"{name}: neighbour points {npop[0].size:,}; times: population {t1-t0:.1f}s, rate histograms (40 nodes) {t2-t1:.1f}s,"
          f" kernel G {t3-t2:.1f}s, counts {t4-t3:.2f}s; ybar (M>1e13) = {np.sum(npop[1]*npop[3]):.4e} arcmin^2/arcmin^2;"
          f" mu(th=0.52',2.3',10.1') = {np.round(R['mu'][[5,15,25]], 4)}", flush=True)
    np.savez(f"{cc.OUT}/G_{name}_coarse.npz", G=G, Pdom=Pdom, mu=R["mu"])

pop_f = cc.population(cc.FID, zedges=ZE)
Nb_b03 = cc.counts_blended(G_b03, pop_f, QE, nz)
Nb_c = cc.counts_blended(Gs["fid"], pop_f, QE, nz)
Nn_f = cc.counts_gauss(pop_f, QE, nz)
P = cc.pull_sigma(Nb_c, Nb_b03, 1.0)
print(f"\nresolution: coarse-population G vs b03-rate G at the fiducial: total {100*(Nb_c.sum()/Nb_b03.sum()-1):+.3f}%,"
      f" max |rel| per bin {100*np.abs(Nb_c/Nb_b03-1).max():.2f}%, full-sky Delta chi2 {np.sum(P**2):.2f} (max|pull| {np.abs(P).max():.2f});"
      f" for comparison naive vs blended: {np.sum(cc.pull_sigma(Nn_f, Nb_b03, 1.0)**2):.1f}")

print("\n(a) blended counts at the true (shifted) cosmology: G(true) vs G(fid)  [coarse G for both]")
for name in ("s8+0.03", "Om+0.02"):
    pop = cc.population(cosmos[name], zedges=ZE)
    Nt = cc.counts_blended(Gs[name], pop, QE, nz)
    Nf = cc.counts_blended(Gs["fid"], pop, QE, nz)
    Nn = cc.counts_gauss(pop, QE, nz)
    for fs in (1.0, 0.4):
        Pf = cc.pull_sigma(Nf, Nt, fs); Pn = cc.pull_sigma(Nn, Nt, fs)
        print(f"  truth {name}, f_sky={fs}: G held at fiducial: total {100*(Nf.sum()/Nt.sum()-1):+.3f}%, Delta chi2 {np.sum(Pf**2):.2f}"
              f" (max|pull| {np.abs(Pf).max():.2f}) | naive model: Delta chi2 {np.sum(Pn**2):.1f}")
    print(f"     blending correction N_b/N_n - 1 (q>5): truth {100*(Nt.sum()/Nn.sum()-1):+.3f}% vs fiducial-G {100*(Nf.sum()/Nn.sum()-1):+.3f}%"
          f" (fiducial cosmology: {100*(Nb_c.sum()/Nn_f.sum()-1):+.3f}%)")

print("\n(b) parameter bias from holding G at the fiducial (free: Om0, sigma8; scaling relation fixed)")


def fit2(D, Gm, fsky, p0):
    p = np.array(p0, float); st = np.array([0.004, 0.004])
    for it in range(10):
        mu = cc.counts_blended(Gm, cc.population(dict(cc.FID, Om0=p[0], sigma8=p[1]), zedges=ZE), QE, nz).ravel() * fsky
        J = np.empty((mu.size, 2))
        for j in range(2):
            dp = np.zeros(2); dp[j] = st[j]
            a = cc.counts_blended(Gm, cc.population(dict(cc.FID, Om0=p[0] + dp[0], sigma8=p[1] + dp[1]), zedges=ZE), QE, nz).ravel()
            b = cc.counts_blended(Gm, cc.population(dict(cc.FID, Om0=p[0] - dp[0], sigma8=p[1] - dp[1]), zedges=ZE), QE, nz).ravel()
            J[:, j] = (a - b) * fsky / (2 * st[j])
        F = (J / mu[:, None]).T @ J
        step = np.linalg.solve(F, J.T @ (D.ravel() * fsky / mu - 1))
        p = p + step
        if np.max(np.abs(step) / np.sqrt(np.diag(np.linalg.inv(F)))) < 1e-3:
            break
    return p, np.sqrt(np.diag(np.linalg.inv(F)))


for name in ("s8+0.03", "Om+0.02"):
    cp = cosmos[name]
    D = cc.counts_blended(Gs[name], cc.population(cp, zedges=ZE), QE, nz)
    for fs in (1.0, 0.4):
        p, s = fit2(D, Gs["fid"], fs, [cp["Om0"], cp["sigma8"]])
        print(f"  truth {name}, f_sky={fs}: best (Om0, sigma8) = {np.round(p, 4)}, truth ({cp['Om0']:.4f}, {cp['sigma8']:.4f}),"
              f" sigma = {np.round(s, 4)}, bias/sigma = {np.round((p - [cp['Om0'], cp['sigma8']]) / s, 3)}")
print("\ntiming summary [s] (population, rates, G, counts):", {k: tuple(round(x, 1) for x in v) for k, v in T.items()})
