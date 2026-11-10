"""Validate the HMF table, the full-sky Poisson sampler, and the expected detection counts.
Also writes the stored catalogue realisations to data/catalogs/.

Usage: python validate_sampler.py [R13] [R14]
  R13 = number of full M>1e13 realisations (default 24, seeds 1..R13; seeds 1-3 saved)
  R14 = number of M>1e14 realisations      (default 200, seeds 101..; seeds 101-110 saved)
"""
import os, sys, time, json
os.environ.setdefault("OMP_NUM_THREADS", "8")
import numpy as np
from scipy import stats
import healpy as hp
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from szsim import cosmo_hmf as ch, signal as sg, sampler

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
FIG = os.path.join(ROOT, "figures/build"); CAT = os.path.join(ROOT, "data/catalogs")
os.makedirs(FIG, exist_ok=True); os.makedirs(CAT, exist_ok=True)
R13 = int(sys.argv[1]) if len(sys.argv) > 1 else 24
R14 = int(sys.argv[2]) if len(sys.argv) > 2 else 200
T = ch.default_table(); m = sg.SZModel()
MTH = [1e13, 3e13, 1e14, 3e14]
QC = [4, 5, 6, 7]
t0 = time.time()

# ------------------------------------------------------------------ 1. HMF table
print("=" * 70, "\n1. Full-sky counts (Tinker08, M500c, planck18, 0.01<z<3)")
# independent check: Gauss-Legendre directly on colossus (exact chi, no table)
xz, wz = np.polynomial.legendre.leggauss(300)
def N_GL(Mlo, Mhi=1e16, zlo=0.01, zhi=3.0, n=300):
    xm, wm = np.polynomial.legendre.leggauss(n)
    zs = 0.5 * (zhi - zlo) * (xz + 1) + zlo; wzs = 0.5 * (zhi - zlo) * wz
    lms = 0.5 * np.log(Mhi / Mlo) * (xm + 1) + np.log(Mlo); wms = 0.5 * np.log(Mhi / Mlo) * wm
    tot = 0.0
    for zz, w in zip(zs, wzs):
        dV = (ch.C_KMS / ch.get_cosmology().H0) / ch.Ez(zz) * ch.chi_Mpc_exact(zz) ** 2 * 4 * np.pi
        tot += w * dV * (ch.dndlnM_Mpc3(np.exp(lms), zz) * wms).sum()
    return tot
for M in MTH:
    Nt = T.N_above(M); Ng = N_GL(M); Ns = sampler.expected_count(M)
    print(f"  N(M500>{M:.0e} Msun) = {Nt:14.1f}  [table trapezoid]   {Ng:14.1f} [Gauss-Legendre on colossus]"
          f"  rel diff {Nt/Ng-1:+.2e};  sampler lambda-sum {Ns:14.1f}")
zq = [0.1, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 2.5, 3.0]
print("  dN/dz (full sky) at selected z:")
print("   " + " ".join(f"{'z=' + str(z):>10}" for z in zq))
for M in MTH:
    dndz = T.dN_dz(Mmin=M)
    print(f"  M>{M:.0e}: " + " ".join(f"{np.interp(z, T.z, dndz):10.4g}" for z in zq))
print("  shape summaries:")
for M in MTH:
    dndz = T.dN_dz(Mmin=M); cdf = np.concatenate([[0], np.cumsum(0.5 * (dndz[1:] + dndz[:-1]) * T.dz)]); cdf /= cdf[-1]
    zmed = np.interp(0.5, cdf, T.z); zpk = T.z[np.argmax(dndz)]
    f1 = 1 - np.interp(1.0, T.z, cdf); f2 = 1 - np.interp(2.0, T.z, cdf)
    print(f"  M>{M:.0e}: dN/dz peak at z={zpk:.3f}, median z={zmed:.3f}, frac(z>1)={f1:.3f}, frac(z>2)={f2:.4f}")

fig, ax = plt.subplots(1, 2, figsize=(10, 4))
for M in MTH:
    ax[0].semilogy(T.z, T.dN_dz(Mmin=M), label=f"M500>{M:.0e}")
ax[0].set_xlabel("z"); ax[0].set_ylabel("full-sky dN/dz"); ax[0].legend()
Mg = np.geomspace(1e13, 5e15, 60)
ax[1].loglog(Mg, [T.N_above(x) for x in Mg]); ax[1].set_xlabel("M500 [Msun]"); ax[1].set_ylabel("N(>M500), full sky, z<3")
fig.tight_layout(); fig.savefig(os.path.join(FIG, "dNdz_Nabove.png"), dpi=120); plt.close(fig)

# ------------------------------------------------------------------ helpers: exact marginal CDFs of the sampled model
def marginal_cdfs(Mmin):
    zz, lm, d = sampler._cell_grid(T, Mmin, 1e16, 0.01, 3.0)
    dl = np.diff(lm); dz = np.diff(zz)
    wl = np.zeros(len(lm)); wl[:-1] += dl / 2; wl[1:] += dl / 2
    wzz = np.zeros(len(zz)); wzz[:-1] += dz / 2; wzz[1:] += dz / 2
    mz = d @ wl  # dN/dz at z nodes (exact for bilinear model: linear in z between nodes)
    ml = wzz @ d  # dN/dlnM at lnM nodes
    def mk(x, f):
        cell = 0.5 * (f[1:] + f[:-1]) * np.diff(x); F = np.concatenate([[0], np.cumsum(cell)]); tot = F[-1]
        def cdf(v):
            v = np.asarray(v, dtype=float)
            i = np.clip(np.searchsorted(x, v, side="right") - 1, 0, len(x) - 2)
            h = x[i + 1] - x[i]; t = np.clip((v - x[i]) / h, 0, 1)
            return (F[i] + h * (f[i] * t + 0.5 * (f[i + 1] - f[i]) * t * t)) / tot
        return cdf
    return mk(zz, mz), mk(lm, ml)

def ks(x, cdf, log=False, chunk=2_000_000):
    """One-sample KS test, memory-light (single float64 copy, chunked CDF evaluation)."""
    x = np.asarray(x).astype(np.float64)
    if log:
        np.log(x, out=x)
    x.sort(); n = len(x); D = 0.0
    for a in range(0, n, chunk):
        F = cdf(x[a:a + chunk]); i = np.arange(a, a + len(F))
        D = max(D, np.max((i + 1) / n - F), np.max(F - i / n))
    del x
    return D, stats.kstwo.sf(D, n)


def healpix_counts(nside, ra, dec, chunk=2_000_000):
    cnt = np.zeros(hp.nside2npix(nside))
    for a in range(0, len(ra), chunk):
        cnt += np.bincount(hp.ang2pix(nside, ra[a:a + chunk].astype(float), dec[a:a + chunk].astype(float), lonlat=True),
                           minlength=len(cnt))
    return cnt

cz13, cl13 = marginal_cdfs(1e13)
cz14, cl14 = marginal_cdfs(1e14)
zb = np.array([0.01, 0.25, 0.5, 1.0, 1.5, 2.0, 3.0])
lam_z13 = np.array([T.N_total(Mmin=1e13, zmin=a, zmax=b) for a, b in zip(zb[:-1], zb[1:])])
Ndet_exp13 = np.array(m.expected_detections(QC, Mmin=1e13))
Ndet_exp14 = np.array(m.expected_detections(QC, Mmin=1e14))
print("  expected detections computed", f"[{time.time()-t0:.0f}s]")

# ------------------------------------------------------------------ 2. M>1e13 realisations
print("=" * 70, f"\n2. {R13} full-sky realisations of the M500>1e13 population")
NSIDE = 16
res13 = dict(N=[], Nth=[], Nz=[], Ndet=[], Ndet_lt14=[], ksz=[], ksl=[], pixchi=[], cl=[], ksY=[], ksq=[])
for r in range(R13):
    seed = 1 + r
    save = seed <= 3
    c = sampler.draw_catalog(seed, Mmin=1e13, fields=sampler.FIELDS if save else ("ra", "dec", "z", "M500", "q_obs"))
    if save:
        save_path = os.path.join(CAT, f"cat_M1e13_seed{seed}.npz"); sampler.save_catalog(c, save_path)
        # Y scatter and q_obs-q noise checks on a 1e6 subsample
        idx = np.random.default_rng(seed).choice(len(c["z"]), 1_000_000, replace=False)
        zz = c["z"][idx].astype(float); MM = c["M500"][idx].astype(float)
        dlnY = np.log(c["Y500"][idx].astype(float) / m.Y500_med(MM, zz)) / m.sigma_lnY
        res13["ksY"].append(stats.kstest(dlnY, "norm").pvalue)
        res13["ksq"].append(stats.kstest((c["q_obs"][idx].astype(float) - c["q"][idx]), "norm").pvalue)
        dth = np.max(np.abs(c["theta500"][idx] / m.theta500(MM, zz) - 1))
        print(f"  seed {seed}: saved {save_path} ({os.path.getsize(save_path)/1e6:.0f} MB); "
              f"KS lnY-scatter p={res13['ksY'][-1]:.3f}, KS (q_obs-q) p={res13['ksq'][-1]:.3f}, max|theta500 rel err|={dth:.1e}")
    M = c["M500"]; z = c["z"]
    res13["N"].append(len(M))
    res13["Nth"].append([np.count_nonzero(M >= x) for x in MTH])
    res13["Nz"].append(np.histogram(z, zb)[0])
    res13["Ndet"].append([np.count_nonzero(c["q_obs"] > q) for q in QC])
    res13["Ndet_lt14"].append([np.count_nonzero((c["q_obs"] > q) & (M < 1e14)) for q in QC])
    res13["ksz"].append(ks(z, cz13)); res13["ksl"].append(ks(M, cl13, log=True))
    cnt = healpix_counts(NSIDE, c["ra"], c["dec"])
    nbar = cnt.mean(); res13["pixchi"].append(((cnt - nbar) ** 2 / nbar).sum())
    res13["cl"].append(hp.anafast(cnt / nbar - 1, lmax=3 * NSIDE - 1) / (4 * np.pi / len(M)))
    del c, M, z
    print(f"  r={r:2d} seed={seed:3d} N={res13['N'][-1]}  KS z p={res13['ksz'][-1][1]:.3f}  KS lnM p={res13['ksl'][-1][1]:.3f}"
          f"  [{time.time()-t0:.0f}s]", flush=True)

def report_counts(label, obs, lam):
    obs = np.asarray(obs, dtype=float); R = len(obs)
    mean = obs.mean(); s2 = obs.var(ddof=1)
    zsig = (mean - lam) / np.sqrt(lam / R)
    disp = (R - 1) * s2 / lam  # ~ chi2_{R-1} if Poisson with mean lam
    p = 2 * min(stats.chi2.cdf(disp, R - 1), stats.chi2.sf(disp, R - 1))
    print(f"  {label:28s} analytic {lam:13.1f}  MC mean {mean:13.1f} +- {np.sqrt(lam/R):8.1f}  ({zsig:+.2f} sigma)"
          f"  var/mean = {s2/lam:.3f} +- {np.sqrt(2/(R-1)):.3f} (chi2 p={p:.2f})")
    return zsig

print("\n  (a,b) counts: mean vs analytic (sigma = sqrt(lambda/R)), Poisson dispersion var/lambda")
zs = []
zs.append(report_counts("N(M>1e13)", res13["N"], sampler.expected_count(1e13)))
for k, x in enumerate(MTH[1:], 1):
    zs.append(report_counts(f"N(M>{x:.0e})", np.array(res13["Nth"])[:, k], sampler.expected_count(x)))
for k in range(len(zb) - 1):
    zs.append(report_counts(f"N(M>1e13, {zb[k]}<z<{zb[k+1]})", np.array(res13["Nz"])[:, k], lam_z13[k]))
print("\n  (4) detections q_obs > q_cut (analytic = selection-function integral, M>1e13 floor)")
for k, q in enumerate(QC):
    zs.append(report_counts(f"N_det(q_obs>{q})", np.array(res13["Ndet"])[:, k], Ndet_exp13[k]))
for k, q in enumerate(QC):
    zs.append(report_counts(f"N_det(q_obs>{q}, M<1e14)", np.array(res13["Ndet_lt14"])[:, k], Ndet_exp13[k] - Ndet_exp14[k]))
zs = np.array(zs)
print(f"  summary of {len(zs)} count z-scores: mean {zs.mean():+.2f}, rms {zs.std():.2f}, max|z| {np.abs(zs).max():.2f}"
      " (note: count bins are correlated across rows)")

print("\n  (c) KS tests of the marginals (exact CDF of the sampled bilinear model)")
pz = np.array([x[1] for x in res13["ksz"]]); pl = np.array([x[1] for x in res13["ksl"]])
Dz = np.array([x[0] for x in res13["ksz"]]); Dl = np.array([x[0] for x in res13["ksl"]])
print(f"  z  : D median {np.median(Dz):.2e} (1/sqrt(N)={1/np.sqrt(np.mean(res13['N'])):.2e}); p-values min {pz.min():.3f};"
      f" KS-uniformity of the {R13} p-values: p={stats.kstest(pz, 'uniform').pvalue:.3f}")
print(f"  lnM: D median {np.median(Dl):.2e}; p-values min {pl.min():.3f}; KS-uniformity of p-values: p={stats.kstest(pl, 'uniform').pvalue:.3f}")
print("\n  (d) isotropy, HEALPix nside=16 (3072 pix):")
pc = np.array(res13["pixchi"]); dof = hp.nside2npix(NSIDE) - 1
print(f"  pixel chi2/dof per realisation: mean {pc.mean()/dof:.4f} +- {np.sqrt(2/dof/R13):.4f} (expect 1);"
      f" combined chi2 p = {stats.chi2.sf(pc.sum(), dof*R13):.3f}")
cl = np.array(res13["cl"])  # C_l / (4pi/N)
ell = np.arange(cl.shape[1])
for L in [1, 2, 3]:
    chi = (2 * L + 1) * cl[:, L].sum(); d_ = (2 * L + 1) * R13
    print(f"  l={L}: <C_l>/C_shot = {cl[:, L].mean():.3f} +- {np.sqrt(2/(2*L+1)/R13):.3f}  (chi2 p={2*min(stats.chi2.cdf(chi,d_),stats.chi2.sf(chi,d_)):.2f})")
wl = (2 * ell[1:] + 1)
avg = (cl[:, 1:] * wl).sum(1) / wl.sum()
print(f"  l=1..{ell[-1]} (2l+1)-weighted <C_l>/C_shot = {avg.mean():.4f} +- {np.sqrt(2/wl.sum()/R13):.4f}")

# ------------------------------------------------------------------ 3. M>1e14 realisations (floor option)
print("=" * 70, f"\n3. {R14} full-sky realisations with mass floor M500>1e14")
res14 = dict(N=[], Nth=[], Ndet=[], ksz=[], ksl=[], pixchi=[])
NS14 = 4
for r in range(R14):
    seed = 101 + r
    save = r < 10
    c = sampler.draw_catalog(seed, Mmin=1e14, fields=sampler.FIELDS if save else ("ra", "dec", "z", "M500", "q_obs"))
    if save:
        sampler.save_catalog(c, os.path.join(CAT, f"cat_M1e14_seed{seed}.npz"))
    M = c["M500"]
    res14["N"].append(len(M)); res14["Nth"].append(np.count_nonzero(M >= 3e14))
    res14["Ndet"].append([np.count_nonzero(c["q_obs"] > q) for q in QC])
    res14["ksz"].append(ks(c["z"], cz14)[1]); res14["ksl"].append(ks(M, cl14, log=True)[1])
    cnt = healpix_counts(NS14, c["ra"], c["dec"])
    res14["pixchi"].append(((cnt - cnt.mean()) ** 2 / cnt.mean()).sum())
print(f"  saved 10 catalogues data/catalogs/cat_M1e14_seed101..110.npz  [{time.time()-t0:.0f}s]")
zs14 = [report_counts("N(M>1e14)", res14["N"], sampler.expected_count(1e14)),
        report_counts("N(M>3e14)", res14["Nth"], sampler.expected_count(3e14))]
for k, q in enumerate(QC):
    zs14.append(report_counts(f"N_det(q_obs>{q}) [M>1e14 floor]", np.array(res14["Ndet"])[:, k], Ndet_exp14[k]))
print(f"  KS z  p-values: KS-uniformity p={stats.kstest(res14['ksz'], 'uniform').pvalue:.3f};"
      f"  KS lnM p-values: KS-uniformity p={stats.kstest(res14['ksl'], 'uniform').pvalue:.3f}")
pc = np.array(res14["pixchi"]); dof = hp.nside2npix(NS14) - 1
print(f"  nside=4 pixel chi2/dof mean {pc.mean()/dof:.4f} +- {np.sqrt(2/dof/R14):.4f}; combined p = {stats.chi2.sf(pc.sum(), dof*R14):.3f}")

# ------------------------------------------------------------------ 4. detection summary
print("=" * 70, "\n4. Expected full-sky detections (q_obs > q_cut; lognormal Y scatter + unit Gaussian noise)")
for k, q in enumerate(QC):
    print(f"  q_cut={q}: N_det = {Ndet_exp13[k]:10.1f} (M>1e13 floor), {Ndet_exp14[k]:10.1f} (M>1e14 floor);"
          f" fraction from M<1e14: {1-Ndet_exp14[k]/Ndet_exp13[k]:.4f}")

summary = dict(N_above={f"{x:.0e}": T.N_above(x) for x in MTH},
               Ndet_expected_floor1e13=dict(zip(map(str, QC), Ndet_exp13.tolist())),
               Ndet_expected_floor1e14=dict(zip(map(str, QC), Ndet_exp14.tolist())),
               Ndet_MC_mean_floor1e13=dict(zip(map(str, QC), np.mean(res13["Ndet"], 0).tolist())),
               N13_MC=res13["N"], R13=R13, R14=R14)
json.dump(summary, open(os.path.join(ROOT, "data/build_summary.json"), "w"), indent=1)

# figure: count residuals
fig, ax = plt.subplots(1, 2, figsize=(10, 4))
ax[0].hist(np.array(res13["N"]) - sampler.expected_count(1e13), bins=8, alpha=0.7)
ax[0].axvline(0, c="k"); ax[0].set_xlabel(r"$N_{\rm MC}-\lambda$ (M>1e13)"); ax[0].set_title(f"{R13} realisations; Poisson sd={np.sqrt(sampler.expected_count(1e13)):.0f}")
ax[1].errorbar(QC, np.mean(res13["Ndet"], 0) / Ndet_exp13 - 1, yerr=np.sqrt(Ndet_exp13 / R13) / Ndet_exp13, fmt="o")
ax[1].axhline(0, c="k"); ax[1].set_xlabel("q_cut"); ax[1].set_ylabel("MC/analytic - 1 (detections)")
fig.tight_layout(); fig.savefig(os.path.join(FIG, "count_validation.png"), dpi=120); plt.close(fig)
print(f"done [{time.time()-t0:.0f}s]")
