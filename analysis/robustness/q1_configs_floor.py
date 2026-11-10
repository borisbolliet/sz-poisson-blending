"""Q1 (B1) robustness with many independent skies: for every sensitivity configuration draw NSKY full skies
with a mass floor M500 > 1e14 (detections only; no neighbour sums needed), and compare
P(>= k other detections within theta | detection) with the Poisson law 1 - sum_{j<k} e^-lam lam^j/j!,
lam = Sigma_det A_cap(theta), Sigma_det = N_det(config, M>1e14) / 4pi (analytic selection integral).
Independent seeds per config (no common random numbers), so the tests are independent across configs.
Usage: python q1_configs_floor.py [NSKY]
"""
import os, sys, json, time
os.environ.setdefault("OMP_NUM_THREADS", "2"); os.environ.setdefault("NUMBA_NUM_THREADS", "2")
import numpy as np
import healpy as hp
from scipy.spatial import cKDTree
from scipy.stats import poisson, chi2

HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import blendlib as bl
import configs as C
from szsim import sampler, signal as sg, cosmo_hmf as ch

NSKY = int(sys.argv[1]) if len(sys.argv) > 1 else 40
QC = 5.0
THETAS = [1.0, 2.0, 5.0, 10.0, 30.0]
ARC = ch.ARCMIN
CFG = {  # name: (beam, noise, sig, hmf, zmax)
    "base": (1.4, 4e-6, 0.173, "tinker08", 3.0),
    "noise_x0.5": (1.4, 2e-6, 0.173, "tinker08", 3.0),
    "noise_x2": (1.4, 8e-6, 0.173, "tinker08", 3.0),
    "beam_5": (5.0, 4e-6, 0.173, "tinker08", 3.0),
    "planck_like": (7.3, C.PLANCK_NOISE_Y, 0.173, "tinker08", 3.0),
    "scatter_0": (1.4, 4e-6, 0.0, "tinker08", 3.0),
    "scatter_0.3": (1.4, 4e-6, 0.3, "tinker08", 3.0),
    "hmf_bocquet16": (1.4, 4e-6, 0.173, "bocquet16", 3.0),
    "hmf_despali16": (1.4, 4e-6, 0.173, "despali16", 3.0),
    "zmax_1": (1.4, 4e-6, 0.173, "tinker08", 1.0),
    "zmax_2": (1.4, 4e-6, 0.173, "tinker08", 2.0),
}


def jk_ratio(num, den):
    Nn, Nd = num.sum(), den.sum(); ok = den > 0
    rj = (Nn - num[ok]) / (Nd - den[ok]); n = ok.sum()
    return Nn / Nd, np.sqrt((n - 1) / n * np.sum((rj - rj.mean()) ** 2))


t0 = time.time()
out = {}
zs_all = []
NREG = hp.nside2npix(2)
for ic, (name, (beam, noise, sig, hmf, zmax)) in enumerate(CFG.items()):
    model = sg.SZModel(beam_fwhm_arcmin=beam, noise_y_arcmin=noise, sigma_lnY=sig)
    table = ch.default_table() if hmf == "tinker08" else ch.HMFTable(model=hmf)
    Nan = float(model.expected_detections(QC, table=table, Mmin=1e14, zmax=zmax))
    Sig = Nan / (4 * np.pi)
    cnt_det = []; cnt = {(th, k): [] for th in THETAS for k in (1, 2, 3)}; Nd = []
    for s in range(NSKY):
        seed = 5000 + 100 * ic + s
        cat = sampler.draw_catalog(seed, Mmin=1e14, zmax=zmax, model=model, table=table, fields=("ra", "dec", "q_obs"))
        sel = cat["q_obs"] > QC
        ra, dec = cat["ra"][sel].astype(np.float64), cat["dec"][sel].astype(np.float64)
        Nd.append(sel.sum())
        lab = hp.ang2pix(2, ra, dec, lonlat=True)
        cnt_det.append(np.bincount(lab, minlength=NREG))
        xyz = np.column_stack(bl.unitvec(ra, dec))
        tree = cKDTree(xyz)
        for th in THETAS:
            nn = tree.query_ball_point(xyz, 2 * np.sin(th * ARC / 2), return_length=True) - 1
            for k in (1, 2, 3):
                cnt[(th, k)].append(np.bincount(lab[nn >= k], minlength=NREG))
    den = np.concatenate(cnt_det)
    Nd = np.array(Nd)
    zN = (Nd.mean() - Nan) / np.sqrt(Nan / NSKY)
    print(f"{name:14s} N_det(M>1e14,q>5): MC {Nd.mean():9.1f} ± {np.sqrt(Nan/NSKY):.1f} vs analytic {Nan:9.1f} ({zN:+.2f}σ); "
          f"var/mean {Nd.var(ddof=1)/Nd.mean():.2f} ± {np.sqrt(2/(NSKY-1)):.2f}; Σ_det = {Sig*ARC**2*3600:.4f}/deg²", flush=True)
    res = {"Nan": Nan, "Nmc": float(Nd.mean()), "zN": zN, "rows": []}
    for th in THETAS:
        lam = Sig * 2 * np.pi * (1 - np.cos(th * ARC))
        for k in (1, 2, 3):
            num = np.concatenate(cnt[(th, k)])
            if num.sum() < 10:
                continue
            r, e = jk_ratio(num, den); pa = poisson.sf(k - 1, lam)
            z = (r - pa) / e; zs_all.append(z)
            res["rows"].append(dict(theta=th, k=k, mc=r, err=e, an=pa, z=z, lam=lam))
            print(f"     θ={th:4.0f}' k≥{k}: MC {r:.4e} ± {e:.1e}  Poisson {pa:.4e}  ({z:+.2f}σ)  [lin. approx Σ_det A = {lam:.4e}]")
    out[name] = res
zs_all = np.array(zs_all)
print(f"\nall configs: {len(zs_all)} B1 tests, rms z = {np.sqrt(np.mean(zs_all**2)):.2f}, max|z| = {np.max(np.abs(zs_all)):.2f}, "
      f"chi2 = {np.sum(zs_all**2):.1f} for {len(zs_all)} (p = {chi2.sf(np.sum(zs_all**2), len(zs_all)):.3f}; tests at different θ, k are correlated)")
json.dump(out, open(os.path.join(ch.DATA_DIR, "robustness", "q1_configs_floor.json"), "w"), indent=1)
print(f"done {time.time()-t0:.0f}s")
