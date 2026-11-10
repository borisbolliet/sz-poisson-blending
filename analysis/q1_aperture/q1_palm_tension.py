"""Repair phase: is the ~2 sigma deficit of P(>=1 other detected within theta | detected centre) at 10'-15'
(found by verifiers in the 40 original full-sky skies, seeds 1001-1040) real or a fluctuation?

Tests on INDEPENDENT seeds:
  (a) 80 new full-sky skies (M>1e13, seeds 1041-1120), same pipeline (q1_mc_fullsky.py)
  (b) many floor-1e14 skies (q1_palm_check.py, data/q1_aperture/palm_new): detected = q_obs>q among M>=1e14
For every sample, two predictions:
  unconditional (Palm/Mecke):  E[# centres with >=1 other] / E[N] = 1 - exp(-mu),  mu = Nbar f,  f = (1-cos theta)/2
  conditional on the sky's realised N (binomial process): P = 1 - (1-f)^(N-1)
Estimator: ratio of sums over skies; error from the sky-to-sky scatter of the per-sky deviation (delta method).
Also: random-centred P(>=1), pair counts (unconditional E = Nbar mu/2, Var = E(1+2mu); conditional E = f N(N-1)/2),
and the decomposition Var[P] = E[Var(P|N)] + Var(E[P|N]) ~ E[P](1-f) + Nbar mu^2  (claim C9).
"""
import os, glob, json
import numpy as np
from scipy import stats
from q1_common import *
from q1_mc_fullsky import FINE

NB = json.load(open(os.path.join(DATA, "nbar.json")))


def fsky(theta):
    return np.sin(np.asarray(theta) * ARCMIN / 2) ** 2          # (1-cos)/2, stable


def ratio_test(num, den, pred_per_unit_den):
    """num_s ~ den_s * P_s;  returns (sum num/sum den, error, z) vs prediction pred (per-sky array or scalar),
    using residuals r_s = num_s - pred_s*den_s; z = sum r / sqrt(R var(r))."""
    num = np.asarray(num, float); den = np.asarray(den, float)
    r = num - pred_per_unit_den * den
    R = len(r)
    est = num.sum() / den.sum()
    se_rel = r.std(ddof=1) * np.sqrt(R) / den.sum()
    z = r.sum() / (r.std(ddof=1) * np.sqrt(R))
    return est, se_rel, z


def report(lab, N, ge1, theta, Nbar, f, extra=""):
    P_unc = 1 - np.exp(-Nbar * f)
    P_cond = 1 - (1 - f) ** (N - 1)
    est, se, zu = ratio_test(ge1, N, P_unc)
    _, _, zc = ratio_test(ge1, N, P_cond)
    print(f"  {lab:24s} theta={theta:5.1f}'  MC={est:.5e} +- {se:.1e}  P_unc={P_unc:.5e} (rel {est/P_unc-1:+.4f}, z={zu:+.2f})"
          f"  z_cond(N)={zc:+.2f} {extra}")
    return zu, zc


# ------------------------------------------------------------------ (a) full-sky skies, original vs new seeds
files = sorted(glob.glob(os.path.join(DATA, "fullsky", "sky_*.npy")))
SK = {int(os.path.basename(f)[4:-4]): f for f in files}
sets = {"original 1001-1040": [s for s in SK if 1001 <= s <= 1040],
        "new 1041-1120": [s for s in SK if 1041 <= s <= 1120]}
sets["all"] = sets["original 1001-1040"] + sets["new 1041-1120"]
cache = {s: np.load(SK[s], allow_pickle=True).item() for s in SK}
TH = [5, 7.5, 10, 15, 20, 30]
print("== (a) full-sky skies (M>1e13 floor): detected-centred P(>=1 other detected within theta) ==")
za = {}
for name, seeds in sets.items():
    print(f"-- {name}: R = {len(seeds)}")
    for q in QCUTS:
        key = f"det{q}|det{q}"
        N = np.array([cache[s]["npop"][f"det{q}"] for s in seeds], float)
        for t in TH:
            j = int(np.argmin(abs(FINE - t)))
            ge1 = np.array([cache[s]["cdf"][key][0, j] for s in seeds]) * N
            za[(name, q, t)] = report(f"det{q}|det{q}", N, ge1, t, NB[f"det{q}"], fsky(t))
    # random-centred, det5, 10'
    N = np.array([cache[s]["npop"]["det5"] for s in seeds], float)
    j = int(np.argmin(abs(FINE - 10)))
    Fr = np.array([cache[s]["cdf"]["rand|det5"][0, j] for s in seeds])
    nr = np.array([cache[s]["ncen"]["rand|det5"] for s in seeds], float)
    est, se, zu = ratio_test(Fr * nr, nr, 1 - np.exp(-NB["det5"] * fsky(10)))
    _, _, zc = ratio_test(Fr * nr, nr, 1 - (1 - fsky(10)) ** N)
    print(f"  rand|det5 theta=10': MC={est:.5e}+-{se:.1e} rel {est/(1-np.exp(-NB['det5']*fsky(10)))-1:+.4f} z_unc={zu:+.2f} z_cond={zc:+.2f}")
    # pair counts det5 at 10'
    j10 = int(np.argmin(abs(THETAS - 10)))
    Pc = np.array([cache[s]["pairs_det5"][j10] for s in seeds])
    mu = NB["det5"] * fsky(10); E = NB["det5"] * mu / 2; V = E * (1 + 2 * mu)
    Pcond = fsky(10) * N * (N - 1) / 2
    zc = (Pc - Pcond).sum() / np.sqrt(len(seeds) * E * (1 - fsky(10)))
    print(f"  det5 pairs<10': MC mean={Pc.mean():.1f}  E={E:.1f}  z_unc={(Pc.mean()-E)/np.sqrt(V/len(seeds)):+.2f}"
          f"  z_cond(N)={zc:+.2f}   mean N(det5)={N.mean():.1f} vs Nbar={NB['det5']:.1f} z={(N.mean()-NB['det5'])/np.sqrt(NB['det5']/len(seeds)):+.2f}")

# ------------------------------------------------------------------ (b) floor-1e14 skies, new seeds
pf = sorted(glob.glob(os.path.join(DATA, os.environ.get("Q1_PALM_DIR", "palm_new"), "sky_*.npy")))
P = [np.load(f, allow_pickle=True).item() for f in pf]
R = len(P)
from q1_palm_check import THP, NRAND
print(f"\n== (b) {R} floor-1e14 skies (seeds {P[0]['seed']}..{P[-1]['seed']}), detected = q_obs>q among M>=1e14 ==")
zb = []
for q in QCUTS:
    Nbar = NB[f"det{q}_floor1e14"]
    N = np.array([s[f"N{q}"] for s in P], float)
    print(f"-- det{q}: mean N = {N.mean():.1f}, Nbar = {Nbar:.1f}, z = {(N.mean()-Nbar)/np.sqrt(Nbar/R):+.2f},"
          f" rel = {N.mean()/Nbar-1:+.5f}, var/mean = {N.var(ddof=1)/Nbar:.3f} +- {np.sqrt(2/(R-1)):.3f}")
    for j, t in enumerate(THP):
        ge1 = np.array([s[f"ge1_det{q}"][j] for s in P], float)
        zu, zc = report(f"det{q}|det{q} >=1", N, ge1, t, Nbar, fsky(t))
        zb.append((q, t, zu, zc))
        # >= 2 others: P = 1 - (1-f)^(N-1) - (N-1) f (1-f)^(N-2)
        ge2 = np.array([s[f"ge2_det{q}"][j] for s in P], float)
        f = fsky(t); mu = Nbar * f
        P2u = 1 - np.exp(-mu) * (1 + mu)
        P2c = 1 - (1 - f) ** (N - 1) - (N - 1) * f * (1 - f) ** (N - 2)
        est2, se2, zu2 = ratio_test(ge2, N, P2u); _, _, zc2 = ratio_test(ge2, N, P2c)
        # random-centred
        g1r = np.array([s[f"ge1_rand{q}"][j] for s in P], float)
        estr, ser, zur = ratio_test(g1r, np.full(R, NRAND, float), 1 - np.exp(-mu))
        _, _, zcr = ratio_test(g1r, np.full(R, NRAND, float), 1 - (1 - f) ** N)
        print(f"  {'':24s}               >=2: MC={est2:.4e}+-{se2:.1e} vs {P2u:.4e} z={zu2:+.2f} z_cond={zc2:+.2f};"
              f"  random-centred >=1: {estr:.4e}+-{ser:.1e} z={zur:+.2f} z_cond={zcr:+.2f}")
zb = np.array(zb)
print(f"\n(b) summary over {len(zb)} (q,theta) detected-centred P(>=1) bins (correlated across q and theta):"
      f" unconditional z: mean {zb[:,2].mean():+.2f}, range [{zb[:,2].min():+.2f},{zb[:,2].max():+.2f}];"
      f" conditional z: mean {zb[:,3].mean():+.2f}, range [{zb[:,3].min():+.2f},{zb[:,3].max():+.2f}]")
for q in QCUTS:
    sel = (zb[:, 0] == q) & np.isin(zb[:, 1], [10, 15])
    print(f"  det{q} at 10'/15': z_unc = {zb[sel,2].round(2)}, z_cond = {zb[sel,3].round(2)}")

# ------------------------------------------------------------------ (c) pair-count variance decomposition (C9)
print("\n== (c) pair counts: E[P] = Nbar mu/2; Var[P] = E(1+2mu) = shot E + N-fluctuation term Nbar mu^2 (=2 mu E) ==")
print("   ratio (N-fluct term)/(shot term) = 2 mu: it exceeds 1 only for mu > 0.5")
for lab, key, nkey, Nbar in [("det5 (M>1e14)", "pairs_det5", "N5", NB["det5_floor1e14"]),
                             ("M>1e14", "pairs_all", "Nall", NB["M1e14"])]:
    Pc = np.array([s[key] for s in P]); N = np.array([s[nkey] for s in P], float)
    for j, t in enumerate(THP):
        f = fsky(t); mu = Nbar * f; E = Nbar * mu / 2; V = E * (1 + 2 * mu)
        resid = Pc[:, j] - f * N * (N - 1) / 2                     # remove the part driven by the realised N
        vr = np.var(resid, ddof=1)
        evar = np.sqrt(2 / (R - 1))
        print(f"  {lab:14s} {t:5.1f}' mu={mu:8.4f} 2mu={2*mu:8.4f}  E={E:11.1f} MC={Pc[:,j].mean():11.1f}"
              f" z={(Pc[:,j].mean()-E)/np.sqrt(V/R):+5.2f}  Var/[E(1+2mu)]={Pc[:,j].var(ddof=1)/V:.3f}+-{evar:.3f}"
              f"  Var/E={Pc[:,j].var(ddof=1)/E:7.3f}  Var(P - fN(N-1)/2)/[E(1-f)]={vr/(E*(1-f)):.3f}+-{evar:.3f}")
print("\n   det5 (full population, Nbar={:.1f}): 2mu at 0.5/1/2/5/10/15/30' = ".format(NB["det5"]) +
      ", ".join(f"{2*NB['det5']*fsky(t):.4f}" for t in [0.5, 1, 2, 5, 10, 15, 30]) +
      f"; mu=0.5 reached at theta = {2*np.arcsin(np.sqrt(0.5/NB['det5']))/ARCMIN:.2f}'")
