"""Repair phase: calibrated goodness-of-fit for the aperture-count claims (C2, C3, C4).

Verifiers pointed out that the per-combo 'diag chi2/dof, p' of q1_compare_fullsky.py sums z^2 over strongly
correlated bins (P(N=0) and P(N=1) are anti-correlated at -0.999 at small theta; nested theta are positively
correlated), so its p-values are not chi2-calibrated.  Two calibrated replacements:

 (1) Centred sky bootstrap of the SAME diagonal statistic: per-sky vectors are shifted so that their mean equals
     the analytic prediction (null hypothesis true, correlation structure kept), skies are resampled with
     replacement, and the statistic is recomputed (same binomial error floor, same bin mask).
     p_boot = fraction of resamples with stat >= observed.  A global version sums the statistic over all combos,
     resampling the same skies for every combo (cross-combo correlations kept).
 (2) Hotelling T^2 on a small non-redundant bin set per combo: P(N>=1) and P(N>=2) at <= 4 radii each
     (only bins with >= 30 events and >= 30 non-events per sky), d <= 8;
     F = (R-d)/(d(R-1)) T^2 ~ F(d, R-d) under Gaussian per-sky means.

The same two tests are applied to the paired Palm differences P(N>=k | detected centre) - P(N>=k | random centre).
usage: Q1_SEEDS=1001-1040 python q1_calibrated_tests.py   (default: all stored skies)
"""
import os, glob, json
import numpy as np
from scipy import stats
from q1_common import *
from q1_mc_fullsky import FINE

NB = json.load(open(os.path.join(DATA, "nbar.json")))
files = sorted(glob.glob(os.path.join(DATA, "fullsky", "sky_*.npy")))
sw = os.environ.get("Q1_SEEDS")
if sw:
    a, b = map(int, sw.split("-"))
    files = [f for f in files if a <= int(os.path.basename(f)[4:-4]) <= b]
S = [np.load(f, allow_pickle=True).item() for f in files]
R = len(S)
iT = np.array([np.argmin(abs(FINE - t)) for t in THETAS])
B = 4000
rng = np.random.default_rng(12345)
BOOT = rng.integers(0, R, size=(B, R))          # the same resamples for every combo -> global test valid
print(f"R = {R} skies (seeds {S[0]['seed']}..{S[-1]['seed']}), {B} centred bootstrap resamples")


def PN_sky(cdf):
    F = cdf[:, iT]
    P = np.empty((KMAX + 1, len(THETAS)))
    P[0] = 1 - F[0]
    for n in range(1, KMAX):
        P[n] = F[n - 1] - F[n]
    P[KMAX] = F[KMAX - 1]
    return P


def PN_analytic(Nbar):
    mu = mu_of(Nbar, THETAS)
    return np.array([P_N(n, mu) for n in range(KMAX)] + [P_geq(KMAX, mu)])


def diag_stat(X, A, floor, ok):
    """X: (R, nbins) per-sky values; A: null mean; floor: binomial error floor; ok: bin mask."""
    m = X.mean(0); sem = np.maximum(X.std(0, ddof=1) / np.sqrt(X.shape[0]), floor)
    z = (m - A)[ok] / sem[ok]
    return float((z ** 2).sum())


def boot_null(X, A, floor, ok):
    Xc = X - X.mean(0) + A
    return np.array([diag_stat(Xc[ii], A, floor, ok) for ii in BOOT])


def hotelling(D):
    """D: (R, d) per-sky deviations from the null mean."""
    Rr, d = D.shape
    m = D.mean(0); Sx = np.cov(D, rowvar=False)
    T2 = Rr * m @ np.linalg.solve(Sx, m)
    F = (Rr - d) / (d * (Rr - 1)) * T2
    return T2, F, float(stats.f.sf(F, d, Rr - d))


def hot_bins(F_an, ncen):
    """choose P(N>=1), P(N>=2) at <= 4 radii each with >= 30 events and non-events per sky."""
    sel = []
    for k in (0, 1):
        good = np.flatnonzero((F_an[k] * ncen >= 30) & ((1 - F_an[k]) * ncen >= 30))
        if len(good) == 0:
            continue
        pick = good[np.unique(np.round(np.linspace(0, len(good) - 1, min(4, len(good)))).astype(int))]
        sel += [(k, j) for j in pick]
    return sel


keys = list(S[0]["cdf"].keys())
print("\n== C2/C4: P(N=n|theta) per combo. diag chi2 (as in q1_compare_fullsky), naive chi2 p, centred-bootstrap p;"
      " Hotelling T^2 on <=8 non-redundant P(N>=k) bins ==")
glob_obs = 0.0; glob_null = np.zeros(B); out = {}
pboot_all, photel_all = [], []
for key in keys:
    c, p = key.split("|")
    X = np.array([PN_sky(s["cdf"][key]) for s in S]).reshape(R, -1)
    A = PN_analytic(NB[p]).ravel()
    ncen = np.mean([s["ncen"][key] for s in S])
    floor = np.sqrt(A * (1 - A) / (ncen * R))
    ok = (A * ncen * R >= 10) & (X.std(0, ddof=1) / np.sqrt(R) + floor > 0)
    obs = diag_stat(X, A, floor, ok); dof = int(ok.sum())
    null = boot_null(X, A, floor, ok)
    pb = (1 + np.sum(null >= obs)) / (B + 1)
    glob_obs += obs; glob_null += null
    # Hotelling on cumulative P(N>=k)
    Fs = np.array([s["cdf"][key][:, iT] for s in S])            # (R, KMAX, nT)
    Fan = np.array([P_geq(k + 1, mu_of(NB[p], THETAS)) for k in range(KMAX)])
    sel = hot_bins(Fan, ncen)
    D = np.array([[Fs[r, k, j] - Fan[k, j] for (k, j) in sel] for r in range(R)])
    T2, Fst, ph = hotelling(D)
    out[key] = dict(diag=obs, dof=dof, p_naive=float(stats.chi2.sf(obs, dof)), p_boot=float(pb),
                    null_mean=float(null.mean()), d=len(sel), T2=float(T2), p_hot=ph)
    pboot_all.append(pb); photel_all.append(ph)
    print(f"{key:28s} diag={obs:6.1f}/{dof:3d} p_naive={out[key]['p_naive']:.3f}  bootstrap null mean={null.mean():6.1f}"
          f" p_boot={pb:.3f} | Hotelling d={len(sel)} T2={T2:6.2f} p={ph:.3f}")
pg = (1 + np.sum(glob_null >= glob_obs)) / (B + 1)
pboot_all = np.array(pboot_all); photel_all = np.array(photel_all)
print(f"GLOBAL (all {len(keys)} combos, same resamples): sum diag = {glob_obs:.1f}, bootstrap null mean = {glob_null.mean():.1f}"
      f" (sd {glob_null.std():.1f}), p_boot = {pg:.3f}")
print(f"per-combo p_boot: min={pboot_all.min():.3f}, #<0.05 = {np.sum(pboot_all<0.05)}/{len(keys)} (expect {0.05*len(keys):.1f}),"
      f" #<0.01 = {np.sum(pboot_all<0.01)};  per-combo Hotelling p: min={photel_all.min():.3f}, #<0.05 = {np.sum(photel_all<0.05)},"
      f" #<0.01 = {np.sum(photel_all<0.01)};  KS-uniformity of Hotelling p's: p={stats.kstest(photel_all, 'uniform').pvalue:.3f}")
print("(Note: combos share skies and populations, so the per-combo p-values are not independent of each other.)")

print("\n== C3 / C4 splits: paired difference P(N>=k | centre population) - P(N>=k | random centre), same sky ==")
pairs = [("det5|det5", "rand|det5"), ("det4|det4", "rand|det4"), ("det6|det6", "rand|det6"),
         ("det5|M3e13", "rand|M3e13"), ("det5|M1e14", "rand|M1e14"),
         ("det5_zc<0.5|det5", "rand|det5"), ("det5_zc>1|det5", "rand|det5"), ("det5_q>20|det5", "rand|det5"),
         ("det5_zc<0.5|M3e13", "rand|M3e13"), ("det5_zc>1|M3e13", "rand|M3e13"), ("det5_q>20|M3e13", "rand|M3e13")]
for (a, b) in pairs:
    Dfull = np.array([s["cdf"][a][:, iT] - s["cdf"][b][:, iT] for s in S])     # (R, KMAX, nT)
    pa = NB[b.split("|")[1]]
    Fan = np.array([P_geq(k + 1, mu_of(pa, THETAS)) for k in range(KMAX)])
    na = np.mean([s["ncen"][a] for s in S]); nb_ = np.mean([s["ncen"][b] for s in S])
    floor = np.sqrt(Fan * (1 - Fan) * (1 / na + 1 / nb_) / R).ravel()
    good = (Fan * (1 - Fan) * min(na, nb_) * R >= 10).ravel()
    X = Dfull.reshape(R, -1)
    ok = good & ((X.std(0, ddof=1) / np.sqrt(R) + floor) > 0)
    obs = diag_stat(X, 0.0, floor, ok); null = boot_null(X, np.zeros(X.shape[1]), floor, ok)
    pb = (1 + np.sum(null >= obs)) / (B + 1)
    sel = hot_bins(Fan, min(na, nb_))
    D = np.array([[Dfull[r, k, j] for (k, j) in sel] for r in range(R)])
    T2, Fst, ph = hotelling(D)
    print(f"{a:20s} - {b:12s}: diag={obs:6.1f}/{ok.sum():3d} p_naive={stats.chi2.sf(obs, ok.sum()):.3f}"
          f" bootstrap null mean={null.mean():6.1f} p_boot={pb:.3f} | Hotelling d={len(sel)} T2={T2:6.2f} p={ph:.3f}")

print("\n== correlation diagnostics (rand|det5): corr over skies of P(N=0) and P(N=1) at 1', 2', 10'; corr of P(N>=1) at"
      " nested radii 5'-10' ==")
X = np.array([PN_sky(s["cdf"]["rand|det5"]) for s in S])
for t in [1, 2, 10]:
    j = int(np.argmin(abs(THETAS - t)))
    print(f"  theta={t}': corr(P0,P1) = {np.corrcoef(X[:, 0, j], X[:, 1, j])[0, 1]:+.3f}")
F = np.array([s["cdf"]["rand|det5"][0, iT] for s in S])
print(f"  corr(P>=1 at 5', P>=1 at 7.5') = {np.corrcoef(F[:, 6], F[:, 7])[0, 1]:+.3f};"
      f" corr(5', 10') = {np.corrcoef(F[:, 6], F[:, 8])[0, 1]:+.3f}")
json.dump(dict(R=R, combos=out, global_p_boot=float(pg)),
          open(os.path.join(DATA, os.environ.get("Q1_CAL_OUT", "q1_calibrated_tests.json")), "w"), indent=1)
