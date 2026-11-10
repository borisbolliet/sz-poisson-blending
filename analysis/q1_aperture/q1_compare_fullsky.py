"""Compare full-sky MC aperture statistics (q1_mc_fullsky.py output) with the Poisson predictions.
Errors are realisation-to-realisation (std over skies / sqrt(R)), so overlap of apertures inside a
sky is accounted for.  Writes figures to figures/q1_aperture/ and a summary json."""
import os, glob, json
import numpy as np
from scipy import stats
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from q1_common import *
from q1_mc_fullsky import FINE, TRI_THETAS, NRAND_BALL

NB = json.load(open(os.path.join(DATA, "nbar.json")))
files = sorted(glob.glob(os.path.join(DATA, "fullsky", "sky_*.npy")))
# repair phase: optional seed window, e.g. Q1_SEEDS=1041-1120 (default: all stored skies)
_sw = os.environ.get("Q1_SEEDS")
if _sw:
    _a, _b = map(int, _sw.split("-"))
    files = [f for f in files if _a <= int(os.path.basename(f)[4:-4]) <= _b]
SUMMARY_NAME = os.environ.get("Q1_SUMMARY", "q1_fullsky_summary.json")
S = [np.load(f, allow_pickle=True).item() for f in files]
R = len(S)
print(f"R = {R} full-sky realisations (seeds {S[0]['seed']}..{S[-1]['seed']})")
iT = np.array([np.argmin(abs(FINE - t)) for t in THETAS]); assert np.allclose(FINE[iT], THETAS)
P3 = 1 - 3 * np.sqrt(3) / (4 * np.pi)
summary = {"R": R}


def zs(mc, sem, an):
    return (mc - an) / sem


# ---------------------------------------------------------------- 0. population sizes
print("\n== 0. realised population sizes vs Nbar (mean, z of mean, var/mean +- sqrt(2/(R-1))) ==")
for p in ["det4", "det5", "det6", "M1e14", "M3e13", "det5_z0.01-0.5", "det5_z0.5-1.0", "det5_z1.0-3.0", "undet5_M1e14"]:
    n = np.array([s["npop"][p] for s in S], float)
    print(f"{p:16s} Nbar={NB[p]:12.1f} mean={n.mean():12.1f} z={(n.mean()-NB[p])/np.sqrt(NB[p]/R):+6.2f}"
          f" var/mean={n.var(ddof=1)/NB[p]:5.2f} +- {np.sqrt(2/(R-1)):.2f}")


# ---------------------------------------------------------------- 1. P(N) per combo
def PN_sky(cdf):
    F = cdf[:, iT]                           # (KMAX, nT): P(N>=k), k=1..6
    P = np.empty((KMAX + 1, len(THETAS)))
    P[0] = 1 - F[0]
    for n in range(1, KMAX):
        P[n] = F[n - 1] - F[n]
    P[KMAX] = F[KMAX - 1]
    return P


def PN_analytic(Nbar):
    mu = mu_of(Nbar, THETAS)
    P = np.array([P_N(n, mu) for n in range(KMAX)] + [P_geq(KMAX, mu)])
    return P, mu


keys = list(S[0]["cdf"].keys())
allz = []
print("\n== 1. P(N=n | theta), n=0..5 and >=6, MC vs Poisson; per combo: diag chi2 over the (theta,n) bins with"
      " >=10 expected events, rms z, max|z| ==")
combo_stats = {}
for key in keys:
    c, p = key.split("|")
    P = np.array([PN_sky(s["cdf"][key]) for s in S])     # (R, 7, nT)
    m = P.mean(0); sem = P.std(0, ddof=1) / np.sqrt(R)
    A, mu = PN_analytic(NB[p])
    ncen = np.mean([s["ncen"][key] for s in S])
    nexp = A * ncen * R
    # error: sky-to-sky scatter, floored at the independent-trial (binomial) error of the analytic P;
    # the sample std of a few events per sky is unreliable (it can be accidentally tiny)
    sem = np.maximum(sem, np.sqrt(A * (1 - A) / (ncen * R)))
    ok = (nexp >= 10) & (sem > 0)
    z = np.where(ok, (m - A) / np.where(sem > 0, sem, 1), np.nan)
    zz = z[ok]
    chi2 = float((zz ** 2).sum()); dof = int(ok.sum())
    # bins with too few expected events: check the observed number of events is consistent (Poisson)
    nobs = m * ncen * R
    low = (~ok) & (nexp < 10)
    plow = stats.poisson.sf(np.round(nobs[low]) - 1, nexp[low]) if low.any() else np.array([1.0])
    combo_stats[key] = dict(chi2=chi2, dof=dof, p=float(stats.chi2.sf(chi2, dof)), rms=float(np.sqrt(np.mean(zz ** 2))),
                            maxz=float(np.max(abs(zz))), ncen=float(ncen))
    allz.append(zz)
    print(f"{key:28s} ncen/sky={ncen:9.0f} chi2/dof={chi2:7.1f}/{dof:3d} p={combo_stats[key]['p']:.3f}"
          f" rms z={combo_stats[key]['rms']:.2f} max|z|={combo_stats[key]['maxz']:.2f}"
          f"  low-count bins: {low.sum()} (min Poisson tail p={plow.min():.3f}, total obs={nobs[low].sum():.1f} vs exp={nexp[low].sum():.1f})")
allz = np.concatenate(allz)
print(f"ALL combos: {len(allz)} z-scores, rms={np.sqrt(np.mean(allz**2)):.3f}, mean={allz.mean():+.3f},"
      f" max|z|={abs(allz).max():.2f}, frac|z|>2 = {np.mean(abs(allz)>2):.3f} (Gauss 0.046), frac|z|>3 = {np.mean(abs(allz)>3):.4f} (0.0027)")
summary["combo_stats"] = combo_stats
summary["allz"] = dict(n=len(allz), rms=float(np.sqrt(np.mean(allz ** 2))), maxabs=float(abs(allz).max()))

# detailed listing for three representative combos
for key in ["rand|det5", "det5|det5", "det5|M3e13"]:
    c, p = key.split("|")
    P = np.array([PN_sky(s["cdf"][key]) for s in S]); m = P.mean(0); sem = P.std(0, ddof=1) / np.sqrt(R)
    A, mu = PN_analytic(NB[p]); nc = np.mean([s["ncen"][key] for s in S])
    sem = np.maximum(sem, np.sqrt(A * (1 - A) / (nc * R)))
    print(f"\n-- {key}: theta, mu, then for n=0..3: MC +- err (analytic) [z]")
    for j, t in enumerate(THETAS):
        s = " ".join(f"{m[n,j]:.4e}+-{sem[n,j]:.1e}({A[n,j]:.4e})[{(m[n,j]-A[n,j])/sem[n,j] if sem[n,j]>0 else np.nan:+.1f}]" for n in range(4))
        print(f"{t:5.2f}' mu={mu[j]:.3e} {s}")

# ---------------------------------------------------------------- 2. Palm: centred on detected == random
print("\n== 2. Slivnyak-Mecke: paired per-sky difference P(N>=k | detected centre) - P(N>=k | random centre) ==")
for (a, b) in [("det5|det5", "rand|det5"), ("det4|det4", "rand|det4"), ("det6|det6", "rand|det6"),
               ("det5|M3e13", "rand|M3e13"), ("det5|M1e14", "rand|M1e14"),
               ("det5_zc<0.5|det5", "rand|det5"), ("det5_zc>1|det5", "rand|det5"), ("det5_q>20|det5", "rand|det5"),
               ("det5_zc<0.5|M3e13", "rand|M3e13"), ("det5_zc>1|M3e13", "rand|M3e13"), ("det5_q>20|M3e13", "rand|M3e13")]:
    D = np.array([s["cdf"][a][:, iT] - s["cdf"][b][:, iT] for s in S])   # (R, 6, nT)
    m = D.mean(0); e = D.std(0, ddof=1) / np.sqrt(R)
    pa = NB[b.split("|")[1]]
    Fan = np.array([stats.gamma.cdf(mu_of(pa, THETAS), k + 1) for k in range(KMAX)])   # P(N>=k)
    na = np.mean([s["ncen"][a] for s in S]); nb_ = np.mean([s["ncen"][b] for s in S])
    efloor = np.sqrt(Fan * (1 - Fan) * (1 / na + 1 / nb_) / R)
    good = Fan * (1 - Fan) * min(na, nb_) * R >= 10        # enough events in both directions
    e = np.maximum(e, efloor)
    z = np.where(e > 0, m / np.where(e > 0, e, 1), np.nan)
    zz = z[np.isfinite(z) & (e > 0) & good]
    # largest relative difference for P(N>=1)
    F1 = np.array([s["cdf"][b][0, iT] for s in S]).mean(0)
    print(f"{a:20s} vs {b:12s}: {len(zz)} (k,theta) bins, chi2={np.sum(zz**2):.1f}, p={stats.chi2.sf(np.sum(zz**2), len(zz)):.3f},"
          f" max|z|={np.max(abs(zz)):.2f};  P(>=1) rel.diff at 1',5',10',60': "
          + ", ".join(f"{m[0,j]/F1[j]:+.4f}+-{e[0,j]/F1[j]:.4f}" for j in [2, 6, 8, 13]))

# self-inclusion pitfall
print("\n-- pitfall: counting the centre itself shifts N by +1: P_incl(N=n) = P(N=n-1); e.g. det5|det5 at 10':"
      f" P_incl(N=1) would be {PN_analytic(NB['det5'])[0][0,8]:.4f} instead of {PN_analytic(NB['det5'])[0][1,8]:.4f}")

# ---------------------------------------------------------------- 3. KS of kNN PIT
print("\n== 3. kNN distance distributions: u_k = P(Pois(mu(d_k))>=k) ~ U(0,1), mu from Nbar (unconditional law)."
      " Per sky: KS on a random subsample of 1000 centres; combined over skies by a KS-uniformity test of the R p-values"
      " (pU) and Fisher (pF); pooled: KS on all R*1000 u-values ==")
ks_summary = {}
for key in ["rand|det5", "det5|det5", "det4|det4", "det6|det6", "rand|M1e14", "det5|M1e14", "rand|M3e13", "det5|M3e13",
            "det5|det5_z1.0-3.0", "det5|undet5_M1e14", "det5_q>20|det5", "det5_zc>1|M3e13"]:
    p = key.split("|")[1]
    line = []
    for k in range(KMAX):
        us = [knn_u(s["dsub"][key][:, k].astype(float), k + 1, NB[p]) for s in S]
        pv = np.array([stats.kstest(u, "uniform").pvalue for u in us])
        pu = stats.kstest(pv, "uniform").pvalue
        fisher = stats.chi2.sf(-2 * np.log(pv).sum(), 2 * R)
        pool = stats.kstest(np.concatenate(us), "uniform")
        line.append(f"k={k+1}: pU={pu:.2f} pF={fisher:.2f} pool D={pool.statistic:.4f} p={pool.pvalue:.2f}")
        ks_summary[f"{key}|k{k+1}"] = dict(pU=float(pu), pF=float(fisher), pool_p=float(pool.pvalue), pool_D=float(pool.statistic))
    print(f"{key:20s} " + "; ".join(line))
allp = np.array([[v["pU"], v["pF"], v["pool_p"]] for v in ks_summary.values()])
print(f"KS summary over {len(allp)} (combo,k) tests: min pooled p = {allp[:,2].min():.3f}; "
      f"KS-uniformity of the pooled p-values: p = {stats.kstest(allp[:,2], 'uniform').pvalue:.3f};"
      f" fraction pooled p<0.05: {np.mean(allp[:,2]<0.05):.3f}")
summary["ks"] = ks_summary
print("\n-- per-sky KS conditional on the realised population size (mu from the sky's own N): within one sky the"
      " neighbours are a binomial process with the realised N, so for sparse populations and large k the per-sky"
      " test must use it; the pooled test uses Nbar")
for key in ["det5|det5_z1.0-3.0", "rand|det5_z1.0-3.0", "det5|det5", "rand|M3e13"]:
    p = key.split("|")[1]
    line = []
    for k in range(KMAX):
        pv = np.array([stats.kstest(knn_u(s["dsub"][key][:, k].astype(float), k + 1, s["npop"][p]), "uniform").pvalue
                       for s in S])
        line.append(f"k={k+1}: pU={stats.kstest(pv, 'uniform').pvalue:.2f} pF={stats.chi2.sf(-2*np.log(pv).sum(), 2*R):.2f}")
    print(f"{key:20s} (realised N) " + "; ".join(line))

# ---------------------------------------------------------------- 4. dispersion of ball counts
print("\n== 4. random-aperture counts: mean / mu_cap and var/mean (Poisson: 1) ==")
for p in ["det5", "M1e14", "M3e13"]:
    B = np.array([s[f"ball_{p}"] for s in S])   # (R, nT, 2)
    mu = mu_of(NB[p], THETAS)
    rm = B[:, :, 0] / mu; rv = B[:, :, 1] / B[:, :, 0]
    for j in range(len(THETAS)):
        if P_geq(2, mu[j]) * NRAND_BALL * R < 100:
            continue          # var/mean estimator is biased when N>=2 events are too rare to be sampled
        print(f"{p:6s} theta={THETAS[j]:5.1f}' mu={mu[j]:9.4f} mean/mu={rm[:,j].mean():.4f}+-{rm[:,j].std(ddof=1)/np.sqrt(R):.4f}"
              f" [z={(rm[:,j].mean()-1)/(rm[:,j].std(ddof=1)/np.sqrt(R)):+.2f}]"
              f"  var/mean={rv[:,j].mean():.4f}+-{rv[:,j].std(ddof=1)/np.sqrt(R):.4f} [z={(rv[:,j].mean()-1)/(rv[:,j].std(ddof=1)/np.sqrt(R)):+.2f}]")

# ---------------------------------------------------------------- 5. survey-wide pairs, triples, n>=k
print("\n== 5. survey-wide unordered pairs with sep<theta: E[P]=Nbar*mu/2, Var=E[P](1+2mu) ==")
for p in ["det4", "det5", "det6", "M1e14"]:
    Pc = np.array([s[f"pairs_{p}"] for s in S]); mu = mu_of(NB[p], THETAS)
    E = NB[p] * mu / 2; V = E * (1 + 2 * mu)
    for j in [0, 2, 4, 6, 8, 11, 13]:
        print(f"{p:6s} {THETAS[j]:5.2f}' E={E[j]:12.2f} MC={Pc[:,j].mean():12.2f} z={(Pc[:,j].mean()-E[j])/np.sqrt(V[j]/R):+6.2f}"
              f"  var_MC/var_th={Pc[:,j].var(ddof=1)/V[j]:.2f}+-{np.sqrt(2/(R-1)):.2f}")
print("\n-- det5 triples (all 3 separations < theta): E = Nbar mu^2 p3 / 6, p3 = 1-3sqrt3/(4pi)")
Tr = np.array([s["tri_det5"] for s in S]); muT = mu_of(NB["det5"], TRI_THETAS)
ET = NB["det5"] * muT ** 2 * P3 / 6
for j, t in enumerate(TRI_THETAS):
    print(f"theta={t:5.1f}' E={ET[j]:9.3f} MC={Tr[:,j].mean():9.3f}+-{Tr[:,j].std(ddof=1)/np.sqrt(R):.3f}"
          f" z={(Tr[:,j].mean()-ET[j])/(Tr[:,j].std(ddof=1)/np.sqrt(R)):+.2f}")
print("\n-- number of det5 clusters with >= n other det5 within theta: E = Nbar P(Pois(mu)>=n)")
G = np.array([s["nge_det5"] for s in S]); mu = mu_of(NB["det5"], THETAS)
for n in range(1, 6):
    E = NB["det5"] * P_geq(n, mu)
    js = [j for j in range(len(THETAS)) if E[j] > 3]
    zz = [(G[:, n-1, j].mean() - E[j]) / (G[:, n-1, j].std(ddof=1) / np.sqrt(R)) for j in js]
    print(f"n>={n}: " + " ".join(f"{THETAS[j]:.1f}':{E[j]:.1f}/{G[:,n-1,j].mean():.1f}[{z:+.1f}]" for j, z in zip(js, zz)))

# ---------------------------------------------------------------- 6. ready-to-use table
print("\n== 6. READY-TO-USE TABLE: detected cluster (q_obs>q_cut), probability of >=1 / >=2 neighbours within theta ==")
print("| q_cut | theta | neighbours | Sigma [deg^-2] | mu | P(>=1) analytic | P(>=1) MC | z | P(>=2) analytic | P(>=2) MC | z |")
print("|---|---|---|---|---|---|---|---|---|---|---|")
table = []
for q in QCUTS:
    for t in [1, 2, 5, 10]:
        j = int(np.argmin(abs(FINE - t)))
        for lab, p in [("(a) detected", f"det{q}"), ("(b) M>1e14", "M1e14"), ("(c) M>3e13", "M3e13")]:
            key = f"det{q}|{p}"
            F = np.array([s["cdf"][key][:, j] for s in S])
            mu = float(mu_of(NB[p], t))
            a1, a2 = P_geq(1, mu), P_geq(2, mu)
            nc = np.mean([s["ncen"][key] for s in S])
            m1, e1 = F[:, 0].mean(), max(F[:, 0].std(ddof=1) / np.sqrt(R), np.sqrt(a1 * (1 - a1) / (nc * R)))
            m2, e2 = F[:, 1].mean(), max(F[:, 1].std(ddof=1) / np.sqrt(R), np.sqrt(a2 * (1 - a2) / (nc * R)))
            z1 = (m1 - a1) / e1 if e1 > 0 else np.nan
            z2 = (m2 - a2) / e2 if e2 > 0 else np.nan
            table.append(dict(q=q, theta=t, pop=lab, mu=mu, P1=a1, P1mc=m1, e1=e1, P2=a2, P2mc=m2, e2=e2, z1=z1, z2=z2))
            print(f"| {q} | {t}' | {lab} | {NB[p]/SKY_DEG2:.3f} | {mu:.3e} | {a1:.3e} | {m1:.3e} ± {e1:.1e} | {z1:+.2f} |"
                  f" {a2:.3e} | {m2:.3e} ± {e2:.1e} | {z2:+.2f} |")
summary["table"] = table
z1 = np.array([r["z1"] for r in table]); z2 = np.array([r["z2"] for r in table if np.isfinite(r["z2"])])
print(f"table z-scores: P>=1 rms={np.sqrt(np.mean(z1**2)):.2f} max={abs(z1).max():.2f} (n={len(z1)}); "
      f"P>=2 rms={np.sqrt(np.mean(z2**2)):.2f} max={abs(z2).max():.2f} (n={len(z2)})")
json.dump(summary, open(os.path.join(DATA, SUMMARY_NAME), "w"), indent=1, default=float)

# ---------------------------------------------------------------- figures
os.makedirs(FIG, exist_ok=True)
cols = plt.cm.viridis(np.linspace(0, 0.9, KMAX + 1))
fig, axs = plt.subplots(2, 3, figsize=(15, 8), sharex=True, gridspec_kw=dict(height_ratios=[3, 1.3]))
for ax, rax, key, title in zip(axs[0], axs[1], ["rand|det5", "det5|det5", "det5|M3e13"],
                               ["random aperture, detected (q>5)", "centred on detected, other detected (Palm)",
                                "centred on detected, halos M>3e13"]):
    c, p = key.split("|")
    P = np.array([PN_sky(s["cdf"][key]) for s in S]); m = P.mean(0); sem = P.std(0, ddof=1) / np.sqrt(R)
    A, mu = PN_analytic(NB[p])
    tt = np.geomspace(0.3, 100, 300); mm = mu_of(NB[p], tt)
    for n in range(KMAX + 1):
        an = P_N(n, mm) if n < KMAX else P_geq(KMAX, mm)
        ax.plot(tt, an, color=cols[n], lw=1.2, label=f"N={n}" if n < KMAX else "N>=6")
        good = m[n] > 0
        ax.errorbar(THETAS[good], m[n][good], sem[n][good], fmt="o", ms=3.5, color=cols[n])
        ok = (A[n] * R * np.mean([s["ncen"][key] for s in S]) >= 10) & (sem[n] > 0)
        rax.plot(THETAS[ok] * (1 + 0.02 * n), ((m[n] - A[n]) / np.where(sem[n] > 0, sem[n], 1))[ok], "o", ms=3.5, color=cols[n])
    ax.set(xscale="log", yscale="log", ylim=(1e-7, 1.5), title=title)
    rax.axhspan(-2, 2, color="0.9"); rax.axhline(0, color="k", lw=0.8)
    rax.set(xscale="log", ylim=(-4, 4), xlabel="aperture radius theta [arcmin]", ylabel="(MC-Poisson)/sigma")
axs[0, 0].set_ylabel("P(N = n)"); axs[0, 0].legend(fontsize=8, ncol=2)
fig.suptitle(f"Counts in a circular aperture, {R} full-sky Poisson realisations; lines: Poisson with mu = Sigma 2pi(1-cos theta)")
fig.tight_layout(); fig.savefig(os.path.join(FIG, "PN_vs_theta.png"), dpi=130); plt.close(fig)

# kNN CDF figure
fig, axs = plt.subplots(1, 2, figsize=(12, 4.5))
for ax, key in zip(axs, ["det5|det5", "det5|M1e14"]):
    c, p = key.split("|")
    Fm = np.array([s["cdf"][key] for s in S]); m = Fm.mean(0); e = Fm.std(0, ddof=1) / np.sqrt(R)
    for k in range(KMAX):
        an = stats.gamma.cdf(mu_of(NB[p], FINE), k + 1)
        ax.plot(FINE, an, color=cols[k], lw=1.2, label=f"k={k+1}")
        ax.plot(FINE[::3], m[k][::3], "o", ms=3, color=cols[k])
    ax.set(xscale="log", xlabel="distance to k-th neighbour [arcmin]", ylabel="P(d_k <= theta)",
           title=f"{key.replace('|',' centres, neighbours: ')}")
    ax.legend(fontsize=8)
fig.tight_layout(); fig.savefig(os.path.join(FIG, "knn_cdf.png"), dpi=130); plt.close(fig)

# ready-to-use table figure
fig, axs = plt.subplots(1, 3, figsize=(15, 4.5), sharey=True)
tt = np.geomspace(0.3, 60, 200)
for ax, q in zip(axs, QCUTS):
    for lab, p, cc in [("detected", f"det{q}", "C0"), ("M>1e14", "M1e14", "C1"), ("M>3e13", "M3e13", "C2")]:
        mm = mu_of(NB[p], tt)
        ax.plot(tt, P_geq(1, mm), color=cc, label=f"P(>=1), {lab}")
        ax.plot(tt, P_geq(2, mm), color=cc, ls="--", label=f"P(>=2), {lab}")
        key = f"det{q}|{p}"
        Fm = np.array([s["cdf"][key][:, iT] for s in S])
        for kk, mk in [(0, "o"), (1, "s")]:
            mm_ = Fm[:, kk].mean(0); ee = Fm[:, kk].std(0, ddof=1) / np.sqrt(R)
            g = mm_ > 0
            ax.errorbar(THETAS[g], mm_[g], ee[g], fmt=mk, ms=3.5, color=cc, mfc="none" if kk else cc)
    ax.set(xscale="log", yscale="log", ylim=(1e-7, 1.2), xlabel="theta [arcmin]", title=f"centre: detected, q_obs > {q}")
axs[0].set_ylabel("P(>= n neighbours within theta)"); axs[0].legend(fontsize=7)
fig.tight_layout(); fig.savefig(os.path.join(FIG, "neighbour_prob_table.png"), dpi=130); plt.close(fig)
print("figures written to", FIG)
