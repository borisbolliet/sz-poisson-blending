"""Analysis of (A) floor-1e14 skies: spherical cap vs flat-sky aperture counts up to 180 deg, survey-wide pair
counts and their variance; (B) rare-configuration skies (M>3e14): probability that at least one close pair /
triple exists anywhere on the sky."""
import os, glob, json
import numpy as np
from scipy import stats
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from szsim import sampler, cosmo_hmf as ch, signal as sg
from q1_common import *
from q1_floor_caps_pairs import CAP_DEG
from q1_rare_configs import PA_THETAS, PB_THETAS, TA_THETAS

NB = json.load(open(os.path.join(DATA, "nbar.json")))
P3 = 1 - 3 * np.sqrt(3) / (4 * np.pi)
out = {}

# ============================ (A) floor-1e14 skies
S = [np.load(f, allow_pickle=True).item() for f in sorted(glob.glob(os.path.join(DATA, "floor14", "sky_*.npy")))]
R = len(S)
N = np.array([s["N"] for s in S], float); N5 = np.array([s["N5"] for s in S], float)
Nb, N5b = NB["M1e14"], NB["det5_floor1e14"]
print(f"== A. {R} floor-1e14 skies ==")
for lab, x, e in [("N(M>1e14)", N, Nb), ("N(det5, M>1e14)", N5, N5b)]:
    print(f"{lab:16s} mean={x.mean():.1f} expected={e:.1f} z={(x.mean()-e)/np.sqrt(e/R):+.2f}  var/mean={x.var(ddof=1)/e:.3f} +- {np.sqrt(2/(R-1)):.3f}")

print("\n-- A1. mean count in a cap of radius theta (random centres): MC vs exact cap mu = Nbar(1-cos)/2 and flat mu = Nbar theta^2/4")
cm = np.array([s["cap_mean"] for s in S]); cm2 = np.array([s["cap_m2"] for s in S])
mcap = Nb * (1 - np.cos(np.radians(CAP_DEG))) / 2
mflat = Nb * np.radians(CAP_DEG) ** 2 / 4
rows = []
for j, t in enumerate(CAP_DEG):
    m = cm[:, j].mean(); e = cm[:, j].std(ddof=1) / np.sqrt(R)
    var = (cm2[:, j] - cm[:, j] ** 2)          # per-sky variance over centres (about the sky's own mean)
    # total variance over (sky, centre) = mean within-sky var + var of sky means
    vtot = var.mean() + cm[:, j].var(ddof=1)
    # bootstrap over skies for the error on vtot/mcap
    rng = np.random.default_rng(1)
    bs = []
    for _ in range(300):
        ii = rng.integers(0, R, R)
        bs.append((var[ii].mean() + cm[ii, j].var(ddof=1)) / mcap[j])
    zc = (m - mcap[j]) / e; zf = (m - mflat[j]) / e
    rows.append((t, mcap[j], mflat[j], m, e, zc, zf, vtot / mcap[j], np.std(bs)))
    print(f"theta={t:8.3f} deg  mu_cap={mcap[j]:12.4f} mu_flat={mflat[j]:12.4f} MC={m:12.4f}+-{e:.4f}"
          f"  z(cap)={zc:+7.2f}  z(flat)={zf:+8.2f}  flat/cap-1={mflat[j]/mcap[j]-1:+.2e}  Var/mu_cap={vtot/mcap[j]:.4f}+-{np.std(bs):.4f}")
out["caps"] = rows

print("\n-- A2. survey-wide pairs with separation < theta: E[P] = Nbar^2 f / 2 = Nbar mu / 2 (f=(1-cos)/2), Var = E[P](1+2mu)")
for lab, key, nbar in [("M>1e14", "pairs_M1e14", Nb), ("det5 (M>1e14)", "pairs_det5f", N5b)]:
    Pc = np.array([s[key] for s in S])
    mu = mu_of(nbar, THETAS); E = nbar * mu / 2; V = E * (1 + 2 * mu)
    for j, t in enumerate(THETAS):
        print(f"{lab:14s} {t:5.2f}' E={E[j]:13.2f} MC={Pc[:,j].mean():13.2f} z={(Pc[:,j].mean()-E[j])/np.sqrt(V[j]/R):+6.2f}"
              f"  Var_MC/Var_th={Pc[:,j].var(ddof=1)/V[j]:.3f}+-{np.sqrt(2/(R-1)):.3f}  (Poisson-only Var would be E: ratio {Pc[:,j].var(ddof=1)/E[j]:.2f})")

# ============================ (B) rare configurations
rows = np.concatenate([np.load(f) for f in sorted(glob.glob(os.path.join(DATA, "rare", "rare_*.npy")))])
RR = len(rows)
NA = sampler.expected_count(Mmin=3e14); NBz = sampler.expected_count(Mmin=3e14, zmin=1.0)
nA, nB = rows[:, 1], rows[:, 2]
a = 3; pa = rows[:, a:a + len(PA_THETAS)]; a += len(PA_THETAS)
pb = rows[:, a:a + len(PB_THETAS)]; a += len(PB_THETAS)
ta = rows[:, a:a + len(TA_THETAS)]
print(f"\n== B. {RR} skies, A: M500>3e14 (Nbar={NA:.1f}), B: M500>3e14 & z>1 (Nbar={NBz:.2f}) ==")
print(f"N_A mean={nA.mean():.1f} z={(nA.mean()-NA)/np.sqrt(NA/RR):+.2f} var/mean={nA.var(ddof=1)/NA:.3f}+-{np.sqrt(2/(RR-1)):.3f};"
      f"  N_B mean={nB.mean():.2f} z={(nB.mean()-NBz)/np.sqrt(NBz/RR):+.2f} var/mean={nB.var(ddof=1)/NBz:.3f}")
resB = {}
def block(lab, nbar, th, cnt, kind):
    print(f"\n-- {lab}: {kind}")
    res = []
    for j, t in enumerate(th):
        mu = mu_of(nbar, t)
        if kind == "pairs":
            E = nbar * mu / 2; V = E * (1 + 2 * mu)
        else:
            E = nbar * mu ** 2 * P3 / 6; V = None
        pmc = np.mean(cnt[:, j] > 0); pe = np.sqrt(max(pmc * (1 - pmc), 1 / RR) / RR)
        pan = 1 - np.exp(-E)
        mc = cnt[:, j].mean(); emc = cnt[:, j].std(ddof=1) / np.sqrt(RR)
        zE = (mc - E) / (np.sqrt(V / RR) if V is not None else emc)
        # full count distribution vs Poisson(E): chi2 on bins 0,1,2,>=3
        obs = np.array([np.sum(cnt[:, j] == 0), np.sum(cnt[:, j] == 1), np.sum(cnt[:, j] == 2), np.sum(cnt[:, j] >= 3)])
        ex = RR * np.array([stats.poisson.pmf(0, E), stats.poisson.pmf(1, E), stats.poisson.pmf(2, E), stats.poisson.sf(2, E)])
        okb = ex > 5
        chi2 = np.sum((obs[okb] - ex[okb]) ** 2 / ex[okb]); dof = okb.sum() - 1
        print(f"theta={t:6.2f}'  mu={mu:.3e}  E[count]={E:8.4f}  MC mean={mc:8.4f}+-{emc:.4f} (z={zE:+.2f})"
              f"  P(>=1 exists): Poisson-approx 1-exp(-E)={pan:.4f}  MC={pmc:.4f}+-{pe:.4f}  z={(pmc-pan)/pe:+.2f}"
              f"  count-dist chi2={chi2:.1f}/{dof} p={stats.chi2.sf(chi2, dof) if dof>0 else np.nan:.2f}")
        res.append((t, mu, E, mc, emc, pan, pmc, pe))
    return np.array(res)
resB["pairsA"] = block("A pairs (M>3e14)", NA, PA_THETAS, pa, "pairs")
resB["pairsB"] = block("B pairs (M>3e14, z>1)", NBz, PB_THETAS, pb, "pairs")
resB["triA"] = block("A triples, all 3 separations < theta (M>3e14)", NA, TA_THETAS, ta, "triples")
allzP = np.concatenate([(r[:, 6] - r[:, 5]) / r[:, 7] for r in resB.values()])
print(f"\nP(>=1 exists) z-scores over {len(allzP)} tests: rms={np.sqrt(np.mean(allzP**2)):.2f} max|z|={abs(allzP).max():.2f}")

# ============================ figures
fig, axs = plt.subplots(1, 3, figsize=(16, 4.6))
ax = axs[0]
r = np.array(out["caps"], dtype=float)
ax.errorbar(r[:, 0], r[:, 3] / r[:, 1] - 1, r[:, 4] / r[:, 1], fmt="o", ms=4, label="MC / exact cap - 1")
tt = np.geomspace(1 / 60, 180, 300)
ax.plot(tt, np.radians(tt) ** 2 / 4 / ((1 - np.cos(np.radians(tt))) / 2) - 1, "r-", label="flat pi theta^2 / cap - 1")
ax.plot(tt, np.radians(tt) ** 2 / 12, "r:", label="theta^2/12")
ax.axhline(0, color="k", lw=0.8)
ax.set(xscale="log", yscale="symlog", xlabel="aperture radius [deg]", ylabel="relative difference in mean count",
       title=f"cap vs flat-sky area, M>1e14 ({R} skies)")
ax.set_yscale("symlog", linthresh=1e-6)
ax.legend(fontsize=8)
ax = axs[1]
for lab, rr, c in [("pairs, M>3e14", resB["pairsA"], "C0"), ("pairs, M>3e14 & z>1", resB["pairsB"], "C1"),
                   ("triples, M>3e14", resB["triA"], "C2")]:
    ax.errorbar(rr[:, 0], rr[:, 6], rr[:, 7], fmt="o", color=c, ms=4, label=lab + " (MC)")
    nb_ = NA if "z>1" not in lab else NBz
    t2 = np.geomspace(rr[0, 0] * 0.7, rr[-1, 0] * 1.4, 200); mu2 = mu_of(nb_, t2)
    E2 = nb_ * mu2 / 2 if "pairs" in lab else nb_ * mu2 ** 2 * P3 / 6
    ax.plot(t2, 1 - np.exp(-E2), color=c)
ax.set(xscale="log", xlabel="theta [arcmin]", ylabel="P(at least one on the full sky)",
       title=f"survey-wide close configurations ({RR} skies)")
ax.legend(fontsize=8)
ax = axs[2]
Pc = np.array([s["pairs_det5f"] for s in S]); mu = mu_of(N5b, THETAS); E = N5b * mu / 2; V = E * (1 + 2 * mu)
ax.errorbar(THETAS, Pc.mean(0) / E, np.sqrt(V / R) / E, fmt="o", ms=4, label="mean / E[P]")
ax.errorbar(THETAS * 1.03, Pc.var(0, ddof=1) / V, np.sqrt(2 / (R - 1)), fmt="s", ms=4, label="Var / [E(1+2mu)]")
ax.plot(THETAS, Pc.var(0, ddof=1) / E, "x", label="Var / E  (naive Poisson)")
ax.axhline(1, color="k", lw=0.8)
ax.set(xscale="log", yscale="log", xlabel="theta [arcmin]", title=f"det5 pair counts, {R} skies")
ax.legend(fontsize=8)
fig.tight_layout(); fig.savefig(os.path.join(FIG, "caps_pairs_rare.png"), dpi=130); plt.close(fig)
print("figure written:", os.path.join(FIG, "caps_pairs_rare.png"))
