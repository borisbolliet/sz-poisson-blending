"""[REPAIR v2 of c08: converged model; writes bias_bars_v2.png and mcmc_contours_v2.png; count_residuals.png not redone
(Delta_cat changed by <0.05 Poisson sigma per bin, c11)]
Figures for the cosmology-bias test (reads data_cosmo/fits_v2_final.json, truth_*.npz, mcmc_*.npz).
  figures/cosmo/bias_bars_v2.png       bias/sigma in Om, s8, S8 per fit configuration (q>5, SO-like 40% sky and full sky)
  figures/cosmo/count_residuals.png blending-induced count change vs z and vs q_obs (q>5) for the three truth routes
  figures/cosmo/mcmc_contours_v2.png   emcee posteriors (Om, s8) and (s8, 1-b): naive model on its own expectation vs on
                                    the blended (catalogue P(D)) expectation, config d, 40% sky
"""
import os, sys, json
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import cosmolib as CL

FIG = f"{CL.ROOT}/figures/cosmo"; os.makedirs(FIG, exist_ok=True)
COL = {"cat": "#2a78d6", "map_clean": "#eb6834", "map": "#1baf7a"}
LAB = {"cat": "catalogue P(D) model", "map_clean": "maps, fragments + noise peaks removed", "map": "maps, all detections"}
MK = {"cat": "o", "map_clean": "s", "map": "^"}
plt.rcParams.update({"font.size": 10, "axes.edgecolor": "#52514e", "axes.labelcolor": "#0b0b0b",
                     "xtick.color": "#52514e", "ytick.color": "#52514e", "axes.spines.top": False, "axes.spines.right": False})

rows = json.load(open(f"{CL.DC}/fits_v2_final.json"))
CFG = ["a", "b", "c", "d", "e"]
CFG_LAB = {"a": "a: Ωm, σ8", "b": "b: + (1−b)", "c": "c: + (1−b), σlnY", "d": "d: + (1−b), α, σlnY\n   (1−b) prior 4%",
           "e": "e: + (1−b), α, σlnY\n   no prior"}


def get(qc, sv, cn, dn):
    for r in rows:
        if r["qcut"] == qc and r["survey"] == sv and r["config"] == cn and r["data"] == dn:
            return r


# ---------------- figure 1: bias bars
fig, axs = plt.subplots(2, 3, figsize=(12, 7.2), sharey=True)
for irow, sv in enumerate(["SO40", "full"]):
    for ip, p in enumerate(["Om", "s8", "S8"]):
        ax = axs[irow, ip]
        ax.axvspan(-1, 1, color="#e9e8e4", zorder=0)
        ax.axvline(0, color="#52514e", lw=0.8)
        for k, dn in enumerate(["cat", "map_clean", "map"]):
            ys, xs, es = [], [], []
            for j, cn in enumerate(CFG):
                r = get(5.0, sv, cn, dn)
                ys.append(j + (k - 1) * 0.22); xs.append(r[f"b_{p}"])
                es.append(r["boot_std"][ip] if "boot_std" in r else 0.0)
            ax.errorbar(xs, ys, xerr=es, fmt=MK[dn], ms=7, color=COL[dn], mec="white", mew=1.0, elinewidth=1.5,
                        capsize=0, label=LAB[dn] if (irow == 0 and ip == 0) else None, zorder=3)
        ax.set_yticks(range(len(CFG))); ax.set_yticklabels([CFG_LAB[c] for c in CFG])
        ax.invert_yaxis()
        ax.set_xlabel(f"bias in {CL.LABEL[p]}  [σ]")
        ax.grid(axis="x", color="#e9e8e4", lw=0.6)
        if irow == 0:
            ax.set_title(CL.LABEL[p])
    fig.text(0.01, 0.955 - 0.47 * irow, "SO-like 16,500 deg²" if sv == "SO40" else "full sky (41,253 deg²)",
             fontsize=11, fontweight="bold", va="bottom")
fig.legend(loc="upper center", ncol=3, frameon=False, bbox_to_anchor=(0.55, 1.0))
fig.text(0.55, 0.005, "Naive (no-blending) model fitted to the blended expectation (Asimov), q_obs > 5; σ = converged Fisher. Grey: |bias| < 1σ. "
         "Bars: patch bootstrap of map truth.", ha="center", fontsize=8.5, color="#52514e")
fig.tight_layout(rect=(0, 0.02, 1, 0.95))
fig.savefig(f"{FIG}/bias_bars_v2.png", dpi=150); plt.close(fig)

# ---------------- figure 3: MCMC contours
fn_s, fn_c = f"{CL.DC}/mcmc_v2_d_self_SO40.npz", f"{CL.DC}/mcmc_v2_d_cat_SO40.npz"
if os.path.exists(fn_s) and os.path.exists(fn_c):
    fig, axs = plt.subplots(1, 3, figsize=(12, 4))
    pairs = [("Om", "s8"), ("omb", "s8"), ("alpha", "s8")]
    for fn, col, lab in ((fn_s, "#52514e", "naive model on its own expectation"), (fn_c, COL["cat"], "naive model on blended expectation (catalogue P(D))")):
        m = np.load(fn); ch = m["chain"]; free = list(m["free"])
        for ax, (px, py) in zip(axs, pairs):
            x = ch[:, free.index(px)]; y = ch[:, free.index(py)]
            from scipy.ndimage import gaussian_filter
            H, xe, ye = np.histogram2d(x, y, bins=60)
            Hs = gaussian_filter(H.T, 1.3); Hs /= Hs.sum(); o = np.sort(Hs.ravel())[::-1]; c = np.cumsum(o)
            lv = [o[np.searchsorted(c, 0.954)], o[np.searchsorted(c, 0.683)]]
            ax.contour(0.5 * (xe[1:] + xe[:-1]), 0.5 * (ye[1:] + ye[:-1]), Hs, levels=lv, colors=col, linewidths=[1.2, 2])
            ax.plot([], [], color=col, lw=2, label=lab)
            ax.plot(m["xml"][free.index(px)], m["xml"][free.index(py)], "x", color=col, ms=8, mew=2)
    for ax, (px, py) in zip(axs, pairs):
        ax.axvline(CL.FID[CL.PNAMES.index(px)], color="#a3a29d", lw=0.8, ls="--")
        ax.axhline(CL.FID[CL.PNAMES.index(py)], color="#a3a29d", lw=0.8, ls="--")
        ax.set_xlabel(CL.LABEL[px]); ax.set_ylabel(CL.LABEL[py]); ax.grid(color="#e9e8e4", lw=0.6)
    fig.legend(*axs[0].get_legend_handles_labels(), loc="lower center", ncol=2, frameon=False, fontsize=9)
    fig.suptitle("Config d (Ωm, σ8, 1−b, α, σlnY free; 4% prior on 1−b), q > 5, 16,500 deg². 68/95% contours; × = ML; dashed = truth",
                 fontsize=10)
    fig.tight_layout(rect=(0, 0.07, 1, 1)); fig.savefig(f"{FIG}/mcmc_contours_v2.png", dpi=150); plt.close(fig)
print("figures written to", FIG)
