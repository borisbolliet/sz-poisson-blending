"""zA06 (arbiter): figure figures/realistic/step3_szifi_real_arbiter.png.
Top: r(theta) = sigma / sigma_analytic after SZiFi's default noise iteration, mean over the arbiter's 12 realistic tiles
per noise level: SZiFi as configured (S1, the build's runs), with its inverse covariance zeroed outside lrange (S1z),
and also with periodic inpainting (S1zp); step 3a's emulation a01 on the same tiles (dashed) and SZiFi before the
iteration (dotted).  Bottom: the paired excess over a01, r_SZiFi - r_a01, with the error of the mean (bands); grey
dotted: the same difference at iteration 0 (SZiFi's pixel-sampled template; no iteration involved).
Usage: python -I -B zA06_figure.py
"""
import os, sys
sys.dont_write_bytecode = True
ROOT = "/Users/boris/GitHub/sz-poisson-blending"
OUT = os.path.join(ROOT, "data_real", "step3_szifi_real")
FIG = os.path.join(ROOT, "figures", "realistic")
os.environ["MPLCONFIGDIR"] = os.path.join(OUT, "arbiter", "out", "mplconfig")
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

C = dict(blue="#2a78d6", orange="#eb6834", aqua="#1baf7a", gray="#8a8984", ink="#0b0b0b", ink2="#52514e")
plt.rcParams.update({"font.size": 9, "axes.edgecolor": C["ink2"], "axes.labelcolor": C["ink"], "xtick.color": C["ink2"],
                     "ytick.color": C["ink2"], "axes.grid": True, "grid.color": "#e4e3df", "grid.linewidth": 0.6,
                     "axes.spines.top": False, "axes.spines.right": False, "legend.frameon": False,
                     "figure.facecolor": "white", "axes.facecolor": "white"})
REF = dict(np.load(os.path.join(OUT, "reference.npz")))
ANA = dict(np.load(os.path.join(OUT, "analytic_sigma.npz")))
TH = REF["thetas"]
PR = [0, 11, 22, 33, 44, 55, 88, 132, 176, 220, 264, 308]


def load(fn):
    with np.load(fn) as d:
        return {k: d[k] for k in d.files}


def r_of(var, mode, it):
    out = []
    for p in PR:
        fn = os.path.join(OUT, "runs" if var == "S1" else os.path.join("arbiter", "runs"), f"{var}_{mode}_real_{p:03d}.npz")
        out.append(load(fn)[f"it{it}_sig"] / ANA[f"{mode}_S0"])
    return np.array(out)


fig, axs = plt.subplots(2, 2, figsize=(10, 7), sharex=True, gridspec_kw=dict(height_ratios=[1.25, 1]))
SER = (("S1", C["orange"], "-", "SZiFi as configured (S1)"),
       ("S1z", C["blue"], "--", "round-off artifact removed (S1z)"),
       ("S1zp", C["aqua"], "-.", "artifact removed + periodic inpainting (S1zp)"))
for k, mode in enumerate(("baseline", "goal")):
    sY = REF[f"{mode}_sigY"]
    a1 = np.array([load(os.path.join(ROOT, "data_real", "step3", "arbiter", f"a01_real_{p:03d}.npz"))[f"{mode}_szifi_sig1"]
                   for p in PR]) / sY
    a0 = np.array([load(os.path.join(ROOT, "data_real", "step3", "arbiter", f"a01_real_{p:03d}.npz"))[f"{mode}_szifi_sig0"]
                   for p in PR]) / sY
    ax, bx = axs[0, k], axs[1, k]
    r0 = r_of("S1", mode, 0)
    ax.plot(TH, r0.mean(0), color=C["gray"], lw=1.4, ls=":", label="SZiFi before the iteration")
    ax.plot(TH, a1.mean(0), color=C["ink2"], lw=1.6, ls=(0, (5, 2)), label="step 3a's emulation (a01)")
    for var, col, ls, lab in SER:
        r1 = r_of(var, mode, 1)
        ax.plot(TH, r1.mean(0), color=col, lw=2, ls=ls, label=lab)
        d = r1 - a1
        m, e = d.mean(0), d.std(0, ddof=1) / np.sqrt(len(PR))
        bx.fill_between(TH, m - e, m + e, color=col, alpha=0.18, lw=0)
        bx.plot(TH, m, color=col, lw=2, ls=ls, label=lab)
    bx.plot(TH, (r_of("S1zp", mode, 0) - a0).mean(0), color=C["gray"], lw=1.4, ls=":",
            label="same at iteration 0 (template only)")
    bx.axhline(0, color=C["ink2"], lw=0.8)
    ax.set_title(f"SO {mode}: r after SZiFi's default iteration (12 tiles)", fontsize=9.5, loc="left")
    bx.set_title("paired excess over the emulation, r_SZiFi - r_a01", fontsize=9.5, loc="left")
    ax.set_xscale("log"); bx.set_xscale("log")
    bx.set_xlabel("filter scale θ500 [arcmin]")
    ax.set_ylabel("r = σ / σ_analytic"); bx.set_ylabel("Δr")
    for a_ in (ax, bx):
        a_.set_xticks([0.5, 1, 2, 5, 10, 15]); a_.set_xticklabels(["0.5", "1", "2", "5", "10", "15"])
axs[0, 0].legend(fontsize=8, loc="lower left")
h, l = axs[1, 1].get_legend_handles_labels()
fig.legend(h, l, fontsize=8, loc="lower center", ncol=4, title="bottom row")
fig.tight_layout(rect=(0, 0.05, 1, 1))
os.makedirs(FIG, exist_ok=True)
fn = os.path.join(FIG, "step3_szifi_real_arbiter.png")
fig.savefig(fn, dpi=150)
print(f"wrote {fn}")
