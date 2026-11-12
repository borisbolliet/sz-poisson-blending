"""z11: figures for the SZiFi-itself test of step 3a (figures/realistic/step3_szifi_real_*.png).
  _r_theta : r(theta) = sigma / sigma_analytic before and after the noise iteration, SZiFi itself (S1 defaults, S2 l_sep
             12000, S3i SO example with an isotropic covariance) against the step-3a emulations on the same tiles and the
             production finder; step 2's noise normalisations (i) and (iii) for reference;
  _edges   : test (f): q_SZiFi - q_finder at matched iteration-0 detections and the fraction of SZiFi detections with a
             finder partner, versus the distance to the tile edge; the chosen buffer;
  _spectra : SZiFi's own auto spectra after / before the iteration (mean over channels and tiles) versus l.
Usage: python -I -B z11_figures.py
"""
import os, sys
sys.dont_write_bytecode = True
ROOT = "/Users/boris/GitHub/sz-poisson-blending"
OUT = os.path.join(ROOT, "data_real", "step3_szifi_real")
FIG = os.path.join(ROOT, "figures", "realistic")
os.environ["MPLCONFIGDIR"] = os.path.join(OUT, "out", "mplconfig")
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

C = dict(blue="#2a78d6", orange="#eb6834", aqua="#1baf7a", yellow="#eda100", violet="#6250d6", gray="#8a8984",
         ink="#0b0b0b", ink2="#52514e")
plt.rcParams.update({"font.size": 9, "axes.edgecolor": C["ink2"], "axes.labelcolor": C["ink"], "xtick.color": C["ink2"],
                     "ytick.color": C["ink2"], "axes.grid": True, "grid.color": "#e4e3df", "grid.linewidth": 0.6,
                     "axes.spines.top": False, "axes.spines.right": False, "legend.frameon": False,
                     "figure.facecolor": "white", "axes.facecolor": "white"})
REF = dict(np.load(os.path.join(OUT, "reference.npz")))
ANA = dict(np.load(os.path.join(OUT, "analytic_sigma.npz")))
TH = REF["thetas"]
PR = [int(p) for p in REF["pids_real"]]
SETUP = {"S1": "S0", "S2": "S0", "S3i": "S3"}
A01 = os.path.join(ROOT, "data_real", "step3", "arbiter")


def runs(cfg, mode, kind="real"):
    out = []
    for p in PR if kind == "real" else [int(x) for x in REF["pids_noise"]]:
        fn = os.path.join(OUT, "runs", f"{cfg}_{mode}_{kind}_{p:03d}.npz")
        if os.path.exists(fn):
            with np.load(fn) as d:
                out.append((p, {k: d[k] for k in d.files}))
    return out


# ------------------------------------------------------------------ r(theta)
fig, axs = plt.subplots(1, 2, figsize=(10, 4.2), sharey=True)
for ax, mode in zip(axs, ("baseline", "goal")):
    sY = REF[f"{mode}_sigY"]
    ax.plot(TH, REF[f"{mode}_r_iii"], color=C["gray"], lw=1.2, ls=":", label="step 2 (iii) plain std")
    ax.plot(TH, REF[f"{mode}_r_i"], color=C["gray"], lw=1.2, ls="-", label="step 2 (i)")
    S1 = runs("S1", mode)
    if S1:
        r0 = np.array([r["it0_sig"] / ANA[f"{mode}_S0"] for _, r in S1])
        ax.plot(TH, r0.mean(0), color=C["blue"], lw=2, label=f"SZiFi before the iteration ({len(S1)} tiles)")
    for cfg, col, lab, a01v in (("S1", C["orange"], "SZiFi defaults (S1)", "szifi"),
                                ("S2", C["aqua"], "SZiFi, l_sep 12000 (S2)", "szifi_l12000"),
                                ("S3i", C["yellow"], "SZiFi SO example, isotropic (S3i)", None)):
        R = runs(cfg, mode)
        if not R:
            continue
        r1 = np.array([r["it1_sig"] / ANA[f"{mode}_{SETUP[cfg]}"] for _, r in R])
        m, s = r1.mean(0), r1.std(0)
        ax.fill_between(TH, m - s, m + s, color=col, alpha=0.15, lw=0)
        ax.plot(TH, m, color=col, lw=2, label=f"{lab}, after ({len(R)} tiles)")
        if a01v:
            a1 = np.array([np.load(os.path.join(A01, f"a01_real_{p:03d}.npz"))[f"{mode}_{a01v}_sig1"] / sY for p, _ in R])
            ax.plot(TH, a1.mean(0), color=col, lw=1.4, ls="--", label=f"step-3a emulation of {cfg}, same tiles")
    rp = np.array([np.load(os.path.join(ROOT, "data_real", "step3", "tiles", f"tile_{p:03d}.npz"))[f"{mode}_it1_sig"] / sY
                   for p in PR])
    ax.plot(TH, rp.mean(0), color=C["violet"], lw=2, label="production finder, after (whitened)")
    ax.set_xscale("log")
    ax.set_xticks([0.5, 1, 2, 4, 8, 15]); ax.set_xticklabels(["0.5", "1", "2", "4", "8", "15"])
    ax.set_xlabel("filter scale theta [arcmin]")
    ax.set_title(f"SO {mode}", color=C["ink"], loc="left")
axs[0].set_ylabel("r = sigma / sigma_analytic")
h, l = axs[0].get_legend_handles_labels()
fig.legend(h, l, loc="lower center", ncol=3, fontsize=8)
fig.tight_layout(rect=(0, 0.2, 1, 1))
fig.savefig(os.path.join(FIG, "step3_szifi_real_r_theta.png"), dpi=150)
plt.close(fig)

# ------------------------------------------------------------------ edges
E = dict(np.load(os.path.join(OUT, "edge_buffer.npz")))
fig, axs = plt.subplots(1, 2, figsize=(10, 3.8))
for mode, col in (("baseline", C["blue"]), ("goal", C["orange"])):
    st = E[f"stats_{mode}"]
    lo, hi, n, f1, mdq, sdq, ns, f2, nok = st[:, :9].T
    xc = np.sqrt(np.maximum(lo, 0.25) * hi)
    ok = nok >= 2
    err = np.where(nok > 1, sdq / np.sqrt(np.maximum(nok, 1)), np.nan)
    axs[0].errorbar(xc[ok], mdq[ok], yerr=err[ok], color=col, marker="o", ms=4, lw=1.5, capsize=2, label=mode)
    okb = ns >= 1
    axs[1].plot(xc[okb], 100 * f2[okb], color=col, marker="o", ms=4, lw=1.5, label=mode)
for ax in axs:
    ax.axvspan(0.2, float(E["buffer_arcmin"]), color=C["gray"], alpha=0.12, lw=0)
    ax.axvline(float(E["interior_arcmin"]), color=C["ink2"], lw=1, ls="--")
    ax.set_xscale("log"); ax.set_xlim(0.2, 320)
    ax.set_xlabel("distance of the detection to the tile edge [arcmin]")
axs[0].set_yscale("symlog", linthresh=0.5)
axs[0].axhline(0, color=C["ink2"], lw=0.8)
axs[0].set_ylabel("mean q_SZiFi - q_finder (iteration 0)")
axs[0].set_title(f"(a) S/N bias; shaded: edge buffer {float(E['buffer_arcmin']):.0f}', dashed: interior", loc="left", color=C["ink"])
axs[1].set_ylabel("SZiFi q > 4.5 detections with a finder partner [%]")
axs[1].set_title("(b) spurious peaks from the zero-padded convolution", loc="left", color=C["ink"])
axs[0].legend(loc="upper right")
fig.tight_layout()
fig.savefig(os.path.join(FIG, "step3_szifi_real_edges.png"), dpi=150)
plt.close(fig)

# ------------------------------------------------------------------ spectra
fig, axs = plt.subplots(1, 2, figsize=(10, 3.8), sharey=True)
for ax, mode in zip(axs, ("baseline", "goal")):
    for cfg, col, lab in (("S1", C["orange"], "SZiFi defaults (S1): l_sep 3000"), ("S2", C["aqua"], "S2: l_sep 12000"),
                          ("S3i", C["yellow"], "S3i: SO example, isotropic")):
        R = runs(cfg, mode)
        if not R:
            continue
        e = R[0][1]["spec0_ell"]
        rat = np.array([np.mean([r["spec1"][:, i, i] / r["spec0"][:, i, i] for i in range(6)], 0) for _, r in R])
        sel = (e >= 100) & (e <= 20000)
        ax.plot(e[sel], rat.mean(0)[sel], color=col, lw=1.5, label=f"{lab} ({len(R)} tiles)")
    ax.axhline(1, color=C["ink2"], lw=0.8)
    ax.set_xscale("log")
    ax.set_xlabel("multipole l")
    ax.set_title(f"SO {mode}", loc="left", color=C["ink"])
axs[0].set_ylabel("auto spectra, iteration 1 / iteration 0")
axs[0].legend(loc="upper left", fontsize=8)
fig.tight_layout()
fig.savefig(os.path.join(FIG, "step3_szifi_real_spectra.png"), dpi=150)
plt.close(fig)
print("figures written:", [f for f in sorted(os.listdir(FIG)) if f.startswith("step3_szifi_real_")])
