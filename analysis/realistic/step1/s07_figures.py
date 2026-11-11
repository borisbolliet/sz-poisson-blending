"""Step 1 / s07: figures -> figures/realistic/step1_*.png (data from data_real/, written by s01-s06)."""
import os, sys, json
for v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ[v] = "4"
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from szsim import instrument as ins, signal as sg

ROOT = "/Users/boris/GitHub/sz-poisson-blending"
FIG = f"{ROOT}/figures/realistic"
DR = f"{ROOT}/data_real"
os.makedirs(FIG, exist_ok=True)
CAT = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4"]   # project palette (validated, fixed order)
INK, MUTED = "#333333", "#8a8a85"
plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.grid": True, "grid.color": "#e4e3dc", "grid.linewidth": 0.6,
                     "lines.linewidth": 1.8, "legend.frameon": False})
RUNCOL = {"white": CAT[0], "baseline": CAT[1], "goal": CAT[2]}
RUNLAB = {"white": "old: white y-noise 4e-6', 1.4' beam", "baseline": "SO LAT baseline MMF", "goal": "SO LAT goal MMF"}


def fig_noise():
    ell = np.arange(30, 20001, 5).astype(float)
    fig, axs = plt.subplots(1, 2, figsize=(10.5, 3.9))
    ax = axs[0]
    B = ins.beam_l(ell, ins.SO_LAT["fwhm"])
    N = ins.so_lat_noise(ell, "baseline")
    D = ell * (ell + 1) / (2 * np.pi)
    ls = ["-", "--", "-", "--", "-", "--"]
    cols = [CAT[0], CAT[0], CAT[1], CAT[1], CAT[2], CAT[2]]
    for i, nu in enumerate(ins.SO_LAT["freqs"]):
        ax.plot(ell, D * N[i, i] / B[i] ** 2, color=cols[i], ls=ls[i], lw=1.3, label=f"{nu:.0f} GHz")
    ax.plot(ell, D * ins.cmb_cl_TT(ell), color=INK, lw=2, label="lensed CMB TT (CAMB)")
    ax.set_xscale("log"); ax.set_yscale("log"); ax.set_ylim(1, 1e6); ax.set_xlim(30, 2e4)
    ax.set_xlabel("multipole l"); ax.set_ylabel("l(l+1) N_l / 2π / B_l²  [μK²]")
    ax.set_title("SO LAT baseline noise per channel (beam-deconvolved, T)")
    ax.legend(fontsize=7, ncol=2, loc="lower left")
    ax = axs[1]
    for k, run in enumerate(("baseline", "goal")):
        I = ins.Instrument.so_lat(run)
        l = I.ell[100:]
        ax.plot(l, l * (l + 1) / (2 * np.pi) * I.Ny()[100:], color=RUNCOL[run], label=f"N_y, {RUNLAB[run]}")
    Iw = ins.Instrument.white_y()
    lw = Iw.ell[(Iw.ell > 30) & (Iw.ell < 20000)][::20]
    ax.plot(lw, lw * (lw + 1) / (2 * np.pi) * Iw.Ny(lw), color=RUNCOL["white"], label=f"N_w / B_l², {RUNLAB['white']}")
    cv = np.load(f"{DR}/confusion_var.npz")
    lc, Ca, Cs = cv["baseline_ell"], cv["baseline_Cyy_all"], cv["baseline_Cyy_sub"]
    sel = lc >= 30
    ax.plot(lc[sel], lc[sel] * (lc[sel] + 1) / (2 * np.pi) * Ca[sel], color=INK, lw=1.6, label="tSZ Poisson power, all M>1e13")
    ax.plot(lc[sel], lc[sel] * (lc[sel] + 1) / (2 * np.pi) * Cs[sel], color=INK, lw=1.4, ls="--",
            label="tSZ Poisson power, sub-threshold (q<5, baseline)")
    ax.set_xscale("log"); ax.set_yscale("log"); ax.set_xlim(30, 2e4); ax.set_ylim(1e-14, 1e-8)
    ax.set_xlabel("multipole l"); ax.set_ylabel("l(l+1) P_l / 2π   [y²]")
    ax.set_title("Effective y-noise of the MMF vs tSZ confusion power")
    ax.legend(fontsize=6.8)
    fig.tight_layout(); fig.savefig(f"{FIG}/step1_noise_ny.png", dpi=150); plt.close(fig)


def fig_sigma():
    d = np.load(f"{DR}/sigmaY_mmf.npz")
    th = d["theta500"]
    fig, axs = plt.subplots(1, 2, figsize=(10.5, 3.9))
    ax = axs[0]
    for run, key in (("white", "white"), ("baseline", "baseline_l100"), ("goal", "goal_l100")):
        ax.plot(th, d[key], color=RUNCOL[run], label=RUNLAB[run])
    ax.set_xscale("log"); ax.set_yscale("log"); ax.set_xlim(0.1, 100)
    ax.set_xlabel("θ500 [arcmin]"); ax.set_ylabel("σ_Y  [arcmin²]  (Y500)")
    ax.set_title("Matched-filter noise on Y500"); ax.legend(fontsize=7.5)
    ax = axs[1]
    for run, key in (("baseline", "baseline_l100"), ("goal", "goal_l100")):
        ax.plot(th, d[key] / d["white"], color=RUNCOL[run], label=RUNLAB[run])
    ax.plot(th, d["baseline_noCMB"] / d["white"], color=RUNCOL["baseline"], ls="--", lw=1.2, label="baseline without CMB")
    ax.plot(th, d["baseline_noATM"] / d["white"], color=RUNCOL["baseline"], ls=":", lw=1.4, label="baseline without atmosphere")
    ax.axhline(1, color=MUTED, lw=1)
    ax.set_xscale("log"); ax.set_yscale("log"); ax.set_xlim(0.1, 100)
    ax.set_xlabel("θ500 [arcmin]"); ax.set_ylabel("σ_Y(MMF) / σ_Y(old white)")
    ax.set_title("Ratio to the old single-frequency setup"); ax.legend(fontsize=7.5)
    fig.tight_layout(); fig.savefig(f"{FIG}/step1_sigmaY.png", dpi=150); plt.close(fig)


def fig_rho():
    s = np.linspace(0, 40, 801)
    fig, axs = plt.subplots(1, 2, figsize=(10.5, 3.6), sharey=True)
    models = {"white": sg.SZModel(), "baseline": ins.MMFModel(ins.Instrument.so_lat("baseline")),
              "goal": ins.MMFModel(ins.Instrument.so_lat("goal"))}
    for ax, th in zip(axs, (1.0, 4.0)):
        for run, m in models.items():
            r = m.cross_response(th, th, s, nl=80000) if run == "white" else m.cross_response(th, th, s)
            ax.plot(s, r, color=RUNCOL[run], label=RUNLAB[run])
        ax.axhline(0, color=MUTED, lw=1)
        ax.set_xlabel("separation s [arcmin]"); ax.set_title(f"θ_A = θ_B = {th:g}'")
    axs[0].set_ylabel("ρ_AB(s):  Ŷ_A per unit Y500 of B"); axs[0].legend(fontsize=7.5)
    fig.suptitle("Filter response to a neighbour: the MMF is compensated (negative ring, zero DC response)", fontsize=10)
    fig.tight_layout(); fig.savefig(f"{FIG}/step1_rho_profiles.png", dpi=150); plt.close(fig)


def fig_conf():
    cv = np.load(f"{DR}/confusion_var.npz")
    th = cv["thA"]
    fig, axs = plt.subplots(1, 2, figsize=(10.5, 3.9))
    ax = axs[0]
    for run in ("white", "baseline", "goal"):
        ax.plot(th, cv[f"{run}_var_all"], color=RUNCOL[run], label=f"{RUNLAB[run]}: all M>1e13")
        ax.plot(th, cv[f"{run}_var_subthr"], color=RUNCOL[run], ls="--", lw=1.3, label=f"sub-threshold neighbours (q_B<5)")
    ax.axhline(1, color=MUTED, lw=1)
    ax.set_xscale("log"); ax.set_yscale("log"); ax.set_xlim(0.25, 80)
    ax.set_xlabel("θ_A = θ500 of the cluster [arcmin]"); ax.set_ylabel("var_conf / σ_Y²")
    ax.set_title("Poisson confusion variance in units of the filter noise"); ax.legend(fontsize=6.5)
    ax = axs[1]
    ax.plot(th, cv["white_mu"], color=RUNCOL["white"], label="old white filter: μ = ȳ·DC/σ_Y")
    ax.plot(th, np.full_like(th, 1e-3), color=RUNCOL["baseline"], ls="--", lw=1.3,
            label="SO MMF baseline / goal: μ = 0 exactly (compensated; drawn at 1e-3)")
    ax.set_xscale("log"); ax.set_yscale("log"); ax.set_xlim(0.25, 80); ax.set_ylim(5e-4, 50)
    ax.set_xlabel("θ_A [arcmin]"); ax.set_ylabel("Campbell mean shift μ  [σ_Y units]")
    ax.set_title("Response to the mean y of all halos (before mean removal)"); ax.legend(fontsize=7)
    fig.tight_layout(); fig.savefig(f"{FIG}/step1_confusion.png", dpi=150); plt.close(fig)


def fig_fractions():
    runs = [r for r in ("white", "baseline", "goal") if os.path.exists(f"{DR}/pd_fractions_{r}.npz")]
    if not runs:
        print("no pd_fractions files yet"); return
    lab = ["Y boosted\n>10%", "Y boosted\n>20%", "boosted over\nthreshold", "lost\n(of N_single)", "Y lowered\n>10%"]
    panels = [("1e13", "", "neighbours M500 > 1e13 Msun"), ("1e14", "", "neighbours M500 > 1e14 Msun only"),
              ("1e13", "_subthr", "sub-threshold neighbours only (q_B < 5)")]
    panels = [p for p in panels if any(os.path.exists(f"{DR}/pd_fractions_{r}{p[1]}.npz") for r in runs)]
    fig, axs = plt.subplots(1, len(panels), figsize=(5.2 * len(panels), 3.9), sharey=True, squeeze=False)
    axs = axs[0]
    for ax, (fl, tag, title) in zip(axs, panels):
        x = np.arange(len(lab))
        rr = [r for r in runs if os.path.exists(f"{DR}/pd_fractions_{r}{tag}.npz")]
        wbar = 0.8 / len(runs)
        for k, run in enumerate(rr):
            k = runs.index(run)
            d = np.load(f"{DR}/pd_fractions_{run}{tag}.npz")
            cols = list(d["cols"]); row = d[f"rows_{fl}"][1]          # q_cut = 5
            v = [row[cols.index("B2_0.1")], row[cols.index("B2_0.2")], row[cols.index("B3")], row[cols.index("lost")],
                 row[cols.index("B2neg_0.1")]]
            b = ax.bar(x + (k - (len(runs) - 1) / 2) * wbar, np.array(v) * 100, width=wbar * 0.92, color=RUNCOL[run],
                       label=RUNLAB[run])
            for xi, vi in zip(x + (k - (len(runs) - 1) / 2) * wbar, v):
                ax.text(xi, vi * 100 + 0.3, f"{vi*100:.1f}", ha="center", va="bottom", fontsize=6.5, color=INK)
        ax.set_xticks(x); ax.set_xticklabels(lab, fontsize=7.5)
        ax.set_title(f"q > 5: {title}", fontsize=8.5)
        ax.grid(axis="x", visible=False)
    axs[0].set_ylabel("% of detections (lost: % of single-cluster detections)")
    axs[0].legend(fontsize=7)
    fig.tight_layout(); fig.savefig(f"{FIG}/step1_blending_fractions.png", dpi=150); plt.close(fig)


if __name__ == "__main__":
    which = sys.argv[1:] or ["noise", "sigma", "rho", "conf", "fractions"]
    for w in which:
        globals()[f"fig_{w}"]()
        print("done", w, flush=True)
