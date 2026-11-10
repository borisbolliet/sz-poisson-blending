"""Figures for the blending analysis -> figures/blending/*.png"""
import os, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import blendlib as bl

FIG = "/Users/boris/GitHub/sz-poisson-blending/figures/blending"
os.makedirs(FIG, exist_ok=True)
CAT = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300"]
plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.grid": True, "grid.color": "#e4e3dc", "grid.linewidth": 0.6,
                     "lines.linewidth": 2, "legend.frameon": False})

A = dict(np.load(os.path.join(bl.DATA, "analytic.npz")))
C = np.load(os.path.join(bl.DATA, "compare_out.npy"), allow_pickle=True)[0]
QC = A["QCUTS"]; EPS = A["EPS"]; FL = list(A["FLOORS"]); TH = A["th"]

# ---- 1. P(D) CCDF at random points
fig, axs = plt.subplots(1, 3, figsize=(11, 3.6), sharey=True)
thAs = sorted(set(k[1] for k in C if k[0] == "ccdf"))
for ax, fl in zip(axs, FL):
    fi = FL.index(fl)
    for j, thA in enumerate(thAs):
        inode = int(np.argmin(np.abs(TH - thA)))
        ax.plot(A["CCDF_T"], A["CCDF"][fi, inode], color=CAT[j], label=f"θ_A={thA:.2f}'")
        pts = [(t, *C[("ccdf", thA, fl, t)][:2]) for t in A["CCDF_T"]]
        t, p, e = np.array(pts).T
        ok = p > 0
        ax.errorbar(t[ok], p[ok], e[ok], fmt="o", ms=4, color=CAT[j], mec="white", mew=0.8, lw=1)
    ax.set_xscale("log"); ax.set_yscale("log"); ax.set_ylim(1e-6, 1.2)
    ax.set_xlabel("t  [SNR units]"); ax.set_title(f"neighbours M500 > {fl} Msun")
axs[0].set_ylabel("P(δ > t),  δ = D / σ_Y(θ_A)")
axs[0].legend(fontsize=7)
fig.suptitle("Neighbour-sum distribution at random points: FFT compound Poisson (lines) vs MC (points)", fontsize=10)
fig.tight_layout(); fig.savefig(f"{FIG}/PD_ccdf_random_points.png", dpi=150); plt.close(fig)

# ---- 2. mean contribution mu(thA) by neighbour class, + median
fig, axs = plt.subplots(1, 2, figsize=(9, 3.6))
mu = A["mu_exact"]
lbl = ["1e13–3e13", "3e13–1e14", "≥1e14"]
for c in range(3):
    axs[0].plot(TH, mu[:, c], color=CAT[c], label=f"μ from M500 {lbl[c]}")
axs[0].plot(TH, mu.sum(1), color="#333", label="μ total (= monopole shift)")
for fi, fl in enumerate(FL):
    axs[0].plot(TH, A["QUANT"][fi, :, 1], ls="--", color=CAT[fi] if fi < 2 else CAT[2], lw=1.2,
                label=f"median δ, floor {fl}")
axs[0].set_xscale("log"); axs[0].set_yscale("log"); axs[0].set_xlabel("θ_A = θ500 of the primary [arcmin]")
axs[0].set_ylabel("δ [SNR units]"); axs[0].legend(fontsize=6.5); axs[0].set_xlim(0.25, 70)
axs[0].set_title("Mean vs median neighbour boost")
# rate of contributions with x > 1 and x > 0.1
for fi, fl in enumerate(FL):
    cl = list(bl.FLOORS[fl])
    for tcut, ls in ((0.1, "-"), (1.0, "--")):
        lam = np.array([bl.PD.lam_above(A["W"][i, cl].sum(0), A["WX"][i, cl].sum(0), tcut) + A["CAP"][i, cl].sum()
                        for i in range(len(TH))])
        axs[1].plot(TH, lam, ls=ls, color=CAT[fi], label=f"floor {fl}, x > {tcut}")
axs[1].set_xscale("log"); axs[1].set_yscale("log"); axs[1].set_xlim(0.25, 70)
axs[1].set_xlabel("θ_A [arcmin]"); axs[1].set_ylabel("expected # neighbours per filter")
axs[1].set_title("Neighbours contributing x = ρY/σ_A above a level"); axs[1].legend(fontsize=6.5)
fig.tight_layout(); fig.savefig(f"{FIG}/mean_median_boost_vs_theta.png", dpi=150); plt.close(fig)

# ---- 3. B1 vs theta_b
B1a = np.load(os.path.join(bl.DATA, "B1_analytic.npz")); B1m = np.load(os.path.join(bl.DATA, "B1_mc.npz"))
THB = B1a["THB"]
fig, axs = plt.subplots(1, 2, figsize=(9, 3.6))
for k, ax in zip((1, 2), axs):
    for j, q in enumerate(QC):
        ax.plot(THB, B1a[f"fk_13_{int(q)}"][k - 1], color=CAT[j], label=f"q_cut={q:.0f}")
        v = np.array([B1m[f"mc_13_{int(q)}_{k}_{tb}"] for tb in THB])
        ok = v[:, 1] > 0
        ax.errorbar(THB[ok], v[ok, 1], v[ok, 2], fmt="o", ms=4, color=CAT[j], mec="white", mew=0.8, lw=1)
    ax.set_yscale("log"); ax.set_xlabel("θ_b [arcmin]")
    ax.set_ylabel(f"P(≥{k} other detection{'s' if k > 1 else ''} within θ_b)")
    ax.set_title(f"B1: detected cluster with ≥{k} detected neighbour{'s' if k > 1 else ''}")
axs[0].legend(fontsize=7)
fig.suptitle("Resolved-pair blending: Poisson 1−exp(−Σ_det A) (lines) vs 3 full M>1e13 skies (points)", fontsize=10)
fig.tight_layout(); fig.savefig(f"{FIG}/B1_pairs_vs_thetab.png", dpi=150); plt.close(fig)

# ---- 4. B2 vs eps, raw and monopole, floor 1e13
fig, axs = plt.subplots(1, 2, figsize=(9, 3.6), sharey=True)
for c, ax in zip((0, 1), axs):
    for j, q in enumerate(QC):
        row = C[("prim", "1e13", c, q)]
        v = np.array([row[f"B2_{e}"] for e in EPS])
        ax.plot(EPS, v[:, 2], color=CAT[j], label=f"q_cut={q:.0f}")
        ax.errorbar(EPS, v[:, 0], v[:, 1], fmt="o", ms=4, color=CAT[j], mec="white", mew=0.8, lw=1)
        if c == 1:
            vn = np.array([row[f"B2neg_{e}"] for e in EPS])
            ax.plot(EPS, vn[:, 2], color=CAT[j], ls=":", lw=1.4)
            ok = vn[:, 0] > 0
            ax.errorbar(EPS[ok], vn[ok, 0], vn[ok, 1], fmt="s", ms=3.5, color=CAT[j], mfc="white", lw=1)
    ax.set_xscale("log"); ax.set_yscale("log"); ax.set_xlabel("ε (fractional Y bias)"); ax.set_ylim(1e-7, 1.5)
    ax.set_title(["raw sky (filter sees absolute y)", "monopole-removed map (dotted/open: bias < −ε)"][c], fontsize=9)
axs[0].set_ylabel("B2: fraction of detections with bias > ε"); axs[0].legend(fontsize=7)
fig.suptitle("Contamination blending B2 (neighbours M>1e13): analytic P(D) (lines) vs MC (points)", fontsize=10)
fig.tight_layout(); fig.savefig(f"{FIG}/B2_vs_eps.png", dpi=150); plt.close(fig)

# ---- 5. B2 vs primary SNR (q_cut = 5)
QB = [0.5, 3, 5, 7, 10, 15, 20, 40, 100, 3000]
fig, ax = plt.subplots(figsize=(5.5, 3.6))
j = 0
for c in (0, 1):
    for eps in (0.1, 0.2):
        v = np.array([C[("B2vsq", c, eps, lo)] for lo in QB[:-1]])
        xc = np.sqrt(np.array(QB[:-1]) * np.array(QB[1:]))
        ok = v[:, 3] > 30
        ax.plot(xc[ok], v[ok, 2], color=CAT[j], label=f"{['raw', 'monopole'][c]}, ε={eps}")
        ax.errorbar(xc[ok], v[ok, 0], v[ok, 1], fmt="o", ms=4, color=CAT[j], mec="white", mew=0.8, lw=1)
        j += 1
ax.set_xscale("log"); ax.set_yscale("log"); ax.set_xlabel("primary's true SNR q_A"); ax.set_ylabel("P(B2 | detected, q_A)")
ax.set_title("B2 vs own SNR (q_cut=5, neighbours M>1e13)"); ax.legend(fontsize=7)
fig.tight_layout(); fig.savefig(f"{FIG}/B2_vs_primary_snr.png", dpi=150); plt.close(fig)

# ---- 6. B3 / lost / satellites / N_det ratio vs q_cut, floors
fig, axs = plt.subplots(1, 4, figsize=(15, 3.6))
for ic, (name, ttl) in enumerate([("B3", "B3 boosted (would fail q_cut alone)"),
                                  ("B3d_1.0", "B3 with boost δ' > 1 (one noise σ)"), ("B2_0.2", "B2: Y bias > 20%"), ("sat", "satellites (non-dominant, q_obs>q_cut)")]):
    ax = axs[ic]; jj = 0
    for c in ((0, 1) if name != "sat" else (0,)):
        for fi, fl in enumerate(FL):
            v = np.array([C[("prim", fl, c, q)][name] for q in QC])
            ax.plot(QC + 0.05 * fi, v[:, 2], color=CAT[fi], ls=["-", "--"][c],
                    label=f"floor {fl}, {['raw', 'monopole'][c]}")
            ok = v[:, 0] > 0
            ax.errorbar(QC[ok] + 0.05 * fi, v[ok, 0], v[ok, 1], fmt="o", ms=3.5, color=CAT[fi], lw=1,
                        mfc=["white", CAT[fi]][c] if True else None)
    ax.set_yscale("log"); ax.set_xlabel("q_cut"); ax.set_title(ttl, fontsize=9); ax.set_xticks(QC)
axs[0].set_ylabel("fraction")
h, l = axs[0].get_legend_handles_labels()
fig.legend(h, l, loc="lower center", ncol=6, fontsize=7.5, bbox_to_anchor=(0.5, -0.01))
fig.suptitle("Blending fractions vs q_cut and neighbour mass floor: analytic (lines) vs MC (points)", fontsize=10)
fig.tight_layout(rect=(0, 0.07, 1, 1)); fig.savefig(f"{FIG}/B3_B2_sat_vs_qcut_floors.png", dpi=150); plt.close(fig)
print("figures written to", FIG)
