"""Figures for the blending-in-the-likelihood study (figures/cosmo/)."""
import os, json
os.environ.setdefault("NUMBA_NUM_THREADS", "3")
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
import cclib as cc

BLUE, ORANGE, AQUA, INK, MUTED = "#2a78d6", "#eb6834", "#1baf7a", "#0b0b0b", "#8a8984"
plt.rcParams.update({"axes.edgecolor": MUTED, "axes.labelcolor": INK, "xtick.color": INK, "ytick.color": INK,
                     "axes.spines.top": False, "axes.spines.right": False, "font.size": 10})
C = np.load(f"{cc.OUT}/counts_fid.npz"); ZE, QE, Nb, Nn = C["ZE"], C["QE"], C["Nb"], C["Nn"]
zl = [f"{a:g}-{b:g}" for a, b in zip(ZE[:-1], ZE[1:])]; ql = [f"{a:g}-{b:g}" for a, b in zip(QE[:-1], QE[1:])]

# 1. blended/naive per bin and full-sky pulls
fig, ax = plt.subplots(1, 2, figsize=(13, 4.8))
for a, M, lab in ((ax[0], 100 * (Nb / Nn - 1), "blended / naive - 1  [%]"),
                  (ax[1], cc.pull_sigma(Nb, Nn, 1.0), "(N_blended - N_naive) / sqrt(N_naive), full sky")):
    lim = np.abs(M).max()
    im = a.imshow(M, cmap="RdBu_r", norm=TwoSlopeNorm(0, -lim, lim), aspect="auto", origin="lower")
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            a.text(j, i, f"{M[i,j]:.1f}", ha="center", va="center", fontsize=7, color=INK)
    a.set_xticks(range(len(ql)), ql, rotation=45, fontsize=8); a.set_yticks(range(len(zl)), zl, fontsize=8)
    a.set_xlabel("q_obs bin"); a.set_ylabel("z bin (primary)"); a.set_title(lab, fontsize=10)
    plt.colorbar(im, ax=a, shrink=0.85)
plt.tight_layout(); plt.savefig(f"{cc.FIG}/blended_vs_naive_zq.png", dpi=140); plt.close()

# 2. the kernel P(q_obs | q_A, theta)
K = np.load(f"{cc.OUT}/G_fid_b03rates.npz"); G, Pdom = K["G"], K["Pdom"]
t = cc.TGRID; tm = 0.5 * (t[1:] + t[:-1])
fig, ax = plt.subplots(1, 2, figsize=(12, 4.3))
kq = int(np.argmin(np.abs(cc.QNODES - 6.0)))
for i, col in zip((5, 15, 25), (BLUE, ORANGE, AQUA)):
    pdf = np.diff(G[i, kq]) / np.diff(t)
    ax[0].plot(tm, pdf, color=col, lw=2, label=f"theta_A = {K['th'][i]:.2f}'")
    ax[1].semilogy(tm, np.clip(pdf, 1e-7, None), color=col, lw=2, label=f"theta_A = {K['th'][i]:.2f}'")
g = np.exp(-tm ** 2 / 2) / np.sqrt(2 * np.pi)
for a in ax:
    a.plot(tm, g, color=MUTED, lw=1.5, ls="--", label="naive: N(0,1)")
    a.set_xlabel("q_obs - q_A"); a.legend(fontsize=8, frameon=False)
ax[0].set_xlim(-4, 5); ax[0].set_ylabel("P(q_obs | q_A = 6, theta_A)  x P_dom"); ax[1].set_xlim(-5, 25); ax[1].set_ylim(1e-6, 1)
ax[0].set_title("blended kernel (map mean removed): negative median, positive tail", fontsize=10)
ax[1].set_title("same, log scale", fontsize=10)
plt.tight_layout(); plt.savefig(f"{cc.FIG}/kernel.png", dpi=140); plt.close()

# 3. map comparison (q and z marginals) - model x area vs map detections
M = np.load(f"{cc.OUT}/map_compare.npz")
fig, ax = plt.subplots(1, 2, figsize=(12, 4.3))
for a, axis, labs in ((ax[0], 0, ql), (ax[1], 1, zl)):
    x = np.arange(len(labs))
    for H, lab, col, mk in ((M["H_nd"], "maps, non-duplicates", INK, "o"),):
        h = H.sum(0).sum(axis)
        a.errorbar(x, h / M["Nn"].sum(axis) - 1, np.sqrt(h) / M["Nn"].sum(axis), color=col, marker=mk, ls="none", label=lab, ms=5)
    a.plot(x, M["Nb"].sum(axis) / M["Nn"].sum(axis) - 1, color=AQUA, lw=2, marker="s", ms=4, label="blended convolution model")
    a.axhline(0, color=MUTED, lw=1)
    a.set_xticks(x, labs, rotation=45, fontsize=8); a.set_ylabel("relative to naive model")
    a.legend(fontsize=8, frameon=False)
ax[0].set_xlabel("q_obs bin"); ax[1].set_xlabel("z bin (primary)")
ax[0].set_title("map counts include the peak-finding (max over position/scale) excess", fontsize=10)
plt.tight_layout(); plt.savefig(f"{cc.FIG}/map_vs_model.png", dpi=140); plt.close()

# 4. parameter bias, in sigma
fig, ax = plt.subplots(1, 2, figsize=(12, 4.3), sharey=True)
for a, fs in zip(ax, ("1.0", "0.4")):
    B = json.load(open(f"{cc.OUT}/cosmo_bias_fsky{fs}.json"))
    setups = sorted({k.split("|")[1] for k in B})
    x = np.arange(len(setups))
    for off, kind, col, lab in ((-0.2, "naive", ORANGE, "naive likelihood"), (0.0, "eff", BLUE, "naive + 4 free confusion params"),
                                (0.2, "blended", AQUA, "P(D) convolution")):
        om = [B[f"fsky{fs}|{s}|{kind}"]["bias_sigma"][0] for s in setups]
        s8 = [B[f"fsky{fs}|{s}|{kind}"]["bias_sigma"][1] for s in setups]
        a.scatter(x + off, om, color=col, marker="o", s=40, label=f"{lab}: Omega_m", zorder=3)
        a.scatter(x + off, s8, color=col, marker="^", s=40, label=f"{lab}: sigma_8", zorder=3)
    a.axhline(0, color=MUTED, lw=1); a.axhspan(-1, 1, color=MUTED, alpha=0.12, lw=0)
    a.set_xticks(x, [s.split(":")[0] for s in setups]); a.set_title(f"f_sky = {fs}", fontsize=10)
ax[0].set_ylabel("bias / marginalised 1-sigma"); ax[1].legend(fontsize=7, frameon=False, loc="lower right")
plt.tight_layout(); plt.savefig(f"{cc.FIG}/cosmo_bias.png", dpi=140); plt.close()
print("figures written to", cc.FIG)
