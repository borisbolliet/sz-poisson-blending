"""Figures for the cosmology dependence of blending -> figures/cosmo/blend_dependence_*.png"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from c02_grid import S8, OM, OMB, SL, FIDP

D = "/Users/boris/GitHub/sz-poisson-blending/data_cosmo"
FIG = "/Users/boris/GitHub/sz-poisson-blending/figures/cosmo"
os.makedirs(FIG, exist_ok=True)
G = dict(np.load(f"{D}/c02_grid.npz"))
PN = [r"$\sigma_8$", r"$\Omega_m$", r"$1-b$", r"$\sigma_{\ln Y}$"]
NODES = [S8, OM, OMB, SL]
QE = np.array([5, 6, 7, 8, 10, 12.5, 15, 20, 30, 50, 80])


def gidx(p):
    return [tuple(np.round(q, 6)) for q in G["params"]].index(tuple(np.round(p, 6)))


fig, ax = plt.subplots(2, 4, figsize=(15, 7))
for a in range(4):
    js = []
    for v in NODES[a]:
        p = list(FIDP); p[a] = v; js.append(gidx(p))
    x = np.array(NODES[a])
    for key, lab, c in (("ytot", r"$\bar y$", "C0"), ("mu_det", r"$\langle\mu\rangle_{\rm det}$", "C1"),
                        ("var5_det", r"$\langle{\rm Var}(\delta|x<5)\rangle_{\rm det}$", "C2")):
        y = G[key][js] / G[key][js[1]]
        ax[0, a].loglog(x, y, "o-", color=c, label=lab)
    ax[0, a].set_xlabel(PN[a]); ax[0, a].set_ylabel("relative to fiducial")
    ax[0, a].legend(fontsize=8)
    for key, lab, c in (("ratio", r"$N_{\rm det,b}/N_{\rm single}$", "k"), ("B2_10", "B2(>10%)", "C0"),
                        ("B3", "B3 boosted", "C1"), ("lost", "lost", "C2"), ("ybias", r"$\langle(\delta-\mu)/q_A\rangle$", "C3")):
        y = G[key][js] - (1 if key == "ratio" else 0)
        ax[1, a].plot(x, y, "o-", color=c, label=lab + (" $-1$" if key == "ratio" else ""))
    ax[1, a].axhline(0, color="0.7", lw=0.8)
    ax[1, a].set_xlabel(PN[a]); ax[1, a].set_ylabel("fraction")
ax[1, 0].legend(fontsize=7)
fig.suptitle("Blending at q>5 vs one parameter (others fiducial: s8=0.81, Om=0.31, 1-b=0.8, slnY=0.17)")
fig.tight_layout(); fig.savefig(f"{FIG}/blend_dependence_params.png", dpi=130)

fig, ax = plt.subplots(1, 2, figsize=(11, 4))
qc = np.sqrt(QE[:-1] * QE[1:])
for a, axx in ((0, ax[0]), (2, ax[1])):
    for v, c in zip(NODES[a], ("C0", "k", "C3")):
        p = list(FIDP); p[a] = v; j = gidx(p)
        C = G["Nzq_b"][j].sum(0) / G["Nzq_n"][j].sum(0)
        axx.semilogx(qc, C, "o-", color=c, label=f"{PN[a]}={v}")
    axx.axhline(1, color="0.7", lw=0.8)
    axx.set_xlabel(r"$q_{\rm obs}$ bin"); axx.set_ylabel(r"$N_{\rm blended}/N_{\rm single}$ (all z)")
    axx.legend()
fig.tight_layout(); fig.savefig(f"{FIG}/blend_dependence_Cq.png", dpi=130)
print("figures written:", f"{FIG}/blend_dependence_params.png", f"{FIG}/blend_dependence_Cq.png")
