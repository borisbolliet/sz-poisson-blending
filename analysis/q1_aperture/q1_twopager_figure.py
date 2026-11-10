"""Readable version of the Q1 figure for the two-pager.

P(exactly n OTHER detected clusters within theta of a detected q>5 cluster):
dots = 120 simulated full skies, lines = Poisson prediction (nothing fitted).
Output: twopager/f1_readable.png
"""
import os, glob, json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from q1_common import DATA, ROOT, THETAS, KMAX, mu_of, P_N

NB = json.load(open(os.path.join(DATA, "nbar.json")))
files = sorted(glob.glob(os.path.join(DATA, "fullsky", "sky_*.npy")))
S = [np.load(f, allow_pickle=True).item() for f in files]
R = len(S)
key, pop = "det5|det5", "det5"
NMAX = 3  # show n = 0..3


def PN_sky(cdf):
    F = cdf[:, :]  # P(N>=k), k=1..KMAX, on the FINE grid -> pick THETAS below
    return F


from q1_mc_fullsky import FINE
iT = np.array([np.argmin(abs(FINE - t)) for t in THETAS])
P = []
for s in S:
    F = s["cdf"][key][:, iT]
    p = np.empty((NMAX + 1, len(THETAS)))
    p[0] = 1 - F[0]
    for n in range(1, NMAX + 1):
        p[n] = F[n - 1] - F[n]
    P.append(p)
P = np.array(P)
m, sem = P.mean(0), P.std(0, ddof=1) / np.sqrt(R)

tt = np.geomspace(0.3, 100, 400)
mm = mu_of(NB[pop], tt)
cols = ["#1b1b1b", "#2a6fdb", "#e07b00", "#2a9d5c"]
labels = ["no other cluster", "exactly 1 other", "exactly 2 others", "exactly 3 others"]

fig, ax = plt.subplots(figsize=(5.6, 4.4))
for n in range(NMAX + 1):
    ax.plot(tt, P_N(n, mm), color=cols[n], lw=1.8)
    ok = m[n] > 0
    ax.errorbar(THETAS[ok], m[n][ok], sem[n][ok], fmt="o", ms=4.5, mfc="white", mew=1.4, color=cols[n])
ax.set(xscale="log", yscale="log", ylim=(1e-6, 2), xlim=(0.4, 80),
       xlabel="aperture radius around a detected cluster [arcmin]",
       ylabel="probability")
ax.set_title(f"How many other detected clusters (q>5) lie nearby?", fontsize=11)
ax.plot([], [], "o", mfc="white", mec="0.3", label=f"simulation ({R} full skies)")
ax.plot([], [], "-", color="0.3", label="Poisson prediction")
ax.legend(fontsize=8.5, loc="lower right", frameon=True, framealpha=0.9)
for (x, y, s, c) in [(0.5, 0.45, "no other cluster", cols[0]), (0.75, 2.2e-3, "exactly 1 other", cols[1]),
                     (0.6, 1.6e-4, "exactly 2 others", cols[2]), (8.5, 2.5e-5, "exactly 3 others", cols[3])]:
    ax.text(x, y, s, color=c, fontsize=9, fontweight="bold")
i2 = np.argmin(abs(THETAS - 2))
ax.annotate(f"at 2': {100*m[1][i2]:.1f}% have\none detected neighbour", xy=(2, m[1][i2]), xytext=(0.5, 3e-2),
            fontsize=8.5, arrowprops=dict(arrowstyle="->", color="0.3"))
ax.grid(alpha=0.25, which="major")
fig.tight_layout()
out = os.path.join(ROOT, "twopager", "f1_readable.png")
fig.savefig(out, dpi=200)
print("wrote", out, "R =", R)
for n in range(NMAX + 1):
    z = (m[n] - P_N(n, mu_of(NB[pop], THETAS))) / np.where(sem[n] > 0, sem[n], np.inf)
    print(n, "max|z| =", np.nanmax(np.abs(z[sem[n] > 0])).round(2))
