"""Repair phase figure: detected-centred P(>=1 other detected within theta) relative to the Palm/Poisson prediction
1-exp(-mu), for det4/5/6, in four independent sky sets: original 40 full skies (where a ~2 sigma deficit at 10'-15'
was flagged), 80 new full skies, 1000 + 1000 new floor-1e14 skies.  Error bars: sky-to-sky scatter."""
import os, glob, json
import numpy as np
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from q1_common import *
from q1_mc_fullsky import FINE
from q1_palm_check import THP
NB = json.load(open(os.path.join(DATA, "nbar.json")))
f = lambda t: np.sin(np.asarray(t) * ARCMIN / 2) ** 2
def rel(num, den, P):
    r = num - P * den; return num.sum() / den.sum() / P - 1, r.std(ddof=1) * np.sqrt(len(r)) / den.sum() / P
full = {int(os.path.basename(x)[4:-4]): np.load(x, allow_pickle=True).item() for x in glob.glob(os.path.join(DATA, "fullsky", "sky_*.npy"))}
TH = [1, 2, 3, 5, 7.5, 10, 15, 20, 30]
fig, axs = plt.subplots(1, 3, figsize=(15, 4.3), sharey=True)
for ax, q in zip(axs, QCUTS):
    for k, (lab, seeds) in enumerate([("40 original full skies", range(1001, 1041)), ("80 new full skies", range(1041, 1121))]):
        N = np.array([full[s]["npop"][f"det{q}"] for s in seeds], float)
        out = []
        for t in TH:
            j = int(np.argmin(abs(FINE - t))); P = 1 - np.exp(-NB[f"det{q}"] * f(t))
            g = np.array([full[s]["cdf"][f"det{q}|det{q}"][0, j] for s in seeds]) * N
            out.append(rel(g, N, P))
        out = np.array(out); ax.errorbar(np.array(TH) * (1 + 0.03 * k), out[:, 0], out[:, 1], fmt="os"[k], ms=4, label=lab)
    for k, d in enumerate(["palm_new", "palm_rep"]):
        Ps = [np.load(x, allow_pickle=True).item() for x in sorted(glob.glob(os.path.join(DATA, d, "sky_*.npy")))]
        N = np.array([s[f"N{q}"] for s in Ps], float); out = []
        for j, t in enumerate(THP):
            P = 1 - np.exp(-NB[f"det{q}_floor1e14"] * f(t)); g = np.array([s[f"ge1_det{q}"][j] for s in Ps], float)
            out.append(rel(g, N, P))
        out = np.array(out); ax.errorbar(THP * (1 + 0.03 * (k + 2)), out[:, 0], out[:, 1], fmt="^v"[k], ms=4,
                                         label=f"{len(Ps)} floor-1e14 skies ({'40001-41000' if k == 0 else '41001-42000'})")
    ax.axhline(0, color="k", lw=0.8); ax.set(xscale="log", ylim=(-0.03, 0.03), xlabel="theta [arcmin]",
                                             title=f"detected centre, other detected (q_obs>{q})")
axs[0].set_ylabel("MC / [1-exp(-mu)] - 1  for P(>=1)"); axs[0].legend(fontsize=7)
fig.tight_layout(); fig.savefig(os.path.join(FIG, "palm_check_new_seeds.png"), dpi=130)
print("written", os.path.join(FIG, "palm_check_new_seeds.png"))
