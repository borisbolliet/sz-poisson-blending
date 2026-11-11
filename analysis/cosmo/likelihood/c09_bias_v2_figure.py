"""Figure + summary table of the repaired cosmology-bias fits (c05b JSON outputs): bias of (Om0, sigma8, S8cl) in units
of the statistical error of the correct (blended) likelihood at the truth, for naive / naive+4 confusion params."""
import json, glob
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import cclib as cc

R = {}
for f in sorted(glob.glob(f"{cc.OUT}/cosmo_bias_v2_fsky*.json")):
    R.update(json.load(open(f)))
P = {}
for f in sorted(glob.glob(f"{cc.OUT}/cosmo_bias_v2_prior*.json")):
    P.update({k + "|priorshift": v for k, v in json.load(open(f)).items()})
print(f"{'key':40s} {'Om0':>6s} {'s8':>6s} {'S8cl':>6s} | own: {'Om0':>6s} {'s8':>6s} | Dm2l_truth  -2lnL(tot/Pois)  GOFpower")
for k, v in list(R.items()) + list(P.items()):
    print(f"{k:40s} {v['bias_true'][0]:+6.2f} {v['bias_true'][1]:+6.2f} {v['S8cl_bias_true']:+6.2f} |      {v['bias_own'][0]:+6.2f} {v['bias_own'][1]:+6.2f} |"
          f" {v['dm2l_truth']:7.2f}     {v['m2l_tot']:6.2f}/{v['m2l_poisson']:6.2f}   {v['gof_power']:.2f}")
fig, ax = plt.subplots(1, 2, figsize=(12, 4.3), sharey=True)
cols = {"naive": "#d55e00", "eff": "#0072b2", "effexp": "#56b4e9"}
for a, fs in zip(ax, ("1.0", "0.4")):
    for j, s in enumerate(("S0", "S1", "S2")):
        for off, kind in ((-0.22, "naive"), (0.0, "eff"), (0.22, "effexp")):
            k = f"fsky{fs}|{s}|{kind}"
            if k not in R:
                continue
            b = R[k]["bias_true"]
            a.scatter(j + off, b[0], color=cols[kind], marker="o", s=40, zorder=3, label=f"{kind}: Omega_m" if j == 0 else None)
            a.scatter(j + off, b[1], color=cols[kind], marker="^", s=40, zorder=3, label=f"{kind}: sigma_8" if j == 0 else None)
            a.scatter(j + off, R[k]["S8cl_bias_true"], color=cols[kind], marker="s", s=25, zorder=3, label=f"{kind}: S8cl" if j == 0 else None)
    a.axhline(0, color="0.5", lw=1); a.axhspan(-1, 1, color="0.5", alpha=0.12, lw=0)
    a.set_xticks([0, 1, 2], ["S0", "S1", "S2"]); a.set_title(f"f_sky = {fs}", fontsize=10)
ax[0].set_ylabel("bias / sigma (blended likelihood, at truth)")
ax[1].legend(fontsize=7, frameon=False, loc="lower right", ncol=1)
plt.tight_layout(); plt.savefig(f"{cc.FIG}/cosmo_bias_v2.png", dpi=140); plt.close()
print("wrote", f"{cc.FIG}/cosmo_bias_v2.png")
