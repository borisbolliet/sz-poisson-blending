"""V5: re-read the analysis' stored pass files (data/robustness/pass_*.npz) to check C6 at 0.5' (below the beam)
and the per-sky spread behind the C4 despali16 outlier; also the q<0.5 exclusion in the MC N_det."""
import glob, json, os
import numpy as np
DD = "/Users/boris/GitHub/sz-poisson-blending/data/robustness"
ARC = np.pi / 180 / 60


def jk(num, den):
    Nn, Nd = num.sum(), den.sum(); ok = den > 0
    rj = (Nn - num[ok]) / (Nd - den[ok]); n = ok.sum()
    return Nn / Nd, np.sqrt((n - 1) / n * np.sum((rj - rj.mean()) ** 2))


for pname, cname in (("fam", "base"), ("fam", "noise_x0.5"), ("fam", "noise_x2"), ("beam5", "beam_5"), ("planck", "planck_like")):
    fs = sorted(glob.glob(f"{DD}/pass_{pname}_seed*.npz"))
    get = lambda k: np.concatenate([np.load(f)[f"{cname}|{k}"] for f in fs])
    den = get("detb1_5"); Nb = den.sum() / len(fs); Sig = Nb / (4 * np.pi)
    out = []
    for th in (0.5, 1.0, 2.0, 5.0):
        r, e = jk(get(f"B1b_{th:g}_1_5"), den)
        pp = 1 - np.exp(-Sig * 2 * np.pi * (1 - np.cos(th * ARC)))
        out.append(f"{th:g}': {r:.3e}+-{e:.1e} vs {pp:.3e} (x{r/pp:.2f}, {(r-pp)/e:+.1f} sigma)")
    print(f"{cname:12s} blended-catalogue P(>=1 within theta) [n_sky={len(fs)}]: " + "; ".join(out))

print("\nper-sky mean neighbour boost of single detections (q>5): <delta>, mu_sky, difference")
for pname, cname in (("fam", "base"), ("des16", "hmf_despali16"), ("boc16", "hmf_bocquet16"), ("scat03", "scatter_0.3"), ("beam5", "beam_5")):
    fs = sorted(glob.glob(f"{DD}/pass_{pname}_seed*.npz"))
    md = np.array([np.load(f)[f"{cname}|mean_delta_single5"] for f in fs])
    d = md[:, 0] - md[:, 1]
    print(f"   {cname:14s} <delta>-mu_sky per sky: {np.round(d, 4)}; mean {d.mean():+.4f} +- {d.std(ddof=1)/np.sqrt(len(d)):.4f} "
          f"(per-sky rms {d.std(ddof=1):.4f})")
# pooled estimate of per-sky rms of (<delta>-mu) across all 1.4' tinker/HMF configs with 5 skies, scaled
