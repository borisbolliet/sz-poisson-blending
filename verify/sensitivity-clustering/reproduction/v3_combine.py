"""Combine the polar-cap MC results (v3_blend_patch.py) and compare with the claimed numbers.
Errors: delete-one-cap jackknife over independent caps (includes cap-to-cap ybar fluctuations)."""
import glob, json, os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import indep_model as im
ARC = im.ARC


def jk(num, den):
    num = np.asarray(num, float); den = np.asarray(den, float)
    Nn, Nd = num.sum(), den.sum(); n = len(num)
    rj = (Nn - num) / (Nd - den)
    return Nn / Nd, np.sqrt((n - 1) / n * np.sum((rj - rj.mean()) ** 2))


CL = {  # claimed (q=5) B2(0.1) c1, B3 c1, B2 c0, B3 c0, sat ; errors from aggregate.log
    "base": [(0.1629, .0006), (0.0813, .0004), (0.5162, .0006), (0.2588, .0005), (0.0059, .0001)],
    "noise_x0.5": [(0.2149, .0004), (0.0951, .0003), (0.7025, .0003), (0.3356, .0003), (0.0140, .0001)],
    "noise_x2": [(0.1103, .0008), (0.0622, .0006), (0.3358, .0010), (0.1959, .0008), (0.0023, .0001)],
    "floor_1e14": [(0.1280, .0005), (0.0607, .0003), (0.1847, .0006), (0.0984, .0004), (0.0058, .0001)],
    "beam_5": [(0.2248, .0012), (0.1068, .0008), (0.8993, .0006), (0.4649, .0009), (0.0107, .0003)]}
MUCL = {"base": (1.0282, .0028), "noise_x0.5": (1.7139, .0038), "noise_x2": (0.6370, .0037), "floor_1e14": (0.4272, .0027), "beam_5": (2.0081, .0058)}
rat = {}
for beam in (1.4, 5.0):
    fs = sorted(glob.glob(os.path.join(im.CACHE, f"v3_beam{beam:.1f}_seed*.json")))
    if not fs:
        continue
    Rs = [json.load(open(f)) for f in fs]
    print(f"\n===== beam {beam}': {len(fs)} independent caps (3730 deg2 each)")
    for cname in (["base", "noise_x0.5", "noise_x2", "floor_1e14"] if beam == 1.4 else ["base"]):
        key = cname if beam == 1.4 else "beam_5"
        g = lambda k: np.array([R[cname][k] for R in Rs], float)
        vals = [jk(g("b2_1"), g("det1")), jk(g("b3_1"), g("det1")), jk(g("b2_0"), g("det0")), jk(g("b3_0"), g("det0")), jk(g("sat"), g("n_single"))]
        rat[key] = vals
        names = ["B2(0.1) c1", "B3 c1", "B2(0.1) c0", "B3 c0", "sat"]
        print(f"  {key}: N_single total {int(g('n_single').sum())}, N_det_b c1 {int(g('det1').sum())}")
        for nm, (v, e), (c, ce) in zip(names, vals, CL[key]):
            print(f"     {nm:11s}: own {v:.4f} +- {e:.4f} | claimed {c:.4f} +- {ce:.4f} | diff {(v-c)/np.hypot(e,ce):+.2f} sigma ({v/c:.3f}x)")
        m, e = jk(g("sumD_single"), g("n_single")); mu = g("summu_single").sum() / g("n_single").sum()
        c, ce = MUCL[key]
        print(f"     <delta>_single own {m:.4f} +- {e:.4f} vs own Campbell mu {mu:.4f} ({(m-mu)/e:+.2f} sigma); claimed MC {c:.4f} ({(m-c)/np.hypot(e,ce):+.2f} sigma)")
        if cname == "base":
            for tag, lab in (("cl06", "theta_B=0.6' W"), ("cly", "Y-weighted W")):
                b2c, b3c = jk(g(f"b2_{tag}"), g(f"det_{tag}")), jk(g(f"b3_{tag}"), g(f"det_{tag}"))
                b2p, b3p = vals[0][0], vals[1][0]
                print(f"     clustering mean-field ({lab}): B2(0.1) {b2p:.4f}->{b2c[0]:.4f} (x{b2c[0]/b2p:.3f}), B3 {b3p:.4f}->{b3c[0]:.4f} (x{b3c[0]/b3p:.3f}), "
                      f"N_det_b x{g(f'det_{tag}').sum()/g('det1').sum():.4f}  [claimed x1.121, x1.140, x1.039]")
            if beam == 1.4:
                k_ov = g("hasov_single").sum() / g("n_single").sum()
                print(f"     P(single det has 3D-overlapping M>1e13 halo): own MC {k_ov:.3e} +- {np.sqrt(g('hasov_single').sum())/g('n_single').sum():.1e} "
                      f"(claimed MC 2.180e-3, analytic 2.167e-3)")
                b2o = jk(g("b2ovl"), g("det1"))
                print(f"     B2ovl(0.1) own {b2o[0]:.2e} +- {b2o[1]:.1e} (claimed 2.01e-4 +- 1.9e-5)")
            for tag, lab in (("b", "blended (c=1)"), ("s", "single")):
                ncen = g(f"pairs_{tag}_ncen"); ge1 = np.array([R[cname][f"pairs_{tag}_ge1"] for R in Rs], float)
                Sig = g(f"pairs_{tag}_N55").sum() / (len(Rs) * 2 * np.pi * (1 - np.sin(np.radians(55))))
                out = []
                for k, th in enumerate((0.5, 1.0, 2.0, 5.0)):
                    r, e = jk(ge1[:, k], ncen); pp = 1 - np.exp(-Sig * 2 * np.pi * (1 - np.cos(th * ARC)))
                    out.append(f"{th:g}': {r:.3e}+-{e:.1e} vs Poisson {pp:.3e} (x{r/pp:.2f})")
                print(f"     pairs, {lab}: " + "; ".join(out))
if "base" in rat:
    print("\nratios to base (own MC) vs claimed:")
    clr = {"noise_x0.5": (1.319, 1.170), "noise_x2": (0.677, 0.764), "floor_1e14": (0.786, 0.746), "beam_5": (1.380, 1.313)}
    for k, (c2, c3) in clr.items():
        if k in rat:
            r2 = rat[k][0][0] / rat["base"][0][0]; r3 = rat[k][1][0] / rat["base"][1][0]
            print(f"   {k:11s}: B2 x{r2:.3f} (claimed {c2}), B3 x{r3:.3f} (claimed {c3})")
