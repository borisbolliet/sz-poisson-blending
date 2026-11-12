"""w07c: how much of the map-route chi2_min is the map vector's own sampling noise?  For a data vector N0 + Delta_map
whose Delta_map carries bootstrap noise of variance var_i per bin, the Asimov deviance gains about
sum_i f_sky var_i / N0_i (minus what the free parameters absorb).  Compared with chi2_min of the map-route and build fits."""
import os, sys, json
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = "/Users/boris/GitHub/sz-poisson-blending"
d = np.load(os.path.join(HERE, "out", "w04_delta.npz"))
fits = json.load(open(os.path.join(HERE, "out", "w04_fits.json")))
bf = json.load(open(f"{ROOT}/data_real/step2/fits.json"))
out = open(os.path.join(HERE, "w07c_noisechi2.log"), "w")
def log(s):
    print(s); out.write(s + "\n")
for nm, b in (("baseline_f1", "baseline_f1"), ("goal_f1", "goal_f1"), ("white_f1", "white")):
    t = np.load(f"{ROOT}/data_real/step2/truth_{b}.npz")
    N0 = t["N0_q5"]; var = d[f"{nm}_q5_Dboot"].astype(float).var(0)
    for sv, fs in (("SO40", 0.4), ("full", 1.0)):
        extra = float(np.sum(fs * var / np.maximum(N0, 1e-9)))
        for cn in ("a", "d"):
            m = [r for r in fits if r["setup"] == nm and r["config"] == cn and r["survey"] == sv and r["qcut"] == 5.0][0]
            bb = [r for r in bf if r["setup"] == b and r["config"] == cn and r["survey"] == sv and r["qcut"] == 5.0
                  and r["route"] == "cat" and r.get("start", "truth") == "truth"][0]
            log(f"   {nm:12s} {sv:4s} {cn}: map chi2_min {m['dev']:.2f} = build {bb['dev']:.2f} + noise term ~{extra:.2f} "
                f"(sum f_sky var/N0 over 144 bins) -> {bb['dev'] + extra:.2f}")
out.close()
