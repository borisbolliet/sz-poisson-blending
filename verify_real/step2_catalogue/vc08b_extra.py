"""vc08b: (1) step stability of my Fisher / linearised S8 (steps x1, x2, x0.5; configs a, d, baseline q5) and, for configs
c and e, my Fisher sigma(S8) with the build's own finite-difference steps (cosmolib2.STEP2 values, typed here);
(2) maximum goodness-of-fit power over the MMF setups (linearised chi2_min) for configs a and b-e, SO40 and full sky."""
import os, sys, json
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
os.environ["VC_THREADS"] = "1"
import numpy as np
import vclib as V, vcmodel as VM

M = json.load(open(f"{HERE}/out/fits_mine.json"))
print("(1) step stability (baseline_f1, q5, build's Delta): sigma(S8) and linearised S8 bias for steps x1 / x2 / x0.5")
for r in M:
    if r["kind"] == "steps":
        print(f"   {r['survey']:4s} {r['config']}: " + " | ".join(f"x{m}: sigma {r[f'x{m}']['s_S8']:.5f}, lin {r[f'x{m}']['lin_S8']:+.3f}"
                                                          for m in (1.0, 2.0, 0.5)))
B2 = np.array([2e-3, 2e-3, 6e-3, 5e-3, 5e-3])          # the build's STEP2 (cosmolib2.py), typed here
mod = VM.CountModel("baseline", 5.0)
mu_t = mod.counts(V.FID)
for cfg in ("c", "e", "d"):
    for lab, st in (("my steps", np.array([0.001, 0.001, 0.003, 0.003, 0.003])), ("build steps", B2), ("build steps / 4", B2 / 4)):
        Fo = VM.Fit(mod, mu_t * 0.4, cfg, V.FID, 0.4)
        F, J, mu = Fo.fisher(V.FID[Fo.free], st[Fo.free])
        s, sS8, g, C = Fo.sigmas(F)
        print(f"   config {cfg} SO40, {lab:15s}: sigma(S8) {sS8:.5f}, sigma(Om) {s[0]:.5f}")
print("\n(2) maximum goodness-of-fit power (95% test; linearised chi2_min) over MMF setups, q5 and q6")
lin = [r for r in M if r["kind"] == "lin" and r["setup"] != "white"]
for sv in ("SO40", "full"):
    for grp, cf in (("a", ("a",)), ("b-e", ("b", "c", "d", "e"))):
        for dn in ("build", "mine"):
            rr = [r for r in lin if r["survey"] == sv and r["config"] in cf]
            best = max(rr, key=lambda r: r[f"power_{dn}"])
            print(f"   {sv:4s} config {grp:3s} on {dn:5s} Delta: max power {best[f'power_{dn}']:.3f} ({best['setup']}, {best['config']}, "
                  f"q{best['qcut']:.0f}; chi2_min {best[f'chi2lin_{dn}']:.2f})")
