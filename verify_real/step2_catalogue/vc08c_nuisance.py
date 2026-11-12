"""vc08c: config-d nuisance shifts and chi2_min (claim C13) and the config-d/e S8 ranges over fdom and map-sigma variants
(claims C7, C10), from out/fits_mine.json (ML where available, else linearised)."""
import os, sys, json
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
M = json.load(open(f"{HERE}/out/fits_mine.json"))
fits = {(r["setup"], r["qcut"], r["survey"], r["config"], r["data"]): r for r in M if r["kind"] == "fit"}
lins = {(r["setup"], r["qcut"], r["survey"], r["config"]): r for r in M if r["kind"] == "lin"}
print("(C13) config d, q5: bias in sigma of Om / s8 / (1-b) / alpha / slnY ; S8 ; chi2_min (my model + fitter)")
for s in ("baseline_f1", "goal_f1"):
    for sv in ("SO40", "full"):
        for dn in ("build", "mine"):
            r = fits[(s, 5.0, sv, "d", dn)]
            mc = r.get("mc_err", {})
            print(f"   {s:12s} {sv:4s} on {dn:5s}: " + " / ".join(f"{p} {r['ml'][p]:+.2f}" + (f"+-{mc[p]:.2f}" if p in mc else "")
                                                          for p in ("Om", "s8", "omb", "alpha", "slnY"))
                  + f" ; S8 {r['ml']['S8']:+.3f} ; chi2_min {r['dev']:.2f}")


def val(s, sv, c, dn):
    r = fits.get((s, 5.0, sv, c, dn))
    if r is not None:
        return r["ml"]["S8"], "ML"
    return lins[(s, 5.0, sv, c)][f"lin_{dn}"]["S8"], "lin"


for lab, setups in (("C7 fdom 0.5-2, configs d, e", ["baseline_f0.5", "baseline_f1", "baseline_f2", "goal_f0.5", "goal_f1", "goal_f2"]),
                    ("C10 map-sigma, configs d, e", ["baseline_effi", "baseline_effii", "goal_effi", "goal_effii"])):
    for sv in ("SO40", "full"):
        for dn in ("build", "mine"):
            v = [(val(s, sv, c, dn), s, c) for s in setups for c in ("d", "e")]
            lo = min(v, key=lambda t: t[0][0]); hi = max(v, key=lambda t: t[0][0])
            print(f"({lab}) {sv:4s} on {dn:5s} Delta: {lo[0][0]:+.2f} ({lo[1]} {lo[2]}, {lo[0][1]}) to {hi[0][0]:+.2f} ({hi[1]} {hi[2]}, {hi[0][1]})")
