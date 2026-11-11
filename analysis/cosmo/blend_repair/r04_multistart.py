"""Multi-start spread of the converged fits (r02_fits.json), with sigma_lnY folded to |sigma_lnY|
(the model depends on sigma_lnY only through exp(sigma_lnY x) with symmetric x, so -s and +s are identical)."""
import json, numpy as np
J = json.load(open("/Users/boris/GitHub/sz-poisson-blending/data_cosmo/r02_fits.json"))
res = J["res"]
groups = {}
for r in res:
    if r["kind"] in ("ignore", "fixed"):
        groups.setdefault((r["kind"], r["name"], r["case"], r["fs"], r["mask"]), []).append(r)
worst = (0, None); worst_raw = (0, None)
for k, rr in groups.items():
    if len(rr) < 2:
        continue
    T = np.array([r["theta"] for r in rr]); raw = np.abs(T - T[0]).max()
    T[:, 3] = np.abs(T[:, 3])
    sig = np.array(J["sig"][f"{k[2]}_{k[3]}"])
    free = [0, 1] if k[2] == "A" else [0, 1, 2, 3]
    d = (np.abs(T - T[0])[:, free] / sig).max()
    c2 = np.ptp([r["chi2"] for r in rr])
    if d > worst[0]: worst = (d, k, c2)
    if raw > worst_raw[0]: worst_raw = (raw, k, [r["theta"][3] for r in rr])
print(f"multi-start groups: {sum(len(v) > 1 for v in groups.values())}")
print(f"largest raw parameter difference between starts: {worst_raw[0]:.4f} in {worst_raw[1]}; sigma_lnY values {np.round(worst_raw[2], 4)}")
print(f"after folding sigma_lnY -> |sigma_lnY|: max spread {worst[0]:.4f} sigma in {worst[1]}; chi2 spread there {worst[2]:.1e}")
print(f"fits with Newton decrement > 1e-6: {sum(r['dec'] > 1e-6 for r in res)} of {len(res)}; max decrement {max(r['dec'] for r in res):.1e}")
