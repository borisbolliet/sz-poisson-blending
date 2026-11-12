"""f08 (step-3b verification, lens 'fits'): small checks of numbers the fits depend on.
(1) r = sigma_finder / sigma_Y after the iteration (mean over the 324 Q2 tiles) at 0.5', 4.45', 15' per variant, from the
    stored per-tile sigma_finder (halos_VARIANT_MODE.npz 'sig1') and MY sigma_Y (claim F2); the same for the 1,224 new
    tiles (production), which the analyst's rbar does not use.
(2) the analyst's sigma_eff choice: my model's q > 5 counts with rbar from the Q2 tiles vs from all 1,548 tiles (K13-like).
Usage: f08_misc.py
"""
import os, sys
os.environ["VF_THREADS"] = "1"
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import numpy as np
import f00lib as L

print("(1) r = sigma_finder / sigma_Y after the iteration, mean over tiles, at 0.5' / 4.45' / 15' (tile-to-tile std at 0.5')")
for v in ("prod", "szifi3000", "szifi12000"):
    for mode in L.MODES:
        V = np.load(os.path.join(L.D3B, f"halos_{v}_{mode}.npz"))
        r = V["sig1"] / L.sigma_Y(mode, L.THETAS15)[None, :]
        print(f"  {v:10s} {mode:8s} Q2 tiles ({r.shape[0]}): {r.mean(0)[0]:.4f} / {r.mean(0)[9]:.4f} / {r.mean(0)[14]:.4f} (std {r.std(0)[0]:.4f}); "
              f"scale 9 = {L.THETAS15[9]:.3f}'", flush=True)
        if v == "prod":
            fs = sorted(f for f in os.listdir(os.path.join(L.D3B, "newtiles")) if f.endswith(".npz"))
            s1 = np.array([np.load(os.path.join(L.D3B, "newtiles", f))[mode + "_sig1"] for f in fs])
            rn = s1 / L.sigma_Y(mode, L.THETAS15)[None, :]
            print(f"  {'':10s} {'':8s} new tiles ({rn.shape[0]}): {rn.mean(0)[0]:.4f} / {rn.mean(0)[9]:.4f} / {rn.mean(0)[14]:.4f}; "
                  f"all 1548: {np.concatenate([r, rn]).mean(0)[0]:.4f}", flush=True)
            Kt = L.KTable(os.path.join(L.D3B, f"kernel_prod_{mode}.npz"))
            for qc in (5.0,):
                m1 = L.KModel(mode, qc, r.mean(0), Kt, how="alt").counts(L.FID)
                m2 = L.KModel(mode, qc, np.concatenate([r, rn]).mean(0), Kt, how="alt").counts(L.FID)
                print(f"  (2) my model, q > 5: rbar of all 1548 tiles / rbar of the Q2 tiles - 1 = {m2.sum() / m1.sum() - 1:+.4%}; per bin max "
                      f"|rel| (mu > 10) {np.abs(m2 / m1 - 1)[m1 > 10].max():.2e}", flush=True)
print(f"peak RSS {L.rss_gb():.2f} GB")

print("(3) from the stored per-halo tables of the 324 Q2 tiles: credited halos (merge none / closest 1' / fof 10'), catalogue entries "
      "with q > 5 (sum of 'n5'), 'no real cluster' entries (nr_q > 5) as a fraction of them (claims F3, F4)")
for v in ("prod", "szifi3000", "szifi12000"):
    for mode in L.MODES:
        V = np.load(os.path.join(L.D3B, f"halos_{v}_{mode}.npz"))
        cr = [int(np.isfinite(V["qobs_" + m]).sum()) for m in ("none", "closest1", "fof10")]
        n5 = int(V["n5"].sum()); nr = int((V["nr_q"] > 5).sum())
        print(f"  {v:10s} {mode:8s}: credited {cr[0]} / {cr[1]} / {cr[2]}; entries q > 5: {n5}; no-real entries {nr} = {100 * nr / n5:.2f}%", flush=True)
