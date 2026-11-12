"""v06b: SZiFi's literal zero-padded ('same' mode) convolution on the build's periodic tiles: where do the extra
detections sit?  Counts of q > 4.5 and q > 5 detections by distance from the tile edge, SZiFi-convention finder with the
linear convolution (out/v04_lin_*) vs the periodic one (out/v04_per_*) vs the build (stored), final catalogues.
"""
import os, sys
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import numpy as np
import vsz_lib as V

OUT = V.OUTD
N = V.N
EDGES = [0.0, 2.5, 10.0, 30.0, 60.0, 301.0]
TILES = [0, 33, 66, 99, 132, 165, 198, 231, 264, 297]


def dist_edge(y, x):
    return np.minimum.reduce([y, x, N - 1 - y, N - 1 - x]) * V.PIXA


for src, pids in (("tile", TILES), ("noise", [0, 20, 40])):
    for mode in V.MODES:
        rows = {}
        for lab in ("build final", "SZiFi per it1", "SZiFi lin it0", "SZiFi lin it1"):
            rows[lab] = np.zeros((2, len(EDGES) - 1))
        n = 0
        for p in pids:
            fl = os.path.join(OUT, f"v04_lin_{mode}_{src}{p:03d}.npz")
            fp = os.path.join(OUT, f"v04_per_{mode}_{src}{p:03d}.npz")
            if not (os.path.exists(fl) and os.path.exists(fp)):
                continue
            n += 1
            L, P = dict(np.load(fl)), dict(np.load(fp))
            if src == "tile":
                st = V.load_stored_tile(p)
                B = (st[f"{mode}_it1_iy"] * 1.0, st[f"{mode}_it1_ix"] * 1.0, st[f"{mode}_it1_q"])
            else:
                st = V.load_npz(os.path.join(V.DATA3, "noise_only", f"noise_{p:03d}.npz"))
                B = (st[f"{mode}_whitened_it1_iy"] * 1.0, st[f"{mode}_whitened_it1_ix"] * 1.0, st[f"{mode}_whitened_it1_q"])
            cats = {"build final": B,
                    "SZiFi per it1": (P["det1_n0"] - 0.5, P["det1_n1"] - 0.5, P["det1_q"]),
                    "SZiFi lin it0": (L["det0_n0"] - 0.5, L["det0_n1"] - 0.5, L["det0_q"]),
                    "SZiFi lin it1": (L["det1_n0"] - 0.5, L["det1_n1"] - 0.5, L["det1_q"])}
            for lab, (y, x, q) in cats.items():
                d = dist_edge(y, x)
                for k, qc in enumerate((4.5, 5.0)):
                    rows[lab][k] += np.histogram(d[q > qc], bins=EDGES)[0]
        if n == 0:
            continue
        area = np.array([((N - 2 * a / V.PIXA) ** 2 - (N - 2 * b / V.PIXA) ** 2) * (V.PIXA / 60) ** 2
                         for a, b in zip(EDGES[:-1], np.minimum(EDGES[1:], 300.0))]) * n
        print(f"\n{mode}, {n} {src} tiles: detections per 100 deg^2 by distance from the tile edge "
              f"{EDGES[:-1]}..{EDGES[-1]}' (q > 4.5 | q > 5)")
        for lab, r in rows.items():
            print(f"  {lab:15s}: " + "  ".join(f"{100 * a / ar:7.1f} | {100 * b / ar:6.1f}" for a, b, ar in zip(r[0], r[1], area))
                  + f"   total q>5 {int(r[1].sum())}")
