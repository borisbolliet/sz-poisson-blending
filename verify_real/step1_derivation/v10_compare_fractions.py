"""v10: compare my direct per-object fractions (v08, one or more fresh skies) with the builder's stored FFT P(D)
rows (data_real/pd_fractions_*.npz, read as data) and with the numbers quoted in the builder's claims.
Usage: python v10_compare_fractions.py v08_results_seedA_M060.npy [v08_results_seedB_M060.npy ...]
"""
import os, sys
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import numpy as np
import vlib as V

files = sys.argv[1:]
runs = [np.load(os.path.join(HERE, f), allow_pickle=True).item() for f in files]
QC = [4.0, 5.0, 6.0, 7.0]
KEYS = [("N_single", "N_single"), ("N_det_b", "N_det_b"), ("B2_10", "B2_0.1"), ("B2_20", "B2_0.2"), ("B2neg_10", "B2neg_0.1"),
        ("boosted", "B3"), ("lost", "lost"), ("sat", "sat"), ("meanbias", "meanbias")]
ZB = [0.01, 0.1, 0.25, 0.5, 1.0, 1.5, 2.0, 3.0]
claims_z = {"white": [-4.9, -3.0, -1.9, -0.8, 0.1, 1.5, 5.0], "baseline": [0.6, 0.9, 1.0, 1.3, 2.0, 3.1, 5.5],
            "goal": [0.6, 0.9, 1.0, 1.3, 2.2, 3.4, 4.8]}
for mode in ("baseline", "goal", "white"):
    for key, fl, bfile in (("all", "1e13", f"pd_fractions_{mode}.npz"), ("1e14", "1e14", f"pd_fractions_{mode}.npz"),
                           ("sub", "1e13", f"pd_fractions_{mode}_subthr.npz")):
        fn = os.path.join(V.ROOT, "data_real", bfile)
        if not os.path.exists(fn):
            continue
        b = np.load(fn); cols = list(b["cols"]); rows = b[f"rows_{fl}"]
        print(f"\n=== [{mode}] neighbours {key}: mine (mean over {len(runs)} sk{'y' if len(runs)==1 else 'ies'}"
              f"{' +- s.e.' if len(runs) > 1 else ''}) vs builder FFT P(D) ===")
        for qc in QC:
            row = rows[[r[0] for r in rows].index(qc)]
            parts = []
            for mk, bk in KEYS:
                v = np.array([r[mode]["res"][key][qc][mk] for r in runs])
                a = row[cols.index(bk)]
                if mk.startswith("N_"):
                    parts.append(f"{mk} {v.mean():,.0f} ({a:,.0f}, {v.mean()/a-1:+.2%})")
                else:
                    e = f"+-{v.std(ddof=1)/np.sqrt(len(v))*100:.2f}" if len(v) > 1 else ""
                    parts.append(f"{mk} {v.mean()*100:.2f}{e} ({a*100:.2f})")
            net = np.array([r[mode]["res"][key][qc]["net"] for r in runs]).mean()
            parts.append(f"net {net*100:+.2f}% ({(row[cols.index('N_det_b')]/row[cols.index('N_single')]-1)*100:+.2f}%)")
            print(f"  q>{qc:.0f}: " + "; ".join(parts))
        if key == "all":
            zr = np.array([r[mode]["res"]["all"][5.0]["zNd"] / r[mode]["res"]["all"][5.0]["zNs"] - 1 for r in runs]).mean(0)
            print(f"  q>5 Delta N/N per z bin {ZB}: mine " + " ".join(f"{x*100:+.1f}" for x in zr) +
                  "   builder claim " + " ".join(f"{x:+.1f}" for x in claims_z[mode]))
        if key in ("all", "sub") and mode != "white":
            for qc in QC:
                g = np.array([r[mode]["gauss"][key][QC.index(qc)] for r in runs]).mean()
                nd = np.array([r[mode]["res"][key][qc]["N_det_b"] for r in runs]).mean()
                print(f"  Gaussian-equivalent ({key}) q>{qc:.0f}: (N_det_b - N_gauss)/N_gauss = {nd/g-1:+.2%}")
