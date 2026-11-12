"""vc04c: q shape and redshift tilt in the coarse groupings of the build's RESULTS.md table (q 5-7.3, 7.3-10.6, q > 47;
z 0.01-0.1 ... 2-3), Delta/N with MC errors (blocks), mine vs build, q_cut 5, all variants."""
import os, sys
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
os.environ["VC_THREADS"] = "1"
import vclib as V
import numpy as np

dm = np.load(f"{HERE}/out/delta_mine.npz"); ns = len(dm["seeds"])
names = [b[0] for b in V.BVARS if b[0] != "white_f1_muB"]
print(f"{'variant':15s} | q 5-7.3: mine, build | q 7.3-10.6 | q > 47.3 | total")
for k in names:
    bset = "white" if k == "white_f1" else k
    D, N, blk = dm[f"{k}_q5_D"], dm[f"{k}_q5_Ns"], dm[f"{k}_q5_blocks"].astype(float)
    B = np.load(f"{V.ROOT}/data_real/step2/truth_{bset}.npz"); DB, NB = B["D_q5"], B["Nn_q5"]
    out = []
    for sl in (slice(0, 1), slice(1, 2), slice(6, 9)):
        m = D[:, sl].sum() / N[:, sl].sum()
        e = np.sqrt(len(blk) * blk[:, :, sl].sum((1, 2)).var(ddof=1)) / ns / N[:, sl].sum()
        b = DB[:, sl].sum() / NB[:, sl].sum()
        out.append(f"{100*m:+.2f}+-{100*e:.2f} {100*b:+.2f}")
    out.append(f"{100*D.sum()/N.sum():+.3f} {100*DB.sum()/NB.sum():+.3f}")
    print(f"{k:15s} | " + " | ".join(out))
