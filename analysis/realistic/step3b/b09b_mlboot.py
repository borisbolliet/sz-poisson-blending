"""Step 3b / b09b: the bootstrap error of the headline's S8 bias with ML fits instead of the linearised estimator.

For configs d and e the ML fit and the linearised estimator differ by 10-20% at the larger biases (b09, check K15), so the
linearised bootstrap error may not be the ML one.  Here the headline (production finder, merge none, R_s, q_cut 5, SO40)
is refitted by maximum likelihood (s2lib.run_fit, as b09) on the first NB tile-bootstrap resamples of R_s (b07), and the
standard deviation of the ML biases is compared with the linearised one on the same resamples.
Output: data_real/step3b/mlboot.json.  Usage: python b09b_mlboot.py NPROC [NB]
"""
import os, sys, time, json
NPROC = int(sys.argv[1]) if len(sys.argv) > 1 else 6
NB = int(sys.argv[2]) if len(sys.argv) > 2 else 60
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import s3blib as B                                 # noqa: E402
B.threads(1)
import numpy as np                                 # noqa: E402
import s3bcosmo as SC                              # noqa: E402
S2 = SC.S2
_W = {}


def get(mode):
    if mode not in _W:
        _W.clear()
        M = SC.model("prod", mode, 5.0)
        dv = np.load(os.path.join(B.OUT, f"datavec_prod_none_{mode}.npz"))
        _W[mode] = (M, M.counts(SC.FID), dv["Rsboot_q5"][:NB].astype(float), dv["Rs_q5"])
    return _W[mode]


def _job(task):
    mode, cn, b = task
    M, N0, RB, R0 = get(mode)
    R = R0 if b < 0 else RB[b]
    r = S2.run_fit(M, np.clip(N0 * R, 0, None), cn, SC.FID, 0.4, polish=True)
    return dict(mode=mode, config=cn, b=b, b_S8=r["b_S8"], lin_S8=r["lin_S8"], at_bound=r["at_bound"])


if __name__ == "__main__":
    from multiprocessing import get_context
    t0 = time.time()
    tasks = [(m, cn, b) for m in B.MODES for cn in ("d", "e") for b in range(-1, NB)]
    print(f"b09b: {len(tasks)} ML fits (headline, R_s, q_cut 5, SO40; {NB} bootstrap resamples) on {NPROC} processes", flush=True)
    with get_context("spawn").Pool(NPROC) as p:
        rows = list(p.imap_unordered(_job, tasks, chunksize=2))
    json.dump(rows, open(os.path.join(B.OUT, "mlboot.json"), "w"), indent=1)
    for m in B.MODES:
        for cn in ("d", "e"):
            c = [r for r in rows if r["mode"] == m and r["config"] == cn and r["b"] == -1][0]
            v = [r for r in rows if r["mode"] == m and r["config"] == cn and r["b"] >= 0]
            ml = np.array([r["b_S8"] for r in v]); li = np.array([r["lin_S8"] for r in v])
            print(f"  {m:8s} {cn}: central ML {c['b_S8']:+.3f} (lin {c['lin_S8']:+.3f}); over {ml.size} resamples: ML mean {ml.mean():+.3f}, "
                  f"std {ml.std(ddof=1):.3f}; linearised std {li.std(ddof=1):.3f}; ratio {ml.std(ddof=1) / li.std(ddof=1):.2f}; "
                  f"fits at a bound {sum(bool(r['at_bound']) for r in v)}", flush=True)
    print(f"done [{time.time() - t0:.0f} s]")
