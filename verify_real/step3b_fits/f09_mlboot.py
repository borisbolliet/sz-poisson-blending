"""f09 (step-3b verification, lens 'fits'): ML vs linearised scatter over tile-bootstrap resamples (claim P3), reduced.

Headline (prod_none), q_cut 5, SO40, configs d and e, both noise levels: my model + my ML fitter (f04.ml_fit, started at
the linearised point) on NB of the build's stored R_s resamples (datavec 'Rsboot_q5', the first NB) and on NB of my own
R_s resamples (f02).  Printed: std of the ML S8 biases / std of the linearised S8 biases on the same resamples, and the
regression slope of ML on linearised.
Usage: f09_mlboot.py NPROC NB [build,mine]    Output: log, out/f09.json
"""
import os, sys, time, json
os.environ["VF_THREADS"] = "1"
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import numpy as np

NPROC = int(sys.argv[1]) if len(sys.argv) > 1 else 2
NB = int(sys.argv[2]) if len(sys.argv) > 2 else 20


def job(t):
    import f00lib as L
    import f04_fits as F4
    mode, cfg, src, b = t
    M, N0 = F4.model("prod", mode, 5.0)
    if src == "build":
        R = np.load(os.path.join(L.D3B, f"datavec_prod_none_{mode}.npz"))["Rsboot_q5"][b].astype(float)
    else:
        R = np.load(os.path.join(L.OUT, f"dv_prod_none_{mode}.npz"))["Rsboot_q5"][b].astype(float)
    r = F4.ml_fit(M, N0 * R, cfg, 0.4)
    return t, r["ml"]["S8"], r["lin"]["S8"], r["at_bound"]


if __name__ == "__main__":
    t0 = time.time()
    SRCS = sys.argv[3].split(",") if len(sys.argv) > 3 else ["build"]
    tasks = [(mode, cfg, src, b) for mode in ("baseline", "goal") for cfg in ("d", "e") for src in SRCS for b in range(NB)]
    print(f"f09: {len(tasks)} ML fits on {NPROC} processes", flush=True)
    res = []
    from multiprocessing import get_context
    with get_context("spawn").Pool(NPROC, maxtasksperchild=40) as p:
        for k, out in enumerate(p.imap_unordered(job, tasks, chunksize=1)):
            res.append(out)
            if k % 20 == 19:
                print(f"   {k + 1}/{len(tasks)} [{time.time() - t0:.0f} s]", flush=True)
    json.dump([[list(t), a, b_, c] for t, a, b_, c in res], open(os.path.join(HERE, "out", "f09.json"), "w"))
    print("headline, q_cut 5, SO40: std(ML S8 bias) / std(linearised S8 bias) over the same resamples; slope of ML on lin; mean ML - lin")
    for mode in ("baseline", "goal"):
        for cfg in ("d", "e"):
            for src in SRCS:
                v = np.array([(a, b_) for t, a, b_, c in res if t[0] == mode and t[1] == cfg and t[2] == src])
                sl = np.polyfit(v[:, 1], v[:, 0], 1)[0]
                nb = sum(1 for t, a, b_, c in res if t[0] == mode and t[1] == cfg and t[2] == src and c)
                print(f"  {mode:8s} {cfg} {src:5s} resamples ({len(v)}): std ML {v[:, 0].std(ddof=1):.3f}, std lin {v[:, 1].std(ddof=1):.3f}, "
                      f"ratio {v[:, 0].std(ddof=1) / v[:, 1].std(ddof=1):.3f}; slope {sl:.3f}; mean ML - lin {np.mean(v[:, 0] - v[:, 1]):+.3f}; at a bound {nb}", flush=True)
    print(f"total {time.time() - t0:.0f} s")
