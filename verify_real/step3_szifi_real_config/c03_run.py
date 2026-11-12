"""c03: SZiFi runs through the verifier's own driver (vc_szifi.py) and survey file (vc_survey.py), one after the other
in one process.  Each job: CFG:MAP:MODE[:BUFFER[:CHANNELS[:SEED]]] with MAP a file out/maps/MAP.npy, CHANNELS like 2,3,4,5 (or 'all'),
SEED an integer for np.random before the run (only 'orphics' inpainting draws random numbers; default: ours below).
Output: out/runs/<tag>.npz (tag = CFG_MAP, or CFG_MAP_chXXXX), SZiFi's stdout in out/stdout/<tag>.txt.  The full
parameter dictionary is printed once per configuration.  Existing outputs are skipped.
Usage: MallocLargeCache=0 /usr/bin/time -l /Users/boris/GitHub/szifi-venv/bin/python -I -B c03_run.py JOB [JOB ...]
"""
import os, sys, time
sys.dont_write_bytecode = True
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import vc_szifi as V                                 # noqa: E402
import numpy as np                                   # noqa: E402

t0 = time.time()
print(f"c03: {len(sys.argv) - 1} jobs; SZiFi {V.szifi.__file__}; MallocLargeCache="
      f"{os.environ.get('MallocLargeCache', '(default)')}", flush=True)
printed = set()
SEEDS = {"inj_baseline_000": 11, "inj_baseline_033": 12, "inj_goal_000": 13, "inj_goal_033": 14,
         "real_baseline_000": 21, "real_baseline_033": 22, "real_goal_000": 23, "real_goal_033": 24}
for job in sys.argv[1:]:
    f = job.split(":")
    cfg, mname, mode = f[0], f[1], f[2]
    buf = float(f[3]) if len(f) > 3 else 8.
    ch = [int(c) for c in f[4].split(",")] if len(f) > 4 and f[4] != "all" else None
    seed = int(f[5]) if len(f) > 5 else SEEDS.get(mname, 0) + 1000
    tag = f"{cfg}_{mname}" + ("" if ch is None else "_ch" + "".join(str(c) for c in ch)) + ("" if buf == 8. else f"_b{buf:g}") + ("" if len(f) <= 5 else f"_s{seed}")
    if os.path.exists(os.path.join(V.RUND, tag + ".npz")):
        print(f"  {tag}: exists, skipped", flush=True)
        continue
    r = V.run(cfg, mname, mode, buffer_arcmin=buf, channels=ch, tag=tag, seed=seed,
              print_params=(cfg, ch is None) not in printed)
    printed.add((cfg, ch is None))
    msg = []
    for it in range(r["n_it"]):
        q = r[f"it{it}_q"]
        msg.append(f"it{it}: q>4/4.5/5 {(q > 4).sum()}/{(q > 4.5).sum()}/{(q > 5).sum()}, median q {np.median(q) if q.size else np.nan:.2f}, "
                   f"sigma_vec[1,9,14] {r[f'it{it}_sigvec'][1]:.4e}/{r[f'it{it}_sigvec'][9]:.4e}/{r[f'it{it}_sigvec'][14]:.4e}")
    print(f"  {tag} (buffer {buf}'): {r['t_run']:.1f} s, peak RSS {r['peak_rss_gb']:.2f} GB, "
          f"old inverse covariance released {r['released']}x, f {np.round(r['fsky'], 5).tolist()}; " + "; ".join(msg)
          + f"  [{time.strftime('%H:%M:%S')}]", flush=True)
print(f"c03 done [{time.time() - t0:.0f} s]", flush=True)
