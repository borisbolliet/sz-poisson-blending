"""p07: does SZiFi run more than one masking iteration when asked to?  (Review item E6.)

Runs SZiFi itself (unchanged, through the build's driver) with configuration S1 and max_it = 2 on the tile of p00_map.py,
and records every call of SZiFi's cat.identify_clusters(old, new), whose first returned catalogue feeds the convergence
test of the noise-iteration loop (szifi/mmf.py:465-467).  That catalogue is the old masked clusters plus one row with
q_opt = -1 for every new detection not identified with any of them.  For each call it prints the sizes of the old and new
masked lists, the number of new detections not identified with an old one (the -1 rows the test reads), the number of old
clusters not found again (the -1 rows of the second returned catalogue, which the test does not read), the value of the
test as SZiFi writes it, and the value of the same test applied to every entry.
Then it compares the run's catalogues and sigma with the build's stored S1 run (max_it = 1) and counts the passes.
Prints no SZiFi code.

Usage (from the repository root):
  MallocLargeCache=0 /Users/boris/GitHub/szifi-venv/bin/python -I -B analysis/realistic/szifi_review/p07_max_it.py
"""
import os, sys
sys.dont_write_bytecode = True
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
OUT = os.path.join(ROOT, "data_real", "szifi_review", "out")
sys.path.insert(0, os.path.join(ROOT, "analysis", "realistic", "step3_szifi_real"))
import z_szlib as Z                                   # noqa: E402
import numpy as np                                    # noqa: E402

MAP = os.path.join(OUT, "maps", "baseline_real_000.npy")
STORED = os.path.join(ROOT, "data_real", "step3_szifi_real", "runs", "S1_baseline_real_000.npz")
Z.STDD = os.path.join(OUT, "stdout")
Z.RUND = os.path.join(OUT, "runs")
calls = []
_identify = Z.cat.identify_clusters


def n_rows(c):
    return len(np.atleast_1d(c.catalogue.get("q_opt", [])))


def recorder(*a, **k):
    out = _identify(*a, **k)
    q1 = np.asarray(out[0].catalogue.get("q_opt", []), float)
    q2 = np.asarray(out[1].catalogue.get("q_opt", []), float)
    calls.append(dict(n_old=n_rows(a[0]), n_new=n_rows(a[1]), n_rows1=q1.size,
                      n_new_unmatched=int(np.sum(q1 == -1)), n_old_unmatched=int(np.sum(q2 == -1)),
                      as_written=bool(np.all(q1) != -1), every_entry=bool(np.all(q1 != -1))))
    return out


Z.cat.identify_clusters = recorder
Z.CONFIGS["S1_maxit2"] = dict(Z.CONFIGS["S1"], max_it=2)
with np.load(STORED) as d:
    st = {k: d[k] for k in d.files}
r = Z.run_tile("S1_maxit2", "baseline", "real", 0, float(st["buffer_arcmin"]), map_file=MAP, save=False, tag="p07")
print(f"S1 with max_it = 2 on tile 0 (baseline): {r['t_run']:.0f} s; filtering passes {len(r['rec_t_filter'])}; "
      f"noise iterations stored {r['n_it']}")
for i, c in enumerate(calls):
    print(f"  identify_clusters call {i} (loop index i = {i}): old masked clusters {c['n_old']}, new ones {c['n_new']}; "
          f"tested catalogue {c['n_rows1']} rows, of which {c['n_new_unmatched']} new detections not identified with an "
          f"old one (q_opt = -1); old clusters not found again {c['n_old_unmatched']} (not tested); "
          f"convergence test as written: {c['as_written']}; the same test on every entry: {c['every_entry']}")
keys = [k for k in st if k.startswith(("it0_", "it1_"))]
same = [k for k in keys if k in r and np.array_equal(np.asarray(st[k]), np.asarray(r[k]), equal_nan=True)]
print(f"against the build's stored run with max_it = 1: {len(same)} of {len(keys)} catalogue / sigma arrays identical")
