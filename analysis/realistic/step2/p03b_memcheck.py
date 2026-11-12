"""Step 2 / p03b: memory check of p03's parent process after the loader fix (s2lib.rates_from_npz).
p03's first (production) run peaked at a 12.6 GB memory footprint (5.1 GB RSS) in the parent: every d['R_k'][i] access
re-read the whole npz array and the node view kept it alive.  The kernels themselves are unaffected (same values).
This loads the inputs of all 17 setups with the fixed loader and reports the parent's peak RSS; it also checks that
the loaded rates equal the stored arrays."""
import os, sys, resource
os.environ["OMP_NUM_THREADS"] = "1"
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import p03_kernels as P3

names = [n for g in ("fid", "truths", "conv") for n in P3.GROUPS[g]]
U = {n: P3.setup_inputs(n) for n in names}
th, _, R = P3.S.load_s09_rates("baseline")
d = np.load(f"{P3.S.DR}/pd_tables_baseline.npz")
same = all(np.array_equal(R[i][k], d[f"R_{k}"][i]) for i in range(len(th)) for k in P3.S.RKEYS)
print(f"all {len(names)} p03 setups loaded with the fixed loader: parent max RSS {resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1e6:.0f} MB; "
      f"loaded rates identical to the stored arrays at every node: {same}")
