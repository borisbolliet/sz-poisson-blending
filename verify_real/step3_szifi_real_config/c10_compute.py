"""c10: compute and memory of the verifier's SZiFi runs, from the run logs (c03_run_*.log) and the 5-second memory
samples of c00_memwatch.sh (c00_memwatch.log: summed RSS of this verifier's processes).
Usage: /Users/boris/GitHub/szifi-venv/bin/python -I -B c10_compute.py
"""
import os, re, sys, glob
sys.dont_write_bytecode = True
HERE = os.path.dirname(os.path.abspath(__file__))
runs = []
for fn in sorted(glob.glob(os.path.join(HERE, "c03_run_*.log"))):
    txt = open(fn).read()
    for m in re.finditer(r"^  (\S+) \(buffer [^)]*\): ([\d.]+) s, peak RSS ([\d.]+) GB", txt, re.M):
        runs.append((m.group(1), float(m.group(2)), float(m.group(3))))
    mx = re.findall(r"^\s+(\d+)\s+maximum resident set size", txt, re.M)
    fp = re.findall(r"^\s+(\d+)\s+peak memory footprint", txt, re.M)
    print(f"{os.path.basename(fn)}: process maximum RSS {int(mx[0]) / 1e9 if mx else float('nan'):.2f} GB, "
          f"peak footprint {int(fp[0]) / 1e9 if fp else float('nan'):.2f} GB")
print(f"\n{len(runs)} SZiFi runs, {sum(r[1] for r in runs) / 3600:.2f} h of SZiFi wall time in all")
for pre in ("S1g_", "S1_", "S1wg_", "S2g_", "S0g_", "S3ig_", "S3it0g_", "S1wit0", "S1it0gf64"):
    t = [r[1] for r in runs if r[0].startswith(pre)]
    if t:
        print(f"  {pre:10s}: {len(t):2d} runs, mean {sum(t) / len(t):6.1f} s, max {max(t):6.1f} s")
s = [l.split() for l in open(os.path.join(HERE, "c00_memwatch.log")) if l.strip()]
v = [float(x[2]) for x in s]
print(f"\nmemory samples (5 s): {len(v)}, from {s[0][0]} to {s[-1][0]}; maximum summed RSS {max(v):.3f} GB at "
      f"{s[v.index(max(v))][0]}; samples above 2.0 GB: {sum(x > 2.0 for x in v)}; above 1.9 GB: {sum(x > 1.9 for x in v)}")
