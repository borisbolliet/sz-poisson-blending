"""Step 3b arbitration / a14: compute used by the arbiter (logs a01-a13 in this folder), as b12 did for the build.

For every arbiter log with a /usr/bin/time block: wall, CPU (user + sys), largest resident set of one process, the number
of processes it ran, and its interval [file time - wall, file time].  Printed: totals, and over time the largest number
of processes and the largest sum of processes x per-process peak at once (an upper bound: workers do not all peak at
once).  Processes per run are those of the run's command line (pool size); the parent of a pool only hands out tasks.
Usage: python a14_compute.py
"""
import os, re, glob
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
PROCS = {"a02_cross_pipeline_baseline.log": 3, "a02_cross_pipeline_baseline_new400.log": 3, "a06b_fit_checks_fof.log": 2,
         "a12_painter_check.log": 2, "a13_pooled_fits.log": 3}

if __name__ == "__main__":
    rows = []
    for l in sorted(glob.glob(os.path.join(HERE, "a[0-9]*.log"))):
        if os.path.basename(l) == "a14_compute.log":
            continue
        s = open(l).read()
        m = re.findall(r"\n\s+([\d.]+) real\s+([\d.]+) user\s+([\d.]+) sys\n\s+(\d+)\s+maximum resident set size", s)
        if not m:
            continue
        r, u, sy, rss = m[-1]
        end = os.path.getmtime(l)
        wk = re.findall(r"worker peak RSS ([\d.]+) GB", s)
        peak = max([int(rss) / 1e9] + [float(x) for x in wk])
        rows.append(dict(log=os.path.basename(l), wall=float(r), cpu=float(u) + float(sy), peak=peak, n=PROCS.get(os.path.basename(l), 1),
                         t0=end - float(r), t1=end))
    print(f"{'log':42s} {'wall [s]':>8s} {'CPU [s]':>8s} {'peak/process [GB]':>18s} {'processes':>9s}")
    for r in rows:
        print(f"{r['log']:42s} {r['wall']:8.0f} {r['cpu']:8.0f} {r['peak']:18.2f} {r['n']:9d}")
    W, C = sum(r["wall"] for r in rows), sum(r["cpu"] for r in rows)
    t0, t1 = min(r["t0"] for r in rows), max(r["t1"] for r in rows)
    print(f"sum over {len(rows)} logged runs: wall {W:.0f} s ({W / 3600:.2f} h), CPU {C:.0f} s ({C / 3600:.2f} CPU-h); first start to last end "
          f"{(t1 - t0) / 3600:.2f} h")
    grid = np.arange(t0, t1, 2.0)
    procs = np.zeros(grid.size); mem = np.zeros(grid.size)
    for r in rows:
        m = (grid >= r["t0"]) & (grid <= r["t1"])
        procs[m] += r["n"]; mem[m] += r["n"] * r["peak"]
    k = int(np.argmax(mem))
    at = [r["log"] for r in rows if r["t0"] <= grid[k] <= r["t1"]]
    print(f"most processes at once: {procs.max():.0f}; largest sum of processes x per-process peak at once: {mem.max():.2f} GB ({', '.join(at)})")
    print(f"largest per-process peak: {max(r['peak'] for r in rows):.2f} GB; swaps: " +
          ", ".join(sorted(set(re.findall(r'(\d+)\s+swaps', ''.join(open(os.path.join(HERE, r['log'])).read() for r in rows))))))
