"""z12: compute used by this build -- every log in this folder that ends with /usr/bin/time -l output (wall, user + sys
CPU, maximum resident set size, peak memory footprint), the per-run wall times of the SZiFi runs, the start and end of
the runs, and the maxima of the concurrent memory sums recorded by z05_memwatch.
Usage: python3 -I -B z12_compute.py
"""
import os, re, glob, time

HERE = os.path.dirname(os.path.abspath(__file__))
RUND = "/Users/boris/GitHub/sz-poisson-blending/data_real/step3_szifi_real/runs"
rows = []
for fn in sorted(glob.glob(os.path.join(HERE, "*.log"))):
    s = open(fn).read()
    m = re.search(r"([\d.]+) real\s+([\d.]+) user\s+([\d.]+) sys", s)
    if not m:
        continue
    rss = re.search(r"(\d+)\s+maximum resident set size", s)
    fp = re.search(r"(\d+)\s+peak memory footprint", s)
    rows.append((os.path.basename(fn), float(m.group(1)), float(m.group(2)) + float(m.group(3)),
                 int(rss.group(1)) / 1e9 if rss else float("nan"), int(fp.group(1)) / 1e9 if fp else float("nan"),
                 os.path.getmtime(fn)))
print(f"{'log':34s} {'wall [s]':>9s} {'CPU [s]':>9s} {'max RSS [GB]':>12s} {'footprint [GB]':>14s}  finished")
for r in rows:
    print(f"{r[0]:34s} {r[1]:9.0f} {r[2]:9.0f} {r[3]:12.2f} {r[4]:14.2f}  {time.strftime('%H:%M', time.localtime(r[5]))}")
tw = sum(r[1] for r in rows); tc = sum(r[2] for r in rows)
print(f"sum over these logs: wall {tw / 3600:.2f} h (processes partly ran in parallel), CPU {tc / 3600:.2f} h")
szr = [r for r in rows if r[0].startswith(("z02", "z05_run", "z05_queue", "z06", "z04", "z09"))]
print(f"  of which SZiFi processes: wall {sum(r[1] for r in szr) / 3600:.2f} h, CPU {sum(r[2] for r in szr) / 3600:.2f} h, "
      f"largest max RSS {max(r[3] for r in szr):.2f} GB, largest peak footprint {max(r[4] for r in szr):.2f} GB")
# SZiFi runs
import numpy as np  # noqa: E402
ts, cnt = {}, {}
first, last = 1e20, 0
for fn in glob.glob(os.path.join(RUND, "*.npz")):
    b = os.path.basename(fn)
    cfg = b.split("_")[0]
    with np.load(fn) as d:
        t = float(d["t_run"])
    ts.setdefault(cfg, []).append(t)
    first = min(first, os.path.getmtime(fn) - t); last = max(last, os.path.getmtime(fn))
print("\nSZiFi runs stored in runs/ (one tile each): configuration, number, mean and max wall time per run")
for cfg in sorted(ts):
    print(f"  {cfg:16s} {len(ts[cfg]):4d} runs, mean {np.mean(ts[cfg]):6.1f} s, max {np.max(ts[cfg]):6.1f} s")
print(f"first run started {time.strftime('%H:%M', time.localtime(first))}, last finished "
      f"{time.strftime('%H:%M', time.localtime(last))} ({(last - first) / 3600:.2f} h)")
def dsize(p):
    return sum(os.path.getsize(os.path.join(a, f)) for a, _, fs in os.walk(p) for f in fs) / 1e6


D = "/Users/boris/GitHub/sz-poisson-blending/data_real/step3_szifi_real"
print(f"\ndata_real/step3_szifi_real/: {dsize(D):.0f} MB in all; runs/ {dsize(RUND):.0f} MB; analytic_shells.npz "
      f"{os.path.getsize(os.path.join(D, 'analytic_shells.npz')) / 1e6:.0f} MB; out/ {dsize(os.path.join(D, 'out')):.1f} MB")
for fn in sorted(glob.glob(os.path.join(HERE, "z05_memwatch*.log"))):
    for l in open(fn):
        if l.startswith("maximum summed"):
            print(f"{os.path.basename(fn)}: {l.strip()[:120]}")
