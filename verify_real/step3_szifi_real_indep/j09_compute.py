"""j09: the verifier's compute and memory, from the /usr/bin/time -l reports of every process it ran.

SZiFi jobs: out/stdout/TAG.log (one process each, run one at a time by jsz_queue.py with MallocLargeCache=0).
Other scripts: the j*.log files of this folder.  j00, j01, j03, j05, j10, j12, j13 ran alone; the light scripts j04, j06,
j07, j08 and j11 sometimes ran beside one SZiFi job, so the bound on concurrent memory is (largest SZiFi job) + (largest of
those five).
Usage: python3 -I -B j09_compute.py
"""
import os, re, glob
HERE = os.path.dirname(os.path.abspath(__file__))


def tim(fn):
    t = open(fn).read()
    g = lambda pat: float(re.search(pat, t).group(1)) if re.search(pat, t) else float("nan")
    return (g(r"([\d.]+) real"), g(r"([\d.]+) user"), g(r"([\d.]+) sys"), g(r"(\d+)\s+maximum resident set size") / 1e9)


print("j09: compute of the verifier's processes")
rows = {}
for fn in sorted(glob.glob(os.path.join(HERE, "out", "stdout", "*.log"))):
    tag = os.path.basename(fn)[:-4]
    cfg = tag.split("_")[0] + ("_norelease" if tag.endswith("norelease") else "")
    t = tim(fn)
    if t[0] == t[0]:                       # skip a job still running (no /usr/bin/time report yet)
        rows.setdefault(cfg, []).append(t)
tot_w = tot_c = 0.0
big = 0.0
print("  SZiFi jobs (one process each, one at a time): configuration, runs, mean / max wall [s], mean CPU [s], max RSS [GB]")
for cfg, v in sorted(rows.items()):
    w = [x[0] for x in v]
    c = [x[1] + x[2] for x in v]
    m = max(x[3] for x in v)
    big = max(big, m)
    tot_w += sum(w)
    tot_c += sum(c)
    print(f"    {cfg:8s} {len(v):3d} runs: wall {sum(w) / len(w):6.1f} / {max(w):6.1f}; CPU {sum(c) / len(c):6.1f}; max RSS {m:.2f}")
print(f"  SZiFi jobs: {sum(len(v) for v in rows.values())} runs, wall {tot_w / 3600:.2f} h, CPU {tot_c / 3600:.2f} h, largest max RSS {big:.2f} GB")
small = 0.0
for fn in sorted(glob.glob(os.path.join(HERE, "j[01][0-9]_*.log"))):
    w, u, s, m = tim(fn)
    if w == w:
        tot_w += w
        tot_c += u + s
        print(f"    {os.path.basename(fn):24s} wall {w:7.1f} s, CPU {u + s:7.1f} s, max RSS {m:.2f} GB")
        if os.path.basename(fn)[:3] in ("j04", "j06", "j07", "j08", "j11"):     # the scripts that ran beside a SZiFi job
            small = max(small, m)
print(f"  all processes: wall {tot_w / 3600:.2f} h (summed), CPU {tot_c / 3600:.2f} h; largest concurrent memory bound "
      f"{big:.2f} + {small:.2f} = {big + small:.2f} GB (one SZiFi job + one light script)")
