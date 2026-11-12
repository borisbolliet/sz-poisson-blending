"""Step 3b / b12: compute used, from the /usr/bin/time -l block at the end of every log in this folder (wall, user + sys
CPU, maximum resident set size) and, for pooled runs, the largest worker peak RSS printed by the script.
CPU of pooled runs: /usr/bin/time counts the parent and its children (multiprocessing workers are children); its maximum
resident set size is that of the largest single process among them.
Also printed: per-script sums; the span from the first start to the last end; and, for runs that overlapped in time, the
largest number of processes at once and the largest sum of (processes x per-process peak), an upper bound on the memory in
use (workers do not all peak at once; pool parents are light and not included).  Start = file time - wall, end = file
time.  The *_superseded_* logs were copied after their runs, so their file times are not run times and they are left out
of the overlap count (their own processes and peaks are in the table).
Usage: python b12_compute.py
"""
import os, re, glob, time

HERE = os.path.dirname(os.path.abspath(__file__))
rows = []
for fn in sorted(glob.glob(os.path.join(HERE, "*.log"))):
    name = os.path.basename(fn)
    if name.startswith("b12_"):
        continue
    txt = open(fn).read()
    m = re.search(r"^\s*([\d.]+) real\s+([\d.]+) user\s+([\d.]+) sys", txt, re.M)
    if not m:
        rows.append(dict(name=name, real=None))
        continue
    real, user, sys_ = float(m.group(1)), float(m.group(2)), float(m.group(3))
    rss = re.search(r"^\s*(\d+)\s+maximum resident set size", txt, re.M)
    rss = int(rss.group(1)) / 1e9 if rss else 0.0
    w = re.findall(r"(?:worker peak RSS|largest worker peak RSS) ([\d.]+) GB", txt)
    wmax = max(float(x) for x in w) if w else None
    procs = re.search(r"(\d+) process", txt)
    npr = int(procs.group(1)) if procs else 1
    peak = max(rss, wmax or 0.0)
    end = os.path.getmtime(fn)
    sw = re.search(r"^\s*(\d+)\s+swaps", txt, re.M)
    rows.append(dict(name=name, real=real, cpu=user + sys_, rss=rss, wmax=wmax, np=npr, peak=peak, mem=npr * peak,
                     start=end - real, end=end, sup="_superseded_" in name, swaps=int(sw.group(1)) if sw else -1))
print(f"{'log':34s} {'wall [s]':>9s} {'CPU [s]':>9s} {'max RSS [GB]':>12s} {'worker peak [GB]':>16s} {'processes':>9s} "
      f"{'procs x peak [GB]':>17s}")
tw = tc = 0.0
for r in rows:
    if r["real"] is None:
        print(f"{r['name']:34s} {'(no /usr/bin/time block)':>40s}")
        continue
    tw += r["real"]; tc += r["cpu"]
    print(f"{r['name']:34s} {r['real']:9.0f} {r['cpu']:9.0f} {r['rss']:12.2f} {('%.2f' % r['wmax']) if r['wmax'] else '-':>16s} "
          f"{r['np']:9d} {r['mem']:17.2f}")
print(f"{'sum':34s} {tw:9.0f} {tc:9.0f}   = {tw / 3600:.2f} h wall (sum of runs, some ran concurrently), {tc / 3600:.2f} CPU-h")

print("\nper script (all logs of that script, superseded runs included)")
print(f"{'script':8s} {'logs':>4s} {'wall [s]':>9s} {'CPU [s]':>9s} {'largest per-process peak [GB]':>30s} {'max processes':>13s}")
groups = {}
for r in rows:
    if r["real"] is not None:
        groups.setdefault(r["name"].split("_")[0], []).append(r)
for g in sorted(groups):
    rr = groups[g]
    print(f"{g:8s} {len(rr):4d} {sum(r['real'] for r in rr):9.0f} {sum(r['cpu'] for r in rr):9.0f} "
          f"{max(r['peak'] for r in rr):30.2f} {max(r['np'] for r in rr):13d}")

ok = [r for r in rows if r["real"] is not None and not r["sup"]]
print()
for lab, sel in (("runs of >= 60 s", [r for r in ok if r["real"] >= 60]), ("all runs, with the short table reruns", ok)):
    t0, t1 = min(r["start"] for r in sel), max(r["end"] for r in sel)
    print(f"span, {lab} (first start to last end, superseded logs left out): {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(t0))} "
          f"to {time.strftime('%H:%M:%S', time.localtime(t1))} = {(t1 - t0) / 3600:.2f} h")
ev = sorted({r["start"] for r in ok} | {r["end"] for r in ok})
best_p = best_m = (0, 0.0, [])
for t in ev:
    act = [r for r in ok if r["start"] <= t < r["end"]]
    p = sum(r["np"] for r in act); mm = sum(r["mem"] for r in act)
    if p > best_p[0]:
        best_p = (p, mm, [r["name"] for r in act])
    if mm > best_m[1]:
        best_m = (p, mm, [r["name"] for r in act])
print(f"most processes at once: {best_p[0]} ({', '.join(best_p[2])}; sum of procs x peak {best_p[1]:.2f} GB)")
print(f"largest sum of procs x peak at once: {best_m[1]:.2f} GB ({', '.join(best_m[2])}; {best_m[0]} processes)")
al = [r for r in rows if r["real"] is not None and r["real"] >= 5]
rr = max(al, key=lambda r: r["cpu"] / (r["real"] * r["np"]))
print(f"cores used: largest CPU / (wall x processes) over the logs with wall >= 5 s: {rr['cpu'] / (rr['real'] * rr['np']):.3f} ({rr['name']}); "
      f"single-process runs: CPU / wall <= {max(r['cpu'] / r['real'] for r in al if r['np'] == 1):.3f}")
sup = [r for r in rows if r["real"] is not None and r["sup"]]
print("superseded runs, not in the overlap count (processes x per-process peak): " +
      "; ".join(f"{r['name']}: {r['np']} x {r['peak']:.2f} = {r['mem']:.2f} GB" for r in sup))
tl = [r for r in rows if r["real"] is not None]
print(f"swaps (from /usr/bin/time): largest over the {len(tl)} logs {max(r['swaps'] for r in tl)}; logs without a swaps line: "
      f"{sum(r['swaps'] < 0 for r in tl)}")
