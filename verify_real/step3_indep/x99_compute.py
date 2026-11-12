"""x99 (indep lens): compute used by the logged runs of this verification (from the /usr/bin/time -l files)."""
import os, re, glob
HERE = os.path.dirname(os.path.abspath(__file__))
rows = []
for f in sorted(glob.glob(os.path.join(HERE, "*.time"))):
    s = open(f).read()
    m = re.search(r"([\d.]+) real\s+([\d.]+) user", s); r = re.search(r"(\d+)\s+maximum resident set size", s)
    if m and r:
        rows.append((os.path.basename(f), float(m.group(1)), float(m.group(2)), int(r.group(1)) / 1e9))
with open(os.path.join(HERE, "x99_compute.log"), "w") as out:
    for name, w, u, rss in rows:
        line = f"{name:34s} wall {w:8.0f} s   user CPU {u:8.0f} s   max RSS (parent; workers report their own peaks in the run logs) {rss:5.2f} GB"
        print(line); out.write(line + "\n")
    tw, tu = sum(r[1] for r in rows), sum(r[2] for r in rows)
    line = (f"total: {tw:.0f} s wall (sum of runs) = {tw / 3600:.2f} h; {tu:.0f} s user CPU = {tu / 3600:.2f} CPU-h (multiprocessing workers' CPU is included "
            f"in user time on macOS only for waited children; worker counts: x04 3, x03 3 then 2, x03b 2, x10 2, x04b 3, x04c 2)")
    print(line); out.write(line + "\n")
    line = ("peak concurrent memory (worker peaks from the run logs): x04 3 x 1.42 GB = 4.3 GB; x03 3 x 0.93 + x07 0.97 = 3.8 GB; "
            "x10 2 x 1.16 + x03b 2 x 0.92 = 4.2 GB; x04b 3 x 1.50 = 4.5 GB; all <= 5 GB, <= 4 processes at a time")
    print(line); out.write(line + "\n")
