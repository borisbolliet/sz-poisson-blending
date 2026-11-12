"""y09 (sims lens): compute account from the /usr/bin/time -l blocks (*.time) and the workers' peak RSS in the logs."""
import os, re, glob
HERE = os.path.dirname(os.path.abspath(__file__))
out = open(os.path.join(HERE, "y09_compute.log"), "w")


def p(*a):
    s = " ".join(str(x) for x in a)
    print(s); out.write(s + "\n")


tot_w = tot_c = 0.0
p(f"{'run':34s} {'wall [s]':>9s} {'CPU [s]':>9s} {'parent max RSS [GB]':>20s} {'worker peak RSS [GB]':>21s} swaps")
for fn in sorted(glob.glob(os.path.join(HERE, "*.time"))):
    txt = open(fn).read()
    m = re.search(r"([\d.]+) real\s+([\d.]+) user\s+([\d.]+) sys", txt)
    if not m:
        continue
    wall, cpu = float(m.group(1)), float(m.group(2)) + float(m.group(3))
    rss = int(re.search(r"(\d+)\s+maximum resident set size", txt).group(1)) / 1e9
    sw = re.search(r"(\d+)\s+swaps", txt)
    base = os.path.basename(fn)[:-5]
    wk = []
    for lg in glob.glob(os.path.join(HERE, base + "*.stdout")) + glob.glob(os.path.join(HERE, base + "*.log")):
        wk += [float(x) for x in re.findall(r"worker peak RSS ([\d.]+) GB", open(lg).read())]
    tot_w += wall; tot_c += cpu
    p(f"{base:34s} {wall:9.0f} {cpu:9.0f} {rss:20.2f} {max(wk) if wk else float('nan'):21.2f} {sw.group(1) if sw else '?'}")
p(f"{'total':34s} {tot_w:9.0f} {tot_c:9.0f}   ({tot_w / 3600:.2f} h wall summed, {tot_c / 3600:.2f} CPU-h)")
p("Pooled runs used 3 worker processes (1 thread each); the largest concurrent bound is 3 x worker peak + parent.")
out.close()
