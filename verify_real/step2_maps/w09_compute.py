"""w09: compute and memory of the step-2 map verification runs, from the /usr/bin/time -l outputs next to each script."""
import os, re, glob
HERE = os.path.dirname(os.path.abspath(__file__))
out = open(os.path.join(HERE, "w09_compute.log"), "w")
def log(s):
    print(s); out.write(s + "\n")
THREADS = {"w01_setup": "4 threads (alone)", "w02_maps": "3 threads (+ w05 on 1 core)", "w03_noise": "3 threads (+ w05)",
           "w04_delta": "3 BLAS threads, mostly 1 (+ w05)", "w05_poisson": "1 process", "w06_poisson_summary": "1",
           "w08_sameprimaries": "3 BLAS threads", "w08b_allconfigs": "3 BLAS threads"}
tw = tc = 0.0
log("[w09] script | wall [s] | user CPU [s] | max RSS [GB] | peak footprint [GB] | parallelism")
for fn in sorted(glob.glob(os.path.join(HERE, "*.time"))):
    s = open(fn).read()
    m = re.search(r"([\d.]+) real\s+([\d.]+) user", s)
    rss = re.search(r"(\d+)\s+maximum resident set size", s)
    pf = re.search(r"(\d+)\s+peak memory footprint", s)
    if not m:
        continue
    nm = os.path.basename(fn)[:-5]
    w, c = float(m.group(1)), float(m.group(2))
    tw += w; tc += c
    log(f"   {nm:22s} | {w:8.0f} | {c:8.0f} | {int(rss.group(1))/1e9:5.2f} | {int(pf.group(1))/1e9:5.2f} | {THREADS.get(nm, '')}")
log(f"   total (sequential sum) wall {tw/60:.1f} min, CPU {tc/3600:.2f} h; w02 and w05 ran concurrently (3 + 1 cores), then w03/w04 + w05")
log("   note: an earlier 2-patch test of w02 (not kept) reached a 12.2 GB peak memory footprint (3.4 GB RSS) for ~14 s because")
log("   each npz[key][j] access re-read the whole rho table and the node views kept every copy alive (the same trap as the")
log("   build's p03); fixed by reading each array once before the production run.")
out.close()
