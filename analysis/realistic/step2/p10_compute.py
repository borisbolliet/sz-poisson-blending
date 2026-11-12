"""Step 2 / p10: compute used, parsed from the /usr/bin/time -l output at the end of each step-2 log."""
import os, re
HERE = os.path.dirname(os.path.abspath(__file__))
LOGS = ["p01_inputs", "p02_rates_truths", "p03_kernels", "p03_kernels_checks", "p03b_memcheck", "p04_naive_model", "p05_truth",
        "p06_fits", "p06b_decompose", "p07_fixed_correction", "p08_figures", "p09_tables"]
tw = tc = 0.0
print(f"{'log':24s} {'wall [s]':>10s} {'cpu user [s]':>13s} {'max RSS [GB]':>13s} {'peak footprint [GB]':>20s}")
for n in LOGS:
    txt = open(os.path.join(HERE, n + ".log")).read()
    m = re.search(r"([\d.]+) real\s+([\d.]+) user", txt)
    rss = re.search(r"(\d+)\s+maximum resident set size", txt)
    fp = re.search(r"(\d+)\s+peak memory footprint", txt)
    w, c = float(m.group(1)), float(m.group(2))
    tw += w; tc += c
    print(f"{n:24s} {w:10.1f} {c:13.1f} {int(rss.group(1))/1e9:13.2f} {int(fp.group(1))/1e9:20.2f}")
print(f"total: wall {tw:.0f} s = {tw/60:.1f} min; cpu (user) {tc:.0f} s = {tc/3600:.2f} h")
print("(p08 and p09 were also run once before their final run, ~3 s each; not included)")
