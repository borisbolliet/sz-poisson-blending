"""Arbiter / a07: compute used by the arbiter runs, from the `/usr/bin/time -l` output appended to each arbiter log
(as p10_compute.py does for the build).  Output: a07_compute.log
"""
import os, re
HERE = os.path.dirname(os.path.abspath(__file__))
LOGS = ["a01_maps_excess_diag", "a02_map_kernel_sim_000", "a02_map_kernel_sim_030", "a02b_map_kernel_compare",
        "a02c_map_kernel_lowP", "a03_fisher_rounding", "a04_mapsigma_plainstd", "a04b_mapsigma_fdom", "a05_quick_checks",
        "a06_final_tables"]
out = open(os.path.join(HERE, "a07_compute.log"), "w")


def log(s=""):
    print(s); out.write(s + "\n")


log(f"{'log':28s} {'wall [s]':>9s} {'cpu user [s]':>13s} {'max RSS [GB]':>13s} {'peak footprint [GB]':>20s}")
W = U = 0.0
WALL = {}
for name in LOGS:
    t = open(os.path.join(HERE, name + ".log")).read()
    m = re.search(r"([\d.]+) real\s+([\d.]+) user", t)
    r = re.search(r"(\d+)\s+maximum resident set size", t)
    p = re.search(r"(\d+)\s+peak memory footprint", t)
    if not m:
        log(f"{name:28s} (no time output)")
        continue
    w, u = float(m.group(1)), float(m.group(2))
    W += w; U += u; WALL[name] = w
    log(f"{name:28s} {w:9.1f} {u:13.1f} {int(r.group(1))/1e9:13.2f} {int(p.group(1))/1e9:20.2f}")
par = min(WALL.get("a02_map_kernel_sim_000", 0.0), WALL.get("a02_map_kernel_sim_030", 0.0))
log(f"total: wall {W - par:.0f} s = {(W - par)/60:.1f} min (the two a02 simulation runs ran at the same time; summed {W:.0f} s); "
    f"cpu (user) {U:.0f} s = {U/3600:.2f} h")
log("max RSS is that of the main process; for pools (a04, a04b) the workers were sampled with ps during the runs (section 13).")
