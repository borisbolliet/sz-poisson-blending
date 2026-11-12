"""p02: SZiFi's time per stage over all stored runs of SZiFi itself, from the wall-clock timers the driver
(../step3_szifi_real/z_szlib.py and the arbiter's zA_lib.py) put around six of SZiFi's functions in every run.
A second measurement next to p01's single profiled run: many runs, no profiler, other machine load.

Reads data_real/step3_szifi_real/runs/*.npz (the build) and data_real/step3_szifi_real/arbiter/runs/*.npz (the arbiter).
Usage (from the repository root): .venv/bin/python -I -B analysis/realistic/szifi_review/p02_stage_times.py
"""
import os, glob
import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", ".."))
D = os.path.join(ROOT, "data_real", "step3_szifi_real")
STAGES = [("t_filter", "filtering + peak finding"), ("t_cross", "spectra"), ("t_invcov", "inverse covariance"),
          ("t_mask", "cluster mask"), ("t_inpaint", "inpainting"), ("t_blend", "blending")]
groups = {}
for who, pat in (("build", os.path.join(D, "runs", "*.npz")), ("arbiter", os.path.join(D, "arbiter", "runs", "*.npz"))):
    for fn in sorted(glob.glob(pat)):
        with np.load(fn) as d:
            if "t_run" not in d.files or "rec_t_filter" not in d.files:
                continue
            cfg, kind = str(d["tag"]) if "tag" in d.files else str(d["cfg"]), str(d["kind"])
            n_it = int(d["n_it"]) if "n_it" in d.files else 0
            row = [float(d["t_run"])] + [float(np.sum(d["rec_" + k])) for k, _ in STAGES]
        groups.setdefault((who, cfg, kind, n_it), []).append(row)

print("Mean wall time per SZiFi run and per stage [s] (share of the run), summed over the noise iterations;")
print("one thread per run, 1-3 runs at a time on a 10-core laptop.  The arbiter's S1p, S1zp and S2zp runs fill holes on maps padded")
print("by 200 pixels (a diagnostic switch), which makes their blending slower than SZiFi's own.\n")
hdr = f"  {'who':8s} {'config':7s} {'kind':6s} {'it':>2s} {'runs':>4s} {'run':>6s}" + "".join(f" {lab[:18]:>18s}" for _, lab in STAGES) + f" {'other':>12s}"
print(hdr)
tot = {"build": np.zeros(len(STAGES) + 1), "arbiter": np.zeros(len(STAGES) + 1)}
cnt = {"build": 0, "arbiter": 0}
for (who, cfg, kind, n_it), rows in sorted(groups.items()):
    a = np.array(rows)
    m = a.mean(0)
    other = m[0] - m[1:].sum()
    line = f"  {who:8s} {cfg:7s} {kind:6s} {n_it:2d} {len(rows):4d} {m[0]:6.1f}"
    line += "".join(f" {m[i + 1]:9.1f} ({m[i + 1] / m[0] * 100:3.0f}%)" for i in range(len(STAGES)))
    line += f" {other:5.1f} ({other / m[0] * 100:3.0f}%)"
    print(line)
    if n_it == 2:
        tot[who] += a.sum(0)
        cnt[who] += len(rows)
sub = {"build, without the broken-covariance S3 smoke run": lambda w, c: w == "build" and c != "smokeS3",
       "arbiter, without its periodic-padding switch (S1p, S1zp, S2zp)": lambda w, c: w == "arbiter" and c not in ("S1p", "S1zp", "S2zp"),
       "arbiter, periodic-padding runs only": lambda w, c: w == "arbiter" and c in ("S1p", "S1zp", "S2zp")}
print()
for label, keep in sub.items():
    a = np.concatenate([np.array(rows) for (w, c, k, n), rows in groups.items() if n == 2 and keep(w, c)])
    s_ = a.sum(0)
    print(f"  {label}: {a.shape[0]} runs, mean run {s_[0] / a.shape[0]:.1f} s; filtering + peak finding {s_[1] / s_[0] * 100:.1f}%")
print()
for who in ("build", "arbiter"):
    if cnt[who]:
        s = tot[who]
        print(f"  all {cnt[who]} two-iteration runs of the {who}: mean run {s[0] / cnt[who]:.1f} s; filtering + peak finding "
              f"{s[1] / s[0] * 100:.0f}%, spectra + inverse covariance {(s[2] + s[3]) / s[0] * 100:.0f}%, "
              f"mask + inpainting + blending {(s[4] + s[5] + s[6]) / s[0] * 100:.0f}%, other {(s[0] - s[1:].sum()) / s[0] * 100:.0f}%")
