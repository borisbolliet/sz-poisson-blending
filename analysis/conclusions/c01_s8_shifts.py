"""c01: the absolute S8 shifts behind ../../CONCLUSIONS.md (step 3b, our finder, no merging, q > 5), next to the
statistical errors of the same fits, so that 'more significant in sigma' and 'larger in S8' can be told apart.

Reads the arbiter's stored pooled fits, data_real/step3b/arbiter/fits_pooled.json (written by
analysis/realistic/step3b/a13_pooled_fits.py; the headline biases in step3b/RESULTS.md come from the same file).
Usage (from the repository root): .venv/bin/python -I -B analysis/conclusions/c01_s8_shifts.py
"""
import os, json

ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
fits = json.load(open(os.path.join(ROOT, "data_real", "step3b", "arbiter", "fits_pooled.json")))
print("step 3b pooled fits, q > 5: shift in S8 (best fit minus truth), its statistical error, and the bias in sigma")
print(f"  {'mode':8s} {'survey':6s} {'config':6s} {'shift in S8':>12s} {'sigma(S8)':>10s} {'bias':>8s}")
rows = {}
for r in fits:
    if r.get("config") in ("a", "d") and float(r.get("qcut", 5)) == 5.0:
        rows[(r["mode"], r["survey"], r["config"])] = r
        print(f"  {r['mode']:8s} {r['survey']:6s} {r['config']:6s} {r['d_S8']:+12.5f} {r['s_S8']:10.5f} {r['b_S8']:+8.2f}")
for mode in ("baseline", "goal"):
    for survey in sorted({k[1] for k in rows}):
        a, d = rows.get((mode, survey, "a")), rows.get((mode, survey, "d"))
        if a and d:
            print(f"  {mode} {survey}: sigma(S8) of config d / config a = {d['s_S8'] / a['s_S8']:.1f}; "
                  f"|shift| of config d / config a = {abs(d['d_S8']) / abs(a['d_S8']):.1f}")
