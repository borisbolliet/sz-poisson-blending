"""Step 2 / p09: markdown tables for RESULTS.md from data_real/step2/fits.json (p06), fixed_correction.json (p07) and
truth_*.npz (p05).  Every number of RESULTS.md's tables is printed here (log: p09_tables.log).
"""
import os, sys, json
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import s2lib as S                                  # noqa: E402

rows = json.load(open(f"{S.D2}/fits.json"))
LAB = {"Q3_c11": "Q3 data vector (c11), Q3 code", "white": "old white filter (step-2 kernel)",
       "white_prefix": "old white filter, pre-fix b03 rates", "baseline_f1": "SO baseline, fdom 1",
       "baseline_f0.5": "SO baseline, fdom 0.5", "baseline_f2": "SO baseline, fdom 2", "goal_f1": "SO goal, fdom 1",
       "goal_f0.5": "SO goal, fdom 0.5", "goal_f2": "SO goal, fdom 2", "baseline_effi": "SO baseline, map-sigma (i)",
       "baseline_effii": "SO baseline, map-sigma (ii)", "goal_effi": "SO goal, map-sigma (i)",
       "goal_effii": "SO goal, map-sigma (ii)", "lowP_f1": "low pressure (1-b)=0.6, baseline",
       "lowP_effi": "low pressure, baseline, map-sigma (i)"}
ORDER = list(LAB)
CFGS = ["a", "b", "c", "d", "d3", "d5", "e"]


def get(setup, cn, sv, qc, route="cat"):
    for r in rows:
        if r["setup"] == setup and r["config"] == cn and r["survey"] == sv and r["qcut"] == qc and r["route"] == route \
                and r.get("start", "truth") == "truth":
            return r
    return None


def f(v, n=1):
    return f"{v:+.{n}f}"


for qc in (5.0, 6.0):
    for sv, svl in (("SO40", "SO-like, f_sky = 0.4"), ("full", "full sky")):
        print(f"## Headline: bias in sigma, S8 / Om / s8, {svl}, q_cut = {qc:.0f} (naive model fitted to the blended expectation)\n")
        print("| setup | a: Om, s8 | b: + (1-b) | c: + (1-b), slnY | d: + (1-b) 4% prior, alpha, slnY | e: as d, no prior |")
        print("|---|---|---|---|---|---|")
        for s in ORDER:
            cells = []
            for cn in ("a", "b", "c", "d", "e"):
                r = get(s, cn, sv, qc)
                cells.append("–" if r is None else f"{f(r['b_S8'])} / {f(r['b_Om'])} / {f(r['b_s8'])}")
            print(f"| {LAB[s]} | " + " | ".join(cells) + " |")
        print()

print("\n## S8 bias in sigma, every config, q_cut = 5 (SO40 | full)\n")
print("| setup | " + " | ".join(CFGS) + " |")
print("|---|" + "---|" * len(CFGS))
for s in ORDER:
    cells = []
    for cn in CFGS:
        a, b = get(s, cn, "SO40", 5.0), get(s, cn, "full", 5.0)
        cells.append("–" if a is None else f"{f(a['b_S8'])} \\| {f(b['b_S8'])}")
    print(f"| {LAB[s]} | " + " | ".join(cells) + " |")

print("\n## S8 bias in sigma, every config, q_cut = 6 (SO40 | full)\n")
print("| setup | " + " | ".join(CFGS) + " |")
print("|---|" + "---|" * len(CFGS))
for s in ORDER:
    cells = []
    for cn in CFGS:
        a, b = get(s, cn, "SO40", 6.0), get(s, cn, "full", 6.0)
        cells.append("–" if a is None else f"{f(a['b_S8'])} \\| {f(b['b_S8'])}")
    print(f"| {LAB[s]} | " + " | ".join(cells) + " |")

print("\n## Om and s8 bias in sigma, every config, q_cut = 5, SO40 (Om / s8)\n")
print("| setup | " + " | ".join(CFGS) + " |")
print("|---|" + "---|" * len(CFGS))
for s in ORDER:
    cells = []
    for cn in CFGS:
        a = get(s, cn, "SO40", 5.0)
        cells.append("–" if a is None else f"{f(a['b_Om'])} / {f(a['b_s8'])}")
    print(f"| {LAB[s]} | " + " | ".join(cells) + " |")

print("\n## Om and s8 bias in sigma, every config, q_cut = 5, full sky (Om / s8)\n")
print("| setup | " + " | ".join(CFGS) + " |")
print("|---|" + "---|" * len(CFGS))
for s in ORDER:
    cells = []
    for cn in CFGS:
        a = get(s, cn, "full", 5.0)
        cells.append("–" if a is None else f"{f(a['b_Om'])} / {f(a['b_s8'])}")
    print(f"| {LAB[s]} | " + " | ".join(cells) + " |")

print("\n## Nuisance parameters and goodness of fit, config d (q_cut 5): bias in sigma of (1-b), alpha, slnY; chi2_min; power\n")
print("| setup | SO40: (1-b) / alpha / slnY | SO40 chi2_min, power | full: (1-b) / alpha / slnY | full chi2_min, power | at a bound |")
print("|---|---|---|---|---|---|")
for s in ORDER:
    a, b = get(s, "d", "SO40", 5.0), get(s, "d", "full", 5.0)
    if a is None:
        continue
    print(f"| {LAB[s]} | {f(a['b_omb'])} / {f(a['b_alpha'])} / {f(a['b_slnY'])} | {a['dev']:.1f}, {a['power']:.2f} | "
          f"{f(b['b_omb'])} / {f(b['b_alpha'])} / {f(b['b_slnY'])} | {b['dev']:.1f}, {b['power']:.2f} | "
          f"{','.join(sorted(set(a['at_bound'] + b['at_bound']))) or '–'} |")

print("\n## Goodness of fit, every config, q_cut 5: chi2_min (power) SO40 | full\n")
print("| setup | " + " | ".join(CFGS) + " |")
print("|---|" + "---|" * len(CFGS))
for s in ORDER:
    cells = []
    for cn in CFGS:
        a, b = get(s, cn, "SO40", 5.0), get(s, cn, "full", 5.0)
        cells.append("–" if a is None else f"{a['dev']:.1f} ({a['power']:.2f}) \\| {b['dev']:.1f} ({b['power']:.2f})")
    print(f"| {LAB[s]} | " + " | ".join(cells) + " |")

print("\n## Fisher sigma at the truth (q_cut 5): sigma(S8) SO40 | full\n")
print("| model | " + " | ".join(CFGS) + " |")
print("|---|" + "---|" * len(CFGS))
for s in ("white", "baseline_f1", "goal_f1", "baseline_effi", "lowP_f1"):
    cells = []
    for cn in CFGS:
        a, b = get(s, cn, "SO40", 5.0), get(s, cn, "full", 5.0)
        cells.append("–" if a is None else f"{a['s_S8']:.4f} \\| {b['s_S8']:.4f}")
    print(f"| {LAB[s]} | " + " | ".join(cells) + " |")

print("\n## fdom range (0.5 - 2) of the S8 bias, q_cut 5\n")
for run in ("baseline", "goal"):
    for cn in ("a", "d", "e"):
        for sv in ("SO40", "full"):
            v = [get(f"{run}_f{x}", cn, sv, 5.0)["b_S8"] for x in ("0.5", "1", "2")]
            print(f"   {run} config {cn} {sv}: fdom 0.5 / 1 / 2 -> S8 {v[0]:+.2f} / {v[1]:+.2f} / {v[2]:+.2f} sigma")

print("\n## Self route (naive model on its own expectation)\n")
sr = [r for r in rows if r["route"] == "self"]
print(f"   {len(sr)} fits; max |bias| {max(abs(r['b_'+p]) for r in sr for p in r['free'] + ['S8']):.1e} sigma; "
      f"started 2 sigma off: {sum(r.get('start') == 'offset' for r in sr)} fits, max |bias| "
      f"{max(abs(r['b_'+p]) for r in sr if r.get('start') == 'offset' for p in r['free'] + ['S8']):.1e} sigma")

fc = f"{S.D2}/fixed_correction.json"
if os.path.exists(fc):
    fr = json.load(open(fc))

    def g(t, m, cn, sv, qc=5.0):
        for r in fr:
            if r["truth_name"] == t and r["model"] == m and r["config"] == cn and r["survey"] == sv and r["qcut"] == qc:
                return r
    for qc in (5.0, 6.0):
        print(f"\n## Fixed correction C_fid x N_naive vs ignoring blending, truth off the fiducial (SO baseline, fdom 1, q_cut {qc:.0f}); "
              "bias in sigma S8 / Om / s8\n")
        print("| truth | config | survey | ignore blending | fixed correction (C at fiducial) | chi2_min ignore / fixed |")
        print("|---|---|---|---|---|---|")
        for t in ("fid", "S8=0.75", "S8=0.87"):
            for cn in ("a", "b", "c", "d", "e"):
                for sv in ("SO40", "full"):
                    a, b = g(t, "ignore", cn, sv, qc), g(t, "fixed", cn, sv, qc)
                    print(f"| {t} | {cn} | {sv} | {f(a['b_S8'],2)} / {f(a['b_Om'],2)} / {f(a['b_s8'],2)} | "
                          f"{f(b['b_S8'],2)} / {f(b['b_Om'],2)} / {f(b['b_s8'],2)} | {a['dev']:.1f} / {b['dev']:.2f} |")

print("\n## Truth vectors (p05): Delta / N at q > 5 (all primaries), per z range, and totals\n")
print("| setup | N_naive (full sky) | Delta total | z 0.01-0.1 | 0.1-0.5 | 0.5-1 | 1-1.5 | 1.5-2 | 2-3 | q 5-7.3 | q 7.3-10.6 | q > 47 |")
print("|---|---|---|---|---|---|---|---|---|---|---|---|")
for s in ORDER[1:] + ["S8_0.75_f1", "S8_0.87_f1"]:
    fn = f"{S.D2}/truth_{s}.npz"
    if not os.path.exists(fn):
        continue
    t = np.load(fn)
    D, Nn = t["D_q5"], t["Nn_q5"]
    pq = D.sum(0) / Nn.sum(0)
    print(f"| {LAB.get(s, s)} | {t['N0_q5'].sum():,.0f} | {D.sum():+,.0f} ({D.sum()/Nn.sum():+.2%}) | "
          + " | ".join(f"{100*v:+.2f}%" for v in t["tilt"]) + f" | {100*pq[0]:+.2f}% | {100*pq[1]:+.2f}% | "
          f"{100*D[:, 6:].sum()/Nn[:, 6:].sum():+.2f}% |")

print("\n## Linearised vs maximum-likelihood bias (cat route, all configs, surveys, q_cuts)\n")
for grp, ss in (("old white filter", ["Q3_c11", "white", "white_prefix"]), ("SO MMF (all variants)", ORDER[3:])):
    cr = [r for r in rows if r["route"] == "cat" and r["setup"] in ss and r.get("start", "truth") == "truth"]
    d = max(abs(r[f"b_{p}"] - r[f"lin_{p}"]) for r in cr for p in ("Om", "s8", "S8"))
    print(f"   {grp}: {len(cr)} fits; max |ML - linearised| over Om, s8, S8 = {d:.2f} sigma")

print("\n## Appendix: one table per setup (bias in sigma; chi2_min = Asimov deviance at the ML point; power = P(detect misfit at 95%))\n")
for s in ORDER:
    print(f"### {LAB[s]}\n")
    print("| q_cut | config | SO40: S8 / Om / s8 | SO40: (1-b) / alpha / slnY | SO40: chi2_min (power) | full: S8 / Om / s8 | "
          "full: (1-b) / alpha / slnY | full: chi2_min (power) |")
    print("|---|---|---|---|---|---|---|---|")
    for qc in (5.0, 6.0):
        for cn in CFGS:
            cells = []
            for sv in ("SO40", "full"):
                r = get(s, cn, sv, qc)
                nu = " / ".join(f(r[f"b_{p}"]) if f"b_{p}" in r else "fixed" for p in ("omb", "alpha", "slnY"))
                cells += [f"{f(r['b_S8'])} / {f(r['b_Om'])} / {f(r['b_s8'])}", nu, f"{r['dev']:.1f} ({r['power']:.2f})"]
            print(f"| {qc:.0f} | {cn} | " + " | ".join(cells) + " |")
    print()

print("\n## Summary numbers quoted in RESULTS.md\n")
MMF = ORDER[3:]
cat_ = [r for r in rows if r["route"] == "cat" and r.get("start", "truth") == "truth"]
for cn in ("d3", "d5"):
    d = max(abs(get(s, cn, sv, qc)["b_S8"] - get(s, "d", sv, qc)["b_S8"]) for s in MMF for sv in ("SO40", "full") for qc in (5.0, 6.0))
    print(f"   max |b_S8({cn}) - b_S8(d)| over MMF setups, surveys, q_cuts: {d:.3f} sigma")
for sv in ("SO40", "full"):
    pa = max((r["power"], r["setup"], r["config"], r["qcut"]) for r in cat_ if r["setup"] in MMF and r["survey"] == sv and r["config"] == "a")
    pb = max((r["power"], r["setup"], r["config"], r["qcut"]) for r in cat_ if r["setup"] in MMF and r["survey"] == sv and r["config"] != "a")
    print(f"   max goodness-of-fit power, MMF, {sv}: config a {pa[0]:.3f} ({pa[1]}, q_cut {pa[3]:.0f}); configs b-e {pb[0]:.3f} ({pb[1]}, {pb[2]}, q_cut {pb[3]:.0f})")
for sv in ("SO40", "full"):
    v = [get(s, cn, sv, 5.0)["b_S8"] for s in ("baseline_effi", "baseline_effii", "goal_effi", "goal_effii") for cn in ("d", "e")]
    print(f"   map-sigma (baseline, goal; i, ii) configs d-e S8 bias, q_cut 5, {sv}: {min(v):+.3f} to {max(v):+.3f} sigma")
v = [get(s, "a", "SO40", 6.0)["b_S8"] for s in ("baseline_effi", "baseline_effii", "goal_effi", "goal_effii")]
print(f"   map-sigma config a S8 bias, q_cut 6, SO40: {min(v):+.3f} to {max(v):+.3f} sigma")
for s in ("lowP_f1",):
    v = [get(s, cn, sv, qc)["b_S8"] for cn in ("d", "d3", "d5", "e") for sv in ("SO40", "full") for qc in (5.0, 6.0)]
    print(f"   low pressure configs d-e S8 bias (all surveys, cuts): {min(v):+.3f} to {max(v):+.3f} sigma")
d = max(abs(get("white", cn, sv, qc)[f"b_{p}"] - get("Q3_c11", cn, sv, qc)[f"b_{p}"]) for cn in CFGS for sv in ("SO40", "full")
        for qc in (5.0, 6.0) for p in ("Om", "s8", "S8"))
print(f"   old white filter, step-2 data vector vs Q3's data vector: max |bias difference| over Om, s8, S8, all configs, surveys, cuts = {d:.3f} sigma")
for s in ("baseline_f1", "goal_f1"):
    for sv in ("SO40", "full"):
        a5, a6 = get(s, "a", sv, 5.0)["b_S8"], get(s, "a", sv, 6.0)["b_S8"]
        print(f"   {s} config a {sv}: S8 q_cut 5 {a5:+.3f} -> q_cut 6 {a6:+.3f} sigma ({a6/a5-1:+.1%})")
for s in ("Q3_c11",):
    a5, a6 = get(s, "a", "SO40", 5.0)["b_S8"], get(s, "a", "SO40", 6.0)["b_S8"]
    print(f"   {s} config a SO40: S8 q_cut 5 {a5:+.3f} -> q_cut 6 {a6:+.3f} sigma ({a6/a5-1:+.1%})")
m = max(((abs(r[f"b_{p}"] - r[f"lin_{p}"]), p, r) for r in cat_ if r["setup"] in MMF for p in ("Om", "s8", "S8")), key=lambda t: t[0])
print(f"   largest |ML - linearised| (MMF): {m[0]:.3f} sigma in {m[1]} ({m[2]['setup']}, config {m[2]['config']}, {m[2]['survey']}, q_cut {m[2]['qcut']:.0f}: "
      f"ML {m[2]['b_'+m[1]]:+.2f}, linearised {m[2]['lin_'+m[1]]:+.2f})")
if os.path.exists(fc):
    for grp, cfs in (("config a", ("a",)), ("configs b-e", ("b", "c", "d", "d3", "d5", "e"))):
        for p in ("S8", "Om", "s8"):
            v = max((abs(r[f"b_{p}"]), r["truth_name"], r["config"], r["survey"], r["qcut"]) for r in fr
                    if r["model"] == "fixed" and r["truth_name"] != "fid" and r["config"] in cfs)
            print(f"   fixed correction, truths S8 = 0.75 / 0.87, {grp}: max |bias in {p}| = {v[0]:.3f} sigma ({v[1]}, {v[2]}, {v[3]}, q_cut {v[4]:.0f})")
    v = max((abs(r["b_S8"]), r["truth_name"], r["config"], r["survey"], r["qcut"]) for r in fr if r["model"] == "ignore" and r["truth_name"] != "fid")
    print(f"   ignoring blending at the same truths: max |bias in S8| = {v[0]:.3f} sigma ({v[1]}, {v[2]}, {v[3]}, q_cut {v[4]:.0f})")
