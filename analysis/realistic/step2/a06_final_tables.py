"""Arbiter / a06: final tables for RESULTS.md (after the arbitration), from the logged fit files:
data_real/step2/fits.json (build, p06), fits_arbiter_effiii.json (a04), fits_arbiter_fdom_eff.json (a04b).
(1) headline: S8 bias, configs a-e, q_cut 5, fdom 1, sigma_Y normalisation, SO40 | full;
(2) what the analyst does: S8 bias of configs a, d, e for every attribution rule x noise normalisation computed;
(3) envelope of configs d and e over all MMF analyst choices; goodness-of-fit power there.
Output: a06_final_tables.log
"""
import os, json
HERE = os.path.dirname(os.path.abspath(__file__))
D2 = "/Users/boris/GitHub/sz-poisson-blending/data_real/step2"
LOG = open(os.path.join(HERE, "a06_final_tables.log"), "w")


def log(s=""):
    print(s, flush=True); LOG.write(s + "\n"); LOG.flush()


rows = [r for r in json.load(open(f"{D2}/fits.json")) if r["route"] == "cat" and r.get("start", "truth") == "truth"]
for fn in ("fits_arbiter_effiii.json", "fits_arbiter_fdom_eff.json"):
    if os.path.exists(f"{D2}/{fn}"):
        rows += [r for r in json.load(open(f"{D2}/{fn}")) if r["route"] == "cat" and r.get("start", "truth") == "truth"]
F = {(r["setup"], r["qcut"], r["survey"], r["config"]): r for r in rows}


def b(setup, cn, sv, qc=5.0, p="S8"):
    r = F.get((setup, qc, sv, cn))
    return None if r is None else r[f"b_{p}"]


def fmt(v):
    return "–" if v is None else f"{v:+.2f}"


log("## (1) Headline: S8 bias in Fisher sigma, q_cut 5, fdom 1, SNR normalised by sigma_Y (SO40 | full sky)\n")
log("| setup | a | b | c | d | e |")
log("|---|---|---|---|---|---|")
for s, lab in (("Q3_c11", "Q3 data vector (c11), Q3 code"), ("white", "old white filter (step-2 kernel)"),
               ("baseline_f1", "SO baseline MMF"), ("goal_f1", "SO goal MMF"), ("lowP_f1", "low pressure (1-b)=0.6, baseline")):
    log(f"| {lab} | " + " | ".join(f"{fmt(b(s, cn, 'SO40'))} \\| {fmt(b(s, cn, 'full'))}" for cn in ("a", "b", "c", "d", "e")) + " |")

log("\n## (2) What the analyst does: S8 bias of configs a / d / e, q_cut 5 (SO40 | full sky)\n")
log("| instrument | attribution rule | SNR normalised by | a | d | e |")
log("|---|---|---|---|---|---|")
NORM = (("", "sigma_Y (no confusion)"), ("_effi", "map sigma (i): sigma_Y (+) sub-threshold confusion"),
        ("_effii", "map sigma (ii): robust width"), ("_effiii", "map sigma (iii): plain std"))
for run in ("baseline", "goal"):
    for f in ("0.5", "1", "2"):
        for tag, lab in NORM:
            if f == "1":
                s = f"{run}_f1" if tag == "" else f"{run}{tag}"
            else:
                s = f"{run}_f{f}" if tag == "" else f"{run}_f{f}{tag}"
            if (s, 5.0, "SO40", "a") not in F:
                continue
            log(f"| SO {run} | fdom {f} | {lab} | " + " | ".join(f"{fmt(b(s, cn, 'SO40'))} \\| {fmt(b(s, cn, 'full'))}"
                                                            for cn in ("a", "d", "e")) + " |")

log("\n## (3) Envelope over every MMF analyst choice computed (fdom 0.5-2 x normalisation), q_cut 5 and 6")
mmf = [r for r in rows if r["setup"].startswith(("baseline", "goal"))]
for cn in ("a", "b", "c", "d", "e"):
    for sv in ("SO40", "full"):
        for qc in (5.0, 6.0):
            sel = [r for r in mmf if r["config"] == cn and r["survey"] == sv and r["qcut"] == qc]
            lo = min(sel, key=lambda r: r["b_S8"]); hi = max(sel, key=lambda r: r["b_S8"])
            pw = max(r["power"] for r in sel)
            log(f"   config {cn} {sv:4s} q>{qc:.0f}: S8 bias {lo['b_S8']:+.2f} ({lo['setup']}) to {hi['b_S8']:+.2f} ({hi['setup']}) "
                f"over {len(sel)} setups; max goodness-of-fit power {pw:.3f}")
sel = [r for r in mmf if r["config"] in ("d", "d3", "d5", "e") and r["qcut"] == 5.0]
log(f"   configs d, d3, d5, e (q>5): largest |S8 bias| {max(abs(r['b_S8']) for r in sel):.2f}; fits with a parameter at a bound: "
    f"{sum(bool(r['at_bound']) for r in sel)} of {len(sel)}")
log("\n[a06] done")
