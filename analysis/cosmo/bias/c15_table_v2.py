"""[REPAIR v2 of c09] Merge fits_v2_q5/q6.json -> data_cosmo/fits_v2_final.json; print result tables (bias in units
of the CONVERGED Fisher sigma, absolute shifts, nuisance shifts, chi2/power), old-vs-new comparison, MCMC (v2) summary."""
import os, sys, json
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import cosmolib as CL

R = json.load(open(f"{CL.DC}/fits_v2_q5.json")) + json.load(open(f"{CL.DC}/fits_v2_q6.json"))
json.dump(R, open(f"{CL.DC}/fits_v2_final.json", "w"), indent=1)
OLD = json.load(open(f"{CL.DC}/fits_final.json"))
print(f"merged {len(R)} fits")
self_rows = [r for r in R if r["data"] == "self"]
print(f"SANITY self-fits (start at truth): max |bias| {max(abs(r['b_'+p]) for r in self_rows for p in r['free']):.1e} sigma, "
      f"max chi2min {max(r['dev'] for r in self_rows):.1e} ({len(self_rows)} fits)")
nb = [r for r in R if r["data"] != "self"]
print(f"fits ending on a bound: {sum(1 for r in nb if r.get('at_bound'))} of {len(nb)}; bounds hit: "
      f"{sorted(set(b for r in nb for b in r.get('at_bound', [])))}")


def g(rows, qc, sv, cn, dn):
    return next(r for r in rows if r["qcut"] == qc and r["survey"] == sv and r["config"] == cn and r["data"] == dn)


for qc in (5.0, 6.0):
    print(f"\n\n### q_cut = {qc:.0f}: bias in units of the converged Fisher 1-sigma (cat | maps clean | maps all); "
          f"map errors = linearised patch bootstrap")
    print("| survey | fit | sigma(Om, s8, S8) | b(Om) cat / clean / all | b(s8) cat / clean / all | b(S8) cat / clean / all "
          "| abs dS8 cat / clean | nuisance shifts (cat; clean) | chi2min cat / clean / all (ndof) | power cat / clean / all |")
    print("|" + "---|" * 10)
    for sv in ("SO40", "full"):
        for cn in ("a", "b", "c", "d", "d3", "d5", "e"):
            rr = [g(R, qc, sv, cn, dn) for dn in ("cat", "map_clean", "map")]
            r0 = rr[0]
            cells = []
            for k, p in enumerate(("Om", "s8", "S8")):
                cells.append(" / ".join([f"{rr[0]['b_'+p]:+.2f}"] + [f"{r['b_'+p]:+.2f}±{r['boot_std'][k]:.2f}" for r in rr[1:]]))
            nui = "; ".join(", ".join(f"{p} {r['d_'+p]:+.3f}({r['b_'+p]:+.2f}s)" + ("*" if p in r.get("at_bound", []) else "")
                                      for p in r["free"][2:]) or "-" for r in rr[:2])
            print(f"| {sv} | {cn} | {r0['s_Om']:.4f}, {r0['s_s8']:.4f}, {r0['s_S8']:.4f} | " + " | ".join(cells)
                  + f" | {rr[0]['d_S8']:+.4f} / {rr[1]['d_S8']:+.4f} | {nui} | " + " / ".join(f"{r['dev']:.1f}" for r in rr)
                  + f" ({r0['ndof']}) | " + " / ".join(f"{r['power']:.2f}" for r in rr) + " |")
    print("(* = on bound, slnY = 0.001)")
    for sv in ("SO40", "full"):
        print(f"map_raw ({sv}): " + "; ".join(
            f"{cn}: Om {g(R,qc,sv,cn,'map_raw')['b_Om']:+.2f}, s8 {g(R,qc,sv,cn,'map_raw')['b_s8']:+.2f}, S8 {g(R,qc,sv,cn,'map_raw')['b_S8']:+.2f}"
            f" (chi2 {g(R,qc,sv,cn,'map_raw')['dev']:.0f})" for cn in "abcde"))

print("\n### old (cosmolib, under-resolved) vs new (cosmolib2) at SO40: sigma(S8) and b(S8), cat")
for qc in (5.0, 6.0):
    for cn in ("a", "b", "c", "d", "e"):
        o, n = g(OLD, qc, "SO40", cn, "cat"), g(R, qc, "SO40", cn, "cat")
        print(f"  q{qc:.0f} {cn}: sigma_S8 old {o['s_S8']:.4f} new {n['s_S8']:.4f} (x{n['s_S8']/o['s_S8']:.2f}); "
              f"dS8 old {o['d_S8']:+.4f} new {n['d_S8']:+.4f}; b_S8 old {o['b_S8']:+.2f} new {n['b_S8']:+.2f}; "
              f"b_s8 old {o['b_s8']:+.2f} new {n['b_s8']:+.2f}; b_Om old {o['b_Om']:+.2f} new {n['b_Om']:+.2f}")

print("\n### absolute shifts, catalogue truth, q>5 (is the bias 'absorbed' by (1-b)?)")
for sv in ("SO40", "full"):
    for cn in ("a", "b", "c", "d", "e"):
        r = g(R, 5.0, sv, cn, "cat")
        print(f"  {sv} {cn}: dOm {r['d_Om']:+.4f}, ds8 {r['d_s8']:+.4f}, dS8 {r['d_S8']:+.4f}; sigma_S8 {r['s_S8']:.4f}; "
              + ", ".join(f"d{p} {r['d_'+p]:+.4f}" for p in r["free"][2:]))

for dn in ("self", "cat"):
    fn = f"{CL.DC}/mcmc_v2_d_{dn}_SO40.npz"
    if not os.path.exists(fn):
        print("missing", fn)
fs, fc = f"{CL.DC}/mcmc_v2_d_self_SO40.npz", f"{CL.DC}/mcmc_v2_d_cat_SO40.npz"
if os.path.exists(fs) and os.path.exists(fc):
    a, b = np.load(fs), np.load(fc)
    free = list(a["free"])
    rF = g(R, 5.0, "SO40", "d", "cat")
    print("\n### config d, 40% sky, q>5: emcee (converged model) on own expectation vs blended (catalogue) expectation")
    for p in free + ["S8"]:
        if p == "S8":
            xa = CL.S8_of(a["chain"][:, 0], a["chain"][:, 1]); xb = CL.S8_of(b["chain"][:, 0], b["chain"][:, 1])
        else:
            xa = a["chain"][:, free.index(p)]; xb = b["chain"][:, free.index(p)]
        print(f"  {p:5s}: posterior std (unblended) {xa.std():.4f} vs converged Fisher {rF['s_'+p]:.4f} (ratio {xa.std()/rF['s_'+p]:.2f}); "
              f"posterior-mean shift {(xb.mean()-xa.mean())/xa.std():+.2f} std; median shift {(np.median(xb)-np.median(xa))/xa.std():+.2f}; "
              f"ML/Fisher {rF['b_'+p]:+.2f}")
