"""Merge the fit outputs (configs a, b from fits.json; c, d, d3, d5, e from the bounded re-run fits_cde_q*.json) into
data_cosmo/fits_final.json and print the results tables; MCMC-based bias for config d (40% sky)."""
import os, sys, json
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import cosmolib as CL

R0 = json.load(open(f"{CL.DC}/fits.json"))
R = [r for r in R0 if r["config"] in ("a", "b")]
for q in ("5", "6"):
    R += json.load(open(f"{CL.DC}/fits_cde_q{q}.json"))
json.dump(R, open(f"{CL.DC}/fits_final.json", "w"), indent=1)
print(f"merged {len(R)} fits")
self_rows = [r for r in R if r["data"] == "self"]
print(f"SANITY self-fits: max |bias| {max(abs(r['b_'+p]) for r in self_rows for p in r['free']):.1e} sigma, "
      f"max chi2min {max(r['dev'] for r in self_rows):.1e} ({len(self_rows)} fits)")
nb = [r for r in R if r["data"] != "self"]
print(f"L-BFGS-B polish (configs c-e): max lnL gain {max(r['polish_gain'] for r in nb if r['config'] not in 'ab'):.3g}; "
      f"fits ending on a bound: {sum(1 for r in nb if r.get('at_bound'))} of {len(nb)}, bounds hit: "
      f"{sorted(set(b for r in nb for b in r.get('at_bound', [])))}")
for r in nb:
    if r["config"] in ("a", "b") and r["nit"] >= 59:
        print("   (a/b fit at maxit:", r["qcut"], r["survey"], r["config"], r["data"], f"polish gain {r['polish_gain']:.1e})")


def g(qc, sv, cn, dn):
    return next(r for r in R if r["qcut"] == qc and r["survey"] == sv and r["config"] == cn and r["data"] == dn)


for qc in (5.0, 6.0):
    print(f"\n\n### q_cut = {qc:.0f}: bias in units of the Fisher 1-sigma (truth: catalogue P(D) | maps clean | maps all), "
          f"map errors = patch bootstrap")
    print("| survey | fit | free | sigma(Om, s8, S8) | b(Om) cat / clean / all | b(s8) cat / clean / all | b(S8) cat / clean / all "
          "| nuisance shifts (cat; clean; all) | chi2min cat / clean / all (ndof) | power cat / clean / all |")
    print("|" + "---|" * 10)
    for sv in ("SO40", "full"):
        for cn in ("a", "b", "c", "d", "d3", "d5", "e"):
            rr = [g(qc, sv, cn, dn) for dn in ("cat", "map_clean", "map")]
            r0 = rr[0]
            pri = ", prior 1-b " + "/".join(f"{v[1]/0.8*100:.0f}%" for v in r0["priors"].values()) if r0["priors"] else ""
            cells = []
            for k, p in enumerate(("Om", "s8", "S8")):
                cells.append(" / ".join([f"{rr[0]['b_'+p]:+.2f}"] + [f"{r['b_'+p]:+.2f}±{r['boot_std'][k]:.2f}" for r in rr[1:]]))
            nui = "; ".join(", ".join(f"{p} {r['d_'+p]:+.3f}" + ("*" if p in r.get("at_bound", []) else "") for p in r["free"][2:]) or "-"
                            for r in rr)
            print(f"| {sv} | {cn} | {'+'.join(r0['free'])}{pri} | {r0['s_Om']:.4f}, {r0['s_s8']:.4f}, {r0['s_S8']:.4f} | "
                  + " | ".join(cells) + f" | {nui} | " + " / ".join(f"{r['dev']:.1f}" for r in rr) + f" ({r0['ndof']}) | "
                  + " / ".join(f"{r['power']:.2f}" for r in rr) + " |")
    print("(* = parameter on its bound, e.g. slnY at 0.001)")
    print(f"map_raw (maps vs single-cluster expectation, pipeline max-over-scales boost NOT calibrated), SO40: " +
          "; ".join(f"{cn}: S8 {g(qc,'SO40',cn,'map_raw')['b_S8']:+.2f}, s8 {g(qc,'SO40',cn,'map_raw')['b_s8']:+.2f}" for cn in "abcde"))

# ---- MCMC, config d, 40% sky
fs, fc = f"{CL.DC}/mcmc_d_self_SO40.npz", f"{CL.DC}/mcmc_d_cat_SO40.npz"
if os.path.exists(fs) and os.path.exists(fc):
    a, b = np.load(fs), np.load(fc)
    free = list(a["free"])
    print("\n### config d, 40% sky, q>5: emcee posteriors (naive model) on own expectation vs blended (catalogue) expectation")
    for p in free + ["S8"]:
        if p == "S8":
            xa = CL.S8_of(a["chain"][:, 0], a["chain"][:, 1]); xb = CL.S8_of(b["chain"][:, 0], b["chain"][:, 1])
        else:
            xa = a["chain"][:, free.index(p)]; xb = b["chain"][:, free.index(p)]
        print(f"  {p:5s}: posterior mean shift {xb.mean()-xa.mean():+.4f} = {(xb.mean()-xa.mean())/xa.std():+.2f} x (posterior std "
              f"on unblended data {xa.std():.4f}); median shift {(np.median(xb)-np.median(xa))/xa.std():+.2f}")
