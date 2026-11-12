"""y10 (sims lens): own results against the build's claims, with combined errors.
Own: out/fits_<METHOD>.json (y07), out/datavec_*.npz (y06).  Build: data_real/step3b/fits.json, lin.json,
fits_analytic_cl.json (read as data).  Pull = (own - build) / sqrt(own_err^2 + build_err^2), errors = tile-bootstrap
errors of the linearised S8 bias of each side (same estimator, setup, config, survey)."""
import os, sys, json
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import vlib3b as V

ROOT = "/Users/boris/GitHub/sz-poisson-blending"
B3 = f"{ROOT}/data_real/step3b"
log = V.Tee(os.path.join(HERE, "y10_compare.log"))
bml = json.load(open(f"{B3}/fits.json")); blin = json.load(open(f"{B3}/lin.json"))
ban = json.load(open(f"{B3}/fits_analytic_cl.json"))


def b_row(rows, **kw):
    for r in rows:
        if all(r.get(k) == v for k, v in kw.items()):
            return r
    return None


own = {}
for meth in ("whitened", "szifi"):
    fn = os.path.join(HERE, "out", f"fits_{meth}.json")
    if os.path.exists(fn):
        own[meth] = json.load(open(fn))
pulls = []


def compare(label, meth, est, setup, best, mode, cn, sv, qcut=5.0, use_ml=True):
    if meth not in own:
        return
    o_ml = b_row(own[meth]["ml"], mode=mode, est=est, config=cn, survey=sv, qcut=qcut)
    o_lin = b_row(own[meth]["lin"], mode=mode, est=est, config=cn, survey=sv, qcut=qcut)
    if o_lin is None:
        return
    b_ml = b_row(bml, setup=setup, mode=mode, estimator=best, config=cn, survey=sv, qcut=qcut, route="cat")
    b_lin = b_row(blin, setup=setup, mode=mode, estimator=best, config=cn, survey=sv, qcut=qcut)
    if b_lin is None:
        return
    oe, be = o_lin.get("boot_S8", np.nan), b_lin.get("boot_S8", np.nan)
    if not np.isfinite(oe) and est.startswith("Rs_"):          # own R_s of a merge rule: the paired R_p's error
        alt = b_row(own[meth]["lin"], mode=mode, est="Rp_" + est[3:], config=cn, survey=sv, qcut=qcut)
        oe = alt.get("boot_S8", np.nan) if alt else np.nan
    comb = np.hypot(oe, be)
    if use_ml and o_ml is not None and b_ml is not None:
        ov, bv, kind = o_ml["b_S8"], b_ml["b_S8"], "ML"
    else:
        ov, bv, kind = o_lin["lin_S8"], b_lin["lin_S8"], "lin"
    pull = (ov - bv) / comb
    pulls.append((label, mode, cn, sv, pull))
    log(f"  {label:26s} {mode:8s} {cn} {sv:4s} q{qcut:.0f} [{kind:3s}]: own {ov:+6.2f} +- {oe:.2f} | build {bv:+6.2f} +- {be:.2f}"
        f" | diff {ov - bv:+.2f} = {pull:+.1f} combined sigma")


log("S8 bias, own vs build (same estimator; errors = each side's tile bootstrap of the linearised bias)")
for mode in ("baseline", "goal"):
    for cn in ("a", "d", "e"):
        for sv in ("SO40", "full"):
            compare("production R_s", "whitened", "Rs", "prod_none", "Rs", mode, cn, sv)
            compare("production R_p", "whitened", "Rp", "prod_none", "Rp", mode, cn, sv)
            compare("production R", "whitened", "R", "prod_none", "R", mode, cn, sv)
            compare("production R (joint boot)", "whitened", "Rjoint", "prod_none", "R", mode, cn, sv, use_ml=False)
    for cn in ("a", "d", "e"):
        compare("production R_s q_cut 6", "whitened", "Rs", "prod_none", "Rs", mode, cn, "SO40", qcut=6.0)
    compare("closest 1' R_s", "whitened", "Rs_closest1", "prod_closest1", "Rs", mode, "d", "SO40")
    compare("fof 10' R_s", "whitened", "Rs_fof10", "prod_fof10", "Rs", mode, "d", "SO40")
    compare("fof 10' R_s", "whitened", "Rs_fof10", "prod_fof10", "Rs", mode, "a", "SO40")
    compare("+ no real, R_p (lin)", "whitened", "Rp_keepnr", "prod_keepnr", "Rp", mode, "d", "SO40", use_ml=False)
    compare("+ no real, R_p (lin)", "whitened", "Rp_keepnr", "prod_keepnr", "Rp", mode, "a", "SO40", use_ml=False)
    for cn in ("a", "d", "e"):
        for sv in ("SO40", "full"):
            compare("SZiFi defaults R_s", "szifi", "Rs", "szifi3000_none", "Rs", mode, cn, sv)
            compare("SZiFi defaults R_p", "szifi", "Rp", "szifi3000_none", "Rp", mode, cn, sv, use_ml=False)
if pulls:
    p = np.array([x[-1] for x in pulls if np.isfinite(x[-1])])
    log(f"\n{len(p)} comparisons: mean pull {p.mean():+.2f}, rms {np.sqrt(np.mean(p ** 2)):.2f}, |pull| > 2: "
        f"{int(np.sum(np.abs(p) > 2))}, > 3: {int(np.sum(np.abs(p) > 3))}")
    for lab in sorted(set(x[0] for x in pulls)):
        q = np.array([x[-1] for x in pulls if x[0] == lab and np.isfinite(x[-1])])
        if q.size == 0:
            continue
        log(f"   {lab:26s}: n {q.size:2d}, mean pull {q.mean():+.2f}, max |pull| {np.abs(q).max():.2f}")

# analytic analyst
if "whitened" in own:
    log("\nanalytic analyst Phi(sqrt(q'^2 + k) - e), zero below q' = 2, production, SO40, ML S8 bias: own | build "
        "[(i) catalogue, (ii) isolated, (i) - (ii)]")
    for mode in ("baseline", "goal"):
        for k in (0, 2, 3):
            for cn in ("a", "d", "e"):
                oi = b_row(own["whitened"]["ml"], mode=mode, est=f"an{k}_cat", config=cn, survey="SO40")
                oii = b_row(own["whitened"]["ml"], mode=mode, est=f"an{k}_iso", config=cn, survey="SO40")
                bi = b_row(ban, variant="prod", mode=mode, k=float(k), route="blended", config=cn, survey="SO40")
                bii = b_row(ban, variant="prod", mode=mode, k=float(k), route="isolated", config=cn, survey="SO40")
                if oi is None or bi is None:
                    continue
                log(f"  {mode:8s} k={k} {cn}: own {oi['b_S8']:+7.2f} {oii['b_S8']:+7.2f} {oi['b_S8'] - oii['b_S8']:+6.2f} "
                    f"(power {oi['power']:.2f}) | build {bi['b_S8']:+7.2f} {bii['b_S8']:+7.2f} {bi['b_S8'] - bii['b_S8']:+6.2f} "
                    f"(power {bi['power']:.2f}; boot of (i) {bi.get('boot_S8', float('nan')):.2f})")
    rows = [r for r in own["whitened"]["ml"] if r["est"].startswith("anz")]
    if rows:
        from scipy.special import ndtr
        log(f"\nanalytic analyst WITHOUT the zero below q' = 2 (floor Phi(sqrt(k) - 5): k = 2 {ndtr(np.sqrt(2) - 5):.2e}, "
            f"k = 3 {ndtr(np.sqrt(3) - 5):.2e}); own ML fits, SO40:")
        for r in rows:
            log(f"  {r['mode']:8s} {r['est']:9s} {r['config']}: S8 {r['b_S8']:+8.2f} (lin {r['lin_S8']:+8.2f}); at bound "
                f"{r['at_bound']}; polish shift {r['polish_shift']:.2f}; chi2 {r['dev']:.1f}")
        cat = [r for r in rows if r["est"].endswith("cat")]
        log(f"  (i) S8 bias range {min(r['b_S8'] for r in cat):+.2f} to {max(r['b_S8'] for r in cat):+.2f}; fits at a bound "
            f"{sum(bool(r['at_bound']) for r in rows)} of {len(rows)}; with the zero: "
            f"{sum(bool(r['at_bound']) for r in own['whitened']['ml'] if r['est'].startswith('an') and not r['est'].startswith('anz'))} "
            f"of {sum(1 for r in own['whitened']['ml'] if r['est'].startswith('an') and not r['est'].startswith('anz'))}")
