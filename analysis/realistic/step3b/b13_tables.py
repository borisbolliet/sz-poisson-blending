"""Step 3b / b13: the tables of RESULTS.md, from b09 (fits.json, lin.json, lin_convergence.json), b10 (fits_analytic.json)
and step 2's fits (data_real/step2/fits.json, fits_arbiter_fdom_eff.json, fits_arbiter_effiii.json).
  (1) S8 / Om / s8 bias (ML, R_s) +- tile-bootstrap error (linearised) for every setup, config, survey, q_cut.
  (2) The same S8 biases with the three estimators (linearised): R, R_p, R_s.
  (3) Comparison with step 2's default (sigma_Y, fdom 1), step 2's rows that step 3a's preview used (fdom 0.5-1 x map
      sigma (i)) and the nearest row for SZiFi's defaults (fdom 0.5 x plain std (iii)).
  (4) Goodness of fit (power) and nuisance shifts in config d.
  (5) The analytic analyst (b10); (5b) the same with the credit limit (fits_analytic_cl.json), its blending part against
      the measured-kernel analyst, and the floor Phi(sqrt(k) - 5) that the limit removes.
  (6) Kernel Monte Carlo convergence (lin_convergence.json).  (7) The three estimators bin by bin (b07 data vectors).
Usage: python b13_tables.py
"""
import os, sys, json, math
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import s3blib as B                                 # noqa: E402
import numpy as np                                 # noqa: E402

D2 = f"{B.ROOT}/data_real/step2"
S2ROWS = json.load(open(f"{D2}/fits.json")) + json.load(open(f"{D2}/fits_arbiter_fdom_eff.json")) + \
    json.load(open(f"{D2}/fits_arbiter_effiii.json"))
FITS = json.load(open(os.path.join(B.OUT, "fits.json")))
LIN = json.load(open(os.path.join(B.OUT, "lin.json")))
ANA = json.load(open(os.path.join(B.OUT, "fits_analytic.json")))
ANACL = json.load(open(os.path.join(B.OUT, "fits_analytic_cl.json"))) if os.path.exists(os.path.join(B.OUT, "fits_analytic_cl.json")) else []
SETUPS = ["prod_none", "prod_closest1", "prod_fof10", "prod_keepnr", "szifi3000_none", "szifi12000_none"]
SLAB = {"prod_none": "production, merge none (headline)", "prod_closest1": "production, closest 1'", "prod_fof10": "production, fof 10'",
        "prod_keepnr": "production, + 'no real cluster' entries", "szifi3000_none": "SZiFi defaults (l_sep 3000)",
        "szifi12000_none": "SZiFi l_sep 12000"}
CF = ["a", "b", "c", "d", "d3", "d5", "e"]


def f3(setup, mode, cn, sv, qc, est="Rs"):
    return [r for r in FITS if r["setup"] == setup and r["mode"] == mode and r["config"] == cn and r["survey"] == sv
            and r["qcut"] == qc and r["route"] == "cat" and r["estimator"] == est][0]


def l3(setup, mode, cn, sv, qc, est="Rs"):
    return [r for r in LIN if r["setup"] == setup and r["mode"] == mode and r["config"] == cn and r["survey"] == sv
            and r["qcut"] == qc and r["estimator"] == est][0]


def s2(setup, cn, sv, qc=5.0):
    for r in S2ROWS:
        if r["setup"] == setup and r["config"] == cn and r["survey"] == sv and r["qcut"] == qc and r["route"] == "cat" \
                and r.get("start", "truth") == "truth":
            return r


if __name__ == "__main__":
    for qc in (5.0, 6.0):
        print(f"\n(1) S8 bias [sigma], ML fit with R_s, +- tile-bootstrap error; q_cut {qc:.0f}; SO40 | full sky")
        print("| setup | noise | " + " | ".join(CF) + " |")
        print("|---|---|" + "---|" * len(CF))
        for setup in SETUPS:
            for mode in B.MODES:
                cells = []
                for cn in CF:
                    a, b = f3(setup, mode, cn, "SO40", qc), f3(setup, mode, cn, "full", qc)
                    la, lb = l3(setup, mode, cn, "SO40", qc), l3(setup, mode, cn, "full", qc)
                    cells.append(f"{a['b_S8']:+.2f}±{la['boot_S8']:.2f} \\| {b['b_S8']:+.2f}±{lb['boot_S8']:.2f}")
                print(f"| {SLAB[setup]} | {mode} | " + " | ".join(cells) + " |")
    print("\n(1b) Om / s8 bias [sigma], ML with R_s, q_cut 5, SO40 | full sky, configs a, d, e")
    for setup in SETUPS:
        for mode in B.MODES:
            cells = []
            for cn in ("a", "d", "e"):
                a, b = f3(setup, mode, cn, "SO40", 5.0), f3(setup, mode, cn, "full", 5.0)
                cells.append(f"{cn}: {a['b_Om']:+.2f}/{a['b_s8']:+.2f} | {b['b_Om']:+.2f}/{b['b_s8']:+.2f}")
            print(f"  {SLAB[setup]:42s} {mode:8s} " + ";  ".join(cells))
    print("\n(2) S8 bias [sigma], linearised, three estimators: R (N_det/N_exp) | R_p (paired per bin) | R_s (pooled paired), "
          "each +- tile bootstrap; q_cut 5")
    for setup in SETUPS:
        for mode in B.MODES:
            for sv in ("SO40", "full"):
                cells = []
                for cn in ("a", "d", "e"):
                    v = [l3(setup, mode, cn, sv, 5.0, e) for e in ("R", "Rp", "Rs")]
                    cells.append(f"{cn}: " + " | ".join(f"{x['lin_S8']:+.2f}±{x['boot_S8']:.2f}" for x in v))
                print(f"  {SLAB[setup]:42s} {mode:8s} {sv:4s} " + ";  ".join(cells))
    print("\n(2a) the headline on the 324 Q2 tiles alone (linearised; R | R_p | R_s) vs all tiles (R_s)")
    for mode in B.MODES:
        for sv in ("SO40", "full"):
            cells = []
            for cn in ("a", "d", "e"):
                v = [l3("prod_none_q2", mode, cn, sv, 5.0, e) for e in ("R", "Rp", "Rs")]
                w_ = l3("prod_none", mode, cn, sv, 5.0, "Rs")
                cells.append(f"{cn}: " + " | ".join(f"{x['lin_S8']:+.2f}±{x['boot_S8']:.2f}" for x in v) + f"  [all tiles R_s {w_['lin_S8']:+.2f}±{w_['boot_S8']:.2f}]")
            print(f"  {mode:8s} {sv:4s} " + ";  ".join(cells))
    print("\n(2b) headline ML fits with each estimator, S8 [sigma], q_cut 5 (configs a, d, e; SO40 | full)")
    for mode in B.MODES:
        for est in ("R", "Rp", "Rs"):
            print(f"  {mode:8s} {est:2s}: " + ";  ".join(f"{cn}: {f3('prod_none', mode, cn, 'SO40', 5.0, est)['b_S8']:+.2f} | "
                                                       f"{f3('prod_none', mode, cn, 'full', 5.0, est)['b_S8']:+.2f}" for cn in ("a", "d", "e")))
    print("\n(3) comparison with step 2 (S8 bias [sigma], q_cut 5, SO40 | full sky)")
    for mode in B.MODES:
        for cn in ("a", "b", "c", "d", "e"):
            d0 = s2(f"{mode}_f1", cn, "SO40"), s2(f"{mode}_f1", cn, "full")
            pv = [s2(s, cn, "SO40")["b_S8"] for s in (f"{mode}_effi", f"{mode}_f0.5_effi")]
            pvf = [s2(s, cn, "full")["b_S8"] for s in (f"{mode}_effi", f"{mode}_f0.5_effi")]
            n3 = s2(f"{mode}_f0.5_effiii", cn, "SO40"), s2(f"{mode}_f0.5_effiii", cn, "full")
            p3 = f3("prod_none", mode, cn, "SO40", 5.0), f3("prod_none", mode, cn, "full", 5.0)
            z3 = f3("szifi3000_none", mode, cn, "SO40", 5.0), f3("szifi3000_none", mode, cn, "full", 5.0)
            print(f"  {mode:8s} {cn}: step 2 default {d0[0]['b_S8']:+.2f} | {d0[1]['b_S8']:+.2f};  step 3a preview (fdom 0.5-1 x (i)) "
                  f"{min(pv):+.2f} to {max(pv):+.2f} | {min(pvf):+.2f} to {max(pvf):+.2f};  step 3b production {p3[0]['b_S8']:+.2f} | "
                  f"{p3[1]['b_S8']:+.2f};  fdom 0.5 x (iii) {n3[0]['b_S8']:+.2f} | {n3[1]['b_S8']:+.2f};  step 3b SZiFi defaults "
                  f"{z3[0]['b_S8']:+.2f} | {z3[1]['b_S8']:+.2f}")
    print("\n(4) goodness of fit and nuisance shifts (ML, R_s, q_cut 5): chi2_min, power; config d: (1-b) / alpha / slnY [sigma]")
    for setup in SETUPS:
        for mode in B.MODES:
            cells = []
            for sv in ("SO40", "full"):
                pw = {cn: f3(setup, mode, cn, sv, 5.0)["power"] for cn in CF}
                d = f3(setup, mode, "d", sv, 5.0)
                cells.append(f"{sv}: max power a-e {max(pw.values()):.2f} (a {pw['a']:.2f}, d {pw['d']:.2f}); d: chi2 {d['dev']:.1f}, "
                             f"{d['b_omb']:+.2f}/{d['b_alpha']:+.2f}/{d['b_slnY']:+.2f}")
            print(f"  {SLAB[setup]:42s} {mode:8s} " + ";  ".join(cells))
    allp = [r["power"] for r in FITS if r["route"] == "cat" and r["qcut"] == 5.0 and r["estimator"] == "Rs"]
    print(f"  over all R_s fits at q_cut 5: power <= {max(allp):.2f}; configs b-e: "
          f"{max(r['power'] for r in FITS if r['route'] == 'cat' and r['estimator'] == 'Rs' and r['config'] != 'a'):.2f}")
    print("\n(5) analytic analyst Phi(sqrt(q'^2 + k) - e), S8 bias [sigma], q_cut 5: total (i) = finder (ii) + blending (i)-(ii)")
    for v in ("prod", "szifi3000"):
        for mode in B.MODES:
            for sv in ("SO40", "full"):
                cells = []
                for cn in ("a", "d", "e"):
                    parts = []
                    for k in (0.0, 2.0, 3.0):
                        a = [r for r in ANA if r["variant"] == v and r["mode"] == mode and r["survey"] == sv and r["config"] == cn and r["k"] == k]
                        bl = [r for r in a if r["route"] == "blended"][0]; iso = [r for r in a if r["route"] == "isolated"][0]
                        parts.append(f"k{k:.0f} {bl['b_S8']:+.2f}={iso['b_S8']:+.2f}{bl['b_S8'] - iso['b_S8']:+.2f}")
                    cells.append(f"{cn}: " + " ".join(parts))
                print(f"  {v:9s} {mode:8s} {sv:4s} " + ";  ".join(cells))
    if ANACL:
        print("\n(5b) analytic analyst with the credit limit (kernel 0 below q' = 2), S8 bias [sigma], q_cut 5: total (i) = finder (ii) + "
              "blending (i)-(ii); (i) +- tile bootstrap; power of (i) / (ii)")
        for v in ("prod", "szifi3000"):
            for mode in B.MODES:
                for sv in ("SO40", "full"):
                    cells = []
                    for cn in ("a", "d", "e"):
                        parts = []
                        for k in (0.0, 2.0, 3.0):
                            a = [r for r in ANACL if r["variant"] == v and r["mode"] == mode and r["survey"] == sv and r["config"] == cn and r["k"] == k]
                            bl = [r for r in a if r["route"] == "blended"][0]; iso = [r for r in a if r["route"] == "isolated"][0]
                            parts.append(f"k{k:.0f} {bl['b_S8']:+.2f}±{bl['boot_S8']:.2f}={iso['b_S8']:+.2f}{bl['b_S8'] - iso['b_S8']:+.2f} "
                                         f"(p {bl['power']:.2f}/{iso['power']:.2f})")
                        cells.append(f"{cn}: " + " ".join(parts))
                    print(f"  {v:9s} {mode:8s} {sv:4s} " + ";  ".join(cells))
        print(f"  fits with a parameter at a bound: {sum(bool(r['at_bound']) for r in ANACL)} of {len(ANACL)}; without the credit limit "
              f"(5): {sum(bool(r['at_bound']) for r in ANA)} of {len(ANA)}")
        nb = [r["b_S8"] for r in ANA if r["variant"] == "prod" and r["survey"] == "SO40" and r["route"] == "blended" and r["k"] in (2.0, 3.0)]
        print(f"  without the credit limit, k = 2 and 3, production, SO40, (i): S8 bias {min(nb):+.2f} to {max(nb):+.2f} sigma; max ML polish "
              f"shift {max(r['polish_shift'] for r in ANA):.2f} sigma (with the limit: {max(r['polish_shift'] for r in ANACL):.3f})")
        print("  analytic kernel of a halo with q' -> 0 at e = 5 without the credit limit: Phi(sqrt(k) - 5) = " +
              ", ".join(f"{0.5 * math.erfc((5 - math.sqrt(k)) / math.sqrt(2)):.1e} (k = {k:.0f})" for k in (0.0, 2.0, 3.0)))
        for v, setup in (("prod", "prod_none"), ("szifi3000", "szifi3000_none")):
            dmax = {"ad": 0.0, "e": 0.0}
            for mode in B.MODES:
                cells = []
                for cn in ("a", "d", "e"):
                    bp = []
                    for k in (0.0, 2.0, 3.0):
                        a = [r for r in ANACL if r["variant"] == v and r["mode"] == mode and r["survey"] == "SO40" and r["config"] == cn
                             and r["k"] == k]
                        bl = [r for r in a if r["route"] == "blended"][0]; iso = [r for r in a if r["route"] == "isolated"][0]
                        bp.append(bl["b_S8"] - iso["b_S8"])
                    ref = f3(setup, mode, cn, 'SO40', 5.0)['b_S8']
                    kk = "e" if cn == "e" else "ad"
                    dmax[kk] = max(dmax[kk], max(abs(x - ref) for x in bp))
                    cells.append(f"{cn}: {bp[0]:+.2f} / {bp[1]:+.2f} / {bp[2]:+.2f} vs {ref:+.2f}")
                print(f"  blending part (i) - (ii), SO40, k = 0 / 2 / 3 vs the measured-kernel analyst (b09): {v:9s} {mode:8s} " +
                      ";  ".join(cells))
            print(f"  {v}: largest |blending part - measured-kernel bias| over k = 0, 2, 3 and both noise levels (SO40): configs a and d "
                  f"{dmax['ad']:.2f} sigma; config e {dmax['e']:.2f} sigma")
    print("\n(7) the three estimators bin by bin (b07 data vectors, q_cut 5, bins with N_exp >= 10): chi^2 with the bootstrap errors of "
          "the difference; N_iso against N_exp; N_exp-weighted mean of R_p - R_s per q bin")
    np.seterr(divide="ignore", invalid="ignore")     # empty bins (N_exp < 10, excluded from the sums) give 0/0 in the resamples
    for setup in ("prod_none", "szifi3000_none", "szifi12000_none"):
        for mode in B.MODES:
            d = np.load(os.path.join(B.OUT, f"datavec_{setup}_{mode}.npz"))
            Ne, Ni = d["Nexp_q5"], d["Niso_q5"]
            ok = Ne >= 10
            chi = {}
            for lab, A, AB in (("R_p", d["Rp_q5"], d["Rpboot_q5"]), ("R", d["R_q5"], d["Rboot_q5"])):
                pull = (A - d["Rs_q5"]) / (AB.astype(float) - d["Rsboot_q5"].astype(float)).std(0)
                chi[lab] = float(np.sum(pull[ok] ** 2))
            rat = np.where(ok, Ni / np.where(Ne > 0, Ne, 1.0) - 1, 0.0)
            err = (d["Rboot_q5"].astype(float) / d["Rpboot_q5"].astype(float)).std(0)
            chi_i = float(np.sum((rat[ok] / err[ok]) ** 2))
            md = 100 * ((d["Rp_q5"] - d["Rs_q5"]) * Ne).sum(0) / np.maximum(Ne.sum(0), 1e-300)
            print(f"  {setup:16s} {mode:8s} ({int(ok.sum())} bins): R_p vs R_s chi^2 {chi['R_p']:.1f}; R vs R_s {chi['R']:.1f}; N_iso vs N_exp "
                  f"{chi_i:.1f}; mean R_p - R_s per q bin [%]: " + " ".join(f"{v:+.3f}" for v in md))
    cv = json.load(open(os.path.join(B.OUT, "lin_convergence.json")))
    print("\n(6) kernel Monte Carlo convergence (headline, q_cut 5): S8 bias with the even- / odd-tile kernels [R_s, all tiles; R, the "
          "324 Q2 tiles, full kernel in brackets]")
    for mode in B.MODES:
        for sv in ("SO40", "full"):
            for cn in ("a", "d", "e"):
                a = {x["half"]: x for x in cv if x["mode"] == mode and x["survey"] == sv and x["config"] == cn}
                print(f"  {mode:8s} {sv:4s} {cn}: R_s {a['even']['lin_S8']:+.3f} / {a['odd']['lin_S8']:+.3f};  "
                      f"R {a['even']['lin_S8_R']:+.3f} / {a['odd']['lin_S8_R']:+.3f} [{a['even'].get('lin_S8_R_full', float('nan')):+.3f}]")
