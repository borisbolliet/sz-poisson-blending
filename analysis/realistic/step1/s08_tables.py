"""Step 1 / s08: print the markdown tables of RESULTS.md directly from data_real/*.npz (no hand transcription).

Reference b03 numbers (analysis/blending/b03_analytic.log, convention c=1, neighbour floor 1e13, q_cut = 5):
B2(0.1) = 0.1623, B2(0.2) = 0.10514, B3 = 0.08122, lost/N_single = 0.09028, B2neg(0.1) = 0.05733, N_det_b = 84,494.
"""
import os, sys, json
import numpy as np

ROOT = "/Users/boris/GitHub/sz-poisson-blending"
DR = f"{ROOT}/data_real"
RUNS = [("white", "old white (new signed code)"), ("baseline", "SO baseline MMF"), ("goal", "SO goal MMF")]


def frac_table(tag, fl, qcuts=(4.0, 5.0, 6.0, 7.0)):
    print(f"\n#### neighbours: {'sub-threshold only (q_B < 5)' if tag else 'M500 > ' + fl}; mean removed; detected sample\n")
    print("| setup | q_cut | N_single | N_det,blended (change) | Y boosted >10% | Y boosted >20% | boosted over threshold | "
          "lost (of N_single) | Y lowered >10% | mean (δ−μ)/q_A of detections | satellites (of N_single) |")
    print("|---|---|---|---|---|---|---|---|---|---|---|")
    if not tag and fl == "1e13":
        print("| old white (b03 log) | 5 | 85,898 | 84,494 (−1.63%) | 16.23% | 10.51% | 8.12% | 9.03% | 5.73% | – | 0.60% |")
    for run, lab in RUNS:
        f = f"{DR}/pd_fractions_{run}{tag}.npz"
        if not os.path.exists(f):
            continue
        d = np.load(f)
        cols = list(d["cols"])
        for row in d[f"rows_{fl}"]:
            if row[0] not in qcuts:
                continue
            g = lambda k: row[cols.index(k)]
            print(f"| {lab} | {g('qcut'):.0f} | {g('N_single'):,.0f} | {g('N_det_b'):,.0f} ({g('N_det_b')/g('N_single')-1:+.2%}) | "
                  f"{g('B2_0.1'):.2%} | {g('B2_0.2'):.2%} | {g('B3'):.2%} | {g('lost'):.2%} | {g('B2neg_0.1'):.2%} | "
                  f"{g('meanbias'):+.2%} | {g('sat'):.2%} |")


def width_table():
    print("\n#### robust half-width (84th − 16th percentile)/2 of n + δ − μ (noise alone: 1), untruncated P(D)\n")
    th_show = [0.5, 1.0, 2.0, 4.0, 8.0, 16.0, 32.0]
    print("| setup | neighbours | " + " | ".join(f"θ_A≈{t:g}′" for t in th_show) + " |")
    print("|---|---|" + "---|" * len(th_show))
    for run, lab in RUNS:
        for tag, nl, fl in (("", "M>1e13", "1e13"), ("", "M>1e14", "1e14"), ("_subthr", "sub-threshold", "1e13")):
            f = f"{DR}/pd_fractions_{run}{tag}.npz"
            if not os.path.exists(f):
                continue
            d = np.load(f)
            th = d["th"]; Q = d[f"QN_{fl}"]
            idx = [int(np.argmin(np.abs(np.log(th / t)))) for t in th_show]
            print(f"| {lab} | {nl} | " + " | ".join(f"{(Q[i,3]-Q[i,1])/2:.3f} ({th[i]:.2f}′)" for i in idx) + " |")


def sigma_table():
    d = np.load(f"{DR}/sigmaY_mmf.npz")
    # [arbiter fix] exact theta500 values (s03 'theta_exact'); the first version printed the nearest grid nodes
    # (1.995, 5.012, 19.95') under the labels 2, 5, 20'
    th = d["theta_exact"]
    print("\n#### sigma_Y [arcmin^2] at exact theta500 (s03)\n")
    print("| key | " + " | ".join(f"{t:g}′" for t in th) + " |")
    print("|---|" + "---|" * len(th))
    for k in ("white", "white_mmf", "baseline_l100", "goal_l100"):
        print(f"| {k} | " + " | ".join(f"{v:.3g}" for v in d[f"exact_{k}"]) + " |")
    for k in ("baseline_l100", "goal_l100"):
        print(f"| {k}/white | " + " | ".join(f"{a/b:.3f}" for a, b in zip(d[f"exact_{k}"], d["exact_white"])) + " |")
        print(f"| equivalent white y-noise, {k[:-5]} [arcmin] | " + " | ".join(f"{4e-6*a/b:.2g}" for a, b in zip(d[f"exact_{k}"], d["exact_white"])) + " |")


def conf_table():
    cv = np.load(f"{DR}/confusion_var.npz")
    th = cv["thA"]
    show = [0.5, 1.0, 2.0, 4.0, 7.5, 16.0, 32.0]
    idx = [int(np.argmin(np.abs(th - s))) for s in show]
    print("\n#### confusion check table (var_conf / sigma^2)\n")
    print("| key | " + " | ".join(f"{th[i]:.3g}" for i in idx) + " |")
    for k in ("white_var_all", "white_var_subthr", "white_mu", "baseline_var_all", "baseline_var_c2", "baseline_var_subthr",
              "goal_var_all", "goal_var_c2", "goal_var_subthr"):
        print(f"| {k} | " + " | ".join(f"{cv[k][i]:.3g}" for i in idx) + " |")


def gaussian_equivalent():
    """How much of the blended count change would a pipeline whose sigma_Y already contains the confusion
    variance (noise estimated from the data) reproduce?  N_gauss(q_cut) = sum H(theta, q) Phi((q - q_cut)/sqrt(1 + v(theta)))
    with v = Parseval confusion variance (s05; same population), vs the P(D) result N_det,blended (s06b/s06c).
    H = primary population on the same (theta node, q node) grid as s06b (pdlib1.primary_population)."""
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from scipy.special import ndtr
    import pdlib1 as pl
    from szsim import signal as sg
    cv = np.load(f"{DR}/confusion_var.npz")
    print("\n#### Gaussian-equivalent counts: N_det,blended (P(D)) vs N_gauss (sigma -> sigma sqrt(1 + var_conf(theta)))\n")
    print("| setup | neighbours | q_cut | N_single | N_det,blended | N_gauss | (N_det,blended − N_single)/N_single | (N_gauss − N_single)/N_single | (N_det,blended − N_gauss)/N_gauss |")
    print("|---|---|---|---|---|---|---|---|---|")
    for run, lab in RUNS:
        model = sg.SZModel() if run == "white" else pl.make_model(("so", run, 100.0))
        for tag, vkey in (("", "all"), ("_subthr", "subthr")):
            f = f"{DR}/pd_fractions_{run}{tag}.npz"
            if not os.path.exists(f):
                continue
            d = np.load(f)
            th = d["th"]
            H = pl.primary_population(model, th, 0.5)
            v = np.exp(np.interp(np.log(th), np.log(cv["thA"]), np.log(cv[f"{run}_var_{vkey}"])))
            cols = list(d["cols"])
            for row in d["rows_1e13"]:
                qc = row[0]
                Ng = (H * ndtr((pl.QNODES[None, :] - qc) / np.sqrt(1 + v[:, None]))).sum()
                Ns, Nd = row[cols.index("N_single")], row[cols.index("N_det_b")]
                print(f"| {lab} | {'sub-threshold' if tag else 'M>1e13'} | {qc:.0f} | {Ns:,.0f} | {Nd:,.0f} | {Ng:,.0f} | "
                      f"{Nd/Ns-1:+.2%} | {Ng/Ns-1:+.2%} | {Nd/Ng-1:+.2%} |")


def ztilt_table():
    """[arbiter, s09] Delta N/N per z bin at q > 5: P(D) expectation (fdom = 1) and expected per-sky std."""
    print("\n#### Delta N/N = N_det,blended/N_single - 1 per z bin, q > 5, full sky, P(D) expectation (s09); "
          "in brackets the expected full-sky per-sky std\n")
    zb = None
    for run, lab in RUNS:
        f = f"{DR}/pd_tables_{run}.npz"
        if not os.path.exists(f):
            continue
        d = np.load(f)
        if zb is None:
            zb = d["zb"]
            print("| setup | " + " | ".join(f"z {a:g}–{b:g}" for a, b in zip(zb[:-1], zb[1:])) + " |")
            print("|---|" + "---|" * (len(zb) - 1))
        ic = list(d["qcuts"]).index(5.0)
        print(f"| {lab} | " + " | ".join(f"{100*v:+.2f}% (±{100*e:.2f})" for v, e in zip(d["z_dNN"][ic], d["z_sd"][ic])) + " |")
    for run, lab in RUNS:
        f = f"{DR}/pd_tables_{run}.npz"
        if os.path.exists(f):
            d = np.load(f); ic = list(d["qcuts"]).index(5.0)
            print(f"| N_single per bin, {lab} | " + " | ".join(f"{v:,.0f}" for v in d["z_Ns"][ic]) + " |")


def dominance_table(qcuts=(5.0,)):
    """[arbiter, s09] fractions on the detected sample for dominance rules x_B < fdom q_A (fdom = inf: no rule)."""
    print("\n#### dependence on the dominance (attribution) rule x_B < fdom·q_A, neighbours M500 > 1e13 (s09)\n")
    print("| setup | fdom | q_cut | N_det,blended (change) | Y boosted >10% | Y boosted >20% | boosted over threshold | "
          "lost (of N_single) | mean (δ−μ)/q_A | not counted (1−P_dom) |")
    print("|---|---|---|---|---|---|---|---|---|---|")
    for run, lab in RUNS:
        f = f"{DR}/pd_tables_{run}.npz"
        if not os.path.exists(f):
            continue
        d = np.load(f); cols = list(d["cols"])
        for fd in d["fdoms"]:
            for row in d[f"rows_fdom{fd}"]:
                if row[0] not in qcuts:
                    continue
                g = lambda k: row[cols.index(k)]
                print(f"| {lab} | {'none' if not np.isfinite(fd) else f'{fd:g}'} | {g('qcut'):.0f} | {g('N_det_b'):,.0f} "
                      f"({g('N_det_b')/g('N_single')-1:+.2%}) | {g('B2_0.1'):.2%} | {g('B2_0.2'):.2%} | {g('B3'):.2%} | "
                      f"{g('lost'):.2%} | {g('meanbias'):+.2%} | {g('sat'):.2%} |")


if __name__ == "__main__":
    sigma_table()
    conf_table()
    for tag, fl in (("", "1e13"), ("", "1e14"), ("_subthr", "1e13")):
        frac_table(tag, fl)
    width_table()
    gaussian_equivalent()
    ztilt_table()
    dominance_table()
