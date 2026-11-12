"""Step 3b / b09: cosmology bias for the finder's own catalogue.

The analyst's naive model per variant and noise level (s3bcosmo.SmoothModelK): step 2's count model with
sigma_eff = rbar sigma_Y and the variant's no-blending kernel K (b05).  Data = mu_naive(truth) x R(z, q) (Asimov, full
sky; scaled by f_sky in the fit), with R from b07: R (N_det / N_exp), R_p (paired, per bin) or R_s (pooled paired
correction).  Fits as step 2's p06 (s2lib.run_fit: Fisher scoring + bounded L-BFGS-B polish): configs a, b, c, d, d3,
d5, e; SO40 (f_sky 0.4) and full sky; q_cut 5 and 6.
  ML fits: every setup with R_s; the headline (prod_none) also with R and R_p; the self route (data = mu_naive(truth),
  must return the truth) per model.
  Linearised biases F^-1 J^T (R - 1) for every setup, estimator, config, survey and q_cut, with their tile-bootstrap
  errors (the b07 resamples) and p06b's split into amplitude, z tilt and q shape.
  Kernel convergence: the headline's linearised biases with the kernels of the even and of the odd noise-only tiles
  (model with the half kernel; R recomputed with N_exp from the half kernel; R_s is a ratio of model counts).
Outputs: data_real/step3b/fits.json (ML rows), data_real/step3b/lin.json (linearised rows).  Usage: python b09_fits.py NPROC
"""
import os, sys, time, json
NPROC = int(sys.argv[1]) if len(sys.argv) > 1 else 6
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import s3blib as B                                 # noqa: E402
B.threads(1)
import numpy as np                                 # noqa: E402
import s3bcosmo as SC                              # noqa: E402
S2, CL = SC.S2, SC.CL
SETUPS = ["prod_none", "prod_closest1", "prod_fof10", "prod_keepnr", "szifi3000_none", "szifi12000_none"]
LIN_ONLY = ["prod_none_q2"]          # the headline on the 324 Q2 tiles alone (linearised only)
EST = {"R": "R", "Rp": "Rp", "Rs": "Rs"}
_W = {}
VAL = json.load(open(os.path.join(B.OUT, "validate.json")))


def model(variant, mode, qc, kernel="virtual"):
    key = (variant, mode, qc, kernel)
    if key not in _W:
        for k in [k for k in _W if k[0] not in ("dv",)][:-1]:         # keep at most 2 models per process
            del _W[k]
        M = SC.model(variant, mode, qc, kernel=kernel)
        _W[key] = (M, M.counts(SC.FID))
    return _W[key]


def datavec(setup, mode):
    key = ("dv", setup, mode)
    if key not in _W:
        _W[key] = dict(np.load(os.path.join(B.OUT, f"datavec_{setup}_{mode}.npz")))
    return _W[key]


def _job(task):
    setup, mode, est, qc, sv, cn, route = task
    t = time.time()
    variant = setup.split("_")[0]
    M, N0 = model(variant, mode, qc)
    if route == "self":
        data = N0.copy()
    else:
        R = datavec(setup, mode)[f"{est}_q{qc:.0f}"]
        data = N0 * R
    nneg = int(np.sum(data < 0))
    data = np.clip(data, 0, None)
    r = S2.run_fit(M, data, cn, SC.FID, SC.SURVEYS[sv], polish=True)
    r.update(setup=setup, mode=mode, estimator=est if route == "cat" else "-", qcut=qc, survey=sv, route=route, nneg=nneg,
             sec=time.time() - t)
    return r


def line(r):
    nui = " ".join(f"{p}:{r['b_' + p]:+.2f}" for p in r["free"][2:])
    return (f"q{r['qcut']:.0f} {r['survey']:4s} {r['config']:2s} {r['setup']:16s} {r['mode']:8s} {r['route']:4s} {r['estimator']:2s}: "
            f"S8 {r['b_S8']:+.3f} (lin {r['lin_S8']:+.3f})  Om {r['b_Om']:+.2f}  s8 {r['b_s8']:+.2f} | sig(S8) {r['s_S8']:.4f} | {nui} | "
            f"chi2min {r['dev']:.2f} power {r['power']:.3f} [it {r['nit']}, at bound {r['at_bound']}, polish shift {r['polish_shift']:.3f}]")


def linear_all():
    rows = []
    Lc = {}
    for setup in SETUPS + LIN_ONLY:
        variant = setup.split("_")[0]
        for mode in B.MODES:
            dv = datavec(setup, mode)
            for qc in SC.QCUTS:
                M, N0 = model(variant, mode, qc)
                s = f"q{qc:.0f}"
                for sv, fs in SC.SURVEYS.items():
                    for cn in SC.CFGS:
                        if (variant, mode, qc, sv, cn) not in Lc:
                            Lc[(variant, mode, qc, sv, cn)] = SC.linear_parts(M, SC.FID, dv[f"Rs_{s}"], cn, fs)
                        L = Lc[(variant, mode, qc, sv, cn)]
                        for est in EST:
                            R = dv[f"{est}_{s}"]; RB = dv[f"{est}boot_{s}"].astype(float)
                            b, bo, bs, _ = SC.lin_bias(L, R)
                            bb, bob, bsb, _ = SC.lin_bias(L, RB)
                            parts = SC.split_parts(L, R) if cn in ("a", "b", "d", "e") else {}
                            rows.append(dict(setup=setup, mode=mode, estimator=est, qcut=qc, survey=sv, config=cn,
                                             lin_S8=float(b[0]), lin_Om=float(bo[0]), lin_s8=float(bs[0]),
                                             boot_S8=float(bb.std()), boot_Om=float(bob.std()), boot_s8=float(bsb.std()),
                                             boot_mean_S8=float(bb.mean()), s_S8=L["sig"]["S8"], parts=parts))
                        if setup == "prod_none":
                            # the injection-tile reference: N_exp x (1 + delta), delta = pooled t03 + a02 halo-weighted
                            # obs/exp - 1 against the virtual kernel (b06); R_s / (1 + delta), error from delta's error
                            dl = VAL[f"{mode}_pooled"]
                            R = dv[f"Rs_{s}"] / (1 + dl["halo"])
                            b, bo, bs, _ = SC.lin_bias(L, R)
                            dd = SC.lin_bias(L, dv[f"Rs_{s}"] / (1 + dl["halo"] + dl["halo_err"]))[0][0] - b[0]
                            rows.append(dict(setup=setup, mode=mode, estimator="Rs_injref", qcut=qc, survey=sv, config=cn,
                                             lin_S8=float(b[0]), lin_Om=float(bo[0]), lin_s8=float(bs[0]), boot_S8=float(abs(dd)),
                                             delta=dl["halo"], delta_err=dl["halo_err"], s_S8=L["sig"]["S8"], parts={}))
                            b, bo, bs, _ = SC.lin_bias(L, dv[f"Rs_coarse_{s}"])                # cell-grid check
                            rows.append(dict(setup=setup, mode=mode, estimator="Rs_coarse", qcut=qc, survey=sv, config=cn,
                                             lin_S8=float(b[0]), lin_Om=float(bo[0]), lin_s8=float(bs[0]), boot_S8=float("nan"),
                                             s_S8=L["sig"]["S8"], parts={}))
    return rows


def convergence():
    """Headline, q_cut 5: linearised S8 bias with the kernels of the even and of the odd noise-only tiles.  For R the
    expectation N_exp is recomputed with the half kernel on the 324 Q2 tiles (R_half = N_det / N_exp,half; the full-kernel
    value of the same tiles is printed for reference) and the model uses the same half kernel; R_s does not depend on the
    kernel table except through mu (a ratio), so the same R_s is used."""
    import b07_datavec as D7
    out = []
    for mode in B.MODES:
        dv = datavec("prod_none", mode)
        dq2 = datavec("prod_none_q2", mode)          # R with the half kernels: the 324 Q2 tiles (their N_det and halos)
        C = np.load(os.path.join(B.OUT, f"halos_{mode}.npz")); V = np.load(os.path.join(B.OUT, f"halos_prod_{mode}.npz"))
        pid, z, th, qp = C["pid"], C["z"].astype(float), C["theta500"].astype(float), V["qp"].astype(float)
        zed, qed = CL.default_bins(5.0)
        sel = [int(np.argmin(np.abs(B.EDGES - e))) for e in qed[:-1]]
        d = np.load(os.path.join(B.OUT, f"kernel_prod_{mode}.npz"))
        for half in ("even", "odd"):
            Kh = np.zeros((qp.size, B.EDGES.size)); m = qp >= B.CRED_MIN
            Kh[m] = B.Kernel(d["th"], d["QA"], d["K_" + half])(th[m], qp[m])
            Rh = D7.ratio(dq2["Ndet_q5"], D7.per_tile_exp(pid, z, Kh, sel, zed).sum(0))
            M = SC.model("prod", mode, 5.0, kernel=half)
            for sv, fs in SC.SURVEYS.items():
                for cn in ("a", "d", "e"):
                    L = SC.linear_parts(M, SC.FID, dv["Rs_q5"], cn, fs)
                    out.append(dict(mode=mode, half=half, survey=sv, config=cn, lin_S8=float(SC.lin_bias(L, dv["Rs_q5"])[0][0]),
                                    lin_S8_R=float(SC.lin_bias(L, Rh)[0][0]), lin_S8_R_full=float(SC.lin_bias(L, dq2["R_q5"])[0][0])))
    return out


if __name__ == "__main__":
    from multiprocessing import get_context
    t0 = time.time()
    tasks = []
    for setup in SETUPS:
        for mode in B.MODES:
            for qc in SC.QCUTS:
                for sv in SC.SURVEYS:
                    for cn in SC.CFGS:
                        tasks.append((setup, mode, "Rs", qc, sv, cn, "cat"))
                        if setup == "prod_none":
                            tasks += [(setup, mode, "R", qc, sv, cn, "cat"), (setup, mode, "Rp", qc, sv, cn, "cat")]
                        if setup in ("prod_none", "szifi3000_none", "szifi12000_none"):
                            tasks.append((setup, mode, "-", qc, sv, cn, "self"))
    print(f"b09: {len(tasks)} ML fits on {NPROC} processes", flush=True)
    rows = []
    with get_context("spawn").Pool(NPROC) as p:
        for k, r in enumerate(p.imap_unordered(_job, tasks, chunksize=1)):
            rows.append(r)
            if k % 50 == 49:
                print(f"   {k + 1}/{len(tasks)} fits [{time.time() - t0:.0f} s]", flush=True)
    json.dump(rows, open(os.path.join(B.OUT, "fits.json"), "w"), indent=1)
    key = lambda r: (r["qcut"], r["survey"] != "SO40", SC.CFGS.index(r["config"]), r["route"] != "cat", SETUPS.index(r["setup"]),
                     B.MODES.index(r["mode"]), r["estimator"])
    rows.sort(key=key)
    cur = None
    for r in rows:
        if (r["qcut"], r["survey"]) != cur:
            cur = (r["qcut"], r["survey"]); print(f"\n######## q_cut = {r['qcut']:.0f}, {r['survey']}")
        print(line(r))
    sr = [r for r in rows if r["route"] == "self"]
    print(f"\nSANITY: naive model fitted to its own expectation: max |bias| over {len(sr)} fits and all free parameters "
          f"{max(abs(r['b_' + p]) for r in sr for p in r['free']):.2e} sigma; max chi2min {max(r['dev'] for r in sr):.2e}")
    cr = [r for r in rows if r["route"] == "cat"]
    rel = [abs(r["b_S8"] - r["lin_S8"]) / max(abs(r["b_S8"]), 1e-9) for r in cr if abs(r["b_S8"]) >= 0.1]
    big = [(r["setup"], r["mode"], r["estimator"], r["qcut"], r["survey"], r["config"], round(r["b_S8"], 3), round(r["lin_S8"], 3))
           for r in cr if abs(r["b_S8"]) >= 0.1 and abs(r["b_S8"] - r["lin_S8"]) > 0.1 * abs(r["b_S8"])]
    print(f"ML vs linearised S8 bias over {len(rel)} fits with |bias| >= 0.1 sigma: median |ML - lin| / |ML| {np.median(rel):.3f}, "
          f"max {max(rel):.3f}; {len(big)} differ by more than 10%: {big[:12]}")
    absd = max(abs(r["b_S8"] - r["lin_S8"]) for r in cr)
    print(f"   max |ML - lin| over all {len(cr)} fits: {absd:.3f} sigma; fits with a parameter at a bound: "
          f"{sum(bool(r['at_bound']) for r in cr)}; data bins < 0 clipped: {max(r['nneg'] for r in rows)}")
    t1 = time.time()
    L = linear_all()
    json.dump(L, open(os.path.join(B.OUT, "lin.json"), "w"), indent=1)
    print(f"\nlinearised biases and tile-bootstrap errors: {len(L)} rows [{time.time() - t1:.0f} s]")
    print("S8 bias (linearised) +- tile-bootstrap error, q_cut 5 [R | R_p | R_s]:")
    for setup in SETUPS + LIN_ONLY:
        for mode in B.MODES:
            for sv in SC.SURVEYS:
                parts = []
                for cn in ("a", "b", "c", "d", "e"):
                    v = {r["estimator"]: r for r in L if r["setup"] == setup and r["mode"] == mode and r["survey"] == sv
                         and r["config"] == cn and r["qcut"] == 5.0}
                    parts.append(f"{cn}: " + " | ".join(f"{v[e]['lin_S8']:+.2f}+-{v[e]['boot_S8']:.2f}" for e in ("R", "Rp", "Rs")))
                print(f"  {setup:16s} {mode:8s} {sv:4s}: " + ";  ".join(parts))
    print("precision target (bootstrap error of the S8 bias <= 0.1 sigma, SO40, headline configs d and e):")
    for mode in B.MODES:
        nt = int(datavec("prod_none", mode)["ntiles"])
        for cn in ("d", "e"):
            for setup, lab in (("prod_none_q2", "324 Q2 tiles"), ("prod_none", f"{nt} tiles")):
                v = {r["estimator"]: r for r in L if r["setup"] == setup and r["mode"] == mode and r["survey"] == "SO40"
                     and r["config"] == cn and r["qcut"] == 5.0}
                print(f"  {mode} {cn}, {lab:13s}: R {v['R']['boot_S8']:.3f}, R_p {v['Rp']['boot_S8']:.3f}, R_s {v['Rs']['boot_S8']:.3f} -> "
                      f"{'met' if v['Rs']['boot_S8'] <= 0.1 else 'NOT met'} with R_s")
    print("injection-tile reference (pooled t03 + a02, b06): R_s / (1 + delta); linearised S8 bias, q_cut 5 (error: from delta's error)")
    for mode in B.MODES:
        for sv in SC.SURVEYS:
            cells = []
            for cn in ("a", "d", "e"):
                v = [r for r in L if r["setup"] == "prod_none" and r["mode"] == mode and r["survey"] == sv and r["config"] == cn
                     and r["qcut"] == 5.0 and r["estimator"] in ("Rs", "Rs_injref")]
                v = {r["estimator"]: r for r in v}
                cells.append(f"{cn}: {v['Rs']['lin_S8']:+.2f} -> {v['Rs_injref']['lin_S8']:+.2f} +- {v['Rs_injref']['boot_S8']:.2f}")
            print(f"  {mode:8s} {sv:4s} (delta = {100 * v['Rs_injref']['delta']:+.2f} +- {100 * v['Rs_injref']['delta_err']:.2f}%): " + ";  ".join(cells))
    print("cell-grid check (headline): linearised S8 bias with R_s from every other q' node vs the full grid, q_cut 5 / 6")
    for mode in B.MODES:
        for sv in SC.SURVEYS:
            cells = []
            for cn in ("a", "d", "e"):
                for qc in SC.QCUTS:
                    v = {r["estimator"]: r for r in L if r["setup"] == "prod_none" and r["mode"] == mode and r["survey"] == sv
                         and r["config"] == cn and r["qcut"] == qc and r["estimator"] in ("Rs", "Rs_coarse")}
                    cells.append(f"{cn} q{qc:.0f}: {v['Rs_coarse']['lin_S8']:+.3f} vs {v['Rs']['lin_S8']:+.3f}")
            print(f"  {mode:8s} {sv:4s}: " + "; ".join(cells))
    print("split of the linearised S8 bias (R_s, q_cut 5): amplitude + z tilt + q shape = sum")
    for setup in SETUPS:
        for mode in B.MODES:
            for sv in SC.SURVEYS:
                for cn in ("a", "d", "e"):
                    r = [x for x in L if x["setup"] == setup and x["mode"] == mode and x["survey"] == sv and x["config"] == cn
                         and x["qcut"] == 5.0 and x["estimator"] == "Rs"][0]
                    p = r["parts"]
                    print(f"  {setup:16s} {mode:8s} {sv:4s} {cn}: {p['amplitude']:+.3f} {p['z tilt']:+.3f} {p['q shape']:+.3f} = {p['sum']:+.3f}")
    cv = convergence()
    json.dump(cv, open(os.path.join(B.OUT, "lin_convergence.json"), "w"), indent=1)
    print("kernel convergence (headline, q_cut 5): linearised S8 bias with the even- / odd-tile kernels [R_s, all tiles; R, the 324 Q2 "
          "tiles (full kernel in brackets)]:")
    for mode in B.MODES:
        for sv in SC.SURVEYS:
            for cn in ("a", "d", "e"):
                a = [x for x in cv if x["mode"] == mode and x["survey"] == sv and x["config"] == cn]
                e_, o_ = [x for x in a if x["half"] == "even"][0], [x for x in a if x["half"] == "odd"][0]
                print(f"  {mode:8s} {sv:4s} {cn}: even {e_['lin_S8']:+.3f} / odd {o_['lin_S8']:+.3f} (diff {e_['lin_S8'] - o_['lin_S8']:+.3f}); "
                      f"R: {e_['lin_S8_R']:+.3f} / {o_['lin_S8_R']:+.3f} (diff {e_['lin_S8_R'] - o_['lin_S8_R']:+.3f}) [full {e_['lin_S8_R_full']:+.3f}]")
    print(f"total {time.time() - t0:.0f} s; cpu in fits {sum(r['sec'] for r in rows):.0f} s")
