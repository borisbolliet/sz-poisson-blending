"""Step 2 / p05: truth data vectors Delta(z, q) = N_b - N_n in the cosmology bins (16 z x 9 q_obs, q_cut = 5 and 6),
built exactly as c11: for each theta node the blended kernel K_b and the naive kernel Phi(q - e), tabulated on the
SAME q grid, are interpolated onto the fine ln q grid of the count model, differenced into q_obs bins and convolved
with the lognormal Y scatter; counts integrate them over the (z, lnM) nodes (linear in ln theta between nodes).
   data(z, q) = mu_naive(truth) + Delta     (interpolation errors cancel to first order)
Models: old white filter -> cosmolib2.SmoothModel; MMF -> SmoothModelS (MMF sigma_Y); map-sigma -> sigma_eff = r sigma_Y
(the r of the kernel file); low-pressure -> truth (1-b) = 0.6; S8 truths -> sigma8 of the truth (Om fixed).
Checks: (A) white vs Q3's truth_catalogue_v2.npz (and the pre-fix-rate kernel); (B) q_A grid (half grid; 2x finer
grid); (C) theta grid (every other node); (D) primaries with q_A < 0.5 (excluded in step 1); (E) totals and the redshift
tilt vs step 1's s09 P(D) expectation.
Usage: p05_truth.py [setup ...] (default: all kernels present).  Output: data_real/step2/truth_<setup>.npz
"""
import os, sys, time, glob
for v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "NUMBA_NUM_THREADS"):
    os.environ[v] = "4"
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import s2lib as S                                  # noqa: E402
from s2lib import CL                                # noqa: E402

ORDER = ["white", "white_prefix", "baseline_f1", "baseline_f0.5", "baseline_f2", "goal_f1", "goal_f0.5", "goal_f2",
         "baseline_effi", "baseline_effii", "goal_effi", "goal_effii", "lowP_f1", "lowP_effi", "S8_0.75_f1", "S8_0.87_f1",
         "baseline_f1_fineQA"]
G = None


def setup_model(name, qc):
    """count model and truth vector for a setup (kernel file gives th, r)."""
    k = np.load(f"{S.D2}/kernel_{name}.npz")
    truth = S.FID.copy()
    if name.startswith("white"):
        lnsig = S.LnSigma(S.sz_model("white"))
    else:
        run = "goal" if name.startswith("goal") else "baseline"
        eff = bool(k["eff"])
        lnsig = S.LnSigma(S.sz_model(run), k["th"], k["r"]) if eff else S.LnSigma(S.sz_model(run))
        if name.startswith("lowP"):
            truth = S.FID_LOW.copy()
        if name.startswith("S8_"):
            truth[1] = S.s8_from_S8(float(name.split("_")[1]))
    return S.SmoothModelS(qc, lnsig, grid=G), truth, k


def zsum(D, N, zed, a, b):
    s = (zed[:-1] >= a - 1e-9) & (zed[1:] <= b + 1e-9)
    return D[s].sum() / N[s].sum()


def main(names):
    global G
    t0 = time.time()
    G = CL.HMFGrid()
    zed, _ = CL.default_bins(5.0)
    res = {}
    for name in names:
        out = {}
        for qc in (5.0, 6.0):
            s = f"{qc:.0f}"
            M, truth, k = setup_model(name, qc)
            Nb, Nn, N0 = S.delta_counts(M, k["Kb"], k["th"], k["QA"], k["EDGES"], truth)
            Nb5, Nn5, _ = S.delta_counts(M, k["Kb"], k["th"], k["QA"], k["EDGES"], truth, qmin=0.5)
            D = Nb - Nn; D5 = Nb5 - Nn5
            out.update({f"D_q{s}": D, f"Nb_q{s}": Nb, f"Nn_q{s}": Nn, f"N0_q{s}": N0, f"D05_q{s}": D5, f"Nb05_q{s}": Nb5,
                        f"Nn05_q{s}": Nn5})
            pz = D.sum(1) / Nn.sum(1); pq = D.sum(0) / Nn.sum(0)
            sig_full = D / np.sqrt(np.maximum(N0, 1e-9)); sig40 = sig_full * np.sqrt(0.4)
            print(f"=== [{name}] q_cut = {qc:.0f} (truth {np.round(truth, 4).tolist()}) ===")
            print(f"  naive smooth {N0.sum():,.1f} | naive via kernel grid {Nn.sum():,.1f} (rel {Nn.sum()/N0.sum()-1:+.2e}; per-bin max "
                  f"|rel| (mu>10) {np.abs(Nn/N0-1)[N0 > 10].max():.2e}) | blended {Nb.sum():,.1f}")
            print(f"  Delta total {D.sum():+,.1f} = {D.sum()/Nn.sum():+.4%} of N (primaries q_A >= 0.5 only: {D5.sum():+,.1f} = "
                  f"{D5.sum()/Nn5.sum():+.4%}); SO40: {0.4*D.sum():+,.1f}")
            print(f"  Delta/N per z bin: " + " ".join(f"{v:+.4f}" for v in pz))
            print(f"  Delta/N per q bin: " + " ".join(f"{v:+.4f}" for v in pq))
            print(f"  Delta / Poisson sigma per bin: full sky max |.| {np.abs(sig_full).max():.2f}, rms {np.sqrt(np.mean(sig_full**2)):.2f}; "
                  f"SO40 max {np.abs(sig40).max():.2f}, rms {np.sqrt(np.mean(sig40**2)):.2f}; total chi2 (full, Delta^2/N) "
                  f"{np.sum(sig_full**2):.1f}", flush=True)
            if qc == 5.0:
                tilt = [zsum(D, Nn, zed, a, b) for a, b in ((0.01, 0.1), (0.1, 0.5), (0.5, 1.0), (1.0, 1.5), (1.5, 2.0), (2.0, 3.0))]
                tilt5 = [zsum(D5, Nn5, zed, a, b) for a, b in ((0.01, 0.1), (0.1, 0.5), (0.5, 1.0), (1.0, 1.5), (1.5, 2.0), (2.0, 3.0))]
                print(f"  z tilt Delta/N in z = 0.01-0.1, 0.1-0.5, 0.5-1, 1-1.5, 1.5-2, 2-3: " + " ".join(f"{100*v:+.2f}%" for v in tilt)
                      + "  | q_A >= 0.5: " + " ".join(f"{100*v:+.2f}%" for v in tilt5), flush=True)
                out["tilt"] = np.array(tilt); out["tilt05"] = np.array(tilt5)
        np.savez(f"{S.D2}/truth_{name}.npz", truth=truth, desc=str(k["desc"]), **out)
        res[name] = out
        print(f"  saved truth_{name}.npz  [{time.time()-t0:.0f}s]\n", flush=True)

    # ------------------------------------------------------------------ checks
    if "white" in res:
        print("=== (A) regression: old white filter vs Q3 (data_cosmo/truth_catalogue_v2.npz, c11) ===")
        q3 = np.load(f"{S.ROOT}/data_cosmo/truth_catalogue_v2.npz")
        for name in ("white", "white_prefix"):
            if name not in res:
                continue
            for s in ("5", "6"):
                D, Dq = res[name][f"D_q{s}"], q3[f"D_q{s}"]
                N0 = res[name][f"N0_q{s}"]
                dsig = np.abs(D - Dq) * 0.4 / np.sqrt(0.4 * N0 + 1e-9)
                print(f"  [{name}] q>{s}: total Delta {D.sum():+,.1f} vs Q3 {Dq.sum():+,.1f} (diff {D.sum()-Dq.sum():+,.1f} = "
                      f"{(D.sum()-Dq.sum())/N0.sum():+.2e} of N); per-bin |dDelta| in Poisson sigma at 40% sky: max {dsig.max():.3f}, "
                      f"rms {np.sqrt(np.mean(dsig**2)):.3f}; at full sky max {dsig.max()/np.sqrt(0.4):.3f}; max |dDelta|/|Delta_Q3| over "
                      f"bins with |Delta_Q3| > 20: {np.max(np.abs(D-Dq)[np.abs(Dq) > 20]/np.abs(Dq)[np.abs(Dq) > 20]):.3f}")
                print(f"     Delta/N per z bin  step 2: " + " ".join(f"{v:+.4f}" for v in D.sum(1) / res[name][f'Nn_q{s}'].sum(1)))
                print(f"                        Q3    : " + " ".join(f"{v:+.4f}" for v in Dq.sum(1) / q3[f'Nn_q{s}'].sum(1)))
                print(f"     Delta/N per q bin  step 2: " + " ".join(f"{v:+.4f}" for v in D.sum(0) / res[name][f'Nn_q{s}'].sum(0)))
                print(f"                        Q3    : " + " ".join(f"{v:+.4f}" for v in Dq.sum(0) / q3[f'Nn_q{s}'].sum(0)), flush=True)
        if "white_prefix" in res:
            for s in ("5", "6"):
                a, b = res["white"][f"D_q{s}"], res["white_prefix"][f"D_q{s}"]
                N0 = res["white"][f"N0_q{s}"]
                print(f"  fixed vs pre-fix rates (same step-2 code), q>{s}: total {a.sum()-b.sum():+,.1f}; per-bin max |diff| "
                      f"{np.max(np.abs(a-b)*0.4/np.sqrt(0.4*N0)):.3f} Poisson sigma (40% sky)")

    def compare(lab, D, Dref, N0):
        x = np.abs(D - Dref) / np.sqrt(N0 + 1e-9)
        print(f"  {lab}: total {D.sum()-Dref.sum():+.2f} ({(D.sum()-Dref.sum())/N0.sum():+.1e} of N); per bin max |dDelta| "
              f"{np.abs(D-Dref).max():.2f} counts = {x.max():.4f} Poisson sigma (full sky); rms {np.sqrt(np.mean(x**2)):.4f}", flush=True)

    print("\n=== (B) q_A grid convergence (full sky, q_cut 5) ===")
    for name in ("baseline_f1", "white", "goal_f1"):
        if name not in res:
            continue
        M, truth, k = setup_model(name, 5.0)
        Nb, Nn, N0 = S.delta_counts(M, k["Kb"][:, ::2], k["th"], k["QA"][::2], k["EDGES"], truth)
        compare(f"[{name}] half q_A grid (every other node) vs production", Nb - Nn, res[name]["D_q5"], N0)
    if "baseline_f1_fineQA" in res and "baseline_f1" in res:
        compare("[baseline_f1] production vs 2x finer q_A grid", res["baseline_f1"]["D_q5"], res["baseline_f1_fineQA"]["D_q5"],
                res["baseline_f1"]["N0_q5"])
        compare("[baseline_f1] same, q_cut 6", res["baseline_f1"]["D_q6"], res["baseline_f1_fineQA"]["D_q6"], res["baseline_f1"]["N0_q6"])
    print("\n=== (C) theta grid convergence: every other theta node (interpolation error ~4x that of the full grid) ===")
    for name in ("baseline_f1", "goal_f1", "white", "baseline_effi"):
        if name not in res:
            continue
        M, truth, k = setup_model(name, 5.0)
        for off in (0, 1):
            th = k["th"][off::2]
            if name.endswith("effi"):
                M = S.SmoothModelS(5.0, S.LnSigma(S.sz_model("baseline"), th, k["r"][off::2]), grid=G)
            Nb, Nn, N0 = S.delta_counts(M, k["Kb"][off::2], th, k["QA"], k["EDGES"], truth)
            compare(f"[{name}] nodes {off}::2 ({len(th)} nodes) vs all {len(k['th'])}", Nb - Nn, res[name]["D_q5"], N0)
    print("\n=== (D) primaries with q_A < 0.5 (step 1 excluded them): Delta(all) - Delta(q_A >= 0.5) ===")
    for name, o in res.items():
        print(f"  [{name}] q>5: {o['D_q5'].sum()-o['D05_q5'].sum():+.2f} counts ({(o['D_q5'].sum()-o['D05_q5'].sum())/o['N0_q5'].sum():+.1e} "
              f"of N); max per bin {np.abs(o['D_q5']-o['D05_q5']).max():.2f}; q>6: {o['D_q6'].sum()-o['D05_q6'].sum():+.2f}")
    print("\n=== (E) step 1 (s09 P(D) expectation, primaries q_A > 0.5, szsim HMF table) vs step 2 (q_A >= 0.5), q > 5 ===")
    zb09 = [(0.01, 0.1), (0.1, 0.5), (0.5, 1.0), (1.0, 1.5), (1.5, 2.0), (2.0, 3.0)]
    for name, run, f in (("white", "white", 1.0), ("baseline_f1", "baseline", 1.0), ("baseline_f0.5", "baseline", 0.5),
                         ("baseline_f2", "baseline", 2.0), ("goal_f1", "goal", 1.0), ("goal_f0.5", "goal", 0.5),
                         ("goal_f2", "goal", 2.0)):
        if name not in res:
            continue
        d09 = np.load(f"{S.DR}/pd_tables_{run}.npz")
        row = d09[f"rows_fdom{f}"][1]
        o = res[name]
        print(f"  [{name}] N_det,b / N_single - 1: step 1 {row[2]/row[1]-1:+.4%} ({row[2]:,.0f} / {row[1]:,.0f}); step 2 "
              f"{o['Nb05_q5'].sum()/o['Nn05_q5'].sum()-1:+.4%} ({o['Nb05_q5'].sum():,.0f} / {o['Nn05_q5'].sum():,.0f}); all primaries "
              f"{o['Nb_q5'].sum()/o['Nn_q5'].sum()-1:+.4%}")
        if f == 1.0:
            Ns, Nd = d09["z_Ns"][1], d09["z_Nd"][1]
            t09 = [Nd[0] / Ns[0] - 1, (Nd[1] + Nd[2]) / (Ns[1] + Ns[2]) - 1] + [Nd[j] / Ns[j] - 1 for j in range(3, 7)]
            print(f"     z = " + ", ".join(f"{a}-{b}" for a, b in zb09) + ": step 1 " + " ".join(f"{100*v:+.2f}%" for v in t09)
                  + " | step 2 " + " ".join(f"{100*v:+.2f}%" for v in o["tilt05"]), flush=True)
    print(f"total {time.time()-t0:.0f}s")


if __name__ == "__main__":
    names = sys.argv[1:] or [n for n in ORDER if os.path.exists(f"{S.D2}/kernel_{n}.npz")]
    main(names)
