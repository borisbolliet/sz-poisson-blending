"""Arbiter / a04: a third map-sigma variant, (iii) the PLAIN standard deviation of the filtered map, through the build's
kernel route (maps verifier, bugs item 3: with the plain std as sigma_eff, config a moves to -2.8 / -4.3 sigma).

(iii)  sigma_eff = sqrt(sigma_Y^2 + all-halo confusion variance)  ->  r_iii(theta) = sqrt(1 + v_all(theta)),
v_all = Parseval variance of all halos M500 > 1e13 (data_real/step2/inputs.npz <run>_vpars_all, p01): the expectation of
the variance of the filtered map (noise + halos).  Everything else exactly as the build's (i)/(ii) variants:
  kernel K'(theta, q'; e) = P(dom at q_A = r q', q_A + n + delta - mu > r e) (s2lib.kernel_node, fdom 1, s09 rates),
  truth vector Delta = N_b - N_n with the sigma_eff naive model (s2lib.delta_counts, as p05),
  Asimov ML fits of the naive sigma_eff model, configs a, b, c, d, d3, d5, e, SO40 / full, q_cut 5 / 6 (p06's _job).
Output: data_real/step2/kernel_<run>_effiii.npz, truth_<run>_effiii.npz, fits_arbiter_effiii.json; a04_mapsigma_plainstd.log
"""
import os, sys, time, json
NPROC = 6
for v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "NUMBA_NUM_THREADS"):
    os.environ[v] = "1"
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import s2lib as S                                   # noqa: E402
from s2lib import CL                                 # noqa: E402
import p06_fits as P6                                # noqa: E402  (its fit job and data loader; nothing is overwritten)

HERE = os.path.dirname(os.path.abspath(__file__))
RUNS = ("baseline", "goal")


def _kjob(a):
    i, o, mu, extra, qa, thr = a
    K, Pd = S.kernel_node(o, mu, extra, qa, thr, 1.0)
    return i, K, Pd


def main():
    from multiprocessing import get_context
    t0 = time.time()
    LOG = open(os.path.join(HERE, "a04_mapsigma_plainstd.log"), "w")

    def log(s=""):
        print(s, flush=True); LOG.write(s + "\n"); LOG.flush()

    I = np.load(f"{S.D2}/inputs.npz")
    wr = np.load(f"{S.ROOT}/verify_real/step2_maps/out/w03_rvariants.npz")
    # ---------------------------------------------------------------- kernels
    for run in RUNS:
        th, sig, R = S.load_s09_rates(run)
        r = np.sqrt(1.0 + I[f"{run}_vpars_all"])
        mu, extra = I[f"{run}_mu"], I[f"{run}_extra"]
        log(f"[{run}] r_iii = sqrt(1 + v_all): {r.min():.4f}-{r.max():.4f} over the 48 nodes; at 0.25-40.5': "
            f"{r[(th > 0.24) & (th < 41)].min():.4f}-{r[(th > 0.24) & (th < 41)].max():.4f}; r_i {I[f'{run}_r_i'].min():.4f}-"
            f"{I[f'{run}_r_i'].max():.4f}, r_ii {I[f'{run}_r_ii'].min():.4f}-{I[f'{run}_r_ii'].max():.4f}")
        k = f"r_{run}_std"
        if k in wr.files:
            rm = wr[k]
            log(f"   maps verifier's measured plain std (w03, interpolated to the 48 nodes): ratio to r_iii "
                f"{(rm / r)[(th > 0.24) & (th < 41)].min():.4f}-{(rm / r)[(th > 0.24) & (th < 41)].max():.4f} at 0.25-40.5'")
        jobs = [(i, R[i], float(mu[i]), float(extra[i]), S.QA * r[i], S.EDGES * r[i]) for i in range(len(th))]
        K = np.zeros((len(th), len(S.QA), len(S.EDGES))); Pd = np.zeros((len(th), len(S.QA)))
        with get_context("spawn").Pool(NPROC) as p:
            for i, Ki, Pi in p.imap_unordered(_kjob, jobs, chunksize=1):
                K[i] = Ki; Pd[i] = Pi
        assert np.all(np.isfinite(K)) and K.min() >= -1e-12 and K.max() <= 1 + 1e-12
        np.savez(f"{S.D2}/kernel_{run}_effiii.npz", Kb=K, Pdom=Pd, th=th, QA=S.QA, EDGES=S.EDGES, r=r, fdom=1.0, mu=mu,
                 extra=extra, eff=True, desc=f"SO {run} MMF, s09 rates, map-sigma (iii) plain std sqrt(1 + v_all), fdom 1")
        log(f"   kernel_{run}_effiii.npz: min dK/dq {np.diff(K, axis=1).min():+.1e}, max dK/de {np.diff(K, axis=2).max():+.1e} "
            f"[{time.time()-t0:.0f}s]")
    # ---------------------------------------------------------------- truth vectors (as p05)
    G = CL.HMFGrid()
    zed = CL.default_bins(5.0)[0]
    ZG = ((0.01, 0.1), (0.1, 0.5), (0.5, 1.0), (1.0, 1.5), (1.5, 2.0), (2.0, 3.0))
    for run in RUNS:
        k = np.load(f"{S.D2}/kernel_{run}_effiii.npz")
        out = {}
        for qc in (5.0, 6.0):
            s = f"{qc:.0f}"
            M = S.SmoothModelS(qc, S.LnSigma(S.sz_model(run), k["th"], k["r"]), grid=G)
            Nb, Nn, N0 = S.delta_counts(M, k["Kb"], k["th"], k["QA"], k["EDGES"], S.FID)
            Nb5, Nn5, _ = S.delta_counts(M, k["Kb"], k["th"], k["QA"], k["EDGES"], S.FID, qmin=0.5)
            D, D5 = Nb - Nn, Nb5 - Nn5
            out.update({f"D_q{s}": D, f"Nb_q{s}": Nb, f"Nn_q{s}": Nn, f"N0_q{s}": N0, f"D05_q{s}": D5, f"Nb05_q{s}": Nb5,
                        f"Nn05_q{s}": Nn5})
            if qc == 5.0:
                def zs(DD, NN, a, b):
                    sl = (zed[:-1] >= a - 1e-9) & (zed[1:] <= b + 1e-9)
                    return DD[sl].sum() / NN[sl].sum()
                out["tilt"] = np.array([zs(D, Nn, a, b) for a, b in ZG]); out["tilt05"] = np.array([zs(D5, Nn5, a, b) for a, b in ZG])
                pq = D.sum(0) / Nn.sum(0)
                log(f"[{run}_effiii] q>5: naive {N0.sum():,.1f} (vs sigma_Y model {np.load(f'{S.D2}/truth_{run}_f1.npz')['N0_q5'].sum():,.1f}); "
                    f"Delta {D.sum():+,.1f} = {D.sum()/Nn.sum():+.3%} (q_A >= 0.5: {D5.sum()/Nn5.sum():+.3%}); z tilt "
                    + " ".join(f"{100*v:+.2f}%" for v in out["tilt"]) + f"; q 5-7.3 {100*pq[0]:+.2f}%, 7.3-10.6 {100*pq[1]:+.2f}%, "
                    f">47 {100*D[:, 6:].sum()/Nn[:, 6:].sum():+.2f}%")
        np.savez(f"{S.D2}/truth_{run}_effiii.npz", truth=S.FID.copy(), desc=str(k["desc"]), **out)
    log(f"   truth vectors saved [{time.time()-t0:.0f}s]")
    # ---------------------------------------------------------------- fits (p06's job function)
    setups = [f"{run}_effiii" for run in RUNS]
    tasks = [(s, "cat", qc, sv, cn, "truth") for s in setups for qc in P6.QCUTS for sv in P6.SURVEYS for cn in P6.CFGS]
    tasks += [(s, "self", 5.0, sv, cn, "truth") for s in setups for sv in P6.SURVEYS for cn in ("a", "d", "e")]
    tasks += [(s, "self", 5.0, "SO40", cn, "offset") for s in setups for cn in ("a", "d")]
    rows = []
    with get_context("spawn").Pool(NPROC) as p:
        for r in p.imap_unordered(P6._job, tasks, chunksize=1):
            rows.append(r)
    json.dump(rows, open(f"{S.D2}/fits_arbiter_effiii.json", "w"), indent=1)
    rows.sort(key=lambda r: (r["route"] != "cat", r["setup"], r["qcut"], r["survey"], P6.CFGS.index(r["config"]), r["start"]))
    log(f"\n=== fits ({len(rows)}) ===")
    for r in rows:
        log(P6.line(r))
    sr = [r for r in rows if r["route"] == "self"]
    log(f"\nself route: max |bias| {max(abs(r[f'b_{p}']) for r in sr for p in r['free']):.1e} sigma over {len(sr)} fits")
    log("\n=== summary: S8 bias (ML) for the plain-std variant vs the maps verifier's analyst variant 'std' (w07, linearised) ===")
    mv = {("baseline", "a", "SO40"): -2.76, ("baseline", "d", "SO40"): +0.27, ("baseline", "d", "full"): +0.39,
          ("baseline", "e", "SO40"): +0.26, ("baseline", "e", "full"): +0.41, ("goal", "a", "SO40"): -4.26,
          ("goal", "d", "SO40"): +0.48, ("goal", "d", "full"): +0.74, ("goal", "e", "SO40"): +0.52, ("goal", "e", "full"): +0.82}
    for r in rows:
        if r["route"] == "cat" and r["qcut"] == 5.0 and r["config"] in ("a", "b", "c", "d", "e"):
            run = r["setup"].split("_")[0]
            ref = mv.get((run, r["config"], r["survey"]))
            log(f"   [{r['setup']}] {r['survey']:4s} {r['config']}: S8 {r['b_S8']:+.2f} (lin {r['lin_S8']:+.2f}); Om {r['b_Om']:+.2f}, "
                f"s8 {r['b_s8']:+.2f}; chi2min {r['dev']:.1f}, power {r['power']:.2f}" + ("" if ref is None else f" | maps verifier {ref:+.2f}"))
    log(f"\n[a04] done {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
