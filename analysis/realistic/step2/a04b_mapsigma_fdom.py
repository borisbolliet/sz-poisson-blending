"""Arbiter / a04b: the corners of the analyst choices.  The build ran the map-sigma variants only with the default
attribution rule (fdom = 1), and the fdom range only with sigma_Y.  Here both together, through the build's kernel route:
   {baseline, goal} x fdom {0.5, 2} x map-sigma {(i) sub-threshold confusion, (iii) plain std of the filtered map}.
(i) r_i = sqrt(1 + v_sub) (p01; close to a 3-sigma-clipped std of the map, maps verifier w03); (iii) r_iii = sqrt(1 + v_all)
(a04).  Kernels as p03 (s2lib.kernel_node with q_A = r q', thresholds r e, dominance x < fdom q_A), truth vectors as p05,
Asimov ML fits as p06 (configs a, b, c, d, d3, d5, e; SO40 / full; q_cut 5 / 6).
Output: data_real/step2/kernel_<setup>.npz, truth_<setup>.npz, fits_arbiter_fdom_eff.json; a04b_mapsigma_fdom.log
"""
import os, sys, time, json
NPROC = 6
for v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "NUMBA_NUM_THREADS"):
    os.environ[v] = "1"
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import s2lib as S                                   # noqa: E402
from s2lib import CL                                 # noqa: E402
import p06_fits as P6                                # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
SETUPS = [f"{run}_f{f}_eff{v}" for run in ("baseline", "goal") for f in ("0.5", "2") for v in ("i", "iii")]


def _kjob(a):
    name, i, o, mu, extra, qa, thr, fdom = a
    K, Pd = S.kernel_node(o, mu, extra, qa, thr, fdom)
    return name, i, K, Pd


def main():
    from multiprocessing import get_context
    t0 = time.time()
    LOG = open(os.path.join(HERE, "a04b_mapsigma_fdom.log"), "w")

    def log(s=""):
        print(s, flush=True); LOG.write(s + "\n"); LOG.flush()

    I = np.load(f"{S.D2}/inputs.npz")
    U = {}
    jobs = []
    for name in SETUPS:
        run, ftag, vtag = name.split("_")
        fdom = float(ftag[1:]); var = vtag[3:]
        th, sig, R = S.load_s09_rates(run)
        r = I[f"{run}_r_i"] if var == "i" else np.sqrt(1.0 + I[f"{run}_vpars_all"])
        U[name] = dict(th=th, r=r, fdom=fdom, mu=I[f"{run}_mu"], extra=I[f"{run}_extra"], run=run, var=var)
        jobs += [(name, i, R[i], float(I[f"{run}_mu"][i]), float(I[f"{run}_extra"][i]), S.QA * r[i], S.EDGES * r[i], fdom)
                 for i in range(len(th))]
    jobs.sort(key=lambda j: -j[7])
    res = {n: (np.zeros((48, len(S.QA), len(S.EDGES))), np.zeros((48, len(S.QA)))) for n in SETUPS}
    with get_context("spawn").Pool(NPROC) as p:
        for name, i, K, Pd in p.imap_unordered(_kjob, jobs, chunksize=1):
            res[name][0][i] = K; res[name][1][i] = Pd
    for name in SETUPS:
        K, Pd = res[name]; u = U[name]
        assert np.all(np.isfinite(K)) and K.min() >= -1e-12 and K.max() <= 1 + 1e-12
        np.savez(f"{S.D2}/kernel_{name}.npz", Kb=K, Pdom=Pd, th=u["th"], QA=S.QA, EDGES=S.EDGES, r=u["r"], fdom=u["fdom"],
                 mu=u["mu"], extra=u["extra"], eff=True,
                 desc=f"SO {u['run']} MMF, s09 rates, map-sigma ({u['var']}), fdom {u['fdom']}")
    log(f"[a04b] {len(jobs)} node kernels for {SETUPS} [{time.time()-t0:.0f}s]")
    G = CL.HMFGrid()
    zed = CL.default_bins(5.0)[0]
    for name in SETUPS:
        k = np.load(f"{S.D2}/kernel_{name}.npz")
        run = U[name]["run"]
        out = {}
        for qc in (5.0, 6.0):
            s = f"{qc:.0f}"
            M = S.SmoothModelS(qc, S.LnSigma(S.sz_model(run), k["th"], k["r"]), grid=G)
            Nb, Nn, N0 = S.delta_counts(M, k["Kb"], k["th"], k["QA"], k["EDGES"], S.FID)
            Nb5, Nn5, _ = S.delta_counts(M, k["Kb"], k["th"], k["QA"], k["EDGES"], S.FID, qmin=0.5)
            out.update({f"D_q{s}": Nb - Nn, f"Nb_q{s}": Nb, f"Nn_q{s}": Nn, f"N0_q{s}": N0, f"D05_q{s}": Nb5 - Nn5,
                        f"Nb05_q{s}": Nb5, f"Nn05_q{s}": Nn5})
        D, Nn = out["D_q5"], out["Nn_q5"]
        zs = [D[(zed[:-1] >= a - 1e-9) & (zed[1:] <= b + 1e-9)].sum() / Nn[(zed[:-1] >= a - 1e-9) & (zed[1:] <= b + 1e-9)].sum()
              for a, b in ((0.01, 0.1), (2.0, 3.0))]
        log(f"   [{name}] q>5: naive {out['N0_q5'].sum():,.1f}; Delta {D.sum():+,.1f} = {D.sum()/Nn.sum():+.3%}; z<0.1 {100*zs[0]:+.2f}% -> "
            f"z 2-3 {100*zs[1]:+.2f}%; q 5-7.3 {100*D[:, 0].sum()/Nn[:, 0].sum():+.2f}%")
        np.savez(f"{S.D2}/truth_{name}.npz", truth=S.FID.copy(), desc=str(k["desc"]), **out)
    tasks = [(s, "cat", qc, sv, cn, "truth") for s in SETUPS for qc in P6.QCUTS for sv in P6.SURVEYS for cn in P6.CFGS]
    rows = []
    with get_context("spawn").Pool(NPROC) as p:
        for r in p.imap_unordered(P6._job, tasks, chunksize=1):
            rows.append(r)
    json.dump(rows, open(f"{S.D2}/fits_arbiter_fdom_eff.json", "w"), indent=1)
    rows.sort(key=lambda r: (r["setup"], r["qcut"], r["survey"], P6.CFGS.index(r["config"])))
    log(f"\n=== fits ({len(rows)}) ===")
    for r in rows:
        log(P6.line(r))
    log("\n=== summary: S8 bias (ML, q_cut 5), SO40 | full ===")
    for name in SETUPS:
        sel = {(r["config"], r["survey"]): r for r in rows if r["setup"] == name and r["qcut"] == 5.0}
        log(f"   [{name:20s}] " + "; ".join(f"{cn} {sel[(cn, 'SO40')]['b_S8']:+.2f} | {sel[(cn, 'full')]['b_S8']:+.2f}"
                                          for cn in ("a", "b", "c", "d", "e")))
    mp = max(r["power"] for r in rows if r["qcut"] == 5.0 and r["config"] != "a")
    log(f"   max goodness-of-fit power, configs b-e, q_cut 5: {mp:.3f}; bound hits: {sum(bool(r['at_bound']) for r in rows)}")
    log(f"\n[a04b] done {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
