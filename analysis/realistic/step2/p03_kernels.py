"""Step 2 / p03: blended detection kernels K(theta_A, q_A; e) = P(A dominant and q_obs > e) for the 19 cosmology bin
edges e (data_cosmo/kernel_catalogue.npz EDGES), per theta_A node, on the q_A grid of c03/c11 (1065 nodes).

Machinery: step 1's signed compound-Poisson P(D) (pdlib1.PDs, signed_arrays: N = 2^18, dd = 0.001 for x_cut < 20 and
N = 2^19, dd = 0.002 above), convention c = 1 (q_b = q_A + n + delta - mu), dominance x_B < fdom q_A, unit noise and the
Gaussian extra variance inside the FFT (s2lib.kernel_node).  Rates: the stored s09 rate histograms (fiducial sky,
M500 > 1e13, 48 theta_A nodes for the MMF, 40 for the old white filter), rescaled for the low-pressure population, or
p02's rates for the off-fiducial truths.

Setups:
  white            old white filter (s09 rates, post-fix rho_at kernel)                regression to Q3 (c11)
  white_prefix     old white filter with the b03 rates of data/blending/analytic.npz (pre-fix kernel, as c03)
  {baseline,goal}_f{1,0.5,2}   SO MMF, dominance fdom = 1 (default), 0.5, 2
  {baseline,goal}_eff{i,ii}    map-sigma variants, fdom = 1: q_obs = Yhat / sigma_eff, sigma_eff = r(theta) sigma_Y,
                               r_i = sqrt(1 + var_subthr), r_ii = robust half-width of n + delta (p01); K'(theta, q'; e) =
                               P(dom at q_A = r q', q_A + n + delta - mu > r e) tabulated on q' in the same grid
  lowP_f1, lowP_effi           low-pressure population ((1-b) = 0.6 for all halos), baseline MMF
  S8_0.75_f1, S8_0.87_f1       baseline, neighbours at the truths of p02
  baseline_f1_fineQA           convergence test: q_A grid 2x finer (2130 nodes)
Checks (argv 'checks'): (1) K at s09's q nodes and q_cut = 4-7 vs the stored s09 node tables T{fdom}_det (same rates,
s09's stored extra variance); (2) white_prefix vs data_cosmo/kernel_catalogue.npz (c03's own FFT settings differ).
Usage: p03_kernels.py <setup> [<setup> ...] | fid | truths | conv | checks.   Output: data_real/step2/kernel_<setup>.npz
"""
import os, sys, time
NPROC = int(os.environ.get("P03_PROCS", "6"))
for v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "NUMBA_NUM_THREADS"):
    os.environ[v] = "1"
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import s2lib as S                                  # noqa: E402
from s2lib import pl                                # noqa: E402

GROUPS = {"fid": ["white", "white_prefix", "baseline_f1", "baseline_f0.5", "baseline_f2", "goal_f1", "goal_f0.5",
                  "goal_f2", "baseline_effi", "baseline_effii", "goal_effi", "goal_effii", "lowP_f1", "lowP_effi"],
          "truths": ["S8_0.75_f1", "S8_0.87_f1"], "conv": ["baseline_f1_fineQA"]}


def setup_inputs(name):
    """rates, mu, extra, r per node, fdom, q grid and a description for one setup."""
    I = np.load(f"{S.D2}/inputs.npz")
    fdom = 1.0; r = None; qa = S.QA
    if name == "white":
        th, sig, R = S.load_s09_rates("white"); mu, _ = S.moments(R); extra = np.zeros(len(th))
        desc = "old white filter, s09 rates (fixed rho_at), fdom 1"
    elif name == "white_prefix":
        th, sig, R, mu = S.load_prefix_white_rates(); extra = np.zeros(len(th))
        desc = "old white filter, b03 rates of data/blending/analytic.npz (pre-fix rho_at, as c03), fdom 1"
    elif name.startswith(("baseline", "goal")):
        run = name.split("_")[0]
        th, sig, R = S.load_s09_rates(run); mu, extra = I[f"{run}_mu"], I[f"{run}_extra"]
        tag = name.split("_", 1)[1]
        if tag.startswith("f"):
            fdom = float(tag[1:].split("_")[0])
        else:
            r = I[f"{run}_r_{tag[3:]}"]
        if name.endswith("fineQA"):
            qa = S.QA_FINE
        desc = f"SO {run} MMF, s09 rates, " + (f"fdom {fdom}" if r is None else f"map-sigma ({tag[3:]}), fdom 1") + \
            (", q_A grid 2x finer" if name.endswith("fineQA") else "")
    elif name.startswith("lowP"):
        th, sig, R0 = S.load_s09_rates("baseline")
        R = [S.scale_rates(o, S.C_LOW) for o in R0]; mu, extra = I["lowP_mu"], I["lowP_extra"]
        if name.endswith("effi"):
            r = I["lowP_r_i"]
        desc = f"SO baseline MMF, low-pressure population (1-b) = {S.OMB_LOW} (rates x c = {S.C_LOW:.5f})" + \
            (", map-sigma (i)" if r is not None else ", fdom 1")
    elif name.startswith("S8_"):
        d = np.load(f"{S.D2}/rates_{name[:7]}.npz")
        th, sig = d["th"], d["sigA"]
        R = S.rates_from_npz(d)
        mu, extra = d["mu"], d["extra"]
        desc = f"SO baseline MMF, neighbours at S8 = {float(d['S8'])} (sigma8 = {float(d['s8']):.5f}), fdom 1"
    else:
        raise KeyError(name)
    rr = np.ones(len(th)) if r is None else np.asarray(r)
    return dict(th=th, R=R, mu=np.asarray(mu), extra=np.asarray(extra), r=rr, fdom=fdom, qa=qa, desc=desc,
                eff=r is not None)


def _job(args):
    name, i, o, mu, extra, qa, thr, fdom = args
    t = time.time()
    K, Pd = S.kernel_node(o, mu, extra, qa, thr, fdom)
    return name, i, K, Pd, time.time() - t


def run_setups(names):
    from multiprocessing import get_context
    t0 = time.time()
    U = {n: setup_inputs(n) for n in names}
    jobs = []
    for n, u in U.items():
        for i in range(len(u["th"])):
            # map-sigma: q' grid -> q_A = r q', thresholds r e
            jobs.append((n, i, u["R"][i], float(u["mu"][i]), float(u["extra"][i]), u["qa"] * u["r"][i],
                         S.EDGES * u["r"][i], u["fdom"]))
        print(f"[{n}] {u['desc']}; {len(u['th'])} theta nodes {u['th'][0]:.3g}-{u['th'][-1]:.3g}'; {len(u['qa'])} q nodes; "
              f"r range {u['r'].min():.4f}-{u['r'].max():.4f}; max |mu| {np.abs(u['mu']).max():.3g}; max extra {u['extra'].max():.2e}",
              flush=True)
    # expensive jobs first (large fdom, fine grids)
    jobs.sort(key=lambda j: -(len(j[5]) * (1 + (j[7] >= 1)) * (1 + (j[7] >= 2))))
    res = {n: dict(K=np.zeros((len(u["th"]), len(u["qa"]), len(S.EDGES))), Pdom=np.zeros((len(u["th"]), len(u["qa"]))))
           for n, u in U.items()}
    cpu = 0.0
    with get_context("spawn").Pool(NPROC) as p:
        for k, (n, i, K, Pd, dt) in enumerate(p.imap_unordered(_job, jobs, chunksize=1)):
            res[n]["K"][i] = K; res[n]["Pdom"][i] = Pd; cpu += dt
            if k % 48 == 47 or k == len(jobs) - 1:
                print(f"   {k+1}/{len(jobs)} node jobs done  [{time.time()-t0:.0f}s wall, {cpu:.0f}s cpu]", flush=True)
    for n, u in U.items():
        K, Pd = res[n]["K"], res[n]["Pdom"]
        assert np.all(np.isfinite(K)) and K.min() >= -1e-12 and K.max() <= 1 + 1e-12
        mono_q = np.diff(K, axis=1).min()          # K should not decrease with q_A (up to FFT noise)
        mono_e = np.diff(K, axis=2).max()          # nor increase with the edge
        np.savez(f"{S.D2}/kernel_{n}.npz", Kb=K, Pdom=Pd, th=u["th"], QA=u["qa"], EDGES=S.EDGES, r=u["r"], fdom=u["fdom"],
                 mu=u["mu"], extra=u["extra"], eff=u["eff"], desc=u["desc"])
        j5 = int(np.argmin(np.abs(u["qa"] - 5.0)))
        ib = [int(np.argmin(np.abs(u["th"] - s))) for s in (1.0, 4.0, 16.0)]
        print(f"[{n}] saved kernel_{n}.npz; min dK/dq_A {mono_q:+.1e}, max dK/de {mono_e:+.1e}; at q' = 5: P_dom = "
              + ", ".join(f"{Pd[i, j5]:.5f}" for i in ib) + "; K(e=5) = " + ", ".join(f"{K[i, j5, 1]:.4f}" for i in ib)
              + " at theta_A = " + ", ".join(f"{u['th'][i]:.2f}'" for i in ib), flush=True)
    print(f"done {len(jobs)} node jobs in {time.time()-t0:.0f}s wall ({cpu:.0f}s cpu)")


def checks():
    """(1) vs s09 stored node tables; (2) white_prefix vs c03's kernel_catalogue.npz."""
    from scipy.special import ndtr
    t0 = time.time()
    print("=== (1) kernel_node at s09's q nodes, q_cut = 4-7, vs s09 T{fdom}_det (same rates and s09's stored extra) ===")
    for run, nodes, fdoms in (("baseline", (6, 16, 26, 36), (1.0, 0.5, 2.0)), ("goal", (16, 30), (1.0,)),
                              ("white", (5, 20, 35), (1.0,))):
        d = np.load(f"{S.DR}/pd_tables_{run}.npz")
        th, _, R = S.load_s09_rates(run)
        for f in fdoms:
            worst = 0.0; worst_rel = 0.0
            for i in nodes:
                ks = np.nonzero(d["H"][i] != 0)[0]
                K, _ = S.kernel_node(R[i], float(d["mu"][i]), float(d["extra"][i]), pl.QNODES[ks], d["qcuts"], f)
                ref = d[f"T{f}_det"][:, i, ks].T
                worst = max(worst, np.abs(K - ref).max())
                worst_rel = max(worst_rel, (np.abs(K - ref) / np.maximum(ref, 1e-3)).max())
            print(f"   [{run}] fdom={f}: nodes {[round(float(th[i]), 2) for i in nodes]}': max |K - T_det| = {worst:.1e} "
                  f"(max relative, where T_det > 1e-3: {worst_rel:.1e})  [{time.time()-t0:.0f}s]", flush=True)
    print("=== (2) white_prefix vs c03's kernel_catalogue.npz (c03: N = 2^16, dd = 0.01, x_small = 0.02, no 2dd smoothing) ===")
    c = np.load(f"{S.ROOT}/data_cosmo/kernel_catalogue.npz")
    fn = f"{S.D2}/kernel_white_prefix.npz"
    if os.path.exists(fn):
        k = np.load(fn)
        dK = k["Kb"] - c["Kb"]
        print(f"   max |K - K_c03| = {np.abs(dK).max():.1e} (at node {np.unravel_index(np.abs(dK).argmax(), dK.shape)}); "
              f"rms {np.sqrt(np.mean(dK**2)):.1e}; P_dom max |diff| {np.abs(k['Pdom']-c['Pdom']).max():.1e}")
        for e in (5.0, 6.0, 10.5737):
            j = int(np.argmin(np.abs(c["EDGES"] - e)))
            print(f"   edge {c['EDGES'][j]:.3f}: max |dK| {np.abs(dK[:, :, j]).max():.1e}; max |dK| for q_A in 3-8: "
                  f"{np.abs(dK[:, (c['QA'] > 3) & (c['QA'] < 8), j]).max():.1e}")
        kw = np.load(f"{S.D2}/kernel_white.npz")
        dW = kw["Kb"] - k["Kb"]
        print(f"   fixed-kernel (s09) vs pre-fix rates, same code: max |dK| = {np.abs(dW).max():.1e}; at theta_A = 4, 16, 40, 80' "
              f"(q_A = 3-8, e = 5): " + ", ".join(f"{np.abs(dW[int(np.argmin(np.abs(kw['th']-s))), (kw['QA'] > 3) & (kw['QA'] < 8), 1]).max():.1e}"
                                                  for s in (4, 16, 40, 80)))
    print(f"checks {time.time()-t0:.0f}s")


if __name__ == "__main__":
    args = sys.argv[1:]
    if args == ["checks"]:
        checks()
    else:
        names = [n for a in args for n in GROUPS.get(a, [a])]
        run_setups(names)
