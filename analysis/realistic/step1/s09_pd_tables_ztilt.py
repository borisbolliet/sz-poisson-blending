"""Step 1 / s09 (arbiter): per-node P(D) selection tables, redshift-binned count change with its expected per-sky
scatter, and the dependence of the blending fractions on the dominance (attribution) rule.

Same machinery, inputs and resolution as s06b (pdlib1; 48 theta_A nodes = every other rho-table node; QNODES;
neighbour population blendlib.neighbour_population(zsub=1, msub=6, ngh=16), floor M500 > 1e13; primaries q_A > 0.5;
convention c=1, i.e. q_b = q_A + n + delta - mu).

Why (arbitration of the two step-1 verification reports):
 (1) RESULTS quoted the redshift tilt Delta N/N(z) at q > 5 from the 3-sky direct count of s06d, whose quoted errors
     (std of 3 per-sky values / sqrt 3) are not reliable.  The model expectation is the P(D) integrated over the
     z-binned primary population (pdlib1.primary_population(..., zlo, zhi)); computed here at the s06b resolution.
 (2) Expected per-sky scatter of Delta N/N in each z bin (primaries = Poisson process, exact noise marginalisation;
     correlations of delta between primaries through shared neighbours are neglected):
        Var N_det = sum H P_dom det2,   Var N_single = sum H Phi^2,   Cov(N_det, N_single) = sum H Phi P_dom det.
 (3) Dominance-rule dependence (verifier 2): A is counted only if every neighbour has x_B < fdom q_A;
     fdom = 1 (default of s06b/b03), 0.5, 2 and inf (no rule: every primary counted, peaks multiply counted).
 (4) Per-node selection tables for step 2 (s06b did not store them).
Checks: fdom = 1 must reproduce s06b's rows (data_real/pd_fractions_{run}.npz) to rounding; sum_z H_z = H.
Output: data_real/pd_tables_{run}.npz.   Usage: s09_pd_tables_ztilt.py {white|baseline|goal}
"""
import os, sys, time
NTH = int(os.environ.get("S09_THREADS", "3"))
for v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ[v] = "1"
os.environ["NUMBA_NUM_THREADS"] = str(NTH)
import numpy as np
from scipy.special import ndtr

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import pdlib1 as pl                      # noqa: E402
import blendlib as bl                    # noqa: E402
from szsim import signal as sg           # noqa: E402

ROOT = pl.ROOT
QCUTS = [4.0, 5.0, 6.0, 7.0]
CL = (0, 1, 2)
ZB = [0.01, 0.1, 0.25, 0.5, 1.0, 1.5, 2.0, 3.0]
FDOMS = [1.0, 0.5, 2.0, np.inf]
Q_PRIM = 0.5
RKEYS = ("Wp", "WXp", "WX2p", "Wn", "WXn", "WX2n", "capp", "capn")

_PD = {}


def _init():
    _PD["small"] = pl.PDs(N=2 ** 18, dd=0.001, x_small=0.002, off=60.0)
    _PD["big"] = pl.PDs(N=2 ** 19, dd=0.002, x_small=0.004, off=60.0)


def _node_job(args):
    i, fdom, o, mu_i, extra_i, kq = args
    nq, nC, nE = len(pl.QNODES), len(QCUTS), len(pl.EPS)
    Pdom = np.zeros(nq)
    A = {k: np.zeros((nC, nq)) for k in ("det", "b3", "lost", "mean", "det2")}
    B = {k: np.zeros((nC, nq, nE)) for k in ("b2", "b2n")}
    for k in kq:
        qa = pl.QNODES[k]
        pdm, out = pl.node_probs_ext(o, CL, qa, QCUTS, mu_i, _PD["small"], _PD["big"], extra_var=extra_i, fdom=fdom)
        Pdom[k] = pdm
        for ic, qc in enumerate(QCUTS):
            r = out[qc]
            for key in A:
                A[key][ic, k] = r[key] * pdm / (qa if key == "mean" else 1.0)
            for key in B:
                B[key][ic, k] = r[key] * pdm
    return i, fdom, Pdom, A, B


def rows_from(H, T):
    """s06b's summary rows (same formulas): qcut, N_single, N_det_b, B2(eps) x5, B2neg(eps) x5, B3, lost, sat, meanbias."""
    nE = len(pl.EPS)
    Phi = ndtr(pl.QNODES[None, :] - np.array(QCUTS)[:, None])
    rows = []
    for ic, qc in enumerate(QCUTS):
        Ns = (H * Phi[ic][None, :]).sum(); Nd = (H * T["det"][ic]).sum()
        b2 = [(H * T["b2"][ic, :, :, e]).sum() / Nd for e in range(nE)]
        b2n = [(H * T["b2n"][ic, :, :, e]).sum() / Nd for e in range(nE)]
        b3 = (H * T["b3"][ic]).sum() / Nd
        lost = (H * T["lost"][ic]).sum() / Ns
        sat = (H * (1 - T["Pdom"]) * Phi[ic][None, :]).sum() / Ns
        mb = (H * T["mean"][ic]).sum() / Nd
        rows.append([qc, Ns, Nd, *b2, *b2n, b3, lost, sat, mb])
    return np.array(rows)


def main(run):
    from multiprocessing import get_context
    t0 = time.time()
    if run == "white":
        tab = bl.load_rho_table(); model = sg.SZModel()
    else:
        tab = bl.RhoTable(np.load(f"{ROOT}/data_real/rho_mmf_{run}.npz")); model = pl.make_model(("so", run, 100.0))
    TA = tab.th[::2] if tab.NT > 60 else tab.th
    NA = len(TA)
    sigA = model.sigma_Y(TA)
    pop = bl.neighbour_population(zsub=1, msub=6, ngh=16)
    cv = np.load(f"{ROOT}/data_real/confusion_var.npz")
    var_pars = np.exp(np.interp(np.log(TA), np.log(cv["thA"]), np.log(cv[f"{run}_var_all"])))
    print(f"[{run}] {NA} theta_A nodes {TA[0]:.3g}-{TA[-1]:.3g}'; numba threads {NTH}; neighbour points {len(pop[0]):,}",
          flush=True)

    # ---- rate histograms (identical calls to s06b)
    R = []
    for i in range(NA):
        R.append(pl.rate_signed(TA[i], sigA[i], pop, tab))
        if i % 12 == 0:
            print(f"   rate node {i}/{NA} {time.time()-t0:.0f}s", flush=True)
    mu = np.array([pl.moments(R[i], CL)[0] for i in range(NA)])
    var = np.array([pl.moments(R[i], CL)[1] for i in range(NA)])
    extra = np.where(run == "white", 0.0, np.clip(var_pars - var, 0, None))
    # [arbiter fix, as in s06b] no extrapolated Parseval variance outside s05's theta_A range (0.25-80'); the stored
    # data_real/pd_tables_*.npz predate this fix: effect on every fraction <= 3.4e-8 (s10_arbiter_checks.log (E))
    extra[(TA < cv["thA"][0]) | (TA > cv["thA"][-1])] = 0.0
    print(f"[{run}] rate histograms done {time.time()-t0:.0f}s; max added Gaussian var {extra.max():.2e}", flush=True)

    # ---- primary populations: full range and 7 z bins
    H = pl.primary_population(model, TA, Q_PRIM)
    Hz = np.array([pl.primary_population(model, TA, Q_PRIM, zlo=a, zhi=b) for a, b in zip(ZB[:-1], ZB[1:])])
    print(f"[{run}] H: sum over z bins vs full range: max |sum_z H_z - H| / max H = "
          f"{np.abs(Hz.sum(0) - H).max() / np.abs(H).max():.1e}; N_single(q>5) full {(H*ndtr(pl.QNODES-5.0)).sum():,.1f}, "
          f"sum of bins {(Hz*ndtr(pl.QNODES-5.0)).sum():,.1f}  [{time.time()-t0:.0f}s]", flush=True)
    active = (np.abs(H) > 0) | (np.abs(Hz) > 0).any(0)

    # ---- node tables for every dominance variant (process pool, 1 thread each)
    nq, nC, nE = len(pl.QNODES), len(QCUTS), len(pl.EPS)
    TAB = {f: dict(Pdom=np.zeros((NA, nq)), **{k: np.zeros((nC, NA, nq)) for k in ("det", "b3", "lost", "mean", "det2")},
                   **{k: np.zeros((nC, NA, nq, nE)) for k in ("b2", "b2n")}) for f in FDOMS}
    jobs = [(i, f, R[i], mu[i], extra[i], np.nonzero(active[i])[0]) for f in FDOMS for i in range(NA)]
    with get_context("spawn").Pool(NTH, initializer=_init) as p:
        for n, (i, f, Pdom, A, B) in enumerate(p.imap_unordered(_node_job, jobs, chunksize=1)):
            T = TAB[f]
            T["Pdom"][i] = Pdom
            for k in A:
                T[k][:, i] = A[k]
            for k in B:
                T[k][:, i] = B[k]
            if n % 48 == 47:
                print(f"   node tables {n+1}/{len(jobs)} {time.time()-t0:.0f}s", flush=True)

    # ---- (check) fdom = 1 vs s06b
    rows = {f: rows_from(H, TAB[f]) for f in FDOMS}
    ref = np.load(f"{ROOT}/data_real/pd_fractions_{run}.npz")
    cols = list(ref["cols"]); r06 = ref["rows_1e13"]
    dN = np.abs(rows[1.0][:, 1:3] / r06[:, 1:3] - 1).max()
    dF = np.abs(rows[1.0][:, 3:] - r06[:, 3:]).max()
    print(f"\n[{run}] (check) fdom = 1 vs s06b rows (floor 1e13): max rel diff N_single/N_det_b {dN:.1e}; "
          f"max abs diff of all fractions {dF:.1e}", flush=True)

    # ---- (3) dominance-rule variants
    print(f"\n=== [{run}] dominance rule x_B < fdom q_A: fractions on the detected sample (floor 1e13, c=1) ===")
    for f in FDOMS:
        for row in rows[f]:
            g = lambda k: row[cols.index(k)]
            print(f"   fdom={f:>4}: q>{g('qcut'):.0f}: N_det_b {g('N_det_b'):9,.0f} ({g('N_det_b')/g('N_single')-1:+.2%}); "
                  f"Y>10% {g('B2_0.1'):.2%}; Y>20% {g('B2_0.2'):.2%}; boosted {g('B3'):.2%}; lost {g('lost'):.2%}; "
                  f"Y<-10% {g('B2neg_0.1'):.2%}; mean bias {g('meanbias'):+.2%}; not counted (1-P_dom) {g('sat'):.2%}")

    # ---- (1)+(2) z-binned count change and expected per-sky scatter (fdom = 1)
    T = TAB[1.0]
    Phi = ndtr(pl.QNODES[None, :] - np.array(QCUTS)[:, None])
    zr = np.zeros((nC, len(ZB) - 1)); zsd = np.zeros_like(zr); zNs = np.zeros_like(zr); zNd = np.zeros_like(zr)
    trunc = np.zeros_like(zr)
    lowq = pl.QNODES[None, :] < 1.0 - 1e-9
    for ic in range(nC):
        for b in range(len(ZB) - 1):
            Hb = Hz[b]
            Ns = (Hb * Phi[ic][None, :]).sum(); Nd = (Hb * T["det"][ic]).sum()
            vNd = (Hb * T["det2"][ic]).sum(); vNs = (Hb * Phi[ic][None, :] ** 2).sum(); cov = (Hb * Phi[ic][None, :] * T["det"][ic]).sum()
            r = Nd / Ns
            zr[ic, b] = r - 1; zNs[ic, b] = Ns; zNd[ic, b] = Nd
            zsd[ic, b] = np.sqrt(max(vNd - 2 * r * cov + r * r * vNs, 0.0)) / Ns
            trunc[ic, b] = (Hb * (T["det"][ic] - Phi[ic][None, :]) * lowq).sum() / Ns
    print(f"\n=== [{run}] Delta N/N = N_det,blended / N_single - 1 in z bins {ZB} (P(D) expectation, fdom = 1) ===")
    for ic, qc in enumerate(QCUTS):
        print(f"   q>{qc:.0f}: " + "  ".join(f"{100*a:+.2f}%" for a in zr[ic]))
        print(f"         expected per-sky (full-sky) std: " + "  ".join(f"{100*s:.2f}%" for s in zsd[ic]))
        print(f"         N_single per bin: " + "  ".join(f"{v:,.0f}" for v in zNs[ic]))
        print(f"         contribution of primaries with q_A < 1 to Delta N/N: " + "  ".join(f"{100*v:+.4f}%" for v in trunc[ic]))
    np.savez(f"{ROOT}/data_real/pd_tables_{run}.npz", th=TA, sigA=sigA, qnodes=pl.QNODES, qcuts=np.array(QCUTS),
             eps=pl.EPS, zb=np.array(ZB), fdoms=np.array(FDOMS), mu=mu, var=var, extra=extra, H=H, Hz=Hz,
             **{f"R_{k}": np.array([o[k] for o in R]) for k in RKEYS},
             **{f"T{f}_{k}": v for f in FDOMS for k, v in TAB[f].items()},
             **{f"rows_fdom{f}": rows[f] for f in FDOMS}, cols=ref["cols"],
             z_dNN=zr, z_sd=zsd, z_Ns=zNs, z_Nd=zNd, z_trunc_q1=trunc)
    print(f"[{run}] saved data_real/pd_tables_{run}.npz; total {time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main(sys.argv[1])
