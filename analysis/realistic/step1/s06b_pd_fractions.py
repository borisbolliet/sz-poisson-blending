"""Step 1 / s06b: catalogue-level blending fractions (P(D)) with the MMF sigma_Y and cross-response.

Same method and conventions as analysis/blending/b03_analytic.py (REPORT.md sec. 3), convention c=1:
   q_b(A) = q_A + n + delta_A - mu(theta_A),   delta_A = sum_B rho_AB Y_B / sigma_A  (compound Poisson),
   mu = untruncated mean of delta under the same kernel (= map-mean removal; ~0 for the compensated MMF),
   detection: q_b > q_cut AND A dominant (every neighbour has x_B < q_A),
   fractions counted on the detected (blended) sample: B2(eps) = (delta-mu)/q_A > eps (Y boosted by > eps),
   B2neg, B3 = boosted over threshold (q_A + n < q_cut), lost/N_single, satellites/N_single,
   plus the mean fractional Y bias <(delta-mu)/q_A> of detections.
Neighbours: blendlib.neighbour_population(zsub=1, msub=6, ngh=16) (M500 > 1e13; floor 1e14 = class 2 only).
Primaries: q_A > 0.5 on the (theta node, q node) grid, model sigma_Y.

Runs (argv): white  -> old white-noise rho table (data/blending/rho_table.npz) + SZModel: must reproduce b03;
             baseline, goal -> data_real/rho_mmf_{mode}.npz (s06a) + MMFModel(SO, lmin=100).
For the MMF the variance beyond the kernel support (Parseval total from s05 minus binned rate variance) is
added as a Gaussian term (printed).  Untruncated P(D) quantiles of delta and of n + delta (robust widths).
Output: data_real/pd_fractions_{run}.npz and the log.
"""
import os, sys, time
for v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[v] = "1"
os.environ["NUMBA_NUM_THREADS"] = "4"
import numpy as np
from scipy.special import ndtr

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import pdlib1 as pl                      # noqa: E402
import blendlib as bl                    # noqa: E402
from szsim import signal as sg, instrument as ins   # noqa: E402

ROOT = pl.ROOT
QCUTS = [4.0, 5.0, 6.0, 7.0]
FLOORS = {"1e13": (0, 1, 2), "1e14": (2,)}
Q_PRIM = 0.5
TAG = ""                 # output-file tag; s06c sets "_subthr"
VAR_KEY = {"1e13": "all", "1e14": "c2"}   # s05 Parseval keys per floor; s06c sets {"1e13": "subthr"}
POP_FILTER = None        # optional callable (pop, model) -> pop  (s06c: neighbours with q_B < 5 only)


def main(run):
    t0 = time.time()
    if run == "white":
        tab = bl.load_rho_table()
        model = sg.SZModel()
    else:
        tab = bl.RhoTable(np.load(f"{ROOT}/data_real/rho_mmf_{run}.npz"))
        model = pl.make_model(("so", run, 100.0))
    # primary theta_A nodes: every other table node when the table is fine (96 nodes) -> 48 nodes
    TA = tab.th[::2] if tab.NT > 60 else tab.th
    NA = len(TA)
    sigA = model.sigma_Y(TA)
    print(f"[{run}] table: {tab.NT} theta nodes {tab.th[0]:.3g}-{tab.th[-1]:.3g}', NU={tab.NU}, support margin {tab.R_BEAM:.1f}'; "
          f"{NA} primary theta_A nodes; sigma_Y(table) vs model: max rel "
          f"{np.abs(tab.sigY/model.sigma_Y(tab.th)-1).max():.1e}", flush=True)
    pop = bl.neighbour_population(zsub=1, msub=6, ngh=16)
    if POP_FILTER is not None:
        n0 = len(pop[0]); pop = POP_FILTER(pop, model)
        print(f"[{run}] neighbour population filtered ({TAG}): {len(pop[0]):,} of {n0:,} points kept", flush=True)
    # Parseval variances from s05 (same population), interpolated to the table nodes
    cv = np.load(f"{ROOT}/data_real/confusion_var.npz")
    vkey = "white" if run == "white" else run
    var_pars = {fl: np.exp(np.interp(np.log(TA), np.log(cv["thA"]), np.log(cv[f"{vkey}_var_{VAR_KEY[fl]}"])))
                for fl in FLOORS}

    # ---- rate histograms
    R = []
    for i in range(NA):
        R.append(pl.rate_signed(TA[i], sigA[i], pop, tab))
        if i % 8 == 0:
            print(f"   rate node {i}/{NA} {time.time()-t0:.0f}s", flush=True)
    mu = {fl: np.array([pl.moments(R[i], cl)[0] for i in range(NA)]) for fl, cl in FLOORS.items()}
    var = {fl: np.array([pl.moments(R[i], cl)[1] for i in range(NA)]) for fl, cl in FLOORS.items()}
    print(f"[{run}] rate histograms done {time.time()-t0:.0f}s", flush=True)
    dc = model.dc_response(TA) if run != "white" else bl.dc_response(sigA)
    ybar = np.array([np.sum((pop[3] * pop[1])[np.isin(pop[2], cl)]) for cl in FLOORS.values()])
    show = [int(np.argmin(np.abs(TA - s))) for s in (0.5, 1, 2, 4, 8, 16, 32)]
    extra = {}
    for k, (fl, cl) in enumerate(FLOORS.items()):
        mu_formula = ybar[k] * dc / sigA
        extra[fl] = np.where(run == "white", 0.0, np.clip(var_pars[fl] - var[fl], 0, None))
        # [arbiter fix] s05 tabulates the Parseval variance only for theta_A = 0.25-80'; np.interp clamps outside, which
        # gave the nodes theta_A < 0.25' and > 80' a spurious extra variance (<= 0.04).  Effect on every stored fraction
        # <= 3.4e-8 (s10_arbiter_checks.log (E)); the stored npz/logs of s06b and s06c predate this fix.
        extra[fl][(TA < cv["thA"][0]) | (TA > cv["thA"][-1])] = 0.0
        print(f"   floor {fl}: thA      " + " ".join(f"{TA[i]:8.3f}" for i in show))
        print(f"   mean mu (hist)       " + " ".join(f"{mu[fl][i]:8.4f}" for i in show) + "  [noise units]")
        print(f"   mean (Campbell exact)" + " ".join(f"{mu_formula[i]:8.4f}" for i in show))
        print(f"   var (hist, incl caps)" + " ".join(f"{var[fl][i]:8.4f}" for i in show))
        print(f"   var (Parseval, s05)  " + " ".join(f"{var_pars[fl][i]:8.4f}" for i in show))
        print(f"   added Gaussian var   " + " ".join(f"{extra[fl][i]:8.4f}" for i in show), flush=True)
        lam1 = [pl.signed_arrays(R[i], cl, 1.0)[3] for i in show]
        lamn = []
        for i in show:
            o = R[i]; c = list(cl)
            with np.errstate(invalid="ignore", divide="ignore"):
                xn = np.where(o["Wn"][c].sum(0) > 0, o["WXn"][c].sum(0) / o["Wn"][c].sum(0), 0)
            lamn.append(o["Wn"][c].sum(0)[xn < -1.0].sum() + o["capn"][c, 0].sum())
        lam60 = []
        for i in range(NA):
            o = R[i]; c = list(cl)
            with np.errstate(invalid="ignore", divide="ignore"):
                xn = np.where(o["Wn"][c].sum(0) > 0, o["WXn"][c].sum(0) / o["Wn"][c].sum(0), 0)
            lam60.append(o["Wn"][c].sum(0)[xn < -60.0].sum() + o["capn"][c, 0].sum())
        print(f"   Lambda(x>1) per filter" + " ".join(f"{v:8.4f}" for v in lam1))
        print(f"   Lambda(x<-1)          " + " ".join(f"{v:8.4f}" for v in lamn))
        print(f"   Lambda(x<-60) (clipped to -60 in the FFT box): max over theta_A nodes {max(lam60):.2e}", flush=True)

    pd_small = pl.PDs(N=2 ** 18, dd=0.001, x_small=0.002, off=60.0)
    pd_big = pl.PDs(N=2 ** 19, dd=0.002, x_small=0.004, off=60.0)
    pd_un = pl.PDs(N=2 ** 20, dd=0.001, x_small=0.002, off=60.0)
    # ---- untruncated P(D) of delta - mu and of n + delta - mu: quantiles
    QL = [0.02275, 0.15866, 0.5, 0.84134, 0.97725]
    QD = {fl: np.zeros((NA, 5)) for fl in FLOORS}; QN = {fl: np.zeros((NA, 5)) for fl in FLOORS}
    for fl, cl in FLOORS.items():
        for i in range(NA):
            W, X, X2, lam_cap = pl.signed_arrays(R[i], cl)
            for arr, ev in ((QD, 0.0), (QN, 1.0)):
                p = pd_un.pdf(W, X, X2, extra_var=extra[fl][i] + ev)
                cdf = np.cumsum(p) * np.exp(-lam_cap)
                arr[fl][i] = np.interp(QL, cdf, pd_un.grid + 0.5 * pd_un.dd) - mu[fl][i]
        print(f"   floor {fl}: quantiles of delta-mu (2.3/16/50/84/97.7%) and of n+delta-mu at thA = 1, 4, 16':")
        for i in [int(np.argmin(np.abs(TA - s))) for s in (1, 4, 16)]:
            print(f"      thA={TA[i]:6.2f}': delta {np.round(QD[fl][i], 3)}  n+delta {np.round(QN[fl][i], 3)} "
                  f"-> robust half-width (84-16)/2 = {(QN[fl][i,3]-QN[fl][i,1])/2:.3f} (noise alone 1)", flush=True)
    print(f"[{run}] untruncated P(D) done {time.time()-t0:.0f}s", flush=True)

    # ---- primary population
    H = pl.primary_population(model, TA, Q_PRIM)
    for qc in QCUTS:
        a = (H * ndtr(pl.QNODES[None, :] - qc)).sum(); b = model.expected_detections(qc)
        print(f"   single-cluster detections q>{qc:.0f}: grid {a:,.1f} vs expected_detections {b:,.1f} (rel {a/b-1:+.2e})")
    print(f"[{run}] primary population done {time.time()-t0:.0f}s", flush=True)

    # ---- node loop
    nE = len(pl.EPS); nC = len(QCUTS); nq = len(pl.QNODES)
    res = {}
    for fl, cl in FLOORS.items():
        Pdom = np.zeros((NA, nq))
        Pdet = np.zeros((nC, NA, nq)); Pb3 = np.zeros_like(Pdet); Plost = np.zeros_like(Pdet); Pmean = np.zeros_like(Pdet)
        Pb2 = np.zeros((nC, NA, nq, nE)); Pb2n = np.zeros_like(Pb2)
        for i in range(NA):
            for k, qa in enumerate(pl.QNODES):
                if H[i, k] == 0:
                    continue
                pdm, o = pl.node_probs(R[i], cl, qa, QCUTS, mu[fl][i], pd_small, pd_big, extra_var=extra[fl][i])
                Pdom[i, k] = pdm
                for ic, qc in enumerate(QCUTS):
                    r = o[qc]
                    Pdet[ic, i, k] = r["det"] * pdm; Pb3[ic, i, k] = r["b3"] * pdm; Plost[ic, i, k] = r["lost"] * pdm
                    Pmean[ic, i, k] = r["mean"] * pdm / qa
                    Pb2[ic, i, k] = r["b2"] * pdm; Pb2n[ic, i, k] = r["b2n"] * pdm
            if i % 8 == 0:
                print(f"   floor {fl} node {i}/{NA} {time.time()-t0:.0f}s", flush=True)
        Phi = ndtr(pl.QNODES[None, :] - np.array(QCUTS)[:, None])
        print(f"\n=== [{run}] convention c=1 (mean removed), neighbour floor M>{fl}; primaries q_A>{Q_PRIM}; full sky ===")
        rows = []
        for ic, qc in enumerate(QCUTS):
            Ns = (H * Phi[ic][None, :]).sum(); Nd = (H * Pdet[ic]).sum()
            b2 = [(H * Pb2[ic, :, :, e]).sum() / Nd for e in range(nE)]
            b2n = [(H * Pb2n[ic, :, :, e]).sum() / Nd for e in range(nE)]
            b3 = (H * Pb3[ic]).sum() / Nd
            lost = (H * Plost[ic]).sum() / Ns
            sat = (H * (1 - Pdom) * Phi[ic][None, :]).sum() / Ns
            mb = (H * Pmean[ic]).sum() / Nd
            rows.append([qc, Ns, Nd, *b2, *b2n, b3, lost, sat, mb])
            print(f"  q_cut={qc:.0f}: N_single={Ns:,.0f} N_det_b={Nd:,.0f} ({Nd/Ns-1:+.4f}); B2(eps={list(pl.EPS)})={np.round(b2, 5)}; "
                  f"B2neg={np.round(b2n, 5)}; B3(boosted)={b3:.5f}; lost/N_single={lost:.5f}; sat/N_single={sat:.5f}; "
                  f"mean (delta-mu)/q_A of detections={mb:+.5f}", flush=True)
        res[fl] = np.array(rows)
    per_floor = {}
    for fl in FLOORS:
        per_floor.update({f"mu_{fl}": mu[fl], f"var_{fl}": var[fl], f"varpars_{fl}": var_pars[fl], f"extra_{fl}": extra[fl],
                          f"QD_{fl}": QD[fl], f"QN_{fl}": QN[fl], f"rows_{fl}": res[fl]})
    np.savez(f"{ROOT}/data_real/pd_fractions_{run}{TAG}.npz", th=TA, sigA=sigA, **per_floor,
             cols=np.array(["qcut", "N_single", "N_det_b"] + [f"B2_{e}" for e in pl.EPS] + [f"B2neg_{e}" for e in pl.EPS]
                           + ["B3", "lost", "sat", "meanbias"]))
    print(f"[{run}] total {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main(sys.argv[1])
