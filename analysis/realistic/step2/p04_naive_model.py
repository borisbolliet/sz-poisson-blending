"""Step 2 / p04: the naive (no-blending) count model with the MMF, and its checks.

SmoothModelS = cosmolib2.SmoothModel (smooth lognormal-scatter convolution, lnM x4, HMF grid of c01, bins of
cosmolib.default_bins) with sigma_Y(theta500) replaced by
   * the MMF sigma_Y (step 1's MMFModel spline: MMFModel(Instrument.so_lat(mode, lmin=100)).sigma_Y), or
   * sigma_eff = r(theta) sigma_Y for the map-sigma variants (r from p01, linear in ln theta between the 48 nodes).
Checks:
 (1) white: SmoothModelS with the SZModel spline = cosmolib2.SmoothModel (N0 of c11: 85,906.6 at q > 5).
 (2) totals at q > 5, 6 vs step 1's expected_detections.json (MMFModel.expected_detections: szsim's 1201x1201 HMF table,
     48-point Gauss-Hermite scatter) and per z bin vs expected_detections(q, zmin, zmax); vs step 1's N_single (s09 grid).
 (3) low-pressure population ((1-b) = 0.6): totals vs MMFModel(one_minus_b=0.6).expected_detections.
 (4) map-sigma models: totals.
"""
import os, sys, time, json
for v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "NUMBA_NUM_THREADS"):
    os.environ[v] = "4"
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import s2lib as S                                  # noqa: E402
from s2lib import CL, C2, ins, sg                   # noqa: E402

t0 = time.time()
G = CL.HMFGrid()
ed = json.load(open(f"{S.DR}/expected_detections.json"))
I = np.load(f"{S.D2}/inputs.npz")
zed = CL.default_bins(5.0)[0]

print("(1) white: SmoothModelS(SZModel spline) vs cosmolib2.SmoothModel")
for qc in (5.0, 6.0):
    a = S.SmoothModelS(qc, S.LnSigma(S.sz_model("white")), grid=G).counts(S.FID)
    b = C2.SmoothModel(qc, grid=G).counts(S.FID)
    ref = ed["white (old)"][int(qc) - 4]
    print(f"   q>{qc:.0f}: {a.sum():,.1f} vs {b.sum():,.1f} (max per-bin |diff| {np.abs(a-b).max():.1e}); "
          f"expected_detections {ref:,.1f} (rel {a.sum()/ref-1:+.2e})", flush=True)

print("\n(2) SO MMF naive totals (full sky) vs step 1")
for run in ("baseline", "goal"):
    m = S.sz_model(run)
    d09 = np.load(f"{S.DR}/pd_tables_{run}.npz")
    for qc in (5.0, 6.0):
        M = S.SmoothModelS(qc, S.LnSigma(m), grid=G)
        N0 = M.counts(S.FID)
        ref = ed[f"SO {run} lmin100"][int(qc) - 4]
        Ns09 = d09["rows_fdom1.0"][int(qc) - 4, 1]
        print(f"   [{run}] q>{qc:.0f}: SmoothModelS {N0.sum():,.1f}; expected_detections {ref:,.1f} (rel {N0.sum()/ref-1:+.2e}); "
              f"step-1 s09 grid N_single {Ns09:,.1f} (rel {N0.sum()/Ns09-1:+.2e})", flush=True)
        if qc == 5.0:
            perz = np.array([m.expected_detections(qc, zmin=a, zmax=b) for a, b in zip(zed[:-1], zed[1:])])
            rel = N0.sum(1) / perz - 1
            print(f"      per z bin ({len(zed)-1} bins 0.01-3) SmoothModelS / expected_detections - 1: "
                  + " ".join(f"{v:+.1e}" for v in rel), flush=True)
            print(f"      N per z bin: " + " ".join(f"{v:,.0f}" for v in N0.sum(1)), flush=True)
            print(f"      N per q bin ({np.round(M.qed[:-1], 2).tolist()} ...): " + " ".join(f"{v:,.0f}" for v in N0.sum(0)), flush=True)

print("\n(3) low-pressure population, (1-b) = 0.6 (baseline MMF)")
mL = ins.MMFModel(ins.Instrument.so_lat("baseline", lmin=100.0), one_minus_b=S.OMB_LOW)
for qc in (5.0, 6.0):
    M = S.SmoothModelS(qc, S.LnSigma(S.sz_model("baseline")), grid=G)
    N0 = M.counts(S.FID_LOW)
    ref = mL.expected_detections(qc)
    print(f"   q>{qc:.0f}: SmoothModelS {N0.sum():,.1f}; MMFModel(one_minus_b=0.6).expected_detections {ref:,.1f} "
          f"(rel {N0.sum()/ref-1:+.2e}); SO40: {0.4*N0.sum():,.0f}", flush=True)

print("\n(4) map-sigma models: sigma_eff = r(theta) sigma_Y fixed on the fiducial sky; naive totals")
for run, tag, truth in (("baseline", "i", S.FID), ("baseline", "ii", S.FID), ("goal", "i", S.FID), ("goal", "ii", S.FID),
                        ("lowP", "i", S.FID_LOW)):
    mm = S.sz_model("baseline" if run == "lowP" else run)
    th = I["baseline_th" if run == "lowP" else f"{run}_th"]
    for qc in (5.0, 6.0):
        Mp = S.SmoothModelS(qc, S.LnSigma(mm), grid=G)
        Me = S.SmoothModelS(qc, S.LnSigma(mm, th, I[f"{run}_r_{tag}"]), grid=G)
        a, b = Mp.counts(truth).sum(), Me.counts(truth).sum()
        print(f"   [{run} ({tag})] q>{qc:.0f}: sigma_Y model {a:,.1f}; sigma_eff model {b:,.1f} ({b/a-1:+.2%})", flush=True)
print(f"total {time.time()-t0:.0f}s")
