"""Step 2 / p06b: which feature of Delta(z, q) drives the bias?  Linearised Fisher estimator (exact to first order and
linear in Delta, so the parts add up):
    delta_theta = F^-1 J^T (Delta / N0)        (F, J of the naive model at the truth, incl. the (1-b) prior)
with Delta split into
    amplitude : eps N0(z, q),                  eps = sum Delta / sum N0   (a uniform count change)
    z tilt    : (r_z(z) - eps) N0(z, q),       r_z = sum_q Delta / sum_q N0 (redshift dependence at the naive q shape)
    q shape   : the rest                         (q dependence and z-q structure)
Printed in Fisher sigma for S8, Om, s8 (and the ML bias of p06 for comparison).
"""
import os, sys, json
for v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "NUMBA_NUM_THREADS"):
    os.environ[v] = "4"
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import s2lib as S                                  # noqa: E402
from s2lib import CL                                # noqa: E402
import p06_fits as P6                               # noqa: E402

rows = json.load(open(f"{S.D2}/fits.json"))


def ml(setup, cn, sv, qc):
    for r in rows:
        if r["setup"] == setup and r["config"] == cn and r["survey"] == sv and r["qcut"] == qc and r["route"] == "cat" \
                and r.get("start", "truth") == "truth":
            return r


for setup in ("white", "baseline_f1", "goal_f1", "baseline_effi", "baseline_effii", "lowP_f1"):
    for qc in (5.0,):
        M, truth, N0, D = P6.model_and_data(setup, qc)
        eps = D.sum() / N0.sum()
        rz = D.sum(1) / N0.sum(1)
        parts = {"amplitude": eps * N0, "z tilt": (rz[:, None] - eps) * N0}
        parts["q shape"] = D - parts["amplitude"] - parts["z tilt"]
        print(f"=== [{setup}] q_cut {qc:.0f}: Delta total {100*eps:+.2f}%; z tilt r_z - eps from z<0.1 to z 2.5-3: "
              f"{100*(rz[0]-eps):+.2f}% ... {100*(rz[-1]-eps):+.2f}% ===")
        for sv, fs in P6.SURVEYS.items():
            for cn in ("a", "b", "c", "d", "e"):
                free, pri = S.config(cn, truth)
                fi = [CL.PNAMES.index(p) for p in free]
                F_ = S.Fitter3(M, N0 * fs, free, fsky=fs, priors=pri, truth=truth)
                Ft = F_.fisher(truth[fi]); J = F_.jac(truth[fi])
                sig, Cov, gS = S.derived_sigma3(Ft, fi, truth)
                out = {}
                for k, Dk in list(parts.items()) + [("sum", D)]:
                    x = np.linalg.solve(Ft, J.T @ (Dk * fs / (N0 * fs)).ravel())
                    out[k] = (gS @ x / sig["S8"], x[0] / sig["Om"], x[1] / sig["s8"])
                r = ml(setup, cn, sv, qc)
                print(f"  {sv:4s} {cn}: S8 bias (lin) = amplitude {out['amplitude'][0]:+.2f} + z tilt {out['z tilt'][0]:+.2f} + "
                      f"q shape {out['q shape'][0]:+.2f} = {out['sum'][0]:+.2f} sigma (ML {r['b_S8']:+.2f}); "
                      f"Om: {out['amplitude'][1]:+.2f} {out['z tilt'][1]:+.2f} {out['q shape'][1]:+.2f} = {out['sum'][1]:+.2f} "
                      f"(ML {r['b_Om']:+.2f}); s8: {out['amplitude'][2]:+.2f} {out['z tilt'][2]:+.2f} {out['q shape'][2]:+.2f} = "
                      f"{out['sum'][2]:+.2f} (ML {r['b_s8']:+.2f})", flush=True)
