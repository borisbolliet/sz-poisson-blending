"""Step 1 / s06e: why the stored data/blending/analytic.npz (b03) neighbour-sum variance differs by ~1-2% from the
variance computed with the current blendlib kernel (found while validating pdlib1 on the white case; read-only check).

Hypothesis: blendlib.rho_at was fixed at 20:36 on 2026-11-10 (theta_B > 80' neighbours: the clamped 80' kernel is
now rescaled by amp = (S_tab/S)^2 < 1; analysis/blending/r03b_large_thetaB*.log), after b03 had written the rate
histograms.  [arbiter correction of the provenance argument: analytic.npz is dated 20:39:15 because
analysis/blending/r02_pd_fine.py (log dated 20:39:15, 18 s run) re-saved the file, replacing only QUANT and CCDF;
W, WX, WX2 and CAP were carried over from the b03 run whose log is dated 18:48:54, i.e. before the fix.  The earlier
explanation here ('b03 takes ~1000 s, so the run started before the fix') was wrong; the conclusion stands.  An
independent re-implementation of the pre-fix kernel is in s10_arbiter_checks.py (D).]
Exact test, like for like (b03 stores the rate histogram for x < 500 only, the cap x >= 500 separately):
   pre-fix kernel = fixed kernel / amp  <=>  same kernel with Y_B -> Y_B / amp  (x is linear in Y_B), so
   predicted b03 var(x<500) = var_now(x<500) - var_now(x<500 | thB>80') + var_prefix(x<500 | thB>80'),
   and the same for the x >= 500 cap rate, compared with the stored analytic.npz W, WX2, CAP.
"""
import os, sys, time
for v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ[v] = "1"
os.environ["NUMBA_NUM_THREADS"] = "4"
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import pdlib1 as pl                      # noqa: E402
import blendlib as bl                    # noqa: E402
from szsim import signal as sg           # noqa: E402


def v_lt500(o):
    return o["WX2p"].sum() + o["WX2n"].sum()


def cap_rate(o):
    return o["capp"][:, 0].sum() + o["capn"][:, 0].sum()


def main():
    for f in ("analysis/blending/blendlib.py", "data/blending/analytic.npz", "analysis/blending/b03_analytic.log"):
        print(f"   {f}: modified {time.ctime(os.path.getmtime(os.path.join(pl.ROOT, f)))}")
    pop = bl.neighbour_population(zsub=1, msub=6, ngh=16)
    tab = bl.load_rho_table(); m = sg.SZModel()
    A = np.load(f"{bl.DATA}/analytic.npz")
    thmax = tab.th[-1]
    big = pop[0] > thmax
    thB, YB, cB, wB = (a[big] for a in pop)
    print(f"   neighbours with theta_B > {thmax:.0f}': {wB.sum()*bl.ARCMIN2_FULLSKY:.3f} per full sky (expected)")
    for i in (15, 20, 25, 30):
        thA = tab.th[i]; sA = m.sigma_Y(thA)
        o_all = pl.rate_signed(thA, sA, pop, tab)
        o_fix = pl.rate_signed(thA, sA, (thB, YB, cB, wB), tab)
        amp = ((5 * thA + 5 * thmax + tab.R_BEAM) / (5 * thA + 5 * thB + tab.R_BEAM)) ** 2
        o_pre = pl.rate_signed(thA, sA, (thB, YB / amp, cB, wB), tab)
        v_now = v_lt500(o_all)
        v_pred = v_now - v_lt500(o_fix) + v_lt500(o_pre)
        v_b03 = A["WX2"][i].sum()
        c_now = cap_rate(o_all); c_pred = c_now - cap_rate(o_fix) + cap_rate(o_pre); c_b03 = A["CAP"][i].sum()
        print(f"thA={thA:6.2f}': var(x<500): current kernel {v_now:.4f} | predicted with pre-fix kernel {v_pred:.4f} | "
              f"stored analytic.npz {v_b03:.4f}  (stored/predicted - 1 = {v_b03/v_pred-1:+.1e}; stored/current - 1 = {v_b03/v_now-1:+.1e})")
        print(f"             rate(x>=500): current {c_now:.3e} | predicted pre-fix {c_pred:.3e} | stored {c_b03:.3e}")


if __name__ == "__main__":
    main()
