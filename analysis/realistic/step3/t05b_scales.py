"""Step 3 / t05b: does the number of filter scales matter?  The same 40 realistic tiles (pid 0-39), both noise levels,
with 12, 15 (production) and 20 log-spaced scales between 0.5' and 15' (t04 runs into tiles_s12, tiles, tiles_s20).
Compared: q > 5 detections and credited halos, class fractions, r(theta) after the iteration, the count change.
The count change uses t05's nominal selection (since the arbiter's fix: credit limit, pooled calibration)."""
import os, sys, time
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "NUMBA_NUM_THREADS"):
    os.environ[_v] = "2"
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import t05_analysis as T5                         # noqa: E402
S, F = T5.S, T5.F
t0 = time.time()
PIDS = range(40)
for mode in S.MODES:
    rinj_b = T5.r_inj(mode)
    rinj = T5.SELA[mode + "_rinj"] if T5.SELA is not None else rinj_b      # the nominal calibration of t05 (arbiter fix)
    print(f"==================== {mode}: tiles 0-39")
    res = {}
    for sub, ns in (("tiles_s12", 12), ("tiles", 15), ("tiles_s20", 20)):
        T5.TD = os.path.join(S.OUT, sub)
        T = [T5.tile_stats(p, mode, rinj, rinj_b) for p in PIDS]
        A = {k: np.array([t_[k] for t_ in T]) for k in ("n5", "n_main", "n_none", "n_frag", "b10", "b20", "needs_alone",
                                                         "merge", "wrong", "lost", "n_alone5")}
        Nd = np.array([t_["Ndet"] for t_ in T]); Ne = np.array([t_["Nexp"] for t_ in T])
        th = np.load(os.path.join(T5.TD, "tile_000.npz"))["thetas"]
        r1 = np.array([t_["r1"] for t_ in T]).mean(0)
        res[ns] = (A, Nd, Ne, th, r1)
        nm = A["n_main"].sum()
        tot, eb, Tt, eT = T5.boot_ratio(Nd, Ne)
        print(f"  {ns:2d} scales: q > 5 detections {A['n5'].sum()}, credited halos {Nd.sum():.0f}, N_exp {Ne.sum():.1f}, "
              f"Delta/N {100 * Tt:+.2f} +- {100 * eT:.2f}%")
        print(f"     of main: Y > 10% {100 * A['b10'].sum() / nm:.2f}%, > 20% {100 * A['b20'].sum() / nm:.2f}%, only thanks to neighbours "
              f"{100 * A['needs_alone'].sum() / nm:.2f}%, merges {100 * A['merge'].sum() / nm:.2f}%, wrong size {100 * A['wrong'].sum() / nm:.2f}%; "
              f"of all: no real cluster {100 * A['n_none'].sum() / A['n5'].sum():.2f}%, fragments {100 * A['n_frag'].sum() / A['n5'].sum():.2f}%; "
              f"lost {100 * A['lost'].sum() / A['n_alone5'].sum():.2f}%")
        rr = np.interp(np.log([0.5, 1.0, 2.0, 4.0, 8.0, 15.0]), np.log(th), r1)
        print(f"     r after iteration at 0.5, 1, 2, 4, 8, 15': " + " ".join(f"{v:.4f}" for v in rr))
        res[ns] = res[ns] + (np.array([t_["r1"] for t_ in T]).mean(0),)
    (A15, N15, _, th15, r15, _) = res[15]
    for ns in (12, 20):
        A, Nd, _, thn, rn, _ = res[ns]
        dr = max(abs(rn[0] - r15[0]), abs(rn[-1] - r15[-1]))
        print(f"  {ns} vs 15 scales: credited halos {100 * (Nd.sum() / N15.sum() - 1):+.2f}%, q > 5 detections "
              f"{100 * (A['n5'].sum() / A15['n5'].sum() - 1):+.2f}%; largest change of r at the common nodes 0.5' and 15': {dr:.1e}")
print(f"done [{time.time() - t0:.0f} s]")
