"""f06 (step-3b verification, lens 'fits'): how much the S8 bias depends on the estimator of the blending ratio.

Linearised S8 bias F^-1 J^T (R - 1) with MY count model (f00lib.KModel, 'alt' kernel interpolation) for:
  my R_s (top-hat cells of f02: standard, 'fine' = theta and q' cells halved, 'coarse' = doubled), my R_p, my R (bilinear
  and alt N_exp), and the build's R_s, R, R_p (datavec_*.npz, read as data).
Paired tile-bootstrap errors of the DIFFERENCES between my estimators (the same 2,000 resamples of f02), so that an
estimator difference can be judged against its own statistical scatter.
Also: the kernel's Monte Carlo noise in R and R_s -- the build's even- and odd-tile half kernels (K_even, K_odd of the
kernel files) in my model and in N_exp (R on the 324 Q2 tiles, my code), headline, q_cut 5.
Usage: f06_estimators.py     Output: log; out/f06.json
"""
import os, sys, time, json
os.environ["VF_THREADS"] = "1"
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import numpy as np
import f00lib as L

t00 = time.time()
rows = []
SET = {"prod_none": "prod", "szifi3000_none": "szifi3000", "prod_fof10": "prod"}


def linmap(M, cfg, fs):
    N0 = M.counts(L.FID)
    Fo = L.Fit(M, N0 * fs, cfg, L.FID, fs)
    F, J, mu = Fo.fisher(Fo.truth[Fo.free], L.STEPS[Fo.free])
    sig, sS8, g, C = Fo.sigmas(F)
    A = np.linalg.solve(F, J.T)
    return lambda R: (A @ (np.asarray(R) - 1).reshape(-1, 144).T).T @ g / sS8, N0


print("linearised S8 bias [sigma] with my model; paired bootstrap std of the differences between my estimators")
for setup, variant in SET.items():
    for mode in L.MODES:
        mine = np.load(os.path.join(L.OUT, f"dv_{setup}_{mode}.npz"))
        bld = np.load(os.path.join(L.D3B, f"datavec_{setup}_{mode}.npz"))
        Kt = L.KTable(os.path.join(L.D3B, f"kernel_{variant}_{mode}.npz"))
        for qc in (5.0, 6.0):
            s = f"q{qc:.0f}"
            M = L.KModel(mode, qc, L.rbar(variant, mode), Kt, how="alt")
            for sv, fs in L.SURV.items():
                for cfg in ("a", "d", "e"):
                    f, N0 = linmap(M, cfg, fs)
                    v = {"mine_Rs": f(mine[f"Rs_{s}"])[0], "mine_Rp": f(mine[f"Rp_{s}"])[0], "mine_R": f(mine[f"R_bilin_{s}"])[0],
                         "mine_Ralt": f(mine[f"R_alt_{s}"])[0], "build_Rs": f(bld[f"Rs_{s}"])[0], "build_Rp": f(bld[f"Rp_{s}"])[0],
                         "build_R": f(bld[f"R_{s}"])[0]}
                    for var in ("fine", "coarse"):
                        if f"Rs_{var}_{s}" in mine.files:
                            v[f"mine_Rs_{var}"] = f(mine[f"Rs_{var}_{s}"])[0]
                    bs = f(mine[f"Rsboot_{s}"].astype(float)); bp = f(mine[f"Rpboot_{s}"].astype(float)); br = f(mine[f"Rboot_bilin_{s}"].astype(float))
                    bb = f(bld[f"Rsboot_{s}"].astype(float))
                    e = dict(Rs=float(bs.std()), Rp=float(bp.std()), R=float(br.std()), Rs_minus_Rp=float((bs - bp).std()),
                             Rs_minus_R=float((bs - br).std()), build_Rs=float(bb.std()))
                    rows.append(dict(setup=setup, mode=mode, qcut=qc, survey=sv, config=cfg, lin={k: float(x) for k, x in v.items()}, boot=e))
                    if cfg in ("d", "e") or (cfg == "a" and sv == "SO40"):
                        print(f"  {setup:15s} {mode:8s} q>{qc:.0f} {sv:4s} {cfg}: my R_s {v['mine_Rs']:+.3f} (fine {v.get('mine_Rs_fine', np.nan):+.3f}, "
                              f"coarse {v.get('mine_Rs_coarse', np.nan):+.3f}) +- {e['Rs']:.3f}; R_p {v['mine_Rp']:+.3f} +- {e['Rp']:.3f}; R {v['mine_R']:+.3f} "
                              f"(alt {v['mine_Ralt']:+.3f}) +- {e['R']:.3f} | build R_s {v['build_Rs']:+.3f} +- {e['build_Rs']:.3f}, R_p {v['build_Rp']:+.3f}, "
                              f"R {v['build_R']:+.3f} | my R_s - R_p {v['mine_Rs'] - v['mine_Rp']:+.3f} +- {e['Rs_minus_Rp']:.3f}; "
                              f"build R_s - my R_p {v['build_Rs'] - v['mine_Rp']:+.3f}", flush=True)

# kernel Monte Carlo noise: even / odd half kernels (headline, q_cut 5)
print("\nkernel Monte Carlo noise (headline, q_cut 5): linearised S8 bias with the build's even / odd half kernels in my model; "
      "R recomputed with N_exp from the same half kernel (324 Q2 tiles; my binning, bilinear interpolation)")
for mode in L.MODES:
    kb = np.load(os.path.join(L.D3B, f"kernel_prod_{mode}.npz"))
    H = L.load_halos(mode, "prod", "q2", merges=("none",))
    mine = np.load(os.path.join(L.OUT, f"dv_prod_none_{mode}.npz"))
    mq2 = np.load(os.path.join(L.OUT, f"dv_prod_none_q2_{mode}.npz"))
    ei = L.edge_idx(5.0)
    out = {}
    for half in ("full", "even", "odd"):
        Kt = L.KTable(th=kb["th"], QA=kb["QA"], K=kb["K"] if half == "full" else kb["K_" + half])
        m2 = H["qp"] >= L.CRED
        Kh = np.zeros((H["t"].size, 19)); Kh[m2] = Kt.at(H["th"][m2], H["qp"][m2], np.arange(19), how="bilin")
        Ke = Kh[:, ei]; D = Ke - np.concatenate([Ke[:, 1:], np.zeros((Ke.shape[0], 1))], 1)
        iz = np.searchsorted(L.ZED, H["z"], side="right") - 1; ok = (iz >= 0) & (iz < 16)
        Ne = np.zeros((16, 9))
        for b in range(9):
            Ne[:, b] = np.bincount(iz[ok], weights=D[ok, b], minlength=16)
        Nd = mq2["Ndet_q5"]
        Rh = np.where(Ne >= 1, Nd / np.where(Ne > 0, Ne, 1), 1.0)
        M = L.KModel(mode, 5.0, L.rbar("prod", mode), Kt, how="alt")
        for sv, fs in L.SURV.items():
            for cfg in ("a", "d", "e"):
                f, N0 = linmap(M, cfg, fs)
                out[(half, sv, cfg)] = (float(f(mine["Rs_q5"])[0]), float(f(Rh)[0]), float(Ne.sum()))
    for sv in L.SURV:
        for cfg in ("a", "d", "e"):
            e_, o_, fu = out[("even", sv, cfg)], out[("odd", sv, cfg)], out[("full", sv, cfg)]
            print(f"  {mode:8s} {sv:4s} {cfg}: my R_s: even {e_[0]:+.3f} / odd {o_[0]:+.3f} (diff {e_[0] - o_[0]:+.3f}; full {fu[0]:+.3f}); "
                  f"R (Q2 tiles): even {e_[1]:+.3f} / odd {o_[1]:+.3f} (diff {e_[1] - o_[1]:+.3f}; full {fu[1]:+.3f}); "
                  f"N_exp(q > 5, Q2) even/odd - 1 = {100 * (e_[2] / o_[2] - 1):+.2f}%", flush=True)
    rows.append(dict(kind="halfkernel", mode=mode, res={f"{k[0]}_{k[1]}_{k[2]}": v for k, v in out.items()}))
json.dump(rows, open(os.path.join(L.OUT, "f06.json"), "w"), indent=1)
print(f"done [{time.time() - t00:.0f} s]; peak RSS {L.rss_gb():.2f} GB")
