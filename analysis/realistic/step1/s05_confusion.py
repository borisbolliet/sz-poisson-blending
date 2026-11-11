"""Step 1 / s05: Poisson confusion variance of the matched-filter Y estimate (Campbell), and the mean boost.

Neighbour population: analysis/blending/blendlib.neighbour_population (Tinker08, Planck13 Y-M + 0.173 scatter,
M500 > 1e13, 0.01 < z < 3; independent of the instrument).  For filter A (size theta_A):
   var_conf(A) = sum_B n_B Y_B^2 int d^2s rho_AB(s)^2           (Campbell's theorem, 2nd cumulant)
               = sigma_A^4 (1/2pi) int dl l g_A^2 C_l^yy,P / N_y^2      (Parseval; method 1)
   mu(A)       = sum_B n_B Y_B int d^2s rho_AB = ybar sigma_A^2 RATIO^2 / N_y(0)      (Campbell's 1st cumulant)
Method 2 (real space): int 2 pi s rho_AB(s)^2 ds from MMFModel.cross_response on an s grid out to 900',
summed over the theta_B-binned population.
Splits: neighbour mass classes (1e13-3e13, 3e13-1e14, >1e14) and 'sub-threshold' neighbours (their own mean
q_B = Y_B / sigma_Y(theta_B) < 5 for the same instrument; i.e. detected clusters masked/removed, the
Raghunathan-2022 'confusion from haloes below the detection limit').
Variants: lmin = 36; filter whose noise model includes the tSZ power (N_y -> N_y + C_l^yy,P), as when the
MMF noise is estimated from the data.
Outputs: data_real/confusion_var.npz.
"""
import os, sys, time
for v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMBA_NUM_THREADS"):
    os.environ[v] = "4"
import numpy as np

ROOT = "/Users/boris/GitHub/sz-poisson-blending"
sys.path.insert(0, f"{ROOT}/analysis/blending")
import blendlib as bl                                       # noqa: E402
from szsim import instrument as ins, signal as sg          # noqa: E402

AR = ins.ARCMIN
THA = np.geomspace(0.25, 80.0, 40)                          # same theta_A nodes as the white rho table


def main():
    t0 = time.time()
    m0 = sg.SZModel()
    thB, YB, cB, wB = bl.neighbour_population(zsub=1, msub=6, ngh=16, model=m0)
    print(f"population: {len(thB):,} points; N full sky = {wB.sum()*bl.ARCMIN2_FULLSKY:,.0f}; "
          f"ybar = {np.sum(wB*YB):.4e} (Y500 units; b03: 3.5895e-07)  [{time.time()-t0:.0f}s]")
    insts = {"white": ins.Instrument.white_y(),
             "baseline": ins.Instrument.so_lat("baseline"),
             "goal": ins.Instrument.so_lat("goal"),
             "baseline_l36": ins.Instrument.so_lat("baseline", lmin=36.0),
             "goal_l36": ins.Instrument.so_lat("goal", lmin=36.0)}
    models = {k: ins.MMFModel(I) for k, I in insts.items()}
    out = dict(thA=THA)
    show = [0.5, 1.0, 2.0, 4.0, 8.0, 16.0, 32.0]
    ish = [int(np.argmin(np.abs(THA - s))) for s in show]
    ybar = np.sum(wB * YB)

    # check of the theta-binned C_l^yy against an unbinned direct sum (random subset)
    rng = np.random.default_rng(3)
    sub = rng.choice(len(thB), 3000, replace=False)
    lt = np.array([300.0, 1000.0, 3000.0, 8000.0])
    direct = np.array([np.sum(wB[sub] * YB[sub] ** 2 * sg.template_hankel(L * thB[sub] * AR) ** 2) for L in lt]) * AR ** 2
    binned = ins.poisson_Cyy(lt, thB[sub], YB[sub], wB[sub])
    print(f"C_l^yy binned vs direct sum (3000-point subset) at l={list(lt.astype(int))}: rel diff {np.round(binned/direct-1, 5)}")

    for name, m in models.items():
        I = m.inst
        sigA = m.sigma_Y_exact(THA)
        qB = YB / m.sigma_Y(thB)
        res = {}
        masks = {"all": np.ones(len(thB), bool), "c0": cB == 0, "c1": cB == 1, "c2": cB == 2, "subthr": qB < 5.0}
        for k, msk in masks.items():
            Cyy = ins.poisson_Cyy(I.ell, thB[msk], YB[msk], wB[msk])
            res[k] = m.confusion_var(THA, Cyy) / sigA ** 2          # noise units
            if k == "all":
                Cyy_all = Cyy
            if k == "subthr":
                Cyy_sub = Cyy
        mu = ybar * m.dc_response(THA) / sigA                       # noise units
        out[f"{name}_sigA"] = sigA
        for k in res:
            out[f"{name}_var_{k}"] = res[k]
        out[f"{name}_mu"] = mu
        out[f"{name}_ell"] = I.ell[::10]
        out[f"{name}_Cyy_all"] = Cyy_all[::10]
        out[f"{name}_Cyy_sub"] = Cyy_sub[::10]
        print(f"\n=== {name}: confusion variance in units of the filter noise variance (var_conf / sigma_A^2) ===")
        print("   theta_A [']    " + " ".join(f"{THA[i]:8.2f}" for i in ish))
        for k in res:
            print(f"   {k:13s}  " + " ".join(f"{res[k][i]:8.3g}" for i in ish))
        print(f"   {'mu (DC)':13s}  " + " ".join(f"{mu[i]:8.3g}" for i in ish) + "   (Campbell mean shift, noise units)")
        print(f"   fraction of neighbour Y^2 power in sub-threshold halos at l=3000: "
              f"{np.interp(3000, I.ell, Cyy_sub)/np.interp(3000, I.ell, Cyy_all):.3f}")
        if name == "white":
            L3 = 3000.0
            Dyy = L3 * (L3 + 1) * np.interp(L3, I.ell, Cyy_all) / (2 * np.pi)
            print(f"   tSZ Poisson power of the population: 1e12 l(l+1)C_l^yy/2pi at l=3000 = {Dyy*1e12:.3f}; "
                  f"D_3000 at 145 GHz = {Dyy*ins.tsz_f_uK(145.0)**2:.2f} uK^2; at 150 GHz = {Dyy*ins.tsz_f_uK(150.0)**2:.2f} uK^2")
            print("   (b03 white, class 0/1 variances at thA = 0.52/1.10/2.30/4.15/8.70/18.23': "
                  "0.006/0.008/0.011/0.015/0.022/0.030 and 0.05/0.068/0.103/0.15/0.24/0.37 (x<500 only))")
            for th, a0, a1 in zip([0.52, 1.10, 2.30, 4.15, 8.70, 18.23], [0.006, 0.008, 0.011, 0.015, 0.022, 0.030],
                                  [0.05, 0.068, 0.103, 0.15, 0.24, 0.37]):
                print(f"      thA={th:5.2f}': Parseval c0 {np.interp(np.log(th), np.log(THA), res['c0']):.4f} (b03 {a0}); "
                      f"c1 {np.interp(np.log(th), np.log(THA), res['c1']):.4f} (b03 {a1})")
        if name in ("baseline", "goal"):
            # variant: filter noise model includes the tSZ power
            for lab, P in (("all", Cyy_all), ("subthr", Cyy_sub)):
                Pf = lambda l, P=P: np.interp(l, I.ell, P)
                Ik = ins.Instrument.so_lat(name, extra_y_power=Pf)
                mk = ins.MMFModel(Ik)
                sk = mk.sigma_Y_exact(THA)
                vk = mk.confusion_var(THA, P) / sk ** 2
                out[f"{name}_tszaware_{lab}_sigA"] = sk
                out[f"{name}_tszaware_{lab}_var"] = vk
                print(f"   variant: filter knows C^yy({lab}): sigma_Y/sigma_Y(default) = "
                      + " ".join(f"{sk[i]/sigA[i]:6.3f}" for i in ish) + " ; var_conf/sigma^2 = "
                      + " ".join(f"{vk[i]:7.3g}" for i in ish))
        print(f"   [{time.time()-t0:.0f}s]", flush=True)

    # ---- method 2: real-space int 2 pi s rho^2 ds  vs Parseval rho2_area, then the population sum
    print("\n=== method 2: real-space integral of rho^2 vs Parseval (MMF baseline), 30 theta_B bins ===", flush=True)
    m = models["baseline"]
    lt = np.log(thB)
    edges = np.linspace(lt.min() - 1e-9, lt.max() + 1e-9, 31)
    idx = np.digitize(lt, edges) - 1
    thc = np.exp(0.5 * (edges[1:] + edges[:-1]))
    masks = {"all": np.ones(len(thB), bool), "subthr": YB / m.sigma_Y(thB) < 5}
    S2 = {k: np.bincount(idx[v], weights=(wB * YB ** 2)[v], minlength=30) for k, v in masks.items()}
    for thA in (1.0, 4.0, 16.0):
        sA = m.sigma_Y_exact(thA)[0]
        a_real = np.zeros(30); a_pars = np.zeros(30)
        for b in range(30):
            if S2["all"][b] == 0:
                continue
            tb = thc[b]
            s_in = 5 * thA + 5 * tb + 30.0
            s = np.concatenate([np.linspace(0, s_in, 3000), np.linspace(s_in, s_in + 900.0, 6000)[1:]])
            rho = m.cross_response(thA, tb, s)
            a_real[b] = np.trapezoid(2 * np.pi * s * rho ** 2, s)
            a_pars[b] = m.rho2_area(thA, tb)
        w = S2["all"] * a_pars
        rel = np.abs(a_real / np.where(a_pars > 0, a_pars, 1) - 1)
        print(f"   thA={thA:5.1f}': per-bin |real/Parseval - 1|: max {rel[a_pars > 0].max():.2e}, "
              f"Y^2-weighted mean {np.sum(w * rel) / w.sum():.2e}", flush=True)
        for k in masks:
            tr, tp = np.sum(S2[k] * a_real) / sA ** 2, np.sum(S2[k] * a_pars) / sA ** 2
            print(f"      [{k:6s}] var_conf/sigma^2: real-space {tr:.5g}  Parseval {tp:.5g}  (rel {tr/tp-1:+.2e}); "
                  f"C_l^yy route (1200 bins) {np.interp(np.log(thA), np.log(THA), out['baseline_var_' + k]):.5g}", flush=True)
    np.savez(f"{ROOT}/data_real/confusion_var.npz", **out)
    print(f"total {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
