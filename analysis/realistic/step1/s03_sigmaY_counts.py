"""Step 1 / s03: MMF sigma_Y(theta500) for SO LAT baseline/goal, comparison with the old white-noise filter,
and expected full-sky detections.

(1) Code-path validation: MMFModel(Instrument.white_y()) must reproduce szsim.signal.SZModel (sigma_Y,
    sigma_Y_point, cross_response) - same physics, independent implementation of the l integrals.
(2) sigma_Y_mmf(theta500) for baseline / goal, lmin = 100 (default) and 36 (= 2 pi / 10 deg), lmax = 20000;
    ratio to sigma_Y_white(4e-6, 1.4').  Variants (baseline): no CMB, no atmosphere, no atm. correlation,
    145 GHz only, 93+145 only.
(3) Numerical convergence: dl = 1 vs 0.5; lmax = 20000 vs 30000.
(4) Expected full-sky detections q_obs > 4..7 (SZModel.expected_detections with the MMF sigma_Y; same HMF,
    Y-M relation, scatter); white case must reproduce 85,912 at q>5 (szsim/README).
Outputs: data_real/sigmaY_mmf.npz; numbers in the log.
"""
import os, sys, time, json
for v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMBA_NUM_THREADS"):
    os.environ[v] = "4"
import numpy as np
from szsim import instrument as ins, signal as sg

ROOT = "/Users/boris/GitHub/sz-poisson-blending"
TH = np.geomspace(0.1, 100.0, 61)
QC = [4.0, 5.0, 6.0, 7.0]


def main():
    t0 = time.time()
    old = sg.SZModel()
    wm = ins.MMFModel(ins.Instrument.white_y())
    print("=== (1) white-noise reproduction: MMFModel(white_y) vs SZModel ===")
    r = wm.sigma_Y_exact(TH) / old.sigma_Y_exact(TH) - 1
    print(f"   sigma_Y_exact rel. diff over theta in [0.1, 100]': max |.| = {np.abs(r).max():.2e}")
    print(f"   sigma_Y_point: {wm.sigma_Y_point:.6e} vs {old.sigma_Y_point:.6e}")
    worst = 0
    for a, b in [(0.5, 0.5), (1, 3), (3, 1), (2, 8), (8, 2), (10, 10)]:
        s = np.linspace(0, 5 * a + 5 * b + 4, 50)
        x, y = wm.cross_response(a, b, s), old.cross_response(a, b, s, nl=80000)
        worst = max(worst, np.abs(x - y).max())
    print(f"   cross_response, 6 (thA, thB) x 50 sep: max |rho_new - rho_old| = {worst:.2e}")

    print("\n=== (2) MMF sigma_Y [arcmin^2] ===")
    cfg = {}
    for mode in ("baseline", "goal"):
        for lmin in (100.0, 36.0):
            cfg[f"{mode}_l{int(lmin)}"] = ins.Instrument.so_lat(mode, lmin=lmin)
    cfg["baseline_noCMB"] = ins.Instrument.so_lat("baseline", cmb=False)
    cfg["baseline_noATM"] = ins.Instrument.so_lat("baseline", atm=False)
    cfg["baseline_atmcorr0"] = ins.Instrument.so_lat("baseline", atm_corr=0.0)
    cfg["baseline_145only"] = ins.Instrument.so_lat("baseline", channels=[3])
    cfg["baseline_93+145"] = ins.Instrument.so_lat("baseline", channels=[2, 3])
    cfg["baseline_dl0.5"] = ins.Instrument.so_lat("baseline", dl=0.5)
    cfg["baseline_lmax30000"] = ins.Instrument.so_lat("baseline", lmax=30000)
    # 'white' = the existing SZModel (reference); 'white_mmf' = the MMF code configured as the white single-channel
    # instrument [arbiter fix: the first version stored only SZModel's values, so the file did not document the MMF code]
    sig = {"white": old.sigma_Y_exact(TH), "white_mmf": wm.sigma_Y_exact(TH)}
    for k, I in cfg.items():
        sig[k] = ins.MMFModel(I).sigma_Y_exact(TH)
    show = [0.25, 0.5, 1, 2, 3, 5, 10, 20, 50]
    idx = [int(np.argmin(np.abs(TH - s))) for s in show]
    print("   (grid nodes nearest to 0.25..50'; NOT the exact values 2, 5, 20' -- see the exact-theta table below)")
    print("   theta500 [']   " + " ".join(f"{TH[i]:9.3f}" for i in idx))
    for k, v in sig.items():
        print(f"   {k:20s} " + " ".join(f"{v[i]:9.3e}" for i in idx))
    print("   ratio sigma_mmf / sigma_white:")
    for k in [k for k in sig if k != "white"]:
        print(f"   {k:20s} " + " ".join(f"{sig[k][i]/sig['white'][i]:9.3f}" for i in idx))
    print("   equivalent single-frequency white y-noise (sigma_mmf/sigma_white * 4e-6 arcmin; 1.4' beam):")
    for k in ("baseline_l100", "goal_l100"):
        print(f"   {k:20s} " + " ".join(f"{4e-6*sig[k][i]/sig['white'][i]:9.2e}" for i in idx))
    print(f"\n=== (3) convergence (baseline): max |dl=0.5 / dl=1 - 1| = "
          f"{np.abs(sig['baseline_dl0.5']/sig['baseline_l100']-1).max():.1e}; max |lmax 3e4 / 2e4 - 1| = "
          f"{np.abs(sig['baseline_lmax30000']/sig['baseline_l100']-1).max():.1e} "
          f"(at theta<=0.25': {np.abs(sig['baseline_lmax30000']/sig['baseline_l100']-1)[TH<=0.25].max():.1e}; "
          f"theta>=0.5': {np.abs(sig['baseline_lmax30000']/sig['baseline_l100']-1)[TH>=0.5].max():.1e})")
    # [arbiter fix] sigma_Y at EXACT theta500 values (the grid table above is at the nearest geomspace nodes, e.g.
    # 1.995, 5.012, 19.95', which differ from 2, 5, 20' by up to 0.6% in sigma_Y)
    TEX = np.array([0.25, 0.5, 1.0, 2.0, 3.2, 5.0, 10.0, 20.0, 50.0])
    tex = {"white": old.sigma_Y_exact(TEX), "white_mmf": wm.sigma_Y_exact(TEX),
           "baseline_l100": ins.MMFModel(cfg["baseline_l100"]).sigma_Y_exact(TEX),
           "goal_l100": ins.MMFModel(cfg["goal_l100"]).sigma_Y_exact(TEX)}
    print("\n   sigma_Y at EXACT theta500 [arcmin^2]:")
    print("   theta500 [']   " + " ".join(f"{t:9.3f}" for t in TEX))
    for k, v in tex.items():
        print(f"   {k:20s} " + " ".join(f"{x:9.4e}" for x in v))
    for k in ("baseline_l100", "goal_l100"):
        print(f"   {k+'/white':20s} " + " ".join(f"{a/b:9.3f}" for a, b in zip(tex[k], tex["white"])))
        print(f"   {'equiv. white '+k[:4]:20s} " + " ".join(f"{4e-6*a/b:9.2e}" for a, b in zip(tex[k], tex["white"])))
    print(f"   white_mmf/white - 1 at the exact nodes: max |.| = {np.abs(tex['white_mmf']/tex['white']-1).max():.2e}")
    np.savez(f"{ROOT}/data_real/sigmaY_mmf.npz", theta500=TH, **{k: v for k, v in sig.items()},
             theta_exact=TEX, **{f"exact_{k}": v for k, v in tex.items()})

    print(f"\n=== (4) expected full-sky detections (M>1e13 floor, 0.01<z<3, no blending) [{time.time()-t0:.0f}s] ===")
    models = {"white (old)": old,
              "SO baseline lmin100": ins.MMFModel(cfg["baseline_l100"]),
              "SO goal lmin100": ins.MMFModel(cfg["goal_l100"]),
              "SO baseline lmin36": ins.MMFModel(cfg["baseline_l36"]),
              "SO goal lmin36": ins.MMFModel(cfg["goal_l36"])}
    res = {}
    for k, m in models.items():
        n = m.expected_detections(QC)
        res[k] = [float(x) for x in n]
        print(f"   {k:22s}: q>4 {n[0]:10,.1f}  q>5 {n[1]:10,.1f}  q>6 {n[2]:10,.1f}  q>7 {n[3]:10,.1f}   "
              f"(f_sky=0.4 at q>5: {0.4*n[1]:,.0f})  [{time.time()-t0:.0f}s]")
    print("   reference (szsim/README): white q>4..7 = 132,124.8 / 85,912.3 / 61,629.5 / 46,574.5")
    print("   Raghunathan 2022 Table 3 (S/N>=5, f_sky=0.4, with CIB/radio/kSZ residuals, other Y-M): "
          "SO-Baseline 14,424, SO-Goal 26,445")
    # independent check of the count integral: Monte Carlo over a stored catalogue (M>1e14 floor skies)
    from szsim import sampler
    import glob
    files = sorted(glob.glob(f"{ROOT}/data/catalogs/cat_M1e14_seed1*.npz"))[:5]
    for k in ("SO baseline lmin100", "SO goal lmin100"):
        m = models[k]
        nd = []
        rng = np.random.default_rng(11)
        for f in files:
            c = sampler.load_catalog(f, fields=["Y500", "theta500"])
            q = c["Y500"] / m.sigma_Y(c["theta500"])
            nd.append(np.sum(q + rng.standard_normal(len(q)) > 5.0))
        an = m.expected_detections(5.0, Mmin=1e14)
        print(f"   check {k}: M>1e14 q>5 analytic {an:,.1f} vs {len(files)} stored skies (Y incl. scatter, fresh noise) "
              f"{np.mean(nd):,.1f} +- {np.std(nd)/np.sqrt(len(nd)):.1f}  (pull {(np.mean(nd)-an)/np.sqrt(an/len(nd)):+.2f} Poisson sigma)")
    json.dump(res, open(f"{ROOT}/data_real/expected_detections.json", "w"), indent=1)

    print("\n=== (5) compensation: response to a uniform y background (k = 0 weight) ===")
    for mode in ("baseline", "goal"):
        I0 = ins.Instrument.so_lat(mode, lmin=0.0, lmax=20000)
        m0, m100 = ins.MMFModel(I0), ins.MMFModel(cfg[f"{mode}_l100"])
        l = np.array([1.0, 5.0, 10.0, 50.0])
        v = np.interp(l, I0.ell, I0.inv_Ny)
        th = np.array([1.0, 4.0, 16.0, 50.0])
        print(f"   [{mode}] lmin=0: 1/N_y at l=1,5,10,50 = {np.array2string(v, precision=3)} sr^-1; log-slope (l=1..5) "
              f"{np.log(v[1]/v[0])/np.log(5):.2f} (atmosphere ~ l^-3.5); 1/N_y(l=0) = {I0.inv_Ny[0]} -> DC response "
              f"sigma^2 RATIO^2 / N_y(0) = {m0.dc_response(4.0)[0]} (lmin=100: {m100.dc_response(4.0)[0]})")
        print(f"   [{mode}] sigma_Y(lmin=0)/sigma_Y(lmin=100) at theta500 = {list(th)}: {np.round(m0.sigma_Y_exact(th)/m100.sigma_Y_exact(th), 5)}")
    print(f"   [white] DC response of the old filter at theta500 = 1, 4, 16': {np.round(old.sigma_Y_exact([1, 4, 16])**2 * sg.RATIO_5R500**2 * ins.ARCMIN**2 / old.N_white_sr, 2)} arcmin^2 (not compensated)")
    print(f"total {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
