"""Step 3b / b08: checks of the analyst's naive model (s3bcosmo.SmoothModelK).

  (1) Machinery: with K = Phi(q' - e) (tabulated on a fine q' grid) SmoothModelK must reproduce step 2's SmoothModelS
      with sigma_eff = rbar sigma_Y (p04's map-sigma model), bin by bin.
  (2) The naive model with the variant's kernel K against the tiles: mu_naive(truth) x (tile area / 41,253 deg^2) vs
      sum over the tiles of N_exp (b07; each halo's own Y500 and its tile's sigma_finder), total and per z bin, q > 5,
      in units of the halos' Poisson noise sqrt(N_exp).
  (3) The model uses the mean r over the tiles (rbar), N_exp each tile's r: N_exp recomputed with rbar for every halo.
  (4) rbar per variant and noise level (what the analyst's sigma_eff is).
Usage: python b08_model_checks.py
"""
import os, sys, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import s3blib as B                                 # noqa: E402
B.threads(4)
import numpy as np                                 # noqa: E402
import s3bcosmo as SC                              # noqa: E402
import b07_datavec as D7                           # noqa: E402
F, CL, S2 = B.F, SC.CL, SC.S2
FT = SC.AREA_TILES / SC.AREA_SKY

if __name__ == "__main__":
    t0 = time.time()
    print(f"tiles: 324 x 100 deg^2 = {SC.AREA_TILES:.0f} deg^2 = {FT:.4f} of the sky")
    for variant in B.VARIANTS:
        for mode in B.MODES:
            rb = SC.rbar(variant, mode)
            print(f"==================== {variant} / {mode}: rbar(theta) at 0.5' / 1.04' / 2.15' / 4.45' / 9.23' / 15': "
                  + " / ".join(f"{v:.4f}" for v in rb[[0, 3, 6, 9, 12, 14]]), flush=True)
            for qc in SC.QCUTS:
                Ma = SC.model(variant, mode, qc, kernel="analytic", k=0.0)
                Ms = S2.SmoothModelS(qc, SC.lnsig(variant, mode), grid=SC.grid())
                a, b = Ma.counts(SC.FID), Ms.counts(SC.FID)
                print(f"  (1) q > {qc:.0f}: K-machinery with Phi(q' - e) {a.sum():,.1f} vs SmoothModelS(sigma_eff) {b.sum():,.1f}; "
                      f"max |rel| per bin with > 1 count {np.abs(a / b - 1)[b > 1].max():.1e}", flush=True)
            M = SC.model(variant, mode, 5.0)
            mu = M.counts(SC.FID)
            dv = np.load(os.path.join(B.OUT, f"datavec_{variant}_none_{mode}.npz"))
            nt = int(dv["ntiles"]) if "ntiles" in dv.files else 324
            ne = dv["Nexp_q5"].sum(1); mz = mu.sum(1) * nt * 100.0 / SC.AREA_SKY
            pull = (ne - mz) / np.sqrt(mz)
            print(f"  (2) q > 5, {nt} tiles: model x area {mz.sum():,.1f} vs tiles' N_exp {ne.sum():,.1f}: {100 * (ne.sum() / mz.sum() - 1):+.2f}% "
                  f"({(ne.sum() - mz.sum()) / np.sqrt(mz.sum()):+.2f} Poisson sigma); per z bin pulls: " + " ".join(f"{v:+.1f}" for v in pull)
                  + f"; chi^2 {np.sum(pull ** 2):.1f} for {pull.size} bins; naive model full sky {mu.sum():,.1f}", flush=True)
            zed, qed = CL.default_bins(5.0)
            for a_, b_ in zip(D7.ZC[:-1], D7.ZC[1:]):
                kz = (zed[:-1] >= a_ - 1e-9) & (zed[1:] <= b_ + 1e-9)
                print(f"      z {a_}-{b_}: tiles/model - 1 = {100 * (ne[kz].sum() / mz[kz].sum() - 1):+.2f}% (Poisson {100 / np.sqrt(mz[kz].sum()):.2f}%)")
            # (3) rbar instead of each tile's r (the 324 Q2 tiles)
            C = np.load(os.path.join(B.OUT, f"halos_{mode}.npz")); V = np.load(os.path.join(B.OUT, f"halos_{variant}_{mode}.npz"))
            th, Y, pid, z = C["theta500"].astype(float), C["Y500"].astype(float), C["pid"], C["z"].astype(float)
            Kern = B.Kernel.load(os.path.join(B.OUT, f"kernel_{variant}_{mode}.npz"))
            qbar = Y / (np.interp(np.log(th), np.log(F.THETAS15), rb) * B.sigma_Y(mode, th))
            m = qbar >= 2.0
            nb = Kern(th[m], qbar[m], int(np.argmin(np.abs(B.EDGES - 5.0))))[:, 0].sum()
            m2 = V["qp"].astype(float) >= 2.0
            n0 = Kern(th[m2], V["qp"].astype(float)[m2], int(np.argmin(np.abs(B.EDGES - 5.0))))[:, 0].sum()
            print(f"  (3) Q2 tiles: N_exp(q > 5) with each tile's r {n0:,.1f}, with rbar for every halo {nb:,.1f} ({100 * (nb / n0 - 1):+.3f}%)", flush=True)
    print(f"done [{time.time() - t0:.0f} s]")
