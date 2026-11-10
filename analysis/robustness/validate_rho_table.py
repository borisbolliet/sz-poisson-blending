"""Validate the tabulated cross response rho(thA, thB, sep) against szsim.SZModel.cross_response
(direct l-integral) and check the exact identity int rho d^2sep = sigma_A^2 RATIO^2 / N_w."""
import time, numpy as np
from szsim import signal as sg, cosmo_hmf as ch
from blendlib import RhoTable

rng = np.random.default_rng(7)
for beam in [1.4, 5.0, 7.3]:
    t = time.time()
    rt = RhoTable(beam)
    print(f"beam {beam}': table {rt.rho.shape} ready in {time.time()-t:.1f}s")
    m = sg.SZModel(beam_fwhm_arcmin=beam)
    # random triples
    nt = 60
    thA = np.exp(rng.uniform(np.log(0.3), np.log(40), nt))
    thB = np.exp(rng.uniform(np.log(0.3), np.log(40), nt))
    errs, rel = [], []
    for a, b in zip(thA, thB):
        seps = rng.uniform(0, 1.5 * (a + b) + 2 * beam, 4)
        ex = m.cross_response(a, b, seps)
        tb = rt(np.full(4, a), np.full(4, b), seps)
        errs.append(np.abs(tb - ex))
        rel.append(np.abs(tb - ex)[ex > 0.02] / ex[ex > 0.02])
    errs = np.concatenate(errs); rel = np.concatenate(rel)
    print(f"  {len(errs)} random (thA,thB,sep): max|table-exact| = {errs.max():.2e}, median = {np.median(errs):.2e}; "
          f"for rho>0.02: max rel err = {rel.max():.2e}, median rel = {np.median(rel):.2e}")
    # self-response at zero separation must be 1
    print(f"  rho(th,th,0) on grid nodes: max|rho-1| = {np.max(np.abs(np.diagonal(rt.rho[:, :, 0]) - 1)):.1e}")
    # identity: int rho d^2 sep = sigma_A^2 RATIO^2 / N_w (independent of thB)
    Nw = m.noise_y_arcmin ** 2
    worst = 0
    for ia in [10, 25, 35, 45]:
        sA = m.sigma_Y_exact(rt.th[ia])[0]
        target = sA ** 2 * sg.RATIO_5R500 ** 2 / Nw
        for ib in [5, 20, 35, 45]:
            integ = np.trapezoid(2 * np.pi * rt.sep * rt.rho[ia, ib], rt.sep)
            worst = max(worst, abs(integ / target - 1))
    print(f"  int rho d^2sep vs sigma_A^2 RATIO^2/N_w (16 combos, thA,thB in [0.3,30]'): max rel dev = {worst:.2e}")
