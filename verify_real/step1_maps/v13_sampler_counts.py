"""v13: is the 0.4% deficit of realised single-cluster detections (v08) a sampler bias or a sky fluctuation?

Expected full-sky single-cluster detections computed analytically from OUR sampler's model (cell lambdas from the
HMF table trapezoid cell averages, uniform (z, lnM) inside each cell approximated by 2x2 sub-cell points,
lognormal Y scatter by 24-point Gauss-Hermite), vs the build's expected_detections (HMFTable trapezoid weights).
If they agree, the realised deficit is a fluctuation of the 400-patch realisation.
"""
import os, sys, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import vlib as V                                    # noqa: E402
import numpy as np                                  # noqa: E402
from scipy.special import ndtr                      # noqa: E402
from szsim import instrument as ins, signal as sg   # noqa: E402

log = V.Tee(os.path.join(V.HERE, "v13_sampler_counts.log"))
t0 = time.time()
S = V.PatchSampler(600.0)
lam_full = S.lam / S.frac                                   # full-sky expected counts per cell
x, wg = np.polynomial.hermite_e.hermegauss(24); wg /= wg.sum()
sig = {"white": sg.SZModel().sigma_Y, "baseline": ins.MMFModel(ins.Instrument.so_lat("baseline")).sigma_Y,
       "goal": ins.MMFModel(ins.Instrument.so_lat("goal")).sigma_Y}
EXPECT = {"white": [132124.8, 85912.3, 61629.5, 46574.5], "baseline": [133127.1, 81704.5, 55657.0, 40062.8],
          "goal": [203772.7, 128166.7, 88809.8, 64881.7]}
QC = [4.0, 5.0, 6.0, 7.0]
sub = (np.arange(2) + 0.5) / 2
tot = {m: np.zeros(len(QC)) for m in sig}
for iz in range(S.lam.shape[0]):
    for fz in sub:
        z = S.z[iz] + fz * S.dz
        lnM = S.lnM[:-1, None] + (sub[None, :]) * S.dl          # (nM-1, 4)
        dlm = lnM - np.log(1e14)
        lY = np.interp(z, S.zt, S.ly0) + S.alpha * dlm
        th = np.exp(np.interp(z, S.zt, S.lt0) + dlm / 3.0)
        w = lam_full[iz][:, None] / 4.0
        for m, f in sig.items():
            q0 = np.exp(lY) / f(th)
            q = q0[..., None] * np.exp(S.sig * x)
            for ic, qc in enumerate(QC):
                tot[m][ic] += float((w[..., None] * ndtr(q - qc) * wg).sum())
    if iz % 300 == 0:
        log(f"   z cell {iz} [{time.time()-t0:.0f}s]")
for m in sig:
    log(f"   [{m:8s}] own-sampler model: " + "  ".join(f"q>{qc:.0f} {t:,.1f} (build {e:,.1f}; {t/e-1:+.2e})" for qc, t, e in zip(QC, tot[m], EXPECT[m])))
log(f"[v13] done {time.time()-t0:.0f}s")
