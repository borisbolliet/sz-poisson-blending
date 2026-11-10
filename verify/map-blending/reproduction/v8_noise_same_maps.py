"""V8 (C5/C6): identical white-noise realisations filtered with (a) the claim's continuous-Hankel filter bank
(q2lib.filter_bank) and (b) my discrete real-space templates.  Compare the filtered maps pixel by pixel and the
single-scale (0.5') peak counts, so that an implementation difference can be told apart from sample variance."""
import os, sys
os.environ.setdefault("OMP_NUM_THREADS", "2"); os.environ.setdefault("NUMBA_NUM_THREADS", "2")
import numpy as np
from scipy import fft as sfft
from scipy.ndimage import maximum_filter
sys.path.insert(0, "/Users/boris/GitHub/sz-poisson-blending/analysis/q2_maps")
import q2lib as Q
import indep_lib as I

psis, sigc = Q.filter_bank()
TFT, SIG, _ = I.templates(); TFT = TFT.copy(); TFT[:, 0, 0] = 0
nus = np.array([3.0, 3.5, 4.0, 4.5])
ca = np.zeros(4); cb = np.zeros(4); NM = int(sys.argv[1]) if len(sys.argv) > 1 else 30
for s in range(NM):
    rng = np.random.default_rng(424242 + s)
    F = sfft.rfft2(rng.standard_normal((I.NPIX, I.NPIX)) * I.SIGPIX, workers=2)
    a = sfft.irfft2(F * psis[0], s=(I.NPIX, I.NPIX), workers=2) / sigc[0]
    b = sfft.irfft2(F * TFT[0], s=(I.NPIX, I.NPIX), workers=2) / SIG[0]
    if s == 0:
        print(f"pixelwise: std(a) {a.std():.5f} std(b) {b.std():.5f}  max|a-b| {np.abs(a-b).max():.2e}  corr {np.corrcoef(a.ravel()[::7], b.ravel()[::7])[0,1]:.8f}")
    for m, c in ((a, ca), (b, cb)):
        pk = m[(m >= maximum_filter(m, size=3, mode="wrap")) & (m >= 2.9)]
        c += (pk[:, None] >= nus[None, :]).sum(0)
print(f"{NM} maps, 0.5' nearest-pixel peaks per 100 deg^2:")
for i, v in enumerate(nus):
    print(f"  nu>{v}: claim filter {ca[i]/NM:8.3f} +- {np.sqrt(ca[i])/NM:.3f}   my filter {cb[i]/NM:8.3f}")
