"""Repair check: is the count model converged and is its Fisher matrix step-independent?
Compares the original cosmolib.NaiveModel (121 lnM nodes, 40 GH nodes, STEP) with cosmolib2.SmoothModel
(scatter as Gaussian convolution in ln q0, lnM refined x MSUB, larger steps).
  1. totals vs szsim; per-bin differences old vs new; new model at (msub, h) = (4, 0.002) vs (8, 0.001).
  2. Fisher sigmas (SO40, q>5) for configs a, b, d, e: old model at step x1, x10; new model at step x0.5, 1, 2, 4 and
     the refined (8, 0.001) model.
  3. Profile likelihood in s8 (config b: Om, (1-b) profiled) on the model's own expectation: Delta chi2 vs quadratic.
"""
import os, sys, time
for v in ("OMP_NUM_THREADS", "NUMBA_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(v, "3")
import numpy as np
from scipy.optimize import minimize

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import cosmolib as CL, cosmolib2 as C2
from c05_fits import CONFIGS

t0 = time.time()
from szsim import signal as sg
sz = sg.SZModel()
FS = 0.4
for qc in (5.0, 6.0):
    O = CL.NaiveModel(qc); N = C2.SmoothModel(qc); R = C2.SmoothModel(qc, msub=8, h=0.001)
    No, Nn, Nr = O.counts(CL.FID), N.counts(CL.FID), R.counts(CL.FID)
    ref = sz.expected_detections(qc)
    m = No > 10
    print(f"q>{qc:.0f}: totals old {No.sum():,.1f} new {Nn.sum():,.1f} refined {Nr.sum():,.1f} szsim {ref:,.1f} "
          f"(new/szsim-1 = {Nn.sum()/ref-1:+.2e})")
    print(f"   per bin (mu>10): max|old/new-1| = {np.abs(No/Nn-1)[m].max():.4f}; max|new/refined-1| = {np.abs(Nn/Nr-1)[m].max():.2e}")
print(f"[{time.time()-t0:.0f}s]")

O = CL.NaiveModel(5.0); N = C2.SmoothModel(5.0); R = C2.SmoothModel(5.0, msub=8, h=0.001)
dO, dN, dR = O.counts(CL.FID), N.counts(CL.FID), R.counts(CL.FID)


class OldStep(CL.Fitter):
    def __init__(self, *a, mult=1.0, **k):
        super().__init__(*a, **k); self.mult = mult

    def jac(self, x):
        J = []
        for k, i in enumerate(self.free):
            e = np.zeros(len(x)); e[k] = CL.STEP[i] * self.mult
            J.append((self.mu(x + e) - self.mu(x - e)) / (2 * CL.STEP[i] * self.mult))
        return np.stack(J, -1)[self.mask]


print("\nFisher marginal sigma at the truth, 40% sky, q>5 (Om, s8, S8 | nuisances)")
for cn in ("a", "b", "c", "d", "e"):
    free, pri = CONFIGS[cn]; fi = [CL.PNAMES.index(p) for p in free]
    rows = []
    for lab, F in [("old  step x1 ", OldStep(O, dO * FS, free, fsky=FS, priors=pri, mult=1)),
                   ("old  step x10", OldStep(O, dO * FS, free, fsky=FS, priors=pri, mult=10)),
                   ("new  step x0.5", C2.Fitter2(N, dN * FS, free, fsky=FS, priors=pri, step_mult=0.5)),
                   ("new  step x1 ", C2.Fitter2(N, dN * FS, free, fsky=FS, priors=pri, step_mult=1)),
                   ("new  step x2 ", C2.Fitter2(N, dN * FS, free, fsky=FS, priors=pri, step_mult=2)),
                   ("new  step x4 ", C2.Fitter2(N, dN * FS, free, fsky=FS, priors=pri, step_mult=4)),
                   ("refined x1  ", C2.Fitter2(R, dR * FS, free, fsky=FS, priors=pri, step_mult=1))]:
        s, _ = CL.derived_sigma(F.fisher(CL.FID[fi]), fi)
        print(f"  {cn} {lab}: " + "  ".join(f"{p} {s[p]:.5f}" for p in free + ["S8"]))
print(f"[{time.time()-t0:.0f}s]")


# ---- profile likelihood in s8, config b, own expectation
class Prof(C2.Fitter2):
    def __init__(self, *a, s8=None, **k):
        super().__init__(*a, **k); self.s8 = s8

    def full(self, x):
        th = CL.FID.copy(); th[[0, 2]] = x; th[1] = self.s8
        return th


print("\nProfile likelihood, config b (Om, (1-b) profiled), 40% sky, q>5, data = own expectation")
for lab, M, d in (("old", O, dO), ("new", N, dN)):
    Ff = C2.Fitter2(M, d * FS, ["Om", "s8", "omb"], fsky=FS) if lab == "new" else OldStep(M, d * FS, ["Om", "s8", "omb"], fsky=FS, mult=10)
    s, _ = CL.derived_sigma(Ff.fisher(CL.FID[[0, 1, 2]]), [0, 1, 2])
    out = []
    for ds in (-0.02, -0.01, -0.005, -0.0025, 0.0025, 0.005, 0.01, 0.02):
        P = Prof(M, d * FS, ["Om", "omb"], fsky=FS, s8=CL.FID[1] + ds)
        best = None
        for x0 in (CL.FID[[0, 2]], CL.FID[[0, 2]] + [-ds * 0.9, ds * 2.0]):
            r = minimize(lambda u: -P.lnL(x0 + u * [s['Om'], s['omb']]), np.zeros(2), method="Nelder-Mead",
                         options=dict(xatol=1e-4, fatol=1e-6, maxiter=800))
            best = r.fun if best is None else min(best, r.fun)
        out.append(f"ds8={ds:+.4f}: dchi2={2*best:.3f} (quad {(ds/s['s8'])**2:.3f})")
    print(f"  {lab} (sigma_s8 from {'new step x1' if lab=='new' else 'old step x10'} Fisher = {s['s8']:.5f}):\n    " + "\n    ".join(out))
print(f"total {time.time()-t0:.0f}s")
