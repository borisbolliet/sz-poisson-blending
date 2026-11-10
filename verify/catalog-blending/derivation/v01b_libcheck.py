"""Spot-check the szsim library sigma_Y / cross_response and the analysis rho table against v01 kernel."""
import numpy as np
from szsim import signal as sg
import v01_kernel as K
m = sg.SZModel()
d = np.load("v01_kernel.npz")
for t in (0.25, 1, 3, 10, 30):
    sl = float(m.sigma_Y(t)); sv = K.sigY(t)[0]
    print(f"theta={t:5}': sigma_Y lib {sl:.5e} verifier {sv:.5e} rel {sl/sv-1:+.2e};  DC exact (lib sigma) {sl*sl*K.RATIO**2/K.NW:.3f}")
# analysis table DC printout used linear interp in ln theta of DC -> check
T = np.load("/Users/boris/GitHub/sz-poisson-blending/data/blending/rho_table.npz")
dcn = T["sigY"] ** 2 * K.RATIO ** 2 / K.NW
print("analysis-style interp of DC (linear in ln th):", np.round(np.interp(np.log([0.25, 1, 3, 10, 30]), np.log(T["th"]), dcn), 3))
# rho spot checks: library exact vs verifier exact vs analysis table (trilinear)
import sys; sys.path.insert(0, "/Users/boris/GitHub/sz-poisson-blending/analysis/blending")
rng = np.random.default_rng(3)
errs = []
for k in range(25):
    a, b = np.exp(rng.uniform(np.log(0.3), np.log(40), 2))
    S = 5 * a + 5 * b + K.RB if hasattr(K, 'RB') else 5*a+5*b+5*np.sqrt(2)*K.SB
    s = np.sort(rng.uniform(0, 1, 6) ** 2 * S)
    rl = m.cross_response(a, b, s, nl=80000)
    rv = K.rho_exact(a, b, s, K.sigY(a)[0])
    r0 = rv[0] if s[0] == 0 else K.rho_exact(a, b, np.array([0.0, S]), K.sigY(a)[0])[0]
    errs.append(np.max(np.abs(rl - rv)) / abs(r0))
print(f"library cross_response vs verifier rho: max|diff|/rho(0) over 25 pairs x 6 s: median {np.median(errs):.2e}, max {np.max(errs):.2e}")
