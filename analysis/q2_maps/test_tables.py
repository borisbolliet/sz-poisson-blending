"""Validate the Hankel tables used for painting and for truth decomposition."""
import numpy as np, time
import q2lib as L
t0 = time.time(); T = L.build_tables(); print(f"tables built/loaded in {time.time()-t0:.1f}s")
paint, resp, rP, rR = T["paint"], T["resp"], T["rP"], T["rR"]
# 1) integral of the painted profile = RATIO_5R500
r = np.linspace(0, 1, L.NR)
I = np.array([np.trapezoid(2*np.pi*r*rP[i]*paint[i]*rP[i], r) for i in range(L.NTH)])
print(f"paint profile integral / RATIO_5R500: min {I.min()/L.sg.RATIO_5R500:.6f} max {I.max()/L.sg.RATIO_5R500:.6f}")
# 2) R_k(0; theta_k) = 1 and vs SZModel.cross_response
m = L.MODEL
for k, tk in enumerate(L.THETA_K):
    v0 = L.resp_eval(resp[k], rR, L.LNTH0, L.DLNTH, np.array([np.log(tk)]), np.array([0.0]))[0]
    print(f"k={k} theta_k={tk:.3f}  R_k(0;theta_k)={v0:.5f}")
rng = np.random.default_rng(0); md = 0
for _ in range(25):
    k = rng.integers(len(L.THETA_K)); th = np.exp(rng.uniform(np.log(0.05), np.log(30)))
    sep = np.array([0.0, 0.5, 1.0, 2.0, 4.0, 8.0])
    ref = m.cross_response(L.THETA_K[k], th, sep, nl=60000)
    tab = L.resp_eval(resp[k], rR, L.LNTH0, L.DLNTH, np.full(sep.size, np.log(th)), sep)
    md = max(md, np.abs(tab - ref).max())
print(f"max |R_table - SZModel.cross_response| over 25 random (k,theta) x 6 seps: {md:.2e}")
print("sigma_k (Hankel, arcmin^2):", np.array2string(T['sigk_arcmin2'], precision=4))
psis, sig = L.filter_bank()
print("discrete-grid sigma_k / Hankel sigma_k - 1:", np.array2string(sig/T['sigk_arcmin2']-1, precision=2))
