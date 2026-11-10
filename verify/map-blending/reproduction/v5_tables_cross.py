"""V5 (C2): compare the claim's painting / response tables (data/q2_maps/tables.npz) with my independently built ones
(real-space I0 beam convolution for painting; own Hankel for responses) at random (k, theta, r)."""
import numpy as np
import indep_lib as I

T = np.load("/Users/boris/GitHub/sz-poisson-blending/data/q2_maps/tables.npz")
paint_c, resp_c, rP_c, rR_c = T["paint"], T["resp"], T["rP"], T["rR"]
lnth0_c, dlnth_c = float(T["lnth0"]), float(T["dlnth"])
assert np.allclose(T["theta_k"], I.THK)
PT, PR = I.paint_table(); RT, RR = I.resp_table(I.templates()[1])


def ev_claim(tab, rmax, lnth, r):
    nth, nr = tab.shape
    t = np.clip((lnth - lnth0_c) / dlnth_c, 0, nth - 1.000001); i0 = int(t); w = t - i0
    v = 0.0
    for ia, wa in ((i0, 1 - w), (i0 + 1, w)):
        u = r / rmax[ia] * (nr - 1)
        if u < nr - 1:
            j = int(u); f = u - j; v += wa * ((1 - f) * tab[ia, j] + f * tab[ia, j + 1])
    return v


rng = np.random.default_rng(2024)
dp, dr, refp, refr = [], [], [], []
for _ in range(4000):
    th = np.exp(rng.uniform(np.log(0.05), np.log(40)))
    k = rng.integers(I.K)
    # painting: r in [0, 5 th + 4 sigma_b]
    r = rng.uniform(0, 5 * th + 4 * I.SB)
    a = ev_claim(paint_c, rP_c, np.log(th), r); b = I.tab_eval(PT, PR, np.log(th), r)
    p0 = I.tab_eval(PT, PR, np.log(th), 0.0)
    dp.append((a - b) / p0)
    r = rng.uniform(0, 3 * th + 3 * I.THK[k] + 4 * I.SB)
    a = ev_claim(resp_c[k], rR_c, np.log(th), r); b = I.tab_eval(RT[k], RR, np.log(th), r)
    r0 = I.tab_eval(RT[k], RR, np.log(th), 0.0)
    dr.append((a - b) / r0)
dp, dr = np.abs(dp), np.abs(dr)
print(f"painting table: |claim - mine| / central value: median {np.median(dp):.2e}  99% {np.percentile(dp,99):.2e}  max {dp.max():.2e}")
print(f"response table: |claim - mine| / R(0):          median {np.median(dr):.2e}  99% {np.percentile(dr,99):.2e}  max {dr.max():.2e}")
print("claim sigma_k:", np.array2string(T["sigk_arcmin2"], precision=5))
print("mine  sigma_k:", np.array2string(I.templates()[1], precision=5))
