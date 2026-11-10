"""Why does the analysis var_c (x<500) at thA=1.10' (1.135 for M>1e14) exceed the exact (untruncated) Parseval
variance (1.110)?  Compare int 2 pi s rho^2 ds from the ANALYSIS rho table (trilinear interpolation, re-implemented
here) with the exact Parseval value, as a function of theta_B, then population-weight."""
import numpy as np
import v01_kernel as K
T = np.load("/Users/boris/GitHub/sz-poisson-blending/data/blending/rho_table.npz")
th, t, rho, RBt = T["th"], T["t"], T["rho"], float(T["R_BEAM"])
lth0, dl = np.log(th[0]), np.log(th[1] / th[0])
def tri(thA, thB, s):
    S = 5 * thA + 5 * thB + RBt
    fa = np.clip((np.log(thA) - lth0) / dl, 0, len(th) - 1.000001); fb = np.clip((np.log(thB) - lth0) / dl, 0, len(th) - 1.000001)
    ft = np.clip(np.sqrt(s / S) * (len(t) - 1), 0, len(t) - 1.000001)
    ia, ib, it = int(fa), int(fb), ft.astype(int); wa, wb, wt = fa - ia, fb - ib, ft - it
    r = 0
    for da, xa in ((0, 1 - wa), (1, wa)):
        for db, xb in ((0, 1 - wb), (1, wb)):
            r = r + xa * xb * ((1 - wt) * rho[ia + da, ib + db, it] + wt * rho[ia + da, ib + db, it + 1])
    return np.where(s < S, r, 0.0)
thA = th[np.argmin(abs(th - 1.1))]
sA = K.sigY(thA)[0]
k = np.linspace(0, 6.5 / K.SB, 60000); dk = k[1] - k[0]
base = k * np.exp(-2 * (k * K.SB) ** 2) * K.g(k * thA) ** 2
print(f"thA = {thA:.4f}' (analysis node)")
print(" thB     exact int rho^2 d2s   table int rho^2   ratio   | int rho: table/exact")
tbs = np.geomspace(0.26, 60, 41)
rat = []
for tb in tbs:
    ex = sA ** 4 * np.sum(base * K.g(k * tb) ** 2) * dk / (2 * np.pi) / K.NW ** 2
    S = 5 * thA + 5 * tb + RBt
    s = np.linspace(0, S, 20001)
    r = tri(thA, tb, s)
    tb2 = np.trapezoid(2 * np.pi * s * r * r, s)
    m1 = np.trapezoid(2 * np.pi * s * r, s) / (sA ** 2 * K.RATIO ** 2 / K.NW)
    rat.append(tb2 / ex)
    if np.isclose(np.log(tb) % 0.5, 0, atol=0.06) or tb in tbs[::5]:
        print(f" {tb:6.3f}  {ex:12.5e}  {tb2:12.5e}  {tb2/ex:.4f}  | {m1:.4f}")
rat = np.array(rat)
print(f"ratio table/exact of int rho^2: min {rat.min():.4f} max {rat.max():.4f} median {np.median(rat):.4f}")
