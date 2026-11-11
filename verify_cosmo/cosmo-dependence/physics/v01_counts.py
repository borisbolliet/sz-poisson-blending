"""Compare independent naive counts and mean y with the stored grid; log slopes of ytot and N_single."""
import time, numpy as np
from v_lib import counts
D = "/Users/boris/GitHub/sz-poisson-blending/data_cosmo"
G = dict(np.load(f"{D}/c02_grid.npz")); E = dict(np.load(f"{D}/c02_extra.npz"))
def gi(S, p): return int(np.argmin(np.abs(S["params"] - np.array(p)).sum(1)))
t = time.time()
for p in [(0.81, 0.31, 0.8, 0.17), (0.75, 0.31, 0.8, 0.17), (0.87, 0.31, 0.8, 0.17), (0.81, 0.35, 0.8, 0.17),
          (0.81, 0.31, 0.9, 0.17), (0.81, 0.31, 0.8, 0.3), (0.87, 0.35, 0.9, 0.3)]:
    N, y = counts(*p, want_y=True)
    j = gi(G, p); Ns = G["Nzq_n"][j]
    s = Ns > 10
    print(f"{p}: N_tot mine {N.sum():10.1f} stored {Ns.sum():10.1f} rel {N.sum()/Ns.sum()-1:+.2e}; "
          f"max|bin rel| (N>10) {np.abs(N[s]/Ns[s]-1).max():.2e};  ytot mine {y:.4e} stored {G['ytot'][j]:.4e} rel {y/G['ytot'][j]-1:+.2e}")
print(f"time per point {(time.time()-t)/7:.1f}s")
fid = [0.81, 0.31, 0.8, 0.17]; st = [0.01, 0.01, 0.02, 0.03]
print("log slopes (central FD, my code): param, dlnY/dlnp, dlnN_single/dlnp")
for a, nm in enumerate(["sigma8", "Om", "1-b", "slnY"]):
    pm = [list(fid), list(fid)]; pm[0][a] -= st[a]; pm[1][a] += st[a]
    (N0, y0), (N1, y1) = counts(*pm[0], want_y=True), counts(*pm[1], want_y=True)
    dl = np.log(pm[1][a] / pm[0][a])
    print(f"  {nm:7s} ytot {np.log(y1/y0)/dl:7.3f}   N_naive(q>5) {np.log(N1.sum()/N0.sum())/dl:7.3f}")
