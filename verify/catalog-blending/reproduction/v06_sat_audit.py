"""V06: audit the analysis' dominance flags (xmax) for satellites on stored sky 1: for primaries flagged
as satellites at q_cut=7 (q_obs>7, xmax>=q), find the dominating neighbour(s) by brute force over all halos
with q_B >= q_A and recompute x_B with my independent exact rho; also audit a random sample of
non-satellites with large xmax/q."""
import os, sys
os.environ["OMP_NUM_THREADS"] = "2"; os.environ["NUMBA_NUM_THREADS"] = "2"
import numpy as np
import indep_model as im
sys.path.insert(0, "/Users/boris/GitHub/sz-poisson-blending/analysis/blending")
import blendlib as bl
from szsim import sampler, signal as sg

m = sg.SZModel(); tab = bl.load_rho_table()
d = np.load("/Users/boris/GitHub/sz-poisson-blending/data/blending/mc_sky1.npz")
cat = sampler.load_catalog("/Users/boris/GitHub/sz-poisson-blending/data/catalogs/cat_M1e13_seed1.npz",
                           fields=["ra", "dec", "Y500", "theta500", "q", "M500"])
q = d["p_q"].astype(float); qo = d["p_q_obs"].astype(float); th = d["p_theta500"].astype(float)
sig = m.sigma_Y(th); xm = (d["xmax"] / sig[:, None]).max(1)
idx = d["p_idx"]
for qc in (5.0, 7.0):
    single = qo > qc; sat = single & (xm >= q)
    print(f"sky1 q_cut={qc}: single {single.sum()}, sat {sat.sum()} ({sat.sum()/single.sum():.5f})")
big = cat["q"] > 0.5
B = {k: cat[k][big].astype(float) for k in ("ra", "dec", "Y500", "theta500", "q", "M500")}
Bidx = np.nonzero(big)[0]
del cat
vB = np.column_stack([np.cos(np.radians(B["dec"])) * np.cos(np.radians(B["ra"])), np.cos(np.radians(B["dec"])) * np.sin(np.radians(B["ra"])), np.sin(np.radians(B["dec"]))])
sat_idx = np.nonzero((qo > 7.0) & (xm >= q))[0]
rng = np.random.default_rng(5)
print(f"auditing {min(60, len(sat_idx))} of {len(sat_idx)} satellites at q_cut=7")
agree = 0; rows = []
for a in rng.choice(sat_idx, min(60, len(sat_idx)), replace=False):
    ra, dec = np.radians(float(d["p_ra"][a])), np.radians(float(d["p_dec"][a]))
    vA = np.array([np.cos(dec) * np.cos(ra), np.cos(dec) * np.sin(ra), np.sin(dec)])
    cand = np.nonzero((B["q"] >= q[a]) & (Bidx != idx[a]))[0]
    s = 2 * np.arcsin(np.clip(np.linalg.norm(vB[cand] - vA, axis=1) / 2, 0, 1)) / im.ARCMIN
    ok = s < 5 * th[a] + 5 * B["theta500"][cand] + 6
    c = cand[ok]; s = s[ok]
    xs_mine = np.array([im.rho(th[a], B["theta500"][j], [ss])[0] * B["Y500"][j] / im.sigmaY_fast(th[a]) for j, ss in zip(c, s)]) if len(c) else np.zeros(0)
    xs_tab = np.array([bl.rho_interp_py(tab, th[a], B["theta500"][j], ss) * B["Y500"][j] / sig[a] for j, ss in zip(c, s)]) if len(c) else np.zeros(0)
    k = np.argmax(xs_tab) if len(c) else -1
    rows.append((q[a], th[a], xm[a], xs_tab.max() if len(c) else 0, xs_mine.max() if len(c) else 0,
                 B["theta500"][c[k]] if k >= 0 else 0, s[k] if k >= 0 else 0, B["q"][c[k]] if k >= 0 else 0))
rows = np.array(rows)
print(" q_A  thA  xmax(analysis)  xmax(table, brute)  xmax(my exact rho)  thB  s  q_B")
for r in rows[:25]:
    print("  " + "  ".join(f"{v:8.3f}" for v in r))
print(f"fraction with my exact x_max >= q_A: {(rows[:,4] >= rows[:,0]).mean():.3f}; with table brute: {(rows[:,3] >= rows[:,0]).mean():.3f}")
