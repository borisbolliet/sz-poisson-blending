"""Validate the fast cosmology-dependent pipeline (cosmolib) against the stored baseline b03 results
(data/blending/analytic.npz, read only) at the exact baseline: Planck18 (sigma8=0.8102, Om=0.3111),
(1-b)=0.8, sigma_lnY=0.173, neighbour floor 1e13, map mean removed, q_cut=5.
Also compares the fast P(D) grid (PD_FAST) with the b03 grid (PD_B03)."""
import os, sys, time
os.environ.setdefault("NUMBA_NUM_THREADS", "3"); os.environ.setdefault("OMP_NUM_THREADS", "3")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import cosmolib as cl

ref = np.load(os.path.join(cl.ROOT, "data", "blending", "analytic.npz"))
t0 = time.time()
r = cl.run_point(0.8102, 0.3111, 0.8, 0.173, verbose=True, keep_rates=True)
print(f"fast run {time.time()-t0:.1f}s")
print("neighbour counts per class (full sky): ", np.round(r["cnt_c"], 1),
      " b03 table: [24,697,396.7 4,201,703.4 396,386.9]")
print("ybar_c (sum Y500 / arcmin^2):", r["ybar_c"], " b03:", ref["ybar"])
tab = cl.rho_tab()[0]
print("\nper theta_A node: mu (this / b03), Var(x<500) (this / b03), Lambda(x>1) (this / b03)")
xe = np.concatenate([[0.0], np.logspace(-8, np.log10(500.0), 2601)])
for i in np.searchsorted(tab.th, [0.5, 1, 2, 4, 8, 16]):
    mu = r["mu_c"][i].sum(); mu_b = ref["mu_exact"][i].sum()
    v = r["var500_c"][i].sum(); v_b = ref["WX2"][i].sum()
    Wb = ref["W"][i].sum(0); WXb = ref["WX"][i].sum(0)
    with np.errstate(invalid="ignore", divide="ignore"):
        xb = np.where(Wb > 0, WXb / Wb, 0)
    l_b = Wb[xb >= 1].sum() + 0 * ref["CAP"][i].sum()   # b03 printed Lambda(x>1) incl. cap
    l_b += ref["CAP"][i].sum()
    print(f"  thA={tab.th[i]:6.2f}': mu {mu:.4f} / {mu_b:.4f} ({mu/mu_b-1:+.2e});  var {v:.4f} / {v_b:.4f} "
          f"({v/v_b-1:+.2e});  Lambda(x>1) {r['lam1'][i]:.4f} / {l_b:.4f} ({r['lam1'][i]/l_b-1:+.2e})")
print("max |mu/mu_formula - 1| over nodes:", np.abs(r["mu_c"].sum(1) / r["mu_formula"] - 1).max())
Hb = ref["H"]
print(f"\nprimary population: sum H this {r['H'].sum():,.1f} vs b03 {Hb.sum():,.1f}; max |dH|/max H = "
      f"{np.abs(r['H'].sum(0)-Hb).max()/Hb.max():.2e}")
b03 = dict(N_single=85898.2, N_det_b=84494, ratio=84494/85898.2, B2_10=0.1623, B2_20=0.10514, B3=0.08122, lost=0.09028)
print("\nfractions at q_cut=5 (floor 1e13, mean removed):  fast-PD | b03 log")
for k in ["N_single", "N_det_b", "ratio", "B2_10", "B2_20", "B3", "lost", "ybias"]:
    print(f"  {k:9s} {r[k]:12.5f} | {b03.get(k, float('nan')):12.5f}")
# same with the b03 P(D) grid settings, re-using rates
t0 = time.time()
Pbin2, P2 = cl.blend_probs(r["W"], r["WX"], r["WX2"], r["mut"], r["vart"], r["H"], cl.PD_B03)
r2 = cl.summarize(r["H"], Pbin2, P2)
print(f"\nb03 P(D) grid ({time.time()-t0:.1f}s): fast - b03grid differences")
for k in ["N_det_b", "ratio", "B2_10", "B2_20", "B3", "lost", "ybias"]:
    print(f"  {k:9s} {r2[k]:12.5f}  diff fast-b03grid {r[k]-r2[k]:+.2e}")
print("  max |dN_zq|/N_zq (blended, bins with N>10):",
      np.abs(r["Nzq_b"] - r2["Nzq_b"])[r2["Nzq_b"] > 10].max() / 1,
      np.max(np.abs(r["Nzq_b"] / r2["Nzq_b"] - 1)[r2["Nzq_b"] > 10]))
# b03 per-node Pdet with the stored H (independent check of the PD loop given b03's own rates)
Pd_b03 = ref["Pdet"][0, 1, 1]
print(f"\nN_det_b with this H and b03 Pdet: {(r['H'].sum(0)*Pd_b03).sum():,.1f}; this Pdet with b03 H: "
      f"{(Hb*r['Pdet']).sum():,.1f}")
print("Nzq blended (rows z bins, cols q bins):\n", np.round(r["Nzq_b"], 1))
print("Nzq naive:\n", np.round(r["Nzq_n"], 1))
np.savez(os.path.join(cl.OUT, "c01_baseline.npz"), **{k: v for k, v in r.items() if not isinstance(v, dict)})
