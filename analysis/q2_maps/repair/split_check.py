"""REPAIR: are the patch-bootstrap errors realistic, and do S1 (original 108 patches, skies 1-3) and S2 (216 new
patches, skies 4-9) agree?  Uses the per-patch metric sums saved by analysis_blend.py (PIDS=0:324, TAG=_ALL).
For each disputed metric: S1 and S2 ratios with bootstrap errors, S1-S2 in sigma, and the chi2 of the 9 per-sky ratios
around the pooled value using per-sky bootstrap errors (tests whether bootstrap errors cover sky-to-sky scatter)."""
import os, sys
import numpy as np
from scipy.stats import chi2 as chi2d
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ.setdefault("MPLBACKEND", "Agg")
METRICS = ["n_det", "prim_1", "prim_eq_near", "cont10", "cont20", "gross10", "gross20", "obs10", "obs20", "neg10",
           "nc10", "multi1", "multi2", "multi3", "multiP2", "multiP2_den", "boost", "dup", "nosig_noise", "nosig_conf",
           "boost_real", "clean", "n_real", "cont10r", "cont20r", "neg10r", "nc10r", "boostr", "spur1", "faint12",
           "mP1", "mP1d", "mP3", "mP3d", "faded_lowpos", "single_expect14",
           "h_detectable", "h_detected", "h_merged", "h_unres", "h_faded", "h_gain",
           "cat_cont10", "cat_cont20", "cat_den", "cat_boost", "cat_pos_det", "single_expect"]
mi = {m: i for i, m in enumerate(METRICS)}
ACC = np.load("/Users/boris/GitHub/sz-poisson-blending/data/q2_maps/blend_acc_ALL.npy")   # (VARS x RULES, 324, 4, nmet)
VR = [(v, r) for v in ["V13", "V3e13", "V14"] for r in ["R1", "R2", "R3"]]
rng = np.random.default_rng(7)
def rat(A, num, den, iq, P):
    a = A[P, iq, mi[num]]; b = A[P, iq, mi[den]]
    B = rng.integers(0, len(P), (2000, len(P)))
    return a.sum() / b.sum(), (a[B].sum(1) / b[B].sum(1)).std(ddof=1)
S1, S2 = np.arange(108), np.arange(108, 324)
tests = [("V13", "R2", "cont10r", "n_real"), ("V13", "R2", "cont20r", "n_real"), ("V13", "R2", "neg10r", "n_real"),
         ("V13", "R2", "nc10r", "n_real"), ("V13", "R2", "boost", "n_det"), ("V13", "R2", "boostr", "n_real"),
         ("V13", "R2", "h_unres", "h_detectable"), ("V13", "R2", "h_faded", "h_detectable"), ("V13", "R2", "h_detected", "h_detectable"),
         ("V13", "R2", "multiP2", "multiP2_den"), ("V13", "R2", "mP1", "mP1d"), ("V13", "R2", "mP3", "mP3d"), ("V13", "R2", "dup", "n_det"),
         ("V14", "R2", "cont10r", "n_real"), ("V3e13", "R2", "cont10r", "n_real")]
allz = []; chis = []; dofs = 0
for v, r, num, den in tests:
    A = ACC[VR.index((v, r))]
    line = f"{v:5s} {r} {num:11s}/{den:12s}"
    for iq, qc in enumerate([4, 5, 6, 7]):
        a1, e1 = rat(A, num, den, iq, S1); a2, e2 = rat(A, num, den, iq, S2); a, e = rat(A, num, den, iq, np.arange(324))
        z = (a1 - a2) / np.hypot(e1, e2); allz.append(z)
        vs = [rat(A, num, den, iq, np.arange(s * 36, (s + 1) * 36)) for s in range(9)]
        c2 = sum(((x - a) / ex) ** 2 for x, ex in vs if ex > 0); chis.append(c2); dofs += 8
        line += f" | q>{qc}: S1 {100*a1:.2f}+-{100*e1:.2f} S2 {100*a2:.2f}+-{100*e2:.2f} ({z:+.1f}s) ALL {100*a:.2f}+-{100*e:.2f} sky-chi2 {c2:.1f}/8"
    print(line)
allz = np.array(allz)
print(f"\nS1-S2 z-scores over {allz.size} (metric, q_cut) pairs: mean {allz.mean():+.2f}, rms {np.sqrt((allz**2).mean()):.2f};"
      f" |z|>2: {(np.abs(allz)>2).sum()}, |z|>3: {(np.abs(allz)>3).sum()} (expected for N(0,1): {allz.size*0.0455:.1f}, {allz.size*0.0027:.2f})")
tot = sum(chis)
print(f"per-sky chi2 summed over all entries: {tot:.1f} for {dofs} dof (chi2/dof {tot/dofs:.2f}); entries are correlated, so this is indicative")
