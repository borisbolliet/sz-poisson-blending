"""Step 2 / p01: inputs for the kernels (fiducial sky).

(1) sigma_Y at the 48 theta_A kernel nodes (MMF, s09 nodes) and the 40 white nodes; checks against step 1.
(2) Parseval confusion variances (s05 method: binned C_l^yy, MMFModel.confusion_var) of all neighbours M500 > 1e13 and of
    the sub-threshold ones (own q_B = Y_B / sigma_Y(theta_B) < 5), at the kernel nodes; check vs confusion_var.npz.
(3) Rate-histogram moments of the stored s09 rates; the extra Gaussian variance beyond the kernel support with the
    arbiter's fix (0 outside theta_A = 0.25-80').
(4) Map-sigma ratios r = sigma_eff / sigma_Y per node:
      (i)  r_i  = sqrt(1 + var_subthr)                    (Gaussian-equivalent sub-threshold confusion)
      (ii) r_ii = robust (84-16%)/2 half-width of n + delta - mu, untruncated P(D), all halos (as s06b table D).
(5) Low-pressure population ((1-b) = 0.6 for every halo): Y x c, c = 0.75^1.79.  Rates are the fiducial ones with
    x -> c x (exact: rho and sigma_Y are unchanged); Parseval variances; sub-threshold set c Y_B / sigma_Y < 5; r_i, r_ii.
(6) Poisson tSZ power D_3000 at 143 GHz of the fiducial and low-pressure populations, step 1's method (s10 part B):
    HMF route (binned C_l^yy of the population) and a direct sum over the halos of the stored sky 1 (read only).
Output: data_real/step2/inputs.npz
"""
import os, sys, time
NTH = "4"
for v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "NUMBA_NUM_THREADS"):
    os.environ[v] = NTH
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import s2lib as S                                  # noqa: E402
from s2lib import pl, bl, ins, sg                   # noqa: E402

t0 = time.time()
out = {}
cv = np.load(f"{S.DR}/confusion_var.npz")
pop = bl.neighbour_population(zsub=1, msub=6, ngh=16)
popf = S.neighbour_population_f(None)
same = all(np.array_equal(a, b) for a, b in zip(pop, popf))
print(f"(0) neighbour population: {len(pop[0]):,} points, N full sky = {pop[3].sum()*bl.ARCMIN2_FULLSKY:,.0f}; "
      f"s2lib copy with F = 1 identical to blendlib: {same}  [{time.time()-t0:.0f}s]", flush=True)
assert same
print(f"    EDGES = {np.round(S.EDGES, 4).tolist()}; equal to data_cosmo/kernel_catalogue.npz EDGES: "
      f"{np.allclose(S.EDGES, np.load(f'{S.ROOT}/data_cosmo/kernel_catalogue.npz')['EDGES'])}; QA grid {len(S.QA)} nodes, "
      f"equal to c03's: {np.allclose(S.QA, np.load(f'{S.ROOT}/data_cosmo/kernel_catalogue.npz')['QA'])}")

# ------------------------------------------------------------------ (1) sigma_Y
sy = np.load(f"{S.DR}/sigmaY_mmf.npz")
for run in ("baseline", "goal"):
    m = S.sz_model(run)
    th, sigA_s09, R = S.load_s09_rates(run)
    sA = m.sigma_Y_exact(th)
    spl = m.sigma_Y(sy["theta500"])
    print(f"(1) [{run}] sigma_Y exact at the 48 kernel nodes vs s09 sigA: max rel {np.abs(sA/sigA_s09-1).max():.1e}; "
          f"spline (SmoothModel) vs sigmaY_mmf.npz {run}_l100 (61 nodes 0.1-100'): max rel "
          f"{np.abs(spl/sy[run + '_l100']-1).max():.1e}", flush=True)
    out[f"{run}_th"] = th; out[f"{run}_sigA"] = sA

# ------------------------------------------------------------------ (2)-(4) baseline, goal
pd_un = pl.PDs(N=2 ** 20, dd=0.001, x_small=0.002, off=60.0)
thB, YB, cB, wB = pop
for run in ("baseline", "goal"):
    m = S.sz_model(run)
    th, _, R = S.load_s09_rates(run)
    sub = YB / m.sigma_Y(thB) < 5.0
    v_all, Cyy_all = S.parseval_var(m, th, pop)
    v_sub, Cyy_sub = S.parseval_var(m, th, pop, sub)
    # same computation at s05's 40 nodes: must reproduce confusion_var.npz
    va40 = m.confusion_var(cv["thA"], Cyy_all) / m.sigma_Y_exact(cv["thA"]) ** 2
    vs40 = m.confusion_var(cv["thA"], Cyy_sub) / m.sigma_Y_exact(cv["thA"]) ** 2
    print(f"(2) [{run}] Parseval at s05's 40 nodes vs confusion_var.npz: all max rel {np.abs(va40/cv[run+'_var_all']-1).max():.1e}, "
          f"sub-threshold {np.abs(vs40/cv[run+'_var_subthr']-1).max():.1e}  [{time.time()-t0:.0f}s]", flush=True)
    mu, var = S.moments(R)
    extra = S.fixed_extra(th, var, v_all)
    d09 = np.load(f"{S.DR}/pd_tables_{run}.npz")
    inner = (th >= 0.25) & (th <= 80.0)
    print(f"(3) [{run}] |mu_hist| max {np.abs(mu).max():.1e}; var_hist / Parseval - 1 over inner nodes: "
          f"{np.abs(var/v_all-1)[inner].max():.1e} (max); extra (fixed) max {extra.max():.2e}, vs s09's stored extra at inner "
          f"nodes max |diff| {np.abs(extra-d09['extra'])[inner].max():.1e}; s09's spurious outer values were "
          f"{np.round(d09['extra'][~inner], 4).tolist()}", flush=True)
    r_i = np.sqrt(1.0 + v_sub)
    r_ii = np.zeros(len(th)); QN = np.zeros((len(th), 5))
    for i in range(len(th)):
        r_ii[i], QN[i] = S.robust_halfwidth(R[i], mu[i], extra[i], pd_un)
    pf = np.load(f"{S.DR}/pd_fractions_{run}.npz")
    hw06 = 0.5 * (pf["QN_1e13"][:, 3] - pf["QN_1e13"][:, 1])
    print(f"(4) [{run}] r_i = sqrt(1 + var_subthr) at theta_A = 0.5, 1, 2, 4, 8, 16, 32': "
          + " ".join(f"{np.interp(np.log(s), np.log(th), r_i):.4f}" for s in (0.5, 1, 2, 4, 8, 16, 32))
          + f"; range over nodes {r_i.min():.4f}-{r_i.max():.4f}")
    print(f"    [{run}] r_ii = robust half-width at the same theta_A: "
          + " ".join(f"{np.interp(np.log(s), np.log(th), r_ii):.4f}" for s in (0.5, 1, 2, 4, 8, 16, 32))
          + f"; range {r_ii.min():.4f}-{r_ii.max():.4f}; vs s06b table D (pd_fractions QN_1e13): max |diff| inner nodes "
          f"{np.abs(r_ii-hw06)[inner].max():.1e}, outer nodes {np.abs(r_ii-hw06)[~inner].max():.1e} (s06b had the spurious extra)",
          flush=True)
    out.update({f"{run}_mu": mu, f"{run}_var": var, f"{run}_vpars_all": v_all, f"{run}_vpars_sub": v_sub,
                f"{run}_extra": extra, f"{run}_r_i": r_i, f"{run}_r_ii": r_ii, f"{run}_QN": QN})

# ------------------------------------------------------------------ white (old filter): moments only (extra = 0, s06b)
th_w, _, R_w = S.load_s09_rates("white")
mu_w, var_w = S.moments(R_w)
d09w = np.load(f"{S.DR}/pd_tables_white.npz")
print(f"(3) [white] 40 nodes {th_w[0]:.2f}-{th_w[-1]:.1f}'; mu_hist = s09 mu: max |diff| {np.abs(mu_w-d09w['mu']).max():.1e}; "
      f"mu at 1, 4, 16, 80': " + " ".join(f"{np.interp(np.log(s), np.log(th_w), mu_w):.3f}" for s in (1, 4, 16, 80)), flush=True)
th_p, _, R_p, mu_p = S.load_prefix_white_rates()
mup_h, _ = S.moments(R_p)
v500 = lambda R_: np.array([o["WX2p"].sum() + o["WX2n"].sum() for o in R_])        # variance from |x| < 500
cap = lambda R_: np.array([o["capp"][:, 0].sum() for o in R_])                       # rate of x >= 500
print(f"    [white, pre-fix b03 rates (data/blending/analytic.npz)] mu_exact vs histogram moments: max |diff| "
      f"{np.abs(mu_p-mup_h).max():.1e}; at theta_A = 4, 16, 40, 80': mu (pre-fix) / mu (fixed) - 1 = "
      + " ".join(f"{np.interp(np.log(s), np.log(th_w), mu_p/mu_w-1):+.2e}" for s in (4, 16, 40, 80))
      + "; var(|x|<500) pre-fix / fixed - 1 = "
      + " ".join(f"{np.interp(np.log(s), np.log(th_w), v500(R_p)/v500(R_w)-1):+.2e}" for s in (4, 16, 40, 80))
      + "; rate(x>=500) pre-fix / fixed = "
      + " ".join(f"{np.interp(np.log(s), np.log(th_w), cap(R_p)/np.maximum(cap(R_w), 1e-300)):.3f}" for s in (4, 16, 40, 80)),
      flush=True)
out.update(white_th=th_w, white_mu=mu_w, white_var=var_w, white_prefix_mu=mu_p)

# ------------------------------------------------------------------ (5) low-pressure population
c = S.C_LOW
m = S.sz_model("baseline")
th, _, R = S.load_s09_rates("baseline")
RL = [S.scale_rates(o, c) for o in R]
muL, varL = S.moments(RL)
popL = S.neighbour_population_f(None, Yscale=c)
subL = popL[1] / m.sigma_Y(popL[0]) < 5.0
vL_all, CyyL_all = S.parseval_var(m, th, popL)
vL_sub, CyyL_sub = S.parseval_var(m, th, popL, subL)
print(f"\n(5) low-pressure population: (1-b) = {S.OMB_LOW}, Y x c = (0.6/0.8)^1.79 = {c:.5f}, c^2 = {c*c:.5f}")
print(f"    Parseval (all) low / fiducial at the nodes: {np.min(vL_all/out['baseline_vpars_all']):.6f}-"
      f"{np.max(vL_all/out['baseline_vpars_all']):.6f} (exact c^2 expected); scaled-rate variance / c^2 / fiducial - 1: "
      f"{np.abs(varL/c**2/out['baseline_var']-1).max():.1e}")
print(f"    sub-threshold set (own q_B < 5): fiducial {wB[YB / m.sigma_Y(thB) < 5].sum()/wB.sum():.4f} of all neighbours, "
      f"low-pressure {popL[3][subL].sum()/popL[3].sum():.4f}; sub-threshold variance low / fiducial at 0.5, 1, 2, 4, 8': "
      + " ".join(f"{np.interp(np.log(s), np.log(th), vL_sub/out['baseline_vpars_sub']):.3f}" for s in (0.5, 1, 2, 4, 8)), flush=True)
extraL = S.fixed_extra(th, varL, vL_all)
r_iL = np.sqrt(1.0 + vL_sub)
r_iiL = np.zeros(len(th)); QNL = np.zeros((len(th), 5))
for i in range(len(th)):
    r_iiL[i], QNL[i] = S.robust_halfwidth(RL[i], muL[i], extraL[i], pd_un)
print(f"    var_all / sigma_Y^2 at 0.5, 1, 2, 4, 8, 16': fiducial "
      + " ".join(f"{np.interp(np.log(s), np.log(th), out['baseline_vpars_all']):.3f}" for s in (0.5, 1, 2, 4, 8, 16))
      + "; low-pressure " + " ".join(f"{np.interp(np.log(s), np.log(th), vL_all):.3f}" for s in (0.5, 1, 2, 4, 8, 16)))
print(f"    var_subthr: fiducial " + " ".join(f"{np.interp(np.log(s), np.log(th), out['baseline_vpars_sub']):.3f}" for s in (0.5, 1, 2, 4, 8, 16))
      + "; low-pressure " + " ".join(f"{np.interp(np.log(s), np.log(th), vL_sub):.3f}" for s in (0.5, 1, 2, 4, 8, 16)))
print(f"    r_i (low) " + " ".join(f"{np.interp(np.log(s), np.log(th), r_iL):.4f}" for s in (0.5, 1, 2, 4, 8, 16))
      + f"; r_ii (low) " + " ".join(f"{np.interp(np.log(s), np.log(th), r_iiL):.4f}" for s in (0.5, 1, 2, 4, 8, 16)), flush=True)
out.update(lowP_mu=muL, lowP_var=varL, lowP_vpars_all=vL_all, lowP_vpars_sub=vL_sub, lowP_extra=extraL, lowP_r_i=r_iL,
           lowP_r_ii=r_iiL, lowP_QN=QNL, C_LOW=c)

# ------------------------------------------------------------------ (6) tSZ power D_3000 at 143 GHz
L = 3000.0
D_hmf = {}
for lab, P in (("fiducial (1-b)=0.8", pop), ("low-pressure (1-b)=0.6", popL)):
    Cl = ins.poisson_Cyy(np.array([L]), P[0], P[1], P[3])[0]
    D_hmf[lab] = L * (L + 1) * Cl / (2 * np.pi)
cz = np.load(f"{S.ROOT}/data/catalogs/cat_M1e13_seed1.npz")
Ycat = cz["Y500"]; Tcat = cz["theta500"]
Ssum = 0.0
for a in range(0, len(Ycat), 4_000_000):
    Y = Ycat[a:a + 4_000_000].astype(np.float64) * ins.ARCMIN ** 2
    g = sg.template_hankel(L * Tcat[a:a + 4_000_000].astype(np.float64) * ins.ARCMIN)
    Ssum += np.sum(Y * Y * g * g)
D_dir = L * (L + 1) * Ssum / (4 * np.pi) / (2 * np.pi)
f2 = ins.tsz_f_uK(143.0) ** 2
print(f"\n(6) Poisson tSZ power at l = 3000, 143 GHz (SPT: 3.42 +- 0.54 uK^2, Reichardt et al. 2021):")
for lab, Dyy in D_hmf.items():
    sc = 1.0 if "0.8" in lab else c * c
    print(f"    {lab:24s}: 1e12 l(l+1)C_l/2pi = {1e12*Dyy:.4f} (HMF) / {1e12*D_dir*sc:.4f} (stored sky 1, {len(Ycat):,} halos, Y x {sc**0.5:.4f}); "
          f"D_3000(143) = {Dyy*f2:.2f} / {D_dir*sc*f2:.2f} uK^2 = {Dyy*f2/3.42:.2f} / {D_dir*sc*f2/3.42:.2f} x SPT "
          f"({(Dyy*f2-3.42)/0.54:+.1f} / {(D_dir*sc*f2-3.42)/0.54:+.1f} sigma)", flush=True)
out.update(D3000_hmf_fid=D_hmf["fiducial (1-b)=0.8"] * f2, D3000_hmf_low=D_hmf["low-pressure (1-b)=0.6"] * f2,
           D3000_sky_fid=D_dir * f2, D3000_sky_low=D_dir * c * c * f2)
np.savez(f"{S.D2}/inputs.npz", **out)
print(f"saved {S.D2}/inputs.npz; total {time.time()-t0:.0f}s")
