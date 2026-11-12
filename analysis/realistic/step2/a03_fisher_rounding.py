"""Arbiter / a03: does colossus' integer rounding of Delta_m in Tinker08 distort the Fisher errors and the biases of
configs c and e?  (catalogue verifier, bugs item 1: sigma(S8) of config e is 0.0321-0.0350 depending on the Om step.)

colossus.lss.mass_function.modelTinker08 sets Delta_m = round(500 rho_c(z) / rho_m(z)), so d2N/dz/dlnM has steps of
one unit in Delta_m (~1/1600) as Om changes, and dN/dOm depends on the finite-difference step.  The build's count model
(cosmolib.HMFGrid, from Q3) evaluates colossus at 21 Om nodes 0.0125 apart and splines across them.

Test: the build's count model with the HMF taken DIRECTLY from colossus at the needed (Om, s8) (no grid), with
  (R) colossus' own Tinker08 (rounded Delta_m), and
  (S) the same function with Delta_m NOT rounded (smooth in Om; identical otherwise),
Jacobian by central differences with Om steps h = 0.001-0.0125 (s8 step 0.002, other parameters the build's STEP2).
Fisher sigma(S8) for configs a-e (SO40; prior on (1-b) in d), and the linearised S8 bias F^-1 J^T (Delta/mu) / sigma of
the build's Delta (truth_<setup>.npz), compared with the build's grid-based values (Fitter3, cosmolib.HMFGrid).
Output: a03_fisher_rounding.log
"""
import os, sys, time
for v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "NUMBA_NUM_THREADS"):
    os.environ[v] = "4"
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import s2lib as S                                   # noqa: E402
from s2lib import CL, C2                             # noqa: E402
from colossus.cosmology import cosmology            # noqa: E402
from colossus.lss import mass_function as mfn       # noqa: E402
from colossus.halo import mass_so                   # noqa: E402

t0 = time.time()
HERE = os.path.dirname(os.path.abspath(__file__))
LOG = open(os.path.join(HERE, "a03_fisher_rounding.log"), "w")


def log(s=""):
    print(s, flush=True); LOG.write(s + "\n"); LOG.flush()


ORIG = mfn.modelTinker08


def tinker08_smooth(sigma, z, mdef):
    """colossus modelTinker08 (v. in the venv) with Delta_m NOT rounded to an integer; otherwise identical."""
    cosmo = cosmology.getCurrent()
    Delta_m = mass_so.densityThreshold(z, mdef) / cosmo.rho_m(z)
    fit_Delta = np.array([200, 300, 400, 600, 800, 1200, 1600, 2400, 3200])
    fit_A0 = np.array([0.186, 0.200, 0.212, 0.218, 0.248, 0.255, 0.260, 0.260, 0.260])
    fit_a0 = np.array([1.47, 1.52, 1.56, 1.61, 1.87, 2.13, 2.30, 2.53, 2.66])
    fit_b0 = np.array([2.57, 2.25, 2.05, 1.87, 1.59, 1.51, 1.46, 1.44, 1.41])
    fit_c0 = np.array([1.19, 1.27, 1.34, 1.45, 1.58, 1.80, 1.97, 2.24, 2.44])
    A0 = np.interp(Delta_m, fit_Delta, fit_A0); a0 = np.interp(Delta_m, fit_Delta, fit_a0)
    b0 = np.interp(Delta_m, fit_Delta, fit_b0); c0 = np.interp(Delta_m, fit_Delta, fit_c0)
    alpha = 10 ** (-(0.75 / np.log10(Delta_m / 75.0)) ** 1.2)
    A = A0 * (1.0 + z) ** -0.14; a = a0 * (1.0 + z) ** -0.06; b = b0 * (1.0 + z) ** -alpha; c = c0
    return A * ((sigma / b) ** -a + 1.0) * np.exp(-c / sigma ** 2)


C_KMS = 299792.458


class DirectHMF:
    """Drop-in for cosmolib.HMFGrid: ln d2N/dz/dlnM (full sky), ln E, ln D_A from colossus at the requested (Om, s8)
    on the grid's z and lnM nodes (exactly c01's recipe, without the Om grid and spline)."""

    def __init__(self, G, smooth):
        self.Z, self.LNM, self.rhoc0, self.s8fid = G.Z, G.LNM, G.rhoc0, G.s8fid
        self.OM, self.S8 = G.OM, G.S8
        self.smooth = smooth
        self.cache = {}

    def in_range(self, om, s8):
        return True

    def eval(self, om, s8):
        key = (round(float(om), 12), round(float(s8), 12))
        if key not in self.cache:
            mfn.models["tinker08"].func = tinker08_smooth if self.smooth else ORIG
            try:
                c = cosmology.setCosmology("planck18", Om0=float(om), sigma8=float(s8), persistence="")
                h = c.h
                M = np.exp(self.LNM)
                chi = c.comovingDistance(0.0, self.Z) / h
                Ez = c.Ez(self.Z)
                dV = (C_KMS / c.H0) / Ez * chi ** 2 * 4 * np.pi
                out = np.empty((len(self.Z), len(M)))
                for iz, zz in enumerate(self.Z):
                    out[iz] = mfn.massFunction(M * h, float(zz), mdef="500c", model="tinker08", q_out="dndlnM") * h ** 3 * dV[iz]
            finally:
                mfn.models["tinker08"].func = ORIG
            self.cache[key] = (np.log(out), np.log(Ez), np.log(chi / (1 + self.Z)))
        return self.cache[key]


G = CL.HMFGrid()
DR_ = DirectHMF(G, smooth=False)
DS_ = DirectHMF(G, smooth=True)
truth = S.FID.copy()
STEP2 = C2.STEP2


def model_for(run, grid):
    if run == "white":
        return C2.SmoothModel(qcut=5.0, grid=grid)
    return S.SmoothModelS(5.0, S.LnSigma(S.sz_model(run)), grid=grid)


def jac_fisher(M, hOm, fs=0.4):
    """mu (fs x full sky), J (nbin, 5) at the truth, central differences; Om step hOm, others STEP2."""
    mu = M.counts(truth, fs).ravel()
    J = np.zeros((mu.size, 5))
    for k in range(5):
        hk = hOm if k == 0 else STEP2[k]
        a, b = truth.copy(), truth.copy(); a[k] -= hk; b[k] += hk
        J[:, k] = (M.counts(b, fs).ravel() - M.counts(a, fs).ravel()) / (2 * hk)
    return mu, J


def sigS8_and_lin(mu, J, cn, D=None, fs=0.4):
    free, pri = S.config(cn, truth)
    fi = [CL.PNAMES.index(p) for p in free]
    Jf = J[:, fi]
    F = Jf.T @ (Jf / mu[:, None])
    for p, (m_, s_) in pri.items():
        k = free.index(p); F[k, k] += 1 / s_ ** 2
    sig, _, gS = S.derived_sigma3(F, fi, truth)
    lin = None
    if D is not None:
        x = np.linalg.solve(F, Jf.T @ ((D.ravel() * fs) / mu))
        lin = float(gS @ x / sig["S8"])
    return sig["S8"], lin


# ------------------------------------------------------------------ (0) direct (rounded) = grid at the fiducial
lnG, _, _ = G.eval(truth[0], truth[1]); lnD, _, _ = DR_.eval(truth[0], truth[1]); lnS, _, _ = DS_.eval(truth[0], truth[1])
w = np.exp(lnD) > 1e-2
log(f"[a03] (0) at the fiducial: grid spline vs direct colossus (rounded): max |dlnN| {np.abs(lnG-lnD)[w].max():.2e}; "
    f"smooth vs rounded Tinker08: max |dlnN| {np.abs(lnS-lnD)[w].max():.2e} (d2N > 1e-2)  [{time.time()-t0:.0f}s]")
for z in (0.01, 0.5, 1.0, 2.0):
    c = cosmology.setCosmology("planck18", Om0=0.3111, persistence="")
    dm = mass_so.densityThreshold(z, "500c") / c.rho_m(z)
    log(f"    z = {z}: Delta_m = {dm:.3f} (colossus uses {round(dm)}); dDelta_m/dOm0 ~ "
        f"{(mass_so.densityThreshold(z, '500c') / cosmology.setCosmology('planck18', Om0=0.3121, persistence='').rho_m(z) - dm) / 0.001:.0f} per unit Om")
cosmology.setCosmology("planck18", persistence="")

# ------------------------------------------------------------------ (1) Fisher sigma(S8), SO40
HS = (0.001, 0.002, 0.004, 0.008, 0.0125)
SETUPS = (("baseline", "baseline_f1"), ("goal", "goal_f1"), ("white", "white"))
res = {}
log("\n=== (1) sigma(S8), SO40, q_cut 5: build (grid, Fitter3) | direct rounded, Om step h | direct smooth, Om step h ===")
for run, setup in SETUPS:
    Mg = model_for(run, G)
    D = np.load(f"{S.D2}/truth_{setup}.npz")["D_q5"]
    N0 = Mg.counts(truth)
    for cn in ("a", "b", "c", "d", "e"):
        free, pri = S.config(cn, truth)
        fi = [CL.PNAMES.index(p) for p in free]
        F_ = S.Fitter3(Mg, N0 * 0.4, free, fsky=0.4, priors=pri, truth=truth)
        Ft = F_.fisher(truth[fi]); J = F_.jac(truth[fi])
        sig, _, gS = S.derived_sigma3(Ft, fi, truth)
        mu = F_.mu(truth[fi]).ravel()
        lin = float(gS @ np.linalg.solve(Ft, J.T @ ((D.ravel() * 0.4) / mu)) / sig["S8"])
        res[(run, cn, "grid")] = (sig["S8"], lin)
    for lab, DH in (("rounded", DR_), ("smooth", DS_)):
        Md = model_for(run, DH)
        for h in HS:
            mu, J = jac_fisher(Md, h)
            for cn in ("a", "b", "c", "d", "e"):
                res[(run, cn, lab, h)] = sigS8_and_lin(mu, J, cn, D)
        log(f"   [{run}] {lab} done [{time.time()-t0:.0f}s]")
    for cn in ("a", "b", "c", "d", "e"):
        g = res[(run, cn, "grid")]
        log(f"  [{run:8s}] {cn}: build {g[0]:.5f} | rounded " + " ".join(f"{res[(run, cn, 'rounded', h)][0]:.5f}" for h in HS)
            + " | smooth " + " ".join(f"{res[(run, cn, 'smooth', h)][0]:.5f}" for h in HS))
log(f"    (Om steps h = {HS})")

log("\n=== (2) linearised S8 bias of the build's Delta (SO40, q_cut 5), in Fisher sigma: build grid | rounded h | smooth h ===")
for run, setup in SETUPS:
    for cn in ("a", "b", "c", "d", "e"):
        g = res[(run, cn, "grid")]
        log(f"  [{setup:11s}] {cn}: build {g[1]:+.3f} | rounded " + " ".join(f"{res[(run, cn, 'rounded', h)][1]:+.3f}" for h in HS)
            + " | smooth " + " ".join(f"{res[(run, cn, 'smooth', h)][1]:+.3f}" for h in HS))

log("\n=== (3) summary: smooth (h = 0.002-0.008, mean) / build grid - 1 ===")
for run, setup in SETUPS:
    out = []
    for cn in ("a", "b", "c", "d", "e"):
        s_sm = np.mean([res[(run, cn, "smooth", h)][0] for h in (0.002, 0.004, 0.008)])
        spread = np.ptp([res[(run, cn, "smooth", h)][0] for h in HS]) / s_sm
        l_sm = np.mean([res[(run, cn, "smooth", h)][1] for h in (0.002, 0.004, 0.008)])
        out.append(f"{cn}: sigma {s_sm/res[(run, cn, 'grid')][0]-1:+.3f} (smooth step spread {spread:.3f}), lin bias "
                   f"{l_sm:+.3f} vs {res[(run, cn, 'grid')][1]:+.3f}")
    log(f"  [{setup}] " + "; ".join(out))
log(f"\n[a03] done {time.time()-t0:.0f}s")
