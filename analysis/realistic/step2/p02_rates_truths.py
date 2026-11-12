"""Step 2 / p02: signed rate histograms of the neighbour sum for two truths off the fiducial, S8 = 0.75 and 0.87 at
fixed Om = 0.3111 (sigma8 adjusted), SO baseline MMF (step-1 rho table, 48 theta_A nodes, as s09).

The neighbour population (blendlib.neighbour_population grid: zsub=1, msub=6, ngh=16, M500 > 1e13) keeps its
(theta_B, Y_B) points (fixed Om: same D_A, E(z), R500; same Y-M relation); only the abundances change:
   d2N(sigma8) = d2N(fid) x f_T08(s sigma(M,z); z) / f_T08(sigma(M,z); z),  s = sigma8 / sigma8_fid   (as c01).
Checks: (a) the factor reproduces direct colossus mass functions at the new sigma8; (b) the rate code reproduces the
stored s09 fiducial rates at two nodes; (c) Parseval vs histogram variance for the new populations.
Output: data_real/step2/rates_S8_{0.75,0.87}.npz
"""
import os, sys, time
NTH = os.environ.get("P02_THREADS", "6")
for v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ[v] = "1"
os.environ["NUMBA_NUM_THREADS"] = NTH
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import s2lib as S                                  # noqa: E402
from s2lib import pl, bl, ch                        # noqa: E402

TRUTHS = [0.75, 0.87]


def main():
    t0 = time.time()
    m = S.sz_model("baseline")
    tab = bl.RhoTable(np.load(f"{S.DR}/rho_mmf_baseline.npz"))
    th, sigA_s09, R09 = S.load_s09_rates("baseline")
    assert np.allclose(th, tab.th[::2])
    sigA = m.sigma_Y(th)
    z, lnM = S.population_grid()
    # ---- (a) HMF factor vs direct colossus
    from colossus.cosmology import cosmology
    from colossus.lss import mass_function as mf
    c0 = ch.get_cosmology()
    for S8 in TRUTHS:
        s8 = S.s8_from_S8(S8)
        Fz = S.hmf_factor_s8(s8, np.array([0.1, 0.5, 1.0, 2.0]), lnM)
        cn = cosmology.setCosmology("planck18", sigma8=float(s8), persistence="")
        M = np.exp(lnM)
        worst = 0.0
        for k, zz in enumerate((0.1, 0.5, 1.0, 2.0)):
            a = mf.massFunction(M * cn.h, zz, mdef="500c", model="tinker08", q_out="dndlnM")
            cosmology.setCurrent(c0)
            b = mf.massFunction(M * c0.h, zz, mdef="500c", model="tinker08", q_out="dndlnM")
            cosmology.setCurrent(cn)
            sel = M < 5e15
            worst = max(worst, np.abs((a / b)[sel] / Fz[k][sel] - 1).max())
        cosmology.setCurrent(c0)
        print(f"(a) S8 = {S8}: sigma8 = {s8:.5f}; factor vs direct colossus ratio at z = 0.1-2, M < 5e15: max rel diff "
              f"{worst:.1e}", flush=True)
    assert ch.get_cosmology().sigma8 == c0.sigma8
    # ---- (b) rate code vs stored s09 fiducial rates
    pop = bl.neighbour_population(zsub=1, msub=6, ngh=16)
    for i in (12, 30):
        o = pl.rate_signed(th[i], sigA[i], pop, tab)
        rel = max(np.abs(o[k] - R09[i][k]).max() / max(np.abs(R09[i][k]).max(), 1e-300) for k in S.RKEYS)
        print(f"(b) fiducial node {i} ({th[i]:.2f}'): recomputed rates vs s09 stored: max |diff| / max |value| = {rel:.1e}  "
              f"[{time.time()-t0:.0f}s]", flush=True)
    del pop
    # ---- truths
    for S8 in TRUTHS:
        s8 = S.s8_from_S8(S8)
        F = S.hmf_factor_s8(s8, z, lnM)
        popt = S.neighbour_population_f(F)
        Nt = popt[3].sum() * bl.ARCMIN2_FULLSKY
        print(f"\n=== S8 = {S8} (sigma8 = {s8:.5f}, Om = {S.FID[0]}): neighbours M500 > 1e13 full sky {Nt:,.0f}; factor at "
              f"M = 1e13 / 1e14 / 1e15, z = 0.5: {np.interp(0.5, z, F[:, 0]):.4f} / "
              f"{np.interp(0.5, z, F[:, np.argmin(np.abs(lnM-np.log(1e14)))]):.4f} / "
              f"{np.interp(0.5, z, F[:, np.argmin(np.abs(lnM-np.log(1e15)))]):.4f}", flush=True)
        R = []
        for i in range(len(th)):
            R.append(pl.rate_signed(th[i], sigA[i], popt, tab))
            if i % 12 == 0:
                print(f"   rate node {i}/{len(th)} {time.time()-t0:.0f}s", flush=True)
        mu, var = S.moments(R)
        vpars, _ = S.parseval_var(m, th, popt)
        extra = S.fixed_extra(th, var, vpars)
        inner = (th >= 0.25) & (th <= 80)
        print(f"   |mu_hist| max {np.abs(mu).max():.1e}; var_hist / Parseval - 1 inner nodes max {np.abs(var/vpars-1)[inner].max():.1e}; "
              f"extra max {extra.max():.1e}; var/sigma^2 at 1, 4, 16': "
              + " ".join(f"{np.interp(np.log(s), np.log(th), vpars):.4f}" for s in (1, 4, 16)), flush=True)
        np.savez(f"{S.D2}/rates_S8_{S8}.npz", th=th, sigA=sigA, S8=S8, s8=s8, N_neigh=Nt, mu=mu, var=var, vpars_all=vpars,
                 extra=extra, **{f"R_{k}": np.array([o[k] for o in R]) for k in S.RKEYS})
        print(f"   saved rates_S8_{S8}.npz  [{time.time()-t0:.0f}s]", flush=True)
        del popt
    print(f"total {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
