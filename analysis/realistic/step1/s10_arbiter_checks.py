"""Step 1 / s10 (arbiter): targeted tests for the claims that the two verification reports marked partial or disputed
(light computations; the P(D)/z-tilt items are in s09 and s11).

(A) White-filter reproduction (V1/V2 'partially'): sigma_Y of the old white setup from an essentially exact quadrature
    of the same template interpolant (Gauss-Legendre on every interval between template-table nodes, mapped to l),
    against SZModel.sigma_Y_exact (trapezoid, 20,000 l points) and MMFModel(Instrument.white_y()) (trapezoid on the
    200,001-point white_y grid).  Which code is closer to the exact integral?
(B) Poisson tSZ power of the halo population (V1: '2.6x at the same 143 GHz, not 2.5x'): D_3000 at 143, 145 and
    150 GHz by two routes: the Parseval C_l^yy of s05 (HMF integral, data_real/confusion_var.npz) and a direct sum
    over the halos of the stored full sky data/catalogs/cat_M1e13_seed1.npz (read only).
(C) Lensed-CMB model uncertainty (V1: CLASS differs from the stored CAMB by 5% at l = 6000 and 14-32% at
    l = 8000-20000): (i) CAMB 2.0.5 with the SAME accuracy settings as s01 run B but halofit_version = 'takahashi'
    instead of CAMB's default 'mead2020' (HMcode-2020, used by s01); (ii) the CLASS halofit spectrum of verifier 1
    (verify_real/step1_derivation/cmb_lensed_TT_class.npz, read only); (iii) effect of each spectrum, and of the CMB
    set to zero above l = 6000 or scaled by 1.3 above l = 4000, on the MMF sigma_Y (baseline, goal).
(D) Provenance of data/blending/analytic.npz (claim b03_npz_prefix, checked by neither verifier): (i) the variance
    (x < 500) and the x >= 500 cap rate stored by b03 are recomputed with a PRE-FIX kernel re-implemented here
    (trilinear lookup with theta_B clamped to the last table node and the support S with the true theta_B, no
    rescaling), through an independent numba rate loop (not pdlib1.rate_signed, not the Y_B/amp substitution of s06e);
    (ii) file timeline: which job wrote which keys.
"""
import os, sys, time, json
for v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ[v] = os.environ.get("S10_THREADS", "1")
os.environ["NUMBA_NUM_THREADS"] = os.environ.get("S10_THREADS", "1")
import numpy as np
import numba as nb

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import pdlib1 as pl                                   # noqa: E402
import blendlib as bl                                 # noqa: E402
from szsim import instrument as ins, signal as sg, sampler   # noqa: E402

ROOT = pl.ROOT
AR = ins.ARCMIN


# ------------------------------------------------------------------------------------------------ (A)
def sigma_white_exact(theta_arcmin, ngl=24):
    """sigma_Y [arcmin^2] of the white filter: (1/2pi) int dl l B^2 g(l theta)^2 / N_w, integrated exactly
    (Gauss-Legendre of order ngl on each interval between consecutive template nodes, each split in 4)."""
    m = sg.SZModel()
    sb, Nw = m.sigma_beam_rad, m.N_white_sr
    ug, _ = sg.template_hankel()
    x, w = np.polynomial.legendre.leggauss(ngl)
    out = []
    for th in np.atleast_1d(theta_arcmin):
        t = th * AR
        lm = min(9.0 / sb, ug[-1] / t)                              # same upper limit as SZModel.sigma_Y_exact
        nodes = ug / t
        nodes = np.unique(np.concatenate([nodes[nodes < lm], [lm]]))
        fine = np.concatenate([np.linspace(a, b, 5)[:-1] for a, b in zip(nodes[:-1], nodes[1:])] + [[lm]])
        a, b = fine[:-1], fine[1:]
        L = 0.5 * (b - a)[:, None] * x[None, :] + 0.5 * (a + b)[:, None]
        W = 0.5 * (b - a)[:, None] * w[None, :]
        g = sg.template_hankel(L * t)
        I = np.sum(W * L * np.exp(-(L * sb) ** 2) * g * g) / (2 * np.pi) / Nw
        out.append(1.0 / np.sqrt(I) / AR ** 2)
    return np.array(out)


def part_A():
    print("=== (A) white-filter sigma_Y: exact quadrature vs SZModel vs MMF code (white_y) ===")
    th = np.array([0.1, 0.2, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0, 50.0, 100.0])
    ex = sigma_white_exact(th)
    ex2 = sigma_white_exact(th, ngl=40)
    old = sg.SZModel().sigma_Y_exact(th)
    mmf = ins.MMFModel(ins.Instrument.white_y()).sigma_Y_exact(th)
    print("   theta500 [']        " + " ".join(f"{t:9.2f}" for t in th))
    print("   exact (GL 24)       " + " ".join(f"{v:9.4e}" for v in ex))
    print("   GL 40 / GL 24 - 1   " + " ".join(f"{v:+9.1e}" for v in ex2 / ex - 1))
    print("   SZModel / exact - 1 " + " ".join(f"{v:+9.1e}" for v in old / ex - 1))
    print("   MMF(white)/exact - 1" + " ".join(f"{v:+9.1e}" for v in mmf / ex - 1))
    print(f"   max |SZModel/exact - 1| = {np.abs(old/ex-1).max():.1e};  max |MMF(white)/exact - 1| = {np.abs(mmf/ex-1).max():.1e};"
          f"  max |MMF/SZModel - 1| = {np.abs(mmf/old-1).max():.1e}", flush=True)


# ------------------------------------------------------------------------------------------------ (B)
def part_B():
    print("\n=== (B) Poisson tSZ power of the population at l = 3000 ===")
    cv = np.load(f"{ROOT}/data_real/confusion_var.npz")
    ell, C = cv["baseline_ell"], cv["baseline_Cyy_all"]          # instrument-independent population C_l (s05)
    L = 3000.0
    Dyy_pars = L * (L + 1) * np.interp(L, ell, C) / (2 * np.pi)
    c = sampler.load_catalog(f"{ROOT}/data/catalogs/cat_M1e13_seed1.npz", fields=["Y500", "theta500"])
    S = 0.0
    for a in range(0, len(c["Y500"]), 4_000_000):
        Y = c["Y500"][a:a + 4_000_000].astype(np.float64) * AR ** 2          # sr
        g = sg.template_hankel(L * c["theta500"][a:a + 4_000_000].astype(np.float64) * AR)
        S += np.sum(Y * Y * g * g)
    Cdir = S / (4 * np.pi)
    Dyy_dir = L * (L + 1) * Cdir / (2 * np.pi)
    print(f"   1e12 l(l+1)C_l^yy/2pi at l=3000: Parseval/HMF (s05) {1e12*Dyy_pars:.4f}; direct sum over stored sky 1 "
          f"({len(c['Y500']):,} halos) {1e12*Dyy_dir:.4f} (ratio {Dyy_dir/Dyy_pars:.4f})")
    for nu in (143.0, 145.0, 150.0):
        f2 = ins.tsz_f_uK(nu) ** 2
        print(f"   {nu:5.1f} GHz: D_3000 = {Dyy_pars*f2:.2f} uK^2 (HMF) / {Dyy_dir*f2:.2f} uK^2 (sky 1); "
              f"ratio to SPT 3.42 +- 0.54 uK^2 (143 GHz, Reichardt+2021): {Dyy_pars*f2/3.42:.2f} / {Dyy_dir*f2/3.42:.2f}")
    print("   (the SPT value is quoted at 143 GHz, so the like-for-like ratio is the 143 GHz row)", flush=True)


# ------------------------------------------------------------------------------------------------ (C)
def camb_takahashi():
    import camb
    from colossus.cosmology import cosmology
    meta = json.loads(str(np.load(ins.CMB_FILE)["meta"]))
    P18 = cosmology.cosmologies["planck18"]
    pars = camb.CAMBparams()
    pars.set_cosmology(H0=P18["H0"], ombh2=meta["ombh2"], omch2=meta["omch2"], mnu=0.06, omk=0, tau=0.0544, TCMB=2.7255)
    pars.InitPower.set_params(As=meta["As"], ns=P18["ns"])
    pars.set_matter_power(redshifts=[0.0], kmax=2.0)
    pars.set_for_lmax(26000, lens_potential_accuracy=8)
    pars.set_accuracy(AccuracyBoost=1.5, lAccuracyBoost=1.5)
    pars.NonLinearModel.set_params(halofit_version="takahashi")
    res = camb.get_results(pars)
    cls = res.get_cmb_power_spectra(pars, CMB_unit="muK", raw_cl=True, lmax=26000)["lensed_scalar"][:, 0]
    return np.arange(len(cls)), cls, pars.NonLinearModel.halofit_version, res.get_sigma8_0()


def part_C():
    print("\n=== (C) lensed CMB: non-linear lensing model spread and its effect on the MMF sigma_Y ===", flush=True)
    t0 = time.time()
    d = np.load(ins.CMB_FILE)
    lc, cc = d["ell"].astype(float), d["cl_uK2"]
    lt, ct, hv, s8 = camb_takahashi()
    np.savez(f"{ROOT}/data_real/cmb_lensed_TT_camb_takahashi.npz", ell=lt, cl_uK2=ct,
             meta=json.dumps(dict(halofit_version=hv, sigma8=s8, note="s10 (C): as s01 run B but halofit takahashi")))
    k = np.load(f"{ROOT}/verify_real/step1_derivation/cmb_lensed_TT_class.npz")
    lk, ck, ltrust = k["ell"].astype(float), k["cl_uK2"], float(k["l_trust"])
    print(f"   CAMB takahashi run {time.time()-t0:.0f}s (halofit_version {hv}, sigma8 {s8:.5f}); CLASS (V1) trusted to l = {ltrust:.0f}")
    print("        l    CAMB(mead2020, s01)  takahashi/mead-1  CLASS(halofit)/mead-1  CLASS/takahashi-1")
    for L in (1000, 2000, 3000, 4000, 5000, 6000, 8000, 10000, 12000, 15000, 20000):
        a, b = cc[L], ct[L]
        kc = np.interp(L, lk, ck)
        print(f"   {L:6d}   {L*(L+1)*a/2/np.pi:12.4f} uK^2   {b/a-1:+10.2e}      {kc/a-1:+10.2e}           {kc/b-1:+10.2e}"
              + ("   (CLASS beyond its trusted range)" if L > ltrust else ""))
    spec = {"CAMB mead2020 (s01, default)": None,
            "CAMB takahashi": lambda l: np.interp(l, lt, ct),
            "CLASS halofit (V1)": lambda l: np.interp(l, lk, ck),
            "CMB = 0 above l = 6000": lambda l: np.where(np.asarray(l) > 6000, 0.0, ins_cmb0(l)),
            "CMB x 1.3 above l = 4000": lambda l: np.where(np.asarray(l) > 4000, 1.3, 1.0) * ins_cmb0(l)}
    TH = np.array([0.5, 1.0, 2.0, 5.0, 10.0, 20.0, 50.0])
    for mode in ("baseline", "goal"):
        ref = ins.MMFModel(ins.Instrument.so_lat(mode)).sigma_Y_exact(TH)
        print(f"   [{mode}] sigma_Y(variant)/sigma_Y(s01) - 1 at theta500 = {list(TH)}':")
        for name, f in spec.items():
            if f is None:
                continue
            ins.cmb_cl_TT = lambda l, f=f: np.where(np.asarray(l, float) < 2, 0.0, f(np.asarray(l, float)))
            try:
                v = ins.MMFModel(ins.Instrument.so_lat(mode)).sigma_Y_exact(TH)
            finally:
                ins.cmb_cl_TT = ins_cmb0
            print(f"      {name:28s} " + " ".join(f"{x:+9.1e}" for x in v / ref - 1) + f"   max |.| {np.abs(v/ref-1).max():.1e}")
    print(f"   (C) done {time.time()-t0:.0f}s", flush=True)


ins_cmb0 = ins.cmb_cl_TT


# ------------------------------------------------------------------------------------------------ (D)
@nb.njit(cache=False)
def _rho_prefix(rho, lth0, dlth, R_BEAM, thA, thB, s):
    """Re-implementation of the blendlib.rho_at lookup BEFORE the 2026-11-10 20:36 fix: theta clamped to the
    table, support S = 5 thA + 5 thB + R_BEAM with the TRUE thB, no (S_tab/S)^2 rescaling."""
    S = 5.0 * thA + 5.0 * thB + R_BEAM
    if s >= S:
        return 0.0
    NT = rho.shape[0]; NU = rho.shape[2]
    t = np.sqrt(s / S) * (NU - 1)
    ft = min(max(t, 0.0), NU - 1.000001); it = int(ft); wt = ft - it
    fa = min(max((np.log(thA) - lth0) / dlth, 0.0), NT - 1.000001); ia = int(fa); wa = fa - ia
    fb = min(max((np.log(thB) - lth0) / dlth, 0.0), NT - 1.000001); ib = int(fb); wb = fb - ib
    r = 0.0
    for da in range(2):
        ca = wa if da == 1 else 1.0 - wa
        for db in range(2):
            cb = wb if db == 1 else 1.0 - wb
            r += ca * cb * ((1.0 - wt) * rho[ia + da, ib + db, it] + wt * rho[ia + da, ib + db, it + 1])
    return r


@nb.njit(cache=False)
def _var_cap(thA, sigA, thB, YB, wB, tn, rho, lth0, dlth, R_BEAM):
    v = 0.0; cap = 0.0
    NU = tn.shape[0]; dt = tn[1] - tn[0]
    for k in range(thB.shape[0]):
        S = 5.0 * thA + 5.0 * thB[k] + R_BEAM
        for it in range(1, NU):
            t = tn[it]
            wt = 0.5 * dt if it == NU - 1 else dt
            dA = 4.0 * np.pi * S * S * t * t * t * wt
            x = _rho_prefix(rho, lth0, dlth, R_BEAM, thA, thB[k], S * t * t) * YB[k] / sigA
            w = wB[k] * dA
            if abs(x) >= 500.0:
                cap += w
            else:
                v += w * x * x
    return v, cap


def part_D():
    print("\n=== (D) data/blending/analytic.npz: pre-fix kernel, independent code path; file timeline ===", flush=True)
    t0 = time.time()
    pop = bl.neighbour_population(zsub=1, msub=6, ngh=16)
    tab = bl.load_rho_table(); m = sg.SZModel()
    A = np.load(f"{bl.DATA}/analytic.npz")
    print(f"   rho table: {tab.NT} theta nodes up to {tab.th[-1]:.1f}', R_BEAM {tab.R_BEAM}'; neighbour points {len(pop[0]):,}")
    for i in (15, 20, 25, 30):
        thA = float(tab.th[i]); sA = float(m.sigma_Y(thA))
        v, cap = _var_cap(thA, sA, pop[0], pop[1], pop[3], tab.t, tab.rho, tab.lth0, tab.dlth, tab.R_BEAM)
        vs, cs = A["WX2"][i].sum(), A["CAP"][i].sum()
        print(f"   thA = {thA:6.2f}': var(x<500) pre-fix (this code) {v:.6f} vs stored {vs:.6f} (rel {v/vs-1:+.1e}); "
              f"cap rate {cap:.4e} vs stored {cs:.4e} (rel {cap/cs-1:+.1e})  [{time.time()-t0:.0f}s]", flush=True)
    print("   timeline (file modification times):")
    for f in ("analysis/blending/b03_analytic.log", "analysis/blending/blendlib.py", "analysis/blending/r02_pd_fine.log",
              "data/blending/analytic.npz"):
        print(f"      {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(os.path.getmtime(os.path.join(ROOT, f))))}  {f}")
    print("   r02_pd_fine.py loads analytic.npz, replaces only QUANT and CCDF (keeping QUANT_coarse, CCDF_coarse) and saves it"
          " back (its log has the same mtime as the npz); W, WX, WX2 and CAP are carried over from the b03 run that finished"
          " before the blendlib fix.", flush=True)


# ------------------------------------------------------------------------------------------------ (E)
def part_E():
    """Bug found by the arbiter in s06b (inherited by s06c, s09): the 'missing variance beyond the kernel support',
    extra = Parseval(s05) - rate-histogram variance, is evaluated by np.interp of the s05 variances, which s05 only
    tabulates for theta_A = 0.25-80'.  np.interp clamps outside that range, so the nodes theta_A < 0.25' and > 80'
    received a spurious Gaussian variance (up to 0.014 (baseline) / 0.021 (goal) at 0.1', 0.03-0.04 at 139').
    Inside the range the histogram reproduces Parseval to <= 0.15%, so the correct extra term is 0 everywhere.
    Test: recompute the node tables of the affected nodes with extra = 0 (s09 rate histograms, fdom = 1) and compare
    every q > 4..7 fraction with the stored s06b/s09 values."""
    print("\n=== (E) spurious 'extra' Gaussian variance outside s05's theta_A range (s06b/s06c/s09): effect on the fractions ===",
          flush=True)
    from scipy.special import ndtr
    QCUTS = [4.0, 5.0, 6.0, 7.0]
    RK = ("Wp", "WXp", "WX2p", "Wn", "WXn", "WX2n", "capp", "capn")
    ps = pl.PDs(N=2 ** 18, dd=0.001, x_small=0.002, off=60.0); pb = pl.PDs(N=2 ** 19, dd=0.002, x_small=0.004, off=60.0)
    for run in ("baseline", "goal"):
        t0 = time.time()
        d = np.load(f"{ROOT}/data_real/pd_tables_{run}.npz")
        th, extra, mu, H = d["th"], d["extra"], d["mu"], d["H"]
        cols = list(d["cols"]); ref = d["rows_fdom1.0"]
        nodes = np.nonzero(extra > 0)[0]
        T = {k: d[f"T1.0_{k}"].copy() for k in ("Pdom", "det", "b3", "lost", "mean", "b2", "b2n")}
        for i in nodes:
            o = {k: d[f"R_{k}"][i] for k in RK}
            for k in np.nonzero(H[i] != 0)[0]:
                qa = pl.QNODES[k]
                pdm, out = pl.node_probs_ext(o, (0, 1, 2), qa, QCUTS, mu[i], ps, pb, extra_var=0.0, fdom=1.0)
                T["Pdom"][i, k] = pdm
                for ic, qc in enumerate(QCUTS):
                    r = out[qc]
                    T["det"][ic, i, k] = r["det"] * pdm; T["b3"][ic, i, k] = r["b3"] * pdm; T["lost"][ic, i, k] = r["lost"] * pdm
                    T["mean"][ic, i, k] = r["mean"] * pdm / qa; T["b2"][ic, i, k] = r["b2"] * pdm; T["b2n"][ic, i, k] = r["b2n"] * pdm
        Phi = ndtr(pl.QNODES[None, :] - np.array(QCUTS)[:, None])
        worst_f, worst_n = 0.0, 0.0
        for ic, qc in enumerate(QCUTS):
            Ns = (H * Phi[ic][None, :]).sum(); Nd = (H * T["det"][ic]).sum()
            new = dict(N_det_b=Nd, B3=(H * T["b3"][ic]).sum() / Nd, lost=(H * T["lost"][ic]).sum() / Ns,
                       meanbias=(H * T["mean"][ic]).sum() / Nd, sat=(H * (1 - T["Pdom"]) * Phi[ic][None, :]).sum() / Ns,
                       **{f"B2_{e}": (H * T["b2"][ic, :, :, j]).sum() / Nd for j, e in enumerate(pl.EPS)},
                       **{f"B2neg_{e}": (H * T["b2n"][ic, :, :, j]).sum() / Nd for j, e in enumerate(pl.EPS)})
            row = ref[ic]
            for k, v in new.items():
                if k == "N_det_b":
                    worst_n = max(worst_n, abs(v / row[cols.index(k)] - 1))
                else:
                    worst_f = max(worst_f, abs(v - row[cols.index(k)]))
        share = (H[nodes] * Phi[1][None, :]).sum() / (H * Phi[1][None, :]).sum()
        print(f"   [{run}] affected theta_A nodes: {np.round(th[nodes], 3)} (extra var {np.round(extra[nodes], 4)}); their share of "
              f"N_single(q>5) {share:.1e}; removing the term changes N_det,blended by <= {worst_n:.1e} (relative) and every "
              f"fraction by <= {worst_f:.1e} (absolute), q_cut = 4-7  [{time.time()-t0:.0f}s]", flush=True)


if __name__ == "__main__":
    parts = sys.argv[1:] or ["A", "B", "C", "D", "E"]
    for p in parts:
        {"A": part_A, "B": part_B, "C": part_C, "D": part_D, "E": part_E}[p]()
