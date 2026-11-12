"""Step-2 verification, lens 'catalogue': independent DIRECT per-object route + own count model + own fitter.

Written from scratch. Imports from the project ONLY:
  * szsim.sampler.draw_catalog  (Poisson halo skies), szsim.cosmo_hmf.default_table (the sampler's HMF table, for the
    expected neighbour mean), szsim.signal.SZModel (old white filter sigma_Y and the Y-M relation of the sampler),
  * szsim.instrument.MMFModel / Instrument.so_lat  -> sigma_Y(theta500) of the SO MMF ONLY,
  * the step-1 derivation check's own rho tables verify_real/step1_derivation/rho_table_{white,baseline,goal}.npz
    (built in v07 from that verifier's own N_y and template, without the build's code).
Nothing from analysis/realistic/step2, pdlib1, blendlib, cosmolib(2) or the build's kernels is imported.

Per primary A (every halo of the sky, low-q ones subsampled with weights):
   delta_f(A) = sum_{B != A, s < 5 thA + 5 thB + M0} rho_f(thA, thB, s) Y_B / sigma_f(thA)     (f = white, baseline, goal)
   xmax_f(A)  = max_B of the same contributions (dominance: A counted only if xmax < fdom * q_A)
   q_obs = q_A + n + delta - mu_f(thA),  n ~ N(0,1),  mu_f = EXPECTED mean of the truncated kernel (convention c = 1)
   A's own lognormal Y scatter is integrated per object (randomly shifted 1D lattice, unbiased); the noise analytically.
Histograms: CCDF S(z bin, edge) = sum_A w_A E_scatter[1(dom) Phi(q_A + delta - mu - r e)] and the single-cluster
analogue Phi(q_A / r - e); bin counts are CCDF differences.
"""
import os, sys, math, time
_NT = os.environ.get("VC_THREADS", "4")
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "NUMBA_NUM_THREADS",
           "NUMEXPR_NUM_THREADS"):
    os.environ[_v] = _NT
import numpy as np
import numba as nb

ROOT = "/Users/boris/GitHub/sz-poisson-blending"
HERE = os.path.dirname(os.path.abspath(__file__))
DERIV = f"{ROOT}/verify_real/step1_derivation"
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

AM = np.pi / 180.0 / 60.0
A_SKY_AM2 = 4 * np.pi / AM ** 2                     # full sky in arcmin^2
FID = np.array([0.3111, 0.8102, 0.8, 1.79, 0.173])   # Om, s8, (1-b), alpha, slnY
FID_LOW = np.array([0.3111, 0.8102, 0.6, 1.79, 0.173])
C_LOW = (0.6 / 0.8) ** 1.79
PN = ["Om", "s8", "omb", "alpha", "slnY"]
# cosmology bins (written out here; same numbers as the project's default_bins)
ZED = np.array([0.01, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.25, 1.5, 1.75, 2.0, 2.5, 3.0])
QED5 = np.geomspace(5.0, 100.0, 9)
QED6 = np.geomspace(6.0, 100.0, 9)
EDGES = np.unique(np.round(np.concatenate([QED5, QED6]), 12))     # 17 finite edges
M0 = 30.0                                           # support margin [arcmin]: s < 5 thA + 5 thB + M0
FILTERS = ("white", "baseline", "goal")

# variants of the blended histogram: (name, filter index, fdom, r-kind, scale c)
#   r-kind: 0 none, 1 r_i (baseline), 2 r_ii (baseline), 3 r_i (goal), 4 r_ii (goal), 5 r_i (low pressure)
#   mu-row: 0..2 = my expected mean for filter 0..2; 3 = the build's white mean (diagnostic variant only)
BVARS = [("white_f1", 0, 1.0, 0, 1.0, 0),
         ("baseline_f0.5", 1, 0.5, 0, 1.0, 1), ("baseline_f1", 1, 1.0, 0, 1.0, 1), ("baseline_f2", 1, 2.0, 0, 1.0, 1),
         ("baseline_effi", 1, 1.0, 1, 1.0, 1), ("baseline_effii", 1, 1.0, 2, 1.0, 1),
         ("goal_f0.5", 2, 0.5, 0, 1.0, 2), ("goal_f1", 2, 1.0, 0, 1.0, 2), ("goal_f2", 2, 2.0, 0, 1.0, 2),
         ("goal_effi", 2, 1.0, 3, 1.0, 2), ("goal_effii", 2, 1.0, 4, 1.0, 2),
         ("lowP_f1", 1, 1.0, 0, C_LOW, 1), ("lowP_effi", 1, 1.0, 5, C_LOW, 1),
         ("white_f1_muB", 0, 1.0, 0, 1.0, 3)]
# single-cluster (naive) histograms: (name, filter, r-kind, c)
SVARS = [("white", 0, 0, 1.0), ("baseline", 1, 0, 1.0), ("baseline_effi", 1, 1, 1.0), ("baseline_effii", 1, 2, 1.0),
         ("goal", 2, 0, 1.0), ("goal_effi", 2, 3, 1.0), ("goal_effii", 2, 4, 1.0), ("lowP", 1, 0, C_LOW),
         ("lowP_effi", 1, 5, C_LOW)]
# which single histogram each blended variant is differenced with
B2S = {"white_f1": "white", "baseline_f0.5": "baseline", "baseline_f1": "baseline", "baseline_f2": "baseline",
       "baseline_effi": "baseline_effi", "baseline_effii": "baseline_effii", "goal_f0.5": "goal", "goal_f1": "goal",
       "goal_f2": "goal", "goal_effi": "goal_effi", "goal_effii": "goal_effii", "lowP_f1": "lowP",
       "lowP_effi": "lowP_effi", "white_f1_muB": "white"}
NBLK = 48                                           # 6 bands in sin(dec) x 8 in RA, equal area


def set_threads(n):
    for v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS",
              "NUMBA_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        os.environ[v] = str(n)


def rss_gb():
    import resource
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e9


# ============================================================================ rho tables (derivation's v07)
def load_rho3():
    """Interleaved float32 table R3[iA, iB, s, f] for f = white, baseline, goal (+1 pad); nodes TH, S; table sigma."""
    TH = S = None
    R3 = None
    sig = {}
    for f, name in enumerate(FILTERS):
        d = np.load(f"{DERIV}/rho_table_{name}.npz")
        if R3 is None:
            TH, S = d["th"], d["s"]
            R3 = np.zeros(d["rho"].shape + (4,), np.float32)
        assert np.allclose(d["th"], TH) and np.allclose(d["s"], S)
        R3[..., f] = d["rho"]
        sig[name] = d["sigY"]
    return R3, TH, S, sig


def s_nodes_check(S):
    """the s grid of v07: 0.04' steps to 8', 0.2' to 40', 1' to 200', 4' to 800' (indexing in s_index)."""
    ref = np.concatenate([np.arange(0, 8, 0.04), np.arange(8, 40, 0.2), np.arange(40, 200, 1.0), np.arange(200, 800.01, 4.0)])
    assert len(ref) == len(S) and np.allclose(ref, S)


@nb.njit(inline="always")
def s_index(s):
    if s < 8.0:
        f = s / 0.04
        k = int(f)
        return k, f - k
    elif s < 40.0:
        f = (s - 8.0) / 0.2
        k = int(f)
        return 200 + k, f - k
    elif s < 200.0:
        f = s - 40.0
        k = int(f)
        return 360 + k, f - k
    else:
        f = (s - 200.0) / 4.0
        k = int(f)
        if k >= 150:
            return 669, 1.0
        return 520 + k, f - k


# ============================================================================ sigma_Y
_SIG = {}


def sigma_model(name):
    """'white' -> szsim SZModel (old 4e-6', 1.4' filter); 'baseline'/'goal' -> MMFModel(SO LAT, lmin = 100) [sigma_Y only]."""
    if name not in _SIG:
        from szsim import signal as sg, instrument as ins
        _SIG[name] = sg.SZModel() if name == "white" else ins.MMFModel(ins.Instrument.so_lat(name, lmin=100.0))
        _SIG[name].sigma_Y(1.0)
    return _SIG[name]


def sigma_Y(name, theta):
    return sigma_model(name).sigma_Y(theta)


# ============================================================================ map-sigma ratios (variant definition)
def r_tables():
    """r(theta) of the map-sigma variants, as DEFINED by the build (data_real/step2/inputs.npz, read as data):
    linear in ln theta between the 48 nodes, constant beyond (as the build's LnSigma).  Checked separately (vc06)."""
    I = np.load(f"{ROOT}/data_real/step2/inputs.npz")
    th = I["baseline_th"]
    assert np.allclose(th, I["goal_th"])
    return th, {1: I["baseline_r_i"], 2: I["baseline_r_ii"], 3: I["goal_r_i"], 4: I["goal_r_ii"], 5: I["lowP_r_i"]}


def r_of(theta, th_nodes, r_nodes):
    return np.interp(np.log(theta), np.log(th_nodes), r_nodes)


# ============================================================================ expected neighbour mean (own integral)
def table_at_sigma8(s8):
    """The sampler's HMF table (same z, lnM nodes) with d2N recomputed by colossus at sigma8 = s8 (Om fixed); the
    global colossus cosmology is reset to planck18 afterwards."""
    from szsim import cosmo_hmf as ch
    from colossus.cosmology import cosmology
    from colossus.lss import mass_function as mf
    T0 = ch.default_table()

    class _T:
        pass
    T = _T()
    T.z, T.lnM, T.zmin, T.zmax = T0.z.copy(), T0.lnM.copy(), T0.zmin, T0.zmax
    c = cosmology.setCosmology("planck18", sigma8=float(s8), persistence="")
    h = c.h; M = np.exp(T.lnM)
    chi = c.comovingDistance(0.0, T.z) / h
    dV = (299792.458 / c.H0) / c.Ez(T.z) * chi ** 2 * 4 * np.pi
    T.d2N = np.array([mf.massFunction(M * h, float(z), mdef="500c", model="tinker08", q_out="dndlnM") * h ** 3 * dV[i]
                      for i, z in enumerate(T.z)])
    cosmology.setCosmology("planck18", persistence="")
    T.dz = T.z[1] - T.z[0]; T.dlnM = T.lnM[1] - T.lnM[0]
    T._weights = ch.HMFTable._weights
    return T


def expected_mean_numerators(R3, TH, S, table=None):
    """S_f(theta_A node) = sum_B n_B <Y_B> int_{s < 5thA + 5thB + M0} rho_f d^2s  [arcmin^2 per arcmin^2],
    with n_B from the sampler's own HMF table (szsim default table, trapezoid = the sampler's cell expectation),
    <Y_B> = Y_med e^{slnY^2/2}, theta_B spread over the table's theta nodes with linear weights in ln theta (as the
    trilinear interpolation of the sums).  mu_f(thA) = interp(S_f) / sigma_f(thA)."""
    from szsim import cosmo_hmf as ch, signal as sg
    T = ch.default_table() if table is None else table
    m = sg.SZModel()
    wz = ch.HMFTable._weights(T.z, T.z[0], T.z[-1]); wM = ch.HMFTable._weights(T.lnM, T.lnM[0], T.lnM[-1])
    lth0, dlth = np.log(TH[0]), np.log(TH[1] / TH[0])
    WY = np.zeros(len(TH))
    M = np.exp(T.lnM)
    for iz, z in enumerate(T.z):
        th = m.theta500(M, z)
        y = m.Y500_med(M, z) * np.exp(0.5 * m.sigma_lnY ** 2)
        w = T.d2N[iz] * wz[iz] * wM * y / A_SKY_AM2
        f = np.clip((np.log(th) - lth0) / dlth, 0, len(TH) - 1.000001); i = f.astype(int); a = f - i
        np.add.at(WY, i, w * (1 - a)); np.add.at(WY, i + 1, w * a)
    tw = np.empty_like(S); d = np.diff(S); tw[1:-1] = 0.5 * (d[:-1] + d[1:]); tw[0] = 0.5 * d[0]; tw[-1] = 0.5 * d[-1]
    ring = 2 * np.pi * S * tw
    out = np.zeros((3, len(TH)))
    for i in range(len(TH)):
        inside = (S[None, :] < (5 * TH[i] + 5 * TH + M0)[:, None])            # (iB, s)
        for f in range(3):
            DC = (R3[i, :, :, f].astype(np.float64) * ring[None, :] * inside).sum(1)
            out[f, i] = (WY * DC).sum()
    return out, WY


def realised_mean_numerators(R3, TH, S, Y, th):
    """same with the realised sky (catalogue-weighted, as a map-mean removal would see it)."""
    lth0, dlth = np.log(TH[0]), np.log(TH[1] / TH[0])
    WY = np.zeros(len(TH))
    for c0 in range(0, len(Y), 4_000_000):
        sl = slice(c0, c0 + 4_000_000)
        y = Y[sl].astype(np.float64)
        f = np.clip((np.log(th[sl].astype(np.float64)) - lth0) / dlth, 0, len(TH) - 1.000001); i = f.astype(np.int64); a = f - i
        WY += np.bincount(i, y * (1 - a), len(TH)) + np.bincount(i + 1, y * a, len(TH))
    WY /= A_SKY_AM2
    tw = np.empty_like(S); d = np.diff(S); tw[1:-1] = 0.5 * (d[:-1] + d[1:]); tw[0] = 0.5 * d[0]; tw[-1] = 0.5 * d[-1]
    ring = 2 * np.pi * S * tw
    out = np.zeros((3, len(TH)))
    for k in range(len(TH)):
        inside = (S[None, :] < (5 * TH[k] + 5 * TH + M0)[:, None])
        for ff in range(3):
            out[ff, k] = (WY * (R3[k, :, :, ff].astype(np.float64) * ring[None, :] * inside).sum(1)).sum()
    return out


# ============================================================================ spatial index: dec bands sorted in RA
TIER_TH = np.array([0.0, 1.5, 5.0, 15.0, 1e9])      # theta_B tiers [arcmin]
TIER_H = np.array([0.5, 1.0, 2.0, 5.0])             # band height [deg] per tier


def build_index(ra_deg, dec_deg, th):
    """Permutation that sorts halos by (tier, dec band, RA) and the band start offsets.
    Returns perm, tier_off (ntier+1), band_off (concatenated per tier, each nb_t+1), band_base (ntier), nband (ntier),
    thmax (ntier)."""
    nbt = np.ceil(180.0 / TIER_H).astype(np.int64)
    base = np.concatenate([[0], np.cumsum(nbt)])[:-1]
    n = len(th)
    gb0 = np.empty(n, np.int32)
    for a in range(0, n, 4_000_000):                                    # chunked: few temporaries
        sl = slice(a, a + 4_000_000)
        tier = np.searchsorted(TIER_TH, th[sl], side="right") - 1
        band = np.minimum(((dec_deg[sl].astype(np.float64) + 90.0) / TIER_H[tier]).astype(np.int64), nbt[tier] - 1)
        gb0[sl] = base[tier] + band
    key = gb0.astype(np.float64)
    key *= 400.0
    key += ra_deg                                                       # ra in [0, 360)
    perm = np.argsort(key, kind="stable")
    del key
    gb = gb0[perm]
    del gb0
    tot = int(nbt.sum())
    band_off = np.searchsorted(gb, np.arange(tot + 1)).astype(np.int64)
    tier_off = np.array([band_off[b] for b in np.concatenate([base, [tot]])], np.int64)
    ts = th[perm]
    thmax = np.array([float(ts[tier_off[t]:tier_off[t + 1]].max()) if tier_off[t + 1] > tier_off[t] else 0.0
                      for t in range(len(TIER_H))])
    return perm, tier_off, band_off, base.astype(np.int64), nbt, thmax


@nb.njit(inline="always")
def _lower(a, lo, hi, v):
    # first index in [lo, hi) with a[i] >= v
    while lo < hi:
        mid = (lo + hi) >> 1
        if a[mid] < v:
            lo = mid + 1
        else:
            hi = mid
    return lo


@nb.njit(parallel=True, fastmath=False)
def neighbour_sums(order, pidx, ra, dec, X, Y_, Z, YB, TB, IB, WB, band_off, band_base, nband, tier_h, thmax,
                   R3, IA, WA, sig3, m0):
    """For each primary (sorted index pidx[k]): delta_f and xmax_f, f = 0..2, and the number of pairs.
    order: processing order (random permutation of 0..NP-1, for load balance)."""
    NP = pidx.shape[0]
    out = np.zeros((NP, 6))
    npair = np.zeros(NP, np.int64)
    nthr = nb.get_num_threads()
    chunk = (NP + nthr - 1) // nthr
    deg = np.pi / 180.0
    am = np.pi / 10800.0
    for tid in nb.prange(nthr):
        for kk in range(tid * chunk, min(NP, (tid + 1) * chunk)):
            k = order[kk]
            a = pidx[k]
            xa = np.float64(X[a]); ya = np.float64(Y_[a]); za = np.float64(Z[a])
            raa = np.float64(ra[a]); deca = np.float64(dec[a]); tA = np.float64(TB[a])
            ia = IA[k]; wa = WA[k]
            s0 = sig3[k, 0]; s1 = sig3[k, 1]; s2 = sig3[k, 2]
            d0 = 0.0; d1 = 0.0; d2 = 0.0
            m0_ = -1e300; m1_ = -1e300; m2_ = -1e300
            npr = 0
            for t in range(tier_h.shape[0]):
                Rm = 5.0 * tA + 5.0 * thmax[t] + m0                    # arcmin
                Rr = Rm * am
                if Rr > np.pi:
                    Rr = np.pi
                cR = 2.0 * math.sin(0.5 * Rr)
                cR2 = cR * cR + 1e-14
                Rdeg = Rm / 60.0
                b_lo = int((deca - Rdeg + 90.0) / tier_h[t]); b_hi = int((deca + Rdeg + 90.0) / tier_h[t])
                if b_lo < 0:
                    b_lo = 0
                if b_hi > nband[t] - 1:
                    b_hi = nband[t] - 1
                full = abs(deca) + Rdeg >= 89.999
                if not full:
                    sr = math.sin(Rr) / math.cos(deca * deg)
                    if sr >= 1.0:
                        full = True
                        dra = 180.0
                    else:
                        dra = math.asin(sr) / deg + 1e-4
                else:
                    dra = 180.0
                for b in range(b_lo, b_hi + 1):
                    g = band_base[t] + b
                    lo = band_off[g]; hi = band_off[g + 1]
                    if hi <= lo:
                        continue
                    # one or two RA intervals
                    for piece in range(2):
                        if full:
                            if piece == 1:
                                break
                            p0 = lo; p1 = hi
                        else:
                            r1 = raa - dra; r2 = raa + dra
                            if piece == 0:
                                p0 = _lower(ra, lo, hi, max(r1, 0.0)); p1 = _lower(ra, lo, hi, min(r2, 360.0))
                            else:
                                if r1 < 0.0:
                                    p0 = _lower(ra, lo, hi, r1 + 360.0); p1 = hi
                                elif r2 > 360.0:
                                    p0 = lo; p1 = _lower(ra, lo, hi, r2 - 360.0)
                                else:
                                    break
                        for p in range(p0, p1):
                            if p == a:
                                continue
                            dx = np.float64(X[p]) - xa; dy = np.float64(Y_[p]) - ya; dz = np.float64(Z[p]) - za
                            c2 = dx * dx + dy * dy + dz * dz
                            if c2 > cR2:
                                continue
                            s = 2.0 * math.asin(0.5 * math.sqrt(c2)) / am
                            tb = np.float64(TB[p])
                            if s >= 5.0 * tA + 5.0 * tb + m0:
                                continue
                            ib = IB[p]; wb = np.float64(WB[p])
                            ks, ws = s_index(s)
                            r0 = 0.0; r1_ = 0.0; r2_ = 0.0
                            for da in range(2):
                                fA = wa if da == 1 else 1.0 - wa
                                for db in range(2):
                                    fB = wb if db == 1 else 1.0 - wb
                                    w = fA * fB
                                    w0 = w * (1.0 - ws); w1 = w * ws
                                    r0 += w0 * R3[ia + da, ib + db, ks, 0] + w1 * R3[ia + da, ib + db, ks + 1, 0]
                                    r1_ += w0 * R3[ia + da, ib + db, ks, 1] + w1 * R3[ia + da, ib + db, ks + 1, 1]
                                    r2_ += w0 * R3[ia + da, ib + db, ks, 2] + w1 * R3[ia + da, ib + db, ks + 1, 2]
                            yb = np.float64(YB[p])
                            x0 = r0 * yb / s0; x1 = r1_ * yb / s1; x2 = r2_ * yb / s2
                            d0 += x0; d1 += x1; d2 += x2
                            if x0 > m0_:
                                m0_ = x0
                            if x1 > m1_:
                                m1_ = x1
                            if x2 > m2_:
                                m2_ = x2
                            npr += 1
            out[k, 0] = d0; out[k, 1] = d1; out[k, 2] = d2
            out[k, 3] = m0_; out[k, 4] = m1_; out[k, 5] = m2_
            npair[k] = npr
    return out, npair


# ============================================================================ normal CDF / inverse (numba)
@nb.njit(inline="always")
def ncdf(x):
    return 0.5 * math.erfc(-x * 0.7071067811865476)


@nb.njit
def ninv(p):
    """inverse standard normal CDF (Acklam's rational approximation + one Halley step; |err| < 1e-12)."""
    a1 = -3.969683028665376e+01; a2 = 2.209460984245205e+02; a3 = -2.759285104469687e+02
    a4 = 1.383577518672690e+02; a5 = -3.066479806614716e+01; a6 = 2.506628277459239e+00
    b1 = -5.447609879822406e+01; b2 = 1.615858368580409e+02; b3 = -1.556989798598866e+02
    b4 = 6.680131188771972e+01; b5 = -1.328068155288572e+01
    c1 = -7.784894002430293e-03; c2 = -3.223964580411365e-01; c3 = -2.400758277161838e+00
    c4 = -2.549732539343734e+00; c5 = 4.374664141464968e+00; c6 = 2.938163982698783e+00
    d1 = 7.784695709041462e-03; d2 = 3.224671290700398e-01; d3 = 2.445134137142996e+00; d4 = 3.754408661907416e+00
    plow = 0.02425
    if p < plow:
        q = math.sqrt(-2 * math.log(p))
        x = (((((c1 * q + c2) * q + c3) * q + c4) * q + c5) * q + c6) / ((((d1 * q + d2) * q + d3) * q + d4) * q + 1)
    elif p <= 1 - plow:
        q = p - 0.5
        r = q * q
        x = (((((a1 * r + a2) * r + a3) * r + a4) * r + a5) * r + a6) * q / (((((b1 * r + b2) * r + b3) * r + b4) * r + b5) * r + 1)
    else:
        q = math.sqrt(-2 * math.log(1 - p))
        x = -(((((c1 * q + c2) * q + c3) * q + c4) * q + c5) * q + c6) / ((((d1 * q + d2) * q + d3) * q + d4) * q + 1)
    e = 0.5 * math.erfc(-x / math.sqrt(2.0)) - p
    u = e * math.sqrt(2 * math.pi) * math.exp(x * x / 2)
    return x - u / (1 + x * u / 2)


# ============================================================================ per-object histograms
@nb.njit(inline="always")
def _add_ccdf(H, tid, blk, v, iz, a, rr, edges, w):
    """H[tid, blk, v, iz, j] += w * Phi(a - rr * e_j) for the sorted edges (early exit far in the tails)."""
    for j in range(edges.shape[0]):
        t = a - rr * edges[j]
        if t < -8.5:
            break
        if t > 8.5:
            H[tid, blk, v, iz, j] += w
        else:
            H[tid, blk, v, iz, j] += w * ncdf(t)


@nb.njit(parallel=True)
def accumulate(qmed, delta, xmax, mu, rr, zbin, blk, wgt, u0, slnY, K, edges, bf, bfdom, brk, bc, bmu, sf, srk, sc,
               nblk, nz):
    """qmed (3, NP) median q per filter; delta, xmax, mu (3, NP); rr (6, NP) r per kind (rr[0] = 1); zbin, blk (NP);
    wgt (NP) subsampling weight; u0 (NP) random lattice shift; K lattice size.
    Blended variant v: filter bf[v], fdom bfdom[v], r-kind brk[v], scale bc[v]:
        sum_k 1(xmax < fdom q_k) Phi(c (q_k + delta - mu) - r e)  / K,   q_k = qmed exp(slnY x_k)
    Single variant: Phi(c q_k / r - e).  Returns HB (nthr, nblk, nvb, nz, ne), HS (nthr, nblk, nvs, nz, ne)."""
    NP = qmed.shape[1]
    nthr = nb.get_num_threads()
    nvb = bf.shape[0]; nvs = sf.shape[0]; ne = edges.shape[0]
    HB = np.zeros((nthr, nblk, nvb, nz, ne))
    HS = np.zeros((nthr, nblk, nvs, nz, ne))
    chunk = (NP + nthr - 1) // nthr
    for tid in nb.prange(nthr):
        xs = np.empty(K)
        for k in range(tid * chunk, min(NP, (tid + 1) * chunk)):
            iz = zbin[k]
            if iz < 0:
                continue
            b = blk[k]
            w = wgt[k] / K
            for i in range(K):
                xs[i] = math.exp(slnY * ninv((i + u0[k]) / K))
            for v in range(nvb):
                f = bf[v]
                q0 = qmed[f, k]
                dm = delta[f, k] - mu[bmu[v], k]
                xm = xmax[f, k]
                r = rr[brk[v], k]
                c = bc[v]
                fd = bfdom[v]
                for i in range(K):
                    q = q0 * xs[i]
                    if xm < fd * q:
                        _add_ccdf(HB, tid, b, v, iz, c * (q + dm), r, edges, w)
            for v in range(nvs):
                f = sf[v]
                q0 = qmed[f, k]
                r = rr[srk[v], k]
                c = sc[v]
                for i in range(K):
                    _add_ccdf(HS, tid, b, v, iz, c * q0 * xs[i] / r, 1.0, edges, w)
    return HB, HS


def block_of(ra_deg, dec_deg):
    """48 equal-area blocks: 6 bands in sin(dec), 8 slices in RA."""
    sd = np.sin(np.radians(dec_deg.astype(np.float64)))
    bi = np.clip(((sd + 1.0) / 2.0 * 6).astype(np.int64), 0, 5)
    bj = np.clip((ra_deg.astype(np.float64) / 45.0).astype(np.int64), 0, 7)
    return (bi * 8 + bj).astype(np.int64)


def ccdf_to_bins(C, qed):
    """CCDF over EDGES (..., 17) -> counts in the 9 bins of edges qed (last bin = q > qed[-1])."""
    idx = [int(np.argmin(np.abs(EDGES - e))) for e in qed]
    assert np.allclose(EDGES[idx], qed)
    Cs = C[..., idx]
    out = np.empty(C.shape[:-1] + (len(qed),))
    out[..., :-1] = Cs[..., :-1] - Cs[..., 1:]
    out[..., -1] = Cs[..., -1]
    return out


@nb.njit(parallel=True)
def point_sums(pra, pdec, px, py, pz, pth, ra, dec, X, Y_, Z, YB, TB, IB, WB, SUB, band_off, band_base, nband, tier_h,
               thmax, R3, IA, WA, sig3, m0):
    """delta_f at arbitrary points (no self term): returns (NP, 9) = delta_f (all neighbours), delta_f (only
    neighbours with SUB = 1) and xmax_f (all), f = white, baseline, goal.  Same index, cuts and interpolation as
    neighbour_sums."""
    NP = pra.shape[0]
    out = np.zeros((NP, 9))
    deg = np.pi / 180.0
    am = np.pi / 10800.0
    for k in nb.prange(NP):
        xa = px[k]; ya = py[k]; za = pz[k]; raa = pra[k]; deca = pdec[k]; tA = pth[k]
        ia = IA[k]; wa = WA[k]
        acc = np.zeros(6)
        xm = np.full(3, -1e300)
        for t in range(tier_h.shape[0]):
            Rm = 5.0 * tA + 5.0 * thmax[t] + m0
            Rr = Rm * am
            if Rr > np.pi:
                Rr = np.pi
            cR = 2.0 * math.sin(0.5 * Rr)
            cR2 = cR * cR + 1e-14
            Rdeg = Rm / 60.0
            b_lo = int((deca - Rdeg + 90.0) / tier_h[t]); b_hi = int((deca + Rdeg + 90.0) / tier_h[t])
            if b_lo < 0:
                b_lo = 0
            if b_hi > nband[t] - 1:
                b_hi = nband[t] - 1
            full = abs(deca) + Rdeg >= 89.999
            dra = 180.0
            if not full:
                sr = math.sin(Rr) / math.cos(deca * deg)
                if sr >= 1.0:
                    full = True
                else:
                    dra = math.asin(sr) / deg + 1e-4
            for b in range(b_lo, b_hi + 1):
                g = band_base[t] + b
                lo = band_off[g]; hi = band_off[g + 1]
                if hi <= lo:
                    continue
                for piece in range(2):
                    if full:
                        if piece == 1:
                            break
                        p0 = lo; p1 = hi
                    else:
                        r1 = raa - dra; r2 = raa + dra
                        if piece == 0:
                            p0 = _lower(ra, lo, hi, max(r1, 0.0)); p1 = _lower(ra, lo, hi, min(r2, 360.0))
                        else:
                            if r1 < 0.0:
                                p0 = _lower(ra, lo, hi, r1 + 360.0); p1 = hi
                            elif r2 > 360.0:
                                p0 = lo; p1 = _lower(ra, lo, hi, r2 - 360.0)
                            else:
                                break
                    for p in range(p0, p1):
                        dx = np.float64(X[p]) - xa; dy = np.float64(Y_[p]) - ya; dz = np.float64(Z[p]) - za
                        c2 = dx * dx + dy * dy + dz * dz
                        if c2 > cR2:
                            continue
                        s = 2.0 * math.asin(0.5 * math.sqrt(c2)) / am
                        tb = np.float64(TB[p])
                        if s >= 5.0 * tA + 5.0 * tb + m0:
                            continue
                        ib = IB[p]; wb = np.float64(WB[p])
                        ks, ws = s_index(s)
                        yb = np.float64(YB[p])
                        for f in range(3):
                            r = 0.0
                            for da in range(2):
                                fA = wa if da == 1 else 1.0 - wa
                                for db in range(2):
                                    fB = wb if db == 1 else 1.0 - wb
                                    r += fA * fB * ((1.0 - ws) * R3[ia + da, ib + db, ks, f] + ws * R3[ia + da, ib + db, ks + 1, f])
                            x = r * yb / sig3[k, f]
                            acc[f] += x
                            if x > xm[f]:
                                xm[f] = x
                            if SUB[p]:
                                acc[3 + f] += x
        for j in range(6):
            out[k, j] = acc[j]
        for j in range(3):
            out[k, 6 + j] = xm[j]
    return out
