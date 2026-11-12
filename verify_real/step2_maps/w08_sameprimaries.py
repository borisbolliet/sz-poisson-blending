"""w08: map vs build on the SAME realised primaries (sharper than comparing with the HMF-integrated Delta).

For every primary of w02 (q_A >= 0.5) the build's stored kernel K_b(theta, q_A; e) (data_real/step2/kernel_<setup>.npz:
the blended detection probability incl. dominance, from the P(D) machinery) is evaluated exactly as the build's count
model does (linear in ln theta between nodes, linear in q_A on its grid).  Then per (z, q_obs) bin
   N_b,map   = sum over primaries of the map detection probability (w04's definition)
   N_b,build = sum over the same primaries of K_b
so  N_b,map - N_b,build  tests the P(D) kernel itself; the halo-population sample variance cancels (the single-cluster
term is identical).  Errors: bootstrap over patches.  Also per q_A class and per theta range, and the S8 bias implied by
the difference (linearised, build's naive model).
Output: w08_sameprimaries.log
"""
import os, sys, glob, json, time
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "NUMBA_NUM_THREADS"):
    os.environ[_v] = os.environ.get("W08_THREADS", "3")
import numpy as np
from scipy.special import ndtr

ROOT = "/Users/boris/GitHub/sz-poisson-blending"
HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "out")
sys.path.insert(0, HERE)
t0 = time.time()
f = open(os.path.join(HERE, "w08_sameprimaries.log"), "w")


def log(s=""):
    print(s, flush=True); f.write(s + "\n"); f.flush()


ARCMIN = np.pi / 10800.0
FULLSKY = 4 * np.pi / ARCMIN ** 2
ZED = np.array([0.01, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.25, 1.5, 1.75, 2.0, 2.5, 3.0])
inp = np.load(f"{ROOT}/data_real/step2/inputs.npz")
C_LOW = float(inp["C_LOW"])
SETUPS = {  # name: (mode, kernel file, fdom, r or None, Y scale)
    "baseline_f1": ("baseline", "baseline_f1", 1.0, None, 1.0), "baseline_f0.5": ("baseline", "baseline_f0.5", 0.5, None, 1.0),
    "baseline_f2": ("baseline", "baseline_f2", 2.0, None, 1.0), "goal_f1": ("goal", "goal_f1", 1.0, None, 1.0),
    "goal_f0.5": ("goal", "goal_f0.5", 0.5, None, 1.0), "goal_f2": ("goal", "goal_f2", 2.0, None, 1.0),
    "white_f1": ("white", "white", 1.0, None, 1.0),
    "baseline_effi": ("baseline", "baseline_effi", 1.0, inp["baseline_r_i"], 1.0),
    "baseline_effii": ("baseline", "baseline_effii", 1.0, inp["baseline_r_ii"], 1.0),
    "goal_effi": ("goal", "goal_effi", 1.0, inp["goal_r_i"], 1.0), "goal_effii": ("goal", "goal_effii", 1.0, inp["goal_r_ii"], 1.0),
    "lowP_f1": ("baseline", "lowP_f1", 1.0, None, C_LOW),
}
KER = {}
for nm, (m, kf, fd, r, c) in SETUPS.items():
    k = np.load(f"{ROOT}/data_real/step2/kernel_{kf}.npz")
    Kb = k["Kb"]
    assert np.isclose(float(k["fdom"]), fd)
    KER[nm] = (Kb.astype(np.float64), k["th"], k["QA"], k["EDGES"])
EDGES = KER["baseline_f1"][3]
QE5 = [int(np.argmin(np.abs(EDGES - e))) for e in np.geomspace(5.0, 100.0, 9)]


def kernel_eval(nm, th, qk):
    """K_b at (theta, q) for every primary: linear in ln theta between nodes (clipped), linear in q on the grid."""
    Kb, thn, QA, E = KER[nm]
    lt = np.log(thn)
    fth = np.clip((np.log(th) - lt[0]) / (lt[1] - lt[0]), 0, len(lt) - 1.000001)
    ia = fth.astype(int); wa = fth - ia
    q = np.clip(qk, QA[0], QA[-1])
    iq = np.clip(np.searchsorted(QA, q) - 1, 0, len(QA) - 2)
    wq = (q - QA[iq]) / (QA[iq + 1] - QA[iq])
    out = ((Kb[ia, iq] * (1 - wq)[:, None] + Kb[ia, iq + 1] * wq[:, None]) * (1 - wa)[:, None]
           + (Kb[ia + 1, iq] * (1 - wq)[:, None] + Kb[ia + 1, iq + 1] * wq[:, None]) * wa[:, None])
    return out                                                           # (n, 19)


QCL = [(0.5, 2.0), (2.0, 3.5), (3.5, 5.0), (5.0, 7.0), (7.0, 12.0), (12.0, 1e9)]
THR = [(0.0, 0.5), (0.5, 1.0), (1.0, 2.0), (2.0, 4.0), (4.0, 1e9)]
files = sorted(glob.glob(os.path.join(OUT, "maps", "patch_[0-8]*.npz")))
NP = len(files)
names = list(SETUPS)
# per patch: map and build blended cumulative counts per z bin (16, 19); per q_A class and theta class at q_obs > 5
A_map = np.zeros((NP, len(names), 16, 19)); A_bld = np.zeros_like(A_map); A_sng = np.zeros_like(A_map)
Q_map = np.zeros((NP, len(names), len(QCL))); Q_bld = np.zeros_like(Q_map)
T_map = np.zeros((NP, len(names), len(THR))); T_bld = np.zeros_like(T_map)
area = 0.0
for ip, fn in enumerate(files):
    d = np.load(fn)
    area += float(d["area"])
    cache = {}
    for iv, nm in enumerate(names):
        m, kf, fd, r, c = SETUPS[nm]
        if m not in cache:
            cache[m] = {k: d[f"{m}_{k}"] for k in ("q", "z", "th", "j", "w", "d0", "d1", "x0", "x1")}
        D = cache[m]
        q = D["q"] * c
        keep = q >= 0.5
        q = q[keep]; z = D["z"][keep].astype(float); th = D["th"][keep].astype(float)
        j = D["j"][keep].astype(int); w = D["w"][keep].astype(float)
        d0 = D["d0"][keep] * c; d1 = D["d1"][keep] * c; x0 = D["x0"][keep] * c; x1 = D["x1"][keep] * c
        iz = np.clip(np.searchsorted(ZED, z, side="right") - 1, 0, 15)
        E = EDGES[None, :]
        dom0 = (x0 < fd * q).astype(float); dom1 = (x1 < fd * q).astype(float)
        if r is None:
            pm = ((1 - w) * dom0)[:, None] * ndtr((q + d0)[:, None] - E) + (w * dom1)[:, None] * ndtr((q + d1)[:, None] - E)
            pb = kernel_eval(nm, th, q)
            ps = ndtr(q[:, None] - E)
        else:
            thn = KER[nm][1]
            rA = np.exp(np.interp(np.log(th), np.log(thn), np.log(r)))
            r0 = r[j]; r1 = r[np.minimum(j + 1, len(r) - 1)]
            pm = (((1 - w) * dom0)[:, None] * ndtr((q + d0)[:, None] - r0[:, None] * E)
                  + (w * dom1)[:, None] * ndtr((q + d1)[:, None] - r1[:, None] * E))
            pb = kernel_eval(nm, th, q / rA)                              # K'(theta, q' = Y / sigma_eff)
            ps = ndtr((q / rA)[:, None] - E)
        np.add.at(A_map[ip, iv], iz, pm); np.add.at(A_bld[ip, iv], iz, pb); np.add.at(A_sng[ip, iv], iz, ps)
        e5 = 1                                                             # EDGES[1] = 5
        for k, (a, b) in enumerate(QCL):
            s = (q >= a) & (q < b)
            Q_map[ip, iv, k] = pm[s, e5].sum(); Q_bld[ip, iv, k] = pb[s, e5].sum()
        for k, (a, b) in enumerate(THR):
            s = (th >= a) & (th < b)
            T_map[ip, iv, k] = pm[s, e5].sum(); T_bld[ip, iv, k] = pb[s, e5].sum()
SCALE = FULLSKY / area
rng = np.random.default_rng(7)
NB = 2000
cnt = np.stack([np.bincount(rng.integers(0, NP, NP), minlength=NP) for _ in range(NB)]).astype(float)
log(f"[w08] {NP} patches ({area/3600:,.0f} deg^2); build kernels evaluated on the map's own primaries (q_A >= 0.5)")
log("    per variant, q_obs > 5 (full sky): N_single | N_b map | N_b build(kernel) | (map - build)/N_single with bootstrap 1 sigma; "
    "Delta/N map and build(kernel) on these primaries")
for iv, nm in enumerate(names):
    Ms = A_map[:, iv, :, 1].sum(1); Bs = A_bld[:, iv, :, 1].sum(1); Ss = A_sng[:, iv, :, 1].sum(1)
    dif = (Ms - Bs); db = cnt @ dif; sb = cnt @ Ss
    rel = dif.sum() / Ss.sum(); relb = (db / sb).std()
    log(f"  [{nm:15s}] N_s {SCALE*Ss.sum():9,.0f}  N_b map {SCALE*Ms.sum():9,.0f}  build {SCALE*Bs.sum():9,.0f}  (map-build)/N_s = "
        f"{100*rel:+.3f} +- {100*relb:.3f}% ({rel/relb:+.1f} sigma) | Delta/N map {100*(Ms.sum()/Ss.sum()-1):+.3f}%, build kernel "
        f"{100*(Bs.sum()/Ss.sum()-1):+.3f}%")
    qs = []
    for k, (a, b) in enumerate(QCL):
        dq = Q_map[:, iv, k] - Q_bld[:, iv, k]
        qs.append(f"q_A {a:g}-{b:g}: {SCALE*dq.sum():+7.1f} +- {SCALE*(cnt @ dq).std():5.1f}")
    log("      map - build counts (full sky) by q_A class: " + "; ".join(qs))
    ts = []
    for k, (a, b) in enumerate(THR):
        dt = T_map[:, iv, k] - T_bld[:, iv, k]
        ts.append(f"theta {a:g}-{b:g}': {SCALE*dt.sum():+7.1f} +- {SCALE*(cnt @ dt).std():5.1f}")
    log("      by theta500: " + "; ".join(ts))
    # by z group
    zs = []
    for a, b in ((0.01, 0.1), (0.1, 0.5), (0.5, 1.0), (1.0, 1.5), (1.5, 2.0), (2.0, 3.0)):
        ix = np.nonzero((ZED[:-1] >= a - 1e-9) & (ZED[1:] <= b + 1e-9))[0]
        dz = A_map[:, iv, ix, 1].sum(1) - A_bld[:, iv, ix, 1].sum(1)
        nz = A_sng[:, iv, ix, 1].sum(1)
        zs.append(f"z {a}-{b}: {100*dz.sum()/nz.sum():+.2f}+-{100*((cnt @ dz)/(cnt @ nz)).std():.2f}%")
    log("      (map - build)/N_s by z: " + "; ".join(zs))
np.savez(os.path.join(OUT, "w08_sameprimaries.npz"), names=np.array(names), A_map=A_map, A_bld=A_bld, A_sng=A_sng, scale=SCALE)

# ------------------------------------------------------------------ implied S8 bias of the kernel difference
sys.path.insert(0, f"{ROOT}/analysis/realistic/step2")
import s2lib as S                                       # noqa: E402  (naive model and fitter only)
from s2lib import CL, C2                                # noqa: E402
G = CL.HMFGrid()


def qbins(C, qcut=5.0):
    qe = np.geomspace(qcut, 100.0, 9)
    ix = [int(np.argmin(np.abs(EDGES - e))) for e in qe]
    out = np.empty(C.shape[:-1] + (9,))
    for k in range(8):
        out[..., k] = C[..., ix[k]] - C[..., ix[k + 1]]
    out[..., 8] = C[..., ix[8]]
    return out


log("\n=== implied bias of (map - build kernel) on the same primaries, linearised, S8 / Om / s8 in Fisher sigma (bootstrap) ===")
MOD = {}
for iv, nm in enumerate(names):
    m, kf, fd, r, c = SETUPS[nm]
    truth = S.FID_LOW.copy() if nm.startswith("lowP") else S.FID.copy()
    key = (m, None if r is None else nm, nm.startswith("lowP"))
    if key not in MOD:
        if m == "white":
            M = C2.SmoothModel(qcut=5.0, grid=G)
        elif r is None:
            M = S.SmoothModelS(5.0, S.LnSigma(S.sz_model(m)), grid=G)
        else:
            M = S.SmoothModelS(5.0, S.LnSigma(S.sz_model(m), KER[nm][1], r), grid=G)
        MOD[key] = (M, M.counts(truth))
    M, N0 = MOD[key]
    dD = qbins(A_map[:, iv] - A_bld[:, iv]) * SCALE                     # (NP, 16, 9)
    for sv, fs in (("SO40", 0.4), ("full", 1.0)):
        outs = []
        for cn in ("a", "d"):
            free, pri = S.config(cn, truth)
            fi = [CL.PNAMES.index(p) for p in free]
            F_ = S.Fitter3(M, N0 * fs, free, fsky=fs, priors=pri, truth=truth)
            Ft = F_.fisher(truth[fi]); J = F_.jac(truth[fi])
            sig, Cov, gS = S.derived_sigma3(Ft, fi, truth)
            A = np.linalg.solve(Ft, J.T) / N0.ravel()[None, :]
            v = gS @ A / sig["S8"]
            tot = dD.sum(0).ravel() @ v
            bt = (np.tensordot(cnt, dD, axes=(1, 0)).reshape(NB, -1) @ v).std()
            outs.append(f"config {cn}: {tot:+.3f} +- {bt:.3f}")
        log(f"  [{nm:15s}] {sv:4s}: " + "; ".join(outs))
log(f"\n[w08] done {time.time()-t0:.0f}s")
