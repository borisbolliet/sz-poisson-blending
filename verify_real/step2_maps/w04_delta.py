"""w04: map-level blending data vector Delta_map(z, q_obs) and the cosmology bias it causes (lens: maps).

From w02's per-primary map read-outs (every primary q_A >= 0.5 of every patch), with the instrument noise
marginalised analytically (unit Gaussian in units of sigma_Y(theta_A)):
   single : P(q_obs > e) = Phi(q_A - e)
   blended: (1 - w) 1[x_max,j < fdom q_A] Phi(q_A + delta_j - e) + w 1[x_max,j+1 < fdom q_A] Phi(q_A + delta_j+1 - e)
            (linear in ln theta between the two bracketing nodes, as the build's kernel is interpolated)
   map-sigma variants (sigma_eff = r(theta) sigma_Y, the build's r_i / r_ii):
            single Phi(q_A / r(theta_A) - e);  blended: thresholds r_j e at the nodes (fdom = 1)
   low-pressure population ((1-b) = 0.6, every Y x c = 0.5975): the same skies scaled - delta, x_max and q_A all
            scale with c (the maps are linear in Y), dominance is scale-free; primaries c q_A >= 0.5.
Binned in the build's 16 z bins x 9 q_obs bins (q_cut 5, 6) of the TRUE halo redshift; scaled to the full sky by area.
Errors: bootstrap over patches (independent Poisson skies).
Then, with the BUILD's naive count model and fitter (analysis/realistic/step2/s2lib; model only, the data vector is ours):
   linearised bias  F^-1 J^T (Delta / N0) at the truth (all configs; bootstrap errors), and
   ML fits of the naive model to N0 + Delta_map (central vector; configs a-e; SO40, full sky).
Compared with the build's Delta for primaries q_A >= 0.5 (truth_<setup>.npz D05) and the build's biases (fits.json,
and the same linearised estimator applied to the build's Delta).
Output: out/w04_delta.npz, out/w04_fits.json
Env: NBOOT (2000), FIT_CFGS ("a,b,c,d,e"), EXTRA_R (optional npz with r arrays from w03 for analyst variants)
"""
import os, sys, glob, json, time
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "NUMBA_NUM_THREADS"):
    os.environ[_v] = os.environ.get("W04_THREADS", "3")
import numpy as np
from scipy.special import ndtr

ROOT = "/Users/boris/GitHub/sz-poisson-blending"
HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "out")
sys.path.insert(0, f"{ROOT}/analysis/realistic/step2")
import s2lib as S                                      # noqa: E402  (the build's naive model and fitter only)
from s2lib import CL, C2                               # noqa: E402

NBOOT = int(os.environ.get("NBOOT", "2000"))
FIT_CFGS = os.environ.get("FIT_CFGS", "a,b,c,d,e").split(",")
SURV = {"SO40": 0.4, "full": 1.0}


class Log:
    def __init__(self, path):
        self.f = open(path, "w")

    def __call__(self, *a):
        s = " ".join(str(x) for x in a)
        print(s, flush=True)
        self.f.write(s + "\n"); self.f.flush()


log = Log(os.path.join(HERE, "w04_delta.log"))
t0 = time.time()
ARCMIN = np.pi / 10800.0
FULLSKY = 4 * np.pi / ARCMIN ** 2
ZED = np.array([0.01, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.25, 1.5, 1.75, 2.0, 2.5, 3.0])
EDGES = S.EDGES                                        # the build's 19 q_obs edges (4, 5, 6, 7 and both bin sets)
assert np.allclose(EDGES, np.unique(np.concatenate([np.geomspace(5, 100, 9), np.geomspace(6, 100, 9), [4.0, 7.0]])))


def qbins(C, qcut):
    """cumulative P(q_obs > e) summed per z bin (nz, 19) -> counts in the 9 q_obs bins of cosmolib.default_bins."""
    qe = np.geomspace(qcut, 100.0, 9)
    ix = [int(np.argmin(np.abs(EDGES - e))) for e in qe]
    out = np.empty(C.shape[:-1] + (9,))
    for k in range(8):
        out[..., k] = C[..., ix[k]] - C[..., ix[k + 1]]
    out[..., 8] = C[..., ix[8]]
    return out


inp = np.load(f"{ROOT}/data_real/step2/inputs.npz")
TH = {m: np.load(f"{ROOT}/data_real/pd_tables_{m}.npz")["th"] for m in ("baseline", "goal", "white")}
C_LOW = float(inp["C_LOW"])
# variants: name -> (mode, kind, fdom, r_nodes or None, Yscale)
VAR = {}
for m in ("baseline", "goal", "white"):
    for f in (0.5, 1.0, 2.0):
        VAR[f"{m}_f{f:g}"] = (m, f, None, 1.0)
for m in ("baseline", "goal"):
    VAR[f"{m}_effi"] = (m, 1.0, inp[f"{m}_r_i"], 1.0)
    VAR[f"{m}_effii"] = (m, 1.0, inp[f"{m}_r_ii"], 1.0)
VAR["lowP_f1"] = ("baseline", 1.0, None, C_LOW)
VAR["lowP_effi"] = ("baseline", 1.0, inp["lowP_r_i"], C_LOW)
EXTRA = os.environ.get("EXTRA_R")
if EXTRA:
    ex = np.load(EXTRA)
    for key in ex.files:
        if key.startswith("r_"):
            nm = key[2:]                                    # e.g. baseline_clip
            m = nm.split("_")[0]
            VAR[f"{nm}"] = (m, 1.0, ex[key], 1.0)


def patch_counts(d, mode, fdom, r_nodes, c):
    """(Ns, Nb): (16, 19) cumulative counts per z bin for one patch and variant."""
    q = d[f"{mode}_q"] * c
    keep = q >= 0.5
    q = q[keep]
    z = d[f"{mode}_z"][keep].astype(float)
    th = d[f"{mode}_th"][keep].astype(float)
    j = d[f"{mode}_j"][keep].astype(int)
    w = d[f"{mode}_w"][keep].astype(float)
    d0 = d[f"{mode}_d0"][keep] * c; d1 = d[f"{mode}_d1"][keep] * c
    x0 = d[f"{mode}_x0"][keep] * c; x1 = d[f"{mode}_x1"][keep] * c
    iz = np.clip(np.searchsorted(ZED, z, side="right") - 1, 0, 15)
    dom0 = (x0 < fdom * q).astype(float); dom1 = (x1 < fdom * q).astype(float)
    E = EDGES[None, :]
    if r_nodes is None:
        ps = ndtr(q[:, None] - E)
        pb = ((1 - w) * dom0)[:, None] * ndtr((q + d0)[:, None] - E) + (w * dom1)[:, None] * ndtr((q + d1)[:, None] - E)
    else:
        lt = np.log(TH[mode])
        rA = np.exp(np.interp(np.log(th), lt, np.log(r_nodes)))           # r linear in ln theta, constant beyond
        r0 = r_nodes[j]; r1 = r_nodes[np.minimum(j + 1, len(r_nodes) - 1)]
        ps = ndtr((q / rA)[:, None] - E)
        pb = (((1 - w) * dom0)[:, None] * ndtr((q + d0)[:, None] - r0[:, None] * E)
              + (w * dom1)[:, None] * ndtr((q + d1)[:, None] - r1[:, None] * E))
    Ns = np.zeros((16, EDGES.size)); Nb = np.zeros((16, EDGES.size))
    np.add.at(Ns, iz, ps); np.add.at(Nb, iz, pb)
    return Ns, Nb


files = sorted(glob.glob(os.path.join(OUT, "maps", "patch_[0-8]*.npz")))
NP = len(files)
names = list(VAR)
NS = np.zeros((NP, len(names), 16, EDGES.size)); NB = np.zeros_like(NS)
area = 0.0; nhalo = 0; nskip = 0; nprim = {m: 0 for m in ("baseline", "goal", "white")}
for ip, fn in enumerate(files):
    d = np.load(fn)
    area += float(d["area"]); nhalo += int(d["nhalo"]); nskip += int(d["nskip"])
    for m in nprim:
        nprim[m] += d[f"{m}_q"].size
    for iv, nm in enumerate(names):
        m, fdom, rn, c = VAR[nm]
        NS[ip, iv], NB[ip, iv] = patch_counts(d, m, fdom, rn, c)
SCALE = FULLSKY / area
log(f"[w04] {NP} patches = {area/3600:,.0f} deg^2 (full sky x {1/SCALE:.4f}); {nhalo:,} halos painted, {nskip} with "
    f"theta500 > 60' not painted; primaries q_A >= 0.5: " + ", ".join(f"{m} {n:,}" for m, n in nprim.items())
    + f"; {len(names)} variants; [{time.time()-t0:.0f}s]")

# ------------------------------------------------------------------ bootstrap
rng = np.random.default_rng(2026)
BI = rng.integers(0, NP, (NBOOT, NP))
cnt = np.stack([np.bincount(b, minlength=NP) for b in BI]).astype(float)       # (NBOOT, NP) multiplicities


def boot(X):
    """X (NP, ...) -> central sum and bootstrap sums (NBOOT, ...)."""
    return X.sum(0), np.tensordot(cnt, X, axes=(1, 0))


BUILD = {"baseline_f1": "baseline_f1", "baseline_f0.5": "baseline_f0.5", "baseline_f2": "baseline_f2",
         "goal_f1": "goal_f1", "goal_f0.5": "goal_f0.5", "goal_f2": "goal_f2", "white_f1": "white",
         "baseline_effi": "baseline_effi", "baseline_effii": "baseline_effii", "goal_effi": "goal_effi",
         "goal_effii": "goal_effii", "lowP_f1": "lowP_f1", "lowP_effi": "lowP_effi"}
ZG = [(0.01, 0.1), (0.1, 0.5), (0.5, 1.0), (1.0, 1.5), (1.5, 2.0), (2.0, 3.0)]
zgi = [np.nonzero((ZED[:-1] >= a - 1e-9) & (ZED[1:] <= b + 1e-9))[0] for a, b in ZG]
res = {}
log("\n=== (A) Delta_map = N_blended - N_single (full sky, primaries q_A >= 0.5), q_cut 5, vs the build (same primaries) ===")
log("    Delta/N in %: total | z 0.01-0.1, 0.1-0.5, 0.5-1, 1-1.5, 1.5-2, 2-3 | q 5-7.3, 7.3-10.6, 10.6-15.4, >47; (bootstrap 1 sigma)")
for iv, nm in enumerate(names):
    for qc in (5.0, 6.0):
        s = f"{qc:.0f}"
        Cs, Csb = boot(NS[:, iv]); Cb, Cbb = boot(NB[:, iv])
        Ns_, Nsb = qbins(Cs, qc) * SCALE, qbins(Csb, qc) * SCALE
        Nb_, Nbb = qbins(Cb, qc) * SCALE, qbins(Cbb, qc) * SCALE
        D, Db = Nb_ - Ns_, Nbb - Nsb
        res[f"{nm}_q{s}"] = dict(D=D, Db=Db, Ns=Ns_, Nsb=Nsb, Nb=Nb_)
        if qc != 5.0:
            continue
        rel = lambda A, B: A.sum() / B.sum()
        tot = D.sum() / Ns_.sum(); totb = Db.sum((1, 2)) / Nsb.sum((1, 2))
        zr = [D[ix].sum() / Ns_[ix].sum() for ix in zgi]; zrb = [(Db[:, ix].sum((1, 2)) / Nsb[:, ix].sum((1, 2))).std() for ix in zgi]
        qr = [D[:, k].sum() / Ns_[:, k].sum() for k in (0, 1, 2)] + [D[:, 6:].sum() / Ns_[:, 6:].sum()]
        qrb = [(Db[:, :, k].sum(1) / Nsb[:, :, k].sum(1)).std() for k in (0, 1, 2)] + [(Db[:, :, 6:].sum((1, 2)) / Nsb[:, :, 6:].sum((1, 2))).std()]
        log(f"  [{nm:15s}] map  : N_single {Ns_.sum():9,.0f}  Delta {D.sum():+8,.0f} = {100*tot:+.3f} +- {100*totb.std():.3f}% | "
            + " ".join(f"{100*a:+.2f}({100*b:.2f})" for a, b in zip(zr, zrb)) + " | "
            + " ".join(f"{100*a:+.2f}({100*b:.2f})" for a, b in zip(qr, qrb)))
        if nm in BUILD:
            t = np.load(f"{ROOT}/data_real/step2/truth_{BUILD[nm]}.npz")
            Db5, Nn5, Dall = t["D05_q5"], t["Nn05_q5"], t["D_q5"]
            zr2 = [Db5[ix].sum() / Nn5[ix].sum() for ix in zgi]
            qr2 = [Db5[:, k].sum() / Nn5[:, k].sum() for k in (0, 1, 2)] + [Db5[:, 6:].sum() / Nn5[:, 6:].sum()]
            pull = (tot - Db5.sum() / Nn5.sum()) / totb.std()
            cov = np.cov(Db.reshape(NBOOT, -1).T)
            dd = (D - Db5).ravel()
            var = np.diag(cov)
            ok = var > 0
            chi2d = float(np.sum(dd[ok] ** 2 / var[ok]))
            log(f"  [{nm:15s}] build: N_single {Nn5.sum():9,.0f}  Delta {Db5.sum():+8,.0f} = {100*Db5.sum()/Nn5.sum():+.3f}%"
                f"  (all primaries {100*Dall.sum()/t['Nn_q5'].sum():+.3f}%) | "
                + " ".join(f"{100*a:+.2f}" for a in zr2) + "       | " + " ".join(f"{100*a:+.2f}" for a in qr2)
                + f"\n  {'':17s} map - build: total pull {pull:+.2f} sigma; N_single map/build - 1 = {Ns_.sum()/Nn5.sum()-1:+.4f} "
                f"(bootstrap {Nsb.sum((1,2)).std()/Ns_.sum():.4f}); per-bin chi2 (diagonal bootstrap var) = {chi2d:.1f} for "
                f"{ok.sum()} bins")
log(f"   [{time.time()-t0:.0f}s]")

# ------------------------------------------------------------------ white regression vs Q3 (c11, all primaries)
q3 = np.load(f"{ROOT}/data_cosmo/truth_catalogue_v2.npz")
r_ = res["white_f1_q5"]
log(f"\n=== (B) white filter (old idealised setup): map Delta {r_['D'].sum():+,.0f} +- {r_['Db'].sum((1,2)).std():,.0f} vs Q3 "
    f"(c11, all primaries) {q3['D_q5'].sum():+,.1f} and the build (q_A >= 0.5) "
    f"{np.load(f'{ROOT}/data_real/step2/truth_white.npz')['D05_q5'].sum():+,.1f} (full sky, q > 5)")

# ------------------------------------------------------------------ biases
G = CL.HMFGrid()
MODELS = {}


def model_key(nm):
    m, fdom, rn, c = VAR[nm]
    return (m, "none" if rn is None else nm.replace("lowP_", "lowP-"), nm.startswith("lowP"))


def model_for(nm, qc):
    key = model_key(nm) + (qc,)
    if key in MODELS:
        return MODELS[key]
    m, fdom, rn, c = VAR[nm]
    truth = S.FID.copy()
    if nm.startswith("lowP"):
        truth = S.FID_LOW.copy()
    if m == "white":
        M = C2.SmoothModel(qcut=qc, grid=G)
    elif rn is None:
        M = S.SmoothModelS(qc, S.LnSigma(S.sz_model(m)), grid=G)
    else:
        M = S.SmoothModelS(qc, S.LnSigma(S.sz_model(m), TH[m], rn), grid=G)
    N0 = M.counts(truth)
    MODELS[key] = (M, truth, N0)
    return MODELS[key]


_LINC = {}


def lin_vectors(M, truth, N0, cn, fs):
    key = (id(M), cn, fs)
    if key in _LINC:
        return _LINC[key]
    free, pri = S.config(cn, truth)
    fi = [CL.PNAMES.index(p) for p in free]
    F_ = S.Fitter3(M, N0 * fs, free, fsky=fs, priors=pri, truth=truth)
    Ft = F_.fisher(truth[fi]); J = F_.jac(truth[fi])
    sig, Cov, gS = S.derived_sigma3(Ft, fi, truth)
    A = np.linalg.solve(Ft, J.T) / N0.ravel()[None, :]                # x_lin = A @ Delta (full sky Delta; fs cancels)
    _LINC[key] = dict(S8=gS @ A / sig["S8"], Om=A[0] / sig["Om"], s8=A[1] / sig["s8"], sig=sig, free=free, A=A, fi=fi)
    return _LINC[key]


fits_build = json.load(open(f"{ROOT}/data_real/step2/fits.json"))


def build_ml(setup, cn, sv, qc):
    for r in fits_build:
        if r["setup"] == setup and r["config"] == cn and r["survey"] == sv and r["qcut"] == qc and r["route"] == "cat" \
                and r.get("start", "truth") == "truth":
            return r
    return None


LIN = {}
log("\n=== (C) linearised bias F^-1 J^T (Delta/N0) in Fisher sigma (S8 | Om | s8): map (bootstrap 1 sigma) vs the build's "
    "Delta for the same primaries (D05) and the build's published ML (all primaries) ===")
for nm in names:
    for qc in (5.0, 6.0):
        if qc == 6.0 and not (nm.endswith("_f1") or "eff" in nm):
            continue
        M, truth, N0 = model_for(nm, qc)
        r_ = res[f"{nm}_q{qc:.0f}"]
        bset = BUILD.get(nm)
        tb = np.load(f"{ROOT}/data_real/step2/truth_{bset}.npz") if bset else None
        for sv, fs in SURV.items():
            for cn in ("a", "b", "c", "d", "e"):
                L = lin_vectors(M, truth, N0, cn, fs)
                fac = 1.0                                              # F, J at this f_sky: already in its sigma
                out = {}
                for p in ("S8", "Om", "s8"):
                    v = L[p] * fac
                    out[p] = (float(v @ r_["D"].ravel()), float((r_["Db"].reshape(NBOOT, -1) @ v).std()))
                    if tb is not None:
                        out[p + "_b05"] = float(v @ tb[f"D05_q{qc:.0f}"].ravel())
                        out[p + "_ball"] = float(v @ tb[f"D_q{qc:.0f}"].ravel())
                LIN[(nm, qc, sv, cn)] = out
                ml = build_ml(bset, cn, sv, qc) if bset else None
                bml = f"{ml['b_S8']:+.2f}" if ml else "  -  "
                if tb is not None:
                    log(f"  q{qc:.0f} {sv:4s} {cn} [{nm:15s}] S8 {out['S8'][0]:+.2f} +- {out['S8'][1]:.2f} (build D05 lin "
                        f"{out['S8_b05']:+.2f}, all-primaries lin {out['S8_ball']:+.2f}, ML {bml}; pull "
                        f"{(out['S8'][0]-out['S8_b05'])/out['S8'][1]:+.1f}) | Om {out['Om'][0]:+.2f} +- {out['Om'][1]:.2f} "
                        f"(build {out['Om_b05']:+.2f}) | s8 {out['s8'][0]:+.2f} +- {out['s8'][1]:.2f} (build {out['s8_b05']:+.2f})")
                else:
                    log(f"  q{qc:.0f} {sv:4s} {cn} [{nm:15s}] S8 {out['S8'][0]:+.2f} +- {out['S8'][1]:.2f} | Om {out['Om'][0]:+.2f} "
                        f"+- {out['Om'][1]:.2f} | s8 {out['s8'][0]:+.2f} +- {out['s8'][1]:.2f}")
log(f"   [{time.time()-t0:.0f}s]")

# ------------------------------------------------------------------ linear decomposition (as the build's p06b) of the map Delta
log("\n=== (D) linear decomposition of the map Delta (amplitude / z tilt / q shape), S8 bias, SO40, q_cut 5 (bootstrap 1 sigma) ===")
for nm in ("white_f1", "baseline_f1", "goal_f1", "baseline_effi"):
    M, truth, N0 = model_for(nm, 5.0)
    r_ = res[f"{nm}_q5"]
    for cn in ("a", "b", "d"):
        L = lin_vectors(M, truth, N0, cn, 0.4)
        v = L["S8"]

        def parts(D):
            eps = D.sum() / N0.sum(); rz = D.sum(1) / N0.sum(1)
            amp = eps * N0; zt = (rz[:, None] - eps) * N0
            return [v @ amp.ravel(), v @ zt.ravel(), v @ (D - amp - zt).ravel()]
        c0 = parts(r_["D"])
        cb = np.array([parts(Db) for Db in r_["Db"][:400]])
        log(f"  [{nm:13s}] {cn}: amplitude {c0[0]:+.2f} ({cb[:,0].std():.2f}) + z tilt {c0[1]:+.2f} ({cb[:,1].std():.2f}) + "
            f"q shape {c0[2]:+.2f} ({cb[:,2].std():.2f}) = {sum(c0):+.2f}")

# ------------------------------------------------------------------ ML fits on the central vector
log("\n=== (E) ML fits of the naive model to N0 + Delta_map (central vector), the build's fitter (s2lib.run_fit) ===")
FITS = []
for nm in names:
    for qc in (5.0, 6.0):
        if qc == 6.0 and nm not in ("baseline_f1", "goal_f1", "white_f1"):
            continue
        M, truth, N0 = model_for(nm, qc)
        D = res[f"{nm}_q{qc:.0f}"]["D"]
        for sv, fs in SURV.items():
            for cn in FIT_CFGS:
                if cn not in ("a", "d") and not nm.endswith("_f1"):
                    continue
                r = S.run_fit(M, np.clip(N0 + D, 0, None), cn, truth, fs, polish=True)
                r.update(setup=nm, qcut=qc, survey=sv)
                FITS.append(r)
                bset = BUILD.get(nm)
                ml = build_ml(bset, cn, sv, qc) if bset else None
                nui = " ".join(f"{p}:{r['b_'+p]:+.2f}" for p in r["free"][2:])
                lin = LIN.get((nm, qc, sv, cn))
                log(f"  q{qc:.0f} {sv:4s} {cn} [{nm:15s}] ML S8 {r['b_S8']:+.2f} Om {r['b_Om']:+.2f} s8 {r['b_s8']:+.2f} | {nui} | "
                    f"chi2min {r['dev']:.2f} power {r['power']:.3f}"
                    + (f" | map lin S8 {lin['S8'][0]:+.2f} +- {lin['S8'][1]:.2f}" if lin else "")
                    + (f" || build ML S8 {ml['b_S8']:+.2f} Om {ml['b_Om']:+.2f} s8 {ml['b_s8']:+.2f} "
                       + " ".join(f"{p}:{ml['b_'+p]:+.2f}" for p in ml['free'][2:]) + f" chi2min {ml['dev']:.2f} power {ml['power']:.3f}"
                       if ml else ""))
json.dump(FITS, open(os.path.join(OUT, "w04_fits.json"), "w"), indent=1)
np.savez(os.path.join(OUT, "w04_delta.npz"), names=np.array(names), area=area, scale=SCALE,
         **{f"{k}_D": v["D"] for k, v in res.items()}, **{f"{k}_Ns": v["Ns"] for k, v in res.items()},
         **{f"{k}_Dboot": v["Db"].astype(np.float32) for k, v in res.items()})
json.dump({f"{k[0]}|{k[1]}|{k[2]}|{k[3]}": v for k, v in LIN.items()}, open(os.path.join(OUT, "w04_lin.json"), "w"), indent=0)
log(f"\n[w04] done {time.time()-t0:.0f}s")
