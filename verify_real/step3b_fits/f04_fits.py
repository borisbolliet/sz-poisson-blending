"""f04 (step-3b verification, lens 'fits', tasks c and d): cosmology biases with MY count model and MY fitter.

Model: f00lib.KModel (my HMF grid f01b with smooth Tinker08, sigma_eff = rbar sigma_Y, the build's kernel table with my 'alt' interpolation:
PCHIP in q', linear in ln theta, lognormal scatter by exact hat-function weights).  Data = mu_mine(truth) x R (Asimov,
full sky, x f_sky).  R from f02 (mine: R_s with my top-hat cells, R with the bilinear and the alt N_exp, R_p) or the
build's datavec_*.npz (R_s, R, R_p), read as data.
  ml : ML fit (Nelder-Mead in Fisher-whitened coordinates on the exact Poisson lnL + prior, restarted, started at the
       linearised solution; then Newton with the finite-difference Hessian of lnL).  Bias = (fit - truth) / Fisher sigma
       at the truth (marginal, incl. the prior); S8 = s8 sqrt(Om/0.3).  Deviance and 95% goodness-of-fit power.
  lin: linearised biases F^-1 J^T (R - 1) for every estimator, the tile-bootstrap error (my 2,000 resamples of f02 for my
       estimators; the build's stored resamples for the build's), p06b's split (amplitude, z tilt, q shape).
  mlb: as ml with the model's kernel interpolated bilinearly (the build's scheme), headline, q_cut 5.
  an : the analytic analyst (task d): Phi(sqrt(q'^2 + k) - e), zero below q' = 2, same sigma_eff; fitted to
       (i) mu_K(truth) x R_s (blended) and (ii) mu_K(truth) (isolated clusters only), mu_K = my measured-kernel model.
Usage: f04_fits.py NPROC [kind ...]    Output: out/f04_rows.jsonl (resumable), out/f04_rows.json.
"""
import os, sys, time, json
os.environ["VF_THREADS"] = "1"
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import numpy as np

NPROC = int(sys.argv[1]) if len(sys.argv) > 1 else 3
KINDS = sys.argv[2:] or ["lin", "ml", "an", "self"]
SETUPS = {"prod_none": "prod", "prod_fof10": "prod", "szifi3000_none": "szifi3000", "prod_none_q2": "prod"}
_W = {}


def model(variant, mode, qc, kern="K", k=None, how="alt"):
    import f00lib as L
    key = (variant, mode, qc, kern, k, how)
    if key not in _W:
        for kk in [kk for kk in _W if kk[0] != "dv"][:-3]:
            del _W[kk]
        if kern == "K":
            Kt = L.KTable(os.path.join(L.D3B, f"kernel_{variant}_{mode}.npz"))
        else:
            Kt = L.AnalyticKernel(k, cl=True)
        M = L.KModel(mode, qc, L.rbar(variant, mode), Kt, how=how)
        _W[key] = (M, M.counts(L.FID))
    return _W[key]


def datavec(setup, mode):
    import f00lib as L
    key = ("dv", setup, mode)
    if key not in _W:
        mine = np.load(os.path.join(L.OUT, f"dv_{setup}_{mode}.npz"))
        bld = np.load(os.path.join(L.D3B, f"datavec_{setup}_{mode}.npz"))
        d = {}
        for s in ("q5", "q6"):
            d[("mine", "Rs", s)] = mine[f"Rs_{s}"]; d[("mine", "Rsboot", s)] = mine[f"Rsboot_{s}"]
            d[("mine", "R", s)] = mine[f"R_bilin_{s}"]; d[("mine", "Rboot", s)] = mine[f"Rboot_bilin_{s}"]
            d[("mine", "Ralt", s)] = mine[f"R_alt_{s}"]; d[("mine", "Raltboot", s)] = mine[f"Rboot_alt_{s}"]
            d[("mine", "Rp", s)] = mine[f"Rp_{s}"]; d[("mine", "Rpboot", s)] = mine[f"Rpboot_{s}"]
            for e in ("Rs", "R", "Rp"):
                d[("build", e, s)] = bld[f"{e}_{s}"]; d[("build", e + "boot", s)] = bld[f"{e}boot_{s}"]
        _W[key] = d
    return _W[key]


def biases(Fo, x, sig, sS8):
    import f00lib as L
    th = Fo.full(x)
    b = {L.PN[i]: float((th[i] - Fo.truth[i]) / sig[k]) for k, i in enumerate(Fo.free)}
    b["S8"] = float((L.S8_of(th[0], th[1]) - L.S8_of(Fo.truth[0], Fo.truth[1])) / sS8)
    return b


def ml_fit(M, data_full, cfg, fs, start="lin"):
    import f00lib as L
    data = np.clip(data_full, 0, None) * fs
    Fo = L.Fit(M, data, cfg, L.FID, fs)
    steps = L.STEPS[Fo.free]
    xt = Fo.truth[Fo.free]
    F, J, mu = Fo.fisher(xt, steps)
    sig, sS8, g, C = Fo.sigmas(F)
    ok = mu > 0
    dx = np.linalg.solve(F, J[ok].T @ ((data.ravel()[ok] - mu[ok]) / mu[ok]))
    lin = {L.PN[i]: float(dx[k] / sig[k]) for k, i in enumerate(Fo.free)}
    lin["S8"] = float((g @ dx) / sS8)
    x0 = np.clip(xt + dx, Fo.lo, Fo.hi) if start == "lin" else None
    xnm, Lnm = Fo.fit_nm(F, x0=x0, steps=steps)
    xn, Ln = Fo.newton_polish(xnm, 0.05 * sig)
    out = dict(lin=lin, nm=biases(Fo, xnm, sig, sS8), ml=biases(Fo, xn, sig, sS8), newton_gain=Ln - Lnm,
               at_bound=[L.PN[i] for k, i in enumerate(Fo.free) if xn[k] <= Fo.lo[k] + 1e-9 or xn[k] >= Fo.hi[k] - 1e-9],
               theta=[float(v) for v in Fo.full(xn)], s_S8=sS8, s=dict(zip([L.PN[i] for i in Fo.free], sig.tolist())),
               nm_info=Fo.nm_info, nev=Fo.nev)
    out["dev"] = Fo.deviance(xn)
    out["power"] = L.power_of(out["dev"], data.size - len(Fo.free))
    return out


def task(t):
    import f00lib as L
    t0 = time.time()
    kind = t[0]
    if kind in ("ml", "self", "mlb"):
        _, setup, mode, src, est, qc, sv, cfg = t
        variant = SETUPS[setup]
        M, N0 = model(variant, mode, qc, how="bilin" if kind == "mlb" else "alt")
        if kind == "self":
            data = N0.copy()
        else:
            data = N0 * datavec(setup, mode)[(src, est, f"q{qc:.0f}")]
        r = ml_fit(M, data, cfg, L.SURV[sv])
        if kind == "self" or (setup == "prod_none" and src == "mine" and est == "Rs" and cfg in ("d", "e") and qc == 5.0):
            r2 = ml_fit(M, data, cfg, L.SURV[sv], start="truth")       # convergence: start at the truth
            r["ml_from_truth"] = r2["ml"]; r["nev"] += r2["nev"]
    elif kind == "an":
        _, variant, mode, k, route, src, sv, cfg = t
        Mk, Nk = model(variant, mode, 5.0, kern="A", k=k)
        Mv, Nv = model(variant, mode, 5.0)
        dv = datavec(f"{variant}_none", mode)
        Rs = dv[(src, "Rs", "q5")]
        data = Nv * Rs if route == "blended" else Nv.copy()
        r = ml_fit(Mk, data, cfg, L.SURV[sv])
        if route == "blended":
            # tile-bootstrap error of (i), linearised with the analytic model at the truth
            fs = L.SURV[sv]
            Fo = L.Fit(Mk, Nk * fs, cfg, L.FID, fs)
            F, J, mu = Fo.fisher(Fo.truth[Fo.free], L.STEPS[Fo.free])
            sig, sS8, g, C = Fo.sigmas(F)
            RB = dv[(src, "Rsboot", "q5")].astype(float)
            X = ((RB - Rs) * Nv / np.maximum(Nk, 1e-300)).reshape(RB.shape[0], -1)
            x = np.linalg.solve(F, J.T @ X.T).T
            r["boot_S8"] = float((x @ g / sS8).std())
    elif kind == "lin":
        _, setup, mode, qc, sv, cfg = t
        variant = SETUPS[setup]
        M, N0 = model(variant, mode, qc)
        fs = L.SURV[sv]
        Fo = L.Fit(M, N0 * fs, cfg, L.FID, fs)
        F, J, mu = Fo.fisher(Fo.truth[Fo.free], L.STEPS[Fo.free])
        sig, sS8, g, C = Fo.sigmas(F)
        A = np.linalg.solve(F, J.T)                                        # dx per unit (R - 1) in each bin
        dv = datavec(setup, mode)
        s = f"q{qc:.0f}"
        r = dict(s_S8=sS8, s=dict(zip([L.PN[i] for i in Fo.free], sig.tolist())))
        for src, est in (("mine", "Rs"), ("mine", "R"), ("mine", "Ralt"), ("mine", "Rp"), ("build", "Rs"), ("build", "R"), ("build", "Rp")):
            R = dv[(src, est, s)]; RB = dv[(src, est + "boot", s)].astype(float)
            dx = A @ (R - 1).ravel(); dxb = (RB - 1).reshape(RB.shape[0], -1) @ A.T
            r[f"{src}_{est}"] = dict(S8=float(g @ dx / sS8), Om=float(dx[0] / sig[0]), s8=float(dx[1] / sig[1]),
                                     boot_S8=float((dxb @ g / sS8).std()), boot_mean_S8=float((dxb @ g / sS8).mean()))
            if est == "Rs":                                               # p06b split
                D = N0 * (R - 1)
                eps = D.sum() / N0.sum(); rz = D.sum(1) / np.maximum(N0.sum(1), 1e-300)
                parts = {"amplitude": eps * N0, "ztilt": (rz[:, None] - eps) * N0}
                parts["qshape"] = D - parts["amplitude"] - parts["ztilt"]
                r[f"{src}_{est}"]["parts"] = {k: float(g @ (A @ (Dk / np.where(N0 > 0, N0, 1)).ravel()) / sS8) for k, Dk in parts.items()}
                if cfg in ("d", "e"):
                    r[f"{src}_{est}"]["nuis"] = {L.PN[i]: float(dx[k] / sig[k]) for k, i in enumerate(Fo.free)}
    r["task"] = list(t); r["sec"] = time.time() - t0
    return r


def tasks_for(kinds):
    T = []
    if "lin" in kinds:
        for setup in SETUPS:
            for mode in ("baseline", "goal"):
                for qc in (5.0, 6.0):
                    for sv in ("SO40", "full"):
                        for cfg in ("a", "d", "e"):
                            T.append(("lin", setup, mode, qc, sv, cfg))
    if "ml" in kinds:
        for setup in ("prod_none", "szifi3000_none", "prod_fof10"):
            for mode in ("baseline", "goal"):
                for qc in (5.0, 6.0):
                    for sv in ("SO40", "full"):
                        for cfg in ("a", "d", "e"):
                            T.append(("ml", setup, mode, "mine", "Rs", qc, sv, cfg))
                            if setup == "prod_none":
                                T.append(("ml", setup, mode, "build", "Rs", qc, sv, cfg))
                                if qc == 5.0:
                                    T.append(("ml", setup, mode, "mine", "R", qc, sv, cfg))
                                    T.append(("ml", setup, mode, "mine", "Rp", qc, sv, cfg))
    if "an" in kinds:
        for variant in ("prod", "szifi3000"):
            for mode in ("baseline", "goal"):
                for route in ("blended", "isolated"):
                    for sv in ("SO40", "full"):
                        for cfg in ("a", "d", "e"):
                            T.append(("an", variant, mode, 3.0, route, "mine", sv, cfg))
                            if route == "blended":
                                T.append(("an", variant, mode, 3.0, route, "build", sv, cfg))
    if "mlb" in kinds:                       # the model with the build's bilinear kernel interpolation (headline, q_cut 5)
        for mode in ("baseline", "goal"):
            for sv in ("SO40", "full"):
                for cfg in ("d", "e"):
                    for src in ("build", "mine"):
                        T.append(("mlb", "prod_none", mode, src, "Rs", 5.0, sv, cfg))
    if "self" in kinds:
        for mode in ("baseline", "goal"):
            for cfg in ("a", "d", "e"):
                T.append(("self", "prod_none", mode, "-", "-", 5.0, "SO40", cfg))
    return T


def main():
    t0 = time.time()
    import f00lib as L
    T = tasks_for(KINDS)
    cost = lambda t: (0.1 if t[0] == "lin" else (3 if t[-1] in ("d", "e") else 1) * (2 if t[0] == "self" else 1))
    T.sort(key=lambda t: -cost(t))
    jl = os.path.join(L.OUT, "f04_rows.jsonl")
    rows = [json.loads(l) for l in open(jl)] if os.path.exists(jl) else []
    done = {json.dumps(r["task"]) for r in rows}
    todo = [t for t in T if json.dumps(list(t)) not in done]
    print(f"f04: {len(T)} tasks ({len(T) - len(todo)} already done), kinds {KINDS}, {NPROC} processes", flush=True)

    def record(k, r):
        rows.append(r)
        with open(jl, "a") as fh:
            fh.write(json.dumps(r) + "\n")
        t = r["task"]
        extra = ""
        if t[0] in ("ml", "self", "an", "mlb"):
            extra = f" S8 ml {r['ml']['S8']:+.3f} (lin {r['lin']['S8']:+.3f}) nev {r['nev']} bound {r['at_bound']}"
            if "ml_from_truth" in r:
                extra += f" [from truth {r['ml_from_truth']['S8']:+.3f}]"
        print(f"   {k + 1}/{len(todo)} {t} {r['sec']:.0f}s{extra} [{time.time() - t0:.0f}s]", flush=True)
    if NPROC > 1:
        from multiprocessing import get_context
        with get_context("spawn").Pool(NPROC, maxtasksperchild=40) as p:
            for k, r in enumerate(p.imap_unordered(task, todo, chunksize=1)):
                record(k, r)
    else:
        for k, t in enumerate(todo):
            record(k, task(t))
    json.dump(rows, open(os.path.join(L.OUT, "f04_rows.json"), "w"), indent=1)
    print(f"done {len(rows)} rows in {time.time() - t0:.0f}s; parent peak RSS {L.rss_gb():.2f} GB")


if __name__ == "__main__":
    main()
