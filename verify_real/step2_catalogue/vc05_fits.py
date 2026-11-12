"""vc05: cosmology biases with MY count model and MY fitter, for MY direct-route Delta and for the build's Delta.

data = mu_mine(truth) + Delta (full sky), scaled by fsky;  Delta in {mine (vc04, mean of the skies), build (truth_*.npz)}.
(a) ML: Nelder-Mead in Fisher-whitened coordinates on the exact Poisson lnL (+ prior), then Newton with the
    finite-difference Hessian of lnL;   (b) linearised estimator dtheta = F^-1 J^T (d - mu)/mu.
Bias in sigma = (fit - truth) / Fisher sigma at the truth (marginal, incl. the prior).
MC error of the biases from MY Delta: the linearised map applied to the 48 x n_sky block values (blocks i.i.d.).
Also: Fisher sigmas vs the build; self route (my model on its own expectation, from the truth and 2 sigma off; my model
on the build's naive N0); Q3's own data vector (truth_catalogue_v2) with my white model vs the c12_fits_v2 logs.
Usage: vc05_fits.py [nproc]   Output: out/fits_mine.json + log.
"""
import os, sys, time, json
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
os.environ["VC_THREADS"] = "1"
import vclib as V
import numpy as np

NPROC = int(sys.argv[1]) if len(sys.argv) > 1 else 1
D2 = f"{V.ROOT}/data_real/step2"
SURV = {"SO40": 0.4, "full": 1.0}
STEPS = np.array([0.001, 0.001, 0.003, 0.003, 0.003])          # FD steps for J (Om, s8, omb, alpha, slnY)
# setup -> (filter, r-kind, truth); ML fits for FIT_SETUPS, linearised for all
SET = {"white": ("white", 0, "FID"), "baseline_f1": ("baseline", 0, "FID"), "goal_f1": ("goal", 0, "FID"),
       "baseline_effi": ("baseline", 1, "FID"), "lowP_f1": ("baseline", 0, "LOW"),
       "baseline_f0.5": ("baseline", 0, "FID"), "baseline_f2": ("baseline", 0, "FID"), "goal_f0.5": ("goal", 0, "FID"),
       "goal_f2": ("goal", 0, "FID"), "baseline_effii": ("baseline", 2, "FID"), "goal_effi": ("goal", 3, "FID"),
       "goal_effii": ("goal", 4, "FID"), "lowP_effi": ("baseline", 5, "LOW")}
FIT_SETUPS = ["white", "baseline_f1", "goal_f1", "baseline_effi", "lowP_f1"]
MINE_KEY = {"white": "white_f1"}
_M = {}


def model(setup, qc):
    import vcmodel as VM
    key = (setup, qc)
    if key not in _M:
        filt, rk, tr = SET[setup]
        r = None
        if rk:
            rth, RT = V.r_tables(); r = (rth, RT[rk])
        _M[key] = VM.CountModel(filt, qc, r)
    return _M[key]


def truth_of(setup):
    return V.FID.copy() if SET[setup][2] == "FID" else V.FID_LOW.copy()


def deltas(setup, qc):
    tag = f"{qc:.0f}"
    dm = np.load(f"{HERE}/out/delta_mine.npz")
    k = MINE_KEY.get(setup, setup)
    Dm = dm[f"{k}_q{tag}_D"]; blocks = dm[f"{k}_q{tag}_blocks"].astype(np.float64); nsky = len(dm["seeds"])
    Db = np.load(f"{D2}/truth_{setup}.npz")[f"D_q{tag}"]
    return Dm, blocks, nsky, Db


def biases(Fo, x, sig, sS8):
    th = Fo.full(x)
    import vcmodel as VM
    b = {V.PN[i]: float((th[i] - Fo.truth[i]) / sig[k]) for k, i in enumerate(Fo.free)}
    b["S8"] = float((VM.S8_of(th[0], th[1]) - VM.S8_of(Fo.truth[0], Fo.truth[1])) / sS8)
    return b


def task(t):
    import vcmodel as VM
    kind = t[0]
    t0 = time.time()
    if kind == "fit":
        _, setup, qc, sv, cfg, dname = t
        M = model(setup, qc); truth = truth_of(setup); fs = SURV[sv]
        mu_t = M.counts(truth)
        if dname == "self":
            D = np.zeros_like(mu_t)
        elif dname == "selfN0":
            D = np.load(f"{D2}/truth_{setup}.npz")[f"N0_q{qc:.0f}"] - mu_t
        elif dname == "Q3":
            D = np.load(f"{V.ROOT}/data_cosmo/truth_catalogue_v2.npz")[f"D_q{qc:.0f}"]
        else:
            Dm, blocks, nsky, Db = deltas(setup, qc)
            D = Dm if dname == "mine" else Db
        data = np.clip(mu_t + D, 0, None) * fs
        Fo = VM.Fit(M, data, cfg, truth, fs)
        steps = STEPS[Fo.free]
        dx, F, J, mu = Fo.linearised(steps)
        sig, sS8, g, C = Fo.sigmas(F)
        out = dict(kind="fit", setup=setup, qcut=qc, survey=sv, config=cfg, data=dname, s=dict(zip([V.PN[i] for i in Fo.free], sig.tolist())),
                   s_S8=sS8)
        lin = {V.PN[i]: float(dx[k] / sig[k]) for k, i in enumerate(Fo.free)}
        lin["S8"] = float((g @ dx) / sS8)
        out["lin"] = lin
        starts = [None]
        if dname == "self":
            starts.append(Fo.truth[Fo.free] + 2 * sig * np.array([1, -1, 1, -1, 1][:len(Fo.free)]))
        res = []
        nm_info = []
        for x0 in starts:
            xnm, Lnm = Fo.fit_nm(F, x0=x0, steps=steps)
            nm_info.append(Fo.nm_info)
            xn, Ln = Fo.newton_polish(xnm, 0.05 * sig)
            res.append((x0 is not None, biases(Fo, xnm, sig, sS8), biases(Fo, xn, sig, sS8), Ln - Lnm,
                        [V.PN[i] for k, i in enumerate(Fo.free) if xn[k] <= Fo.lo[k] + 1e-9 or xn[k] >= Fo.hi[k] - 1e-9]))
        out["nm"] = res[0][1]; out["ml"] = res[0][2]; out["newton_gain"] = res[0][3]; out["at_bound"] = res[0][4]
        out["nm_info"] = nm_info
        if len(res) > 1:
            out["ml_offset_start"] = res[1][2]
        xml = Fo.truth[Fo.free] + np.array([res[0][2][V.PN[i]] for i in Fo.free]) * sig
        out["dev"] = Fo.deviance(xml)
        from scipy.stats import chi2 as _c2, ncx2 as _nc
        ndof = data.size - len(Fo.free)
        out["power"] = float(_nc.sf(_c2.ppf(0.95, ndof), ndof, max(out["dev"], 1e-9)))
        out["nev"] = Fo.nev
        if dname == "mine":
            # MC error: linearised S8 (and each parameter) bias applied to the block values (mean over skies)
            Dm, blocks, nsky, Db = deltas(setup, qc)
            A = np.linalg.solve(F, (J / mu[:, None]).T) * fs           # (nfree, nbin): dx per unit full-sky Delta
            P = blocks.reshape(len(blocks), -1) @ A.T                     # (nblk, nfree)
            nb_ = len(blocks)
            errs = np.sqrt(nb_ * P.var(0, ddof=1)) / nsky
            out["mc_err"] = {V.PN[i]: float(errs[k] / sig[k]) for k, i in enumerate(Fo.free)}
            pS8 = P @ g
            out["mc_err"]["S8"] = float(np.sqrt(nb_ * pS8.var(ddof=1)) / nsky / sS8)
            # sky-to-sky check of the S8 error (one value per sky)
            per_sky = pS8.reshape(nsky, -1).sum(1) / sS8
            out["mc_err_sky"] = float(per_sky.std(ddof=1) / np.sqrt(nsky)) if nsky > 1 else None
        out["sec"] = time.time() - t0
        return out
    if kind == "lin":
        _, setup, qc, sv, cfg = t
        M = model(setup, qc); truth = truth_of(setup); fs = SURV[sv]
        mu_t = M.counts(truth)
        Dm, blocks, nsky, Db = deltas(setup, qc)
        out = dict(kind="lin", setup=setup, qcut=qc, survey=sv, config=cfg)
        for dname, D in (("mine", Dm), ("build", Db)):
            Fo = VM.Fit(M, np.clip(mu_t + D, 0, None) * fs, cfg, truth, fs)
            dx, F, J, mu = Fo.linearised(STEPS[Fo.free])
            sig, sS8, g, C = Fo.sigmas(F)
            out[f"lin_{dname}"] = dict({V.PN[i]: float(dx[k] / sig[k]) for k, i in enumerate(Fo.free)}, S8=float((g @ dx) / sS8))
            # linearised chi2_min = r.r - b F^-1 b (r = (d - mu)/sqrt(mu), b = J^T (d - mu)/mu) and the 95% test power
            from scipy.stats import chi2 as _c2, ncx2 as _nc
            rr_ = (Fo.d.ravel() - mu) / np.sqrt(mu); bb_ = J.T @ ((Fo.d.ravel() - mu) / mu)
            c2min = float(rr_ @ rr_ - bb_ @ np.linalg.solve(F, bb_)); ndof = mu.size - len(Fo.free)
            out[f"chi2lin_{dname}"] = c2min
            out[f"power_{dname}"] = float(_nc.sf(_c2.ppf(0.95, ndof), ndof, max(c2min, 1e-9)))
            if dname == "mine":
                A = np.linalg.solve(F, (J / mu[:, None]).T) * fs
                pS8 = (blocks.reshape(len(blocks), -1) @ A.T) @ g
                out["mc_err_S8"] = float(np.sqrt(len(blocks) * pS8.var(ddof=1)) / nsky / sS8)
            out["s_S8"] = sS8
        out["sec"] = time.time() - t0
        return out
    if kind == "decomp":
        # linearised S8 bias split into amplitude, z tilt and q shape (the build's p06b definitions), my J and F
        _, setup, qc, sv, cfg = t
        M = model(setup, qc); truth = truth_of(setup); fs = SURV[sv]
        mu_t = M.counts(truth)
        Dm, blocks, nsky, Db = deltas(setup, qc)
        out = dict(kind="decomp", setup=setup, qcut=qc, survey=sv, config=cfg)
        Fo = VM.Fit(M, mu_t * fs, cfg, truth, fs)
        dx0, F, J, mu = Fo.linearised(STEPS[Fo.free])
        sig, sS8, g, C = Fo.sigmas(F)
        for dname, D in (("mine", Dm), ("build", Db)):
            eps = D.sum() / mu_t.sum(); rz = D.sum(1) / mu_t.sum(1)
            parts = {"amplitude": eps * mu_t, "ztilt": (rz[:, None] - eps) * mu_t}
            parts["qshape"] = D - parts["amplitude"] - parts["ztilt"]
            parts["sum"] = D
            res = {}
            for k, Dk in parts.items():
                x = np.linalg.solve(F, J.T @ ((Dk * fs).ravel() / mu))
                res[k] = dict(S8=float(g @ x / sS8), Om=float(x[0] / sig[0]), s8=float(x[1] / sig[1]))
            out[dname] = res
        return out
    if kind == "steps":
        # Fisher / linearised stability: steps x 2
        _, setup, qc, sv, cfg = t
        M = model(setup, qc); truth = truth_of(setup); fs = SURV[sv]
        mu_t = M.counts(truth)
        Dm, blocks, nsky, Db = deltas(setup, qc)
        out = dict(kind="steps", setup=setup, qcut=qc, survey=sv, config=cfg)
        for mult in (1.0, 2.0, 0.5):
            Fo = VM.Fit(M, np.clip(mu_t + Db, 0, None) * fs, cfg, truth, fs)
            dx, F, J, mu = Fo.linearised(STEPS[Fo.free] * mult)
            sig, sS8, g, C = Fo.sigmas(F)
            out[f"x{mult}"] = dict(s_S8=sS8, lin_S8=float((g @ dx) / sS8))
        return out


def task_pair(t):
    r = task(t)
    if "sec" not in r:
        r["sec"] = 0.0
    return t, r


def main():
    t0 = time.time()
    tasks = []
    for s in FIT_SETUPS:
        for sv in SURV:
            for cfg in ("a", "d"):
                for dn in ("mine", "build"):
                    tasks.append(("fit", s, 5.0, sv, cfg, dn))
            for dn in ("mine", "build"):
                tasks.append(("fit", s, 6.0, sv, "a", dn))
    for sv in SURV:
        for dn in ("mine", "build"):
            for s in ("baseline_f1", "goal_f1"):
                for cfg in ("b", "c", "e"):
                    tasks.append(("fit", s, 5.0, sv, cfg, dn))
            for s in ("baseline_f0.5", "baseline_f2", "goal_f0.5", "goal_f2", "baseline_effii", "goal_effi", "goal_effii", "lowP_effi"):
                tasks.append(("fit", s, 5.0, sv, "a", dn))
            for s in ("baseline_f0.5", "baseline_f2", "goal_f0.5", "goal_f2"):
                tasks.append(("fit", s, 5.0, sv, "d", dn))
            for s in ("goal_f1",):
                tasks.append(("fit", s, 6.0, sv, "d", dn))
    for s in ("white", "baseline_f1", "goal_f1", "baseline_effi", "lowP_f1"):
        for cfg in ("a", "b", "c", "d", "e"):
            tasks.append(("decomp", s, 5.0, "SO40", cfg))
    for s in ("white", "baseline_f1", "goal_f1", "lowP_f1", "baseline_effi"):
        for sv in SURV:
            for cfg in ("a", "d"):
                tasks.append(("fit", s, 5.0, sv, cfg, "self"))
                tasks.append(("fit", s, 5.0, sv, cfg, "selfN0"))
    for qc in (5.0, 6.0):
        for sv in SURV:
            for cfg in ("a", "b", "c", "d", "e"):
                tasks.append(("fit", "white", qc, sv, cfg, "Q3"))
    for s in SET:
        for qc in (5.0, 6.0):
            for sv in SURV:
                for cfg in ("a", "b", "c", "d", "e"):
                    tasks.append(("lin", s, qc, sv, cfg))
    for cfg in ("a", "d"):
        for sv in SURV:
            tasks.append(("steps", "baseline_f1", 5.0, sv, cfg))
    cost = lambda t: (60 if t[0] == "fit" and t[4] in ("d", "e") else 15 if t[0] == "fit" else 2) * (2 if t[0] == "fit" and t[5] == "self" else 1)
    tasks.sort(key=lambda t: -cost(t))
    jl = f"{HERE}/out/fits_mine.jsonl"
    rows = [json.loads(l) for l in open(jl)] if os.path.exists(jl) else []
    done = {json.dumps(r["task"]) for r in rows}
    todo = [t for t in tasks if json.dumps(list(t)) not in done]
    print(f"{len(tasks)} tasks ({len(rows)} already done) on {NPROC} processes", flush=True)

    def record(k, t, r):
        r["task"] = list(t)
        rows.append(r)
        with open(jl, "a") as fh:
            fh.write(json.dumps(r) + "\n")
        extra = ""
        if r.get("kind") == "fit":
            extra = f" nev {r['nev']} nm {r.get('nm_info')} S8 ml {r['ml']['S8']:+.3f} lin {r['lin']['S8']:+.3f}"
        print(f"   {k+1}/{len(todo)} {t} {r.get('sec', 0):.0f}s{extra} [{time.time()-t0:.0f}s]", flush=True)
    if NPROC > 1:
        from multiprocessing import get_context
        with get_context("spawn").Pool(NPROC) as p:
            for k, (t, r) in enumerate(p.imap_unordered(task_pair, todo, chunksize=1)):
                record(k, t, r)
    else:
        for k, t in enumerate(todo):
            record(k, t, task(t))
    json.dump(rows, open(f"{HERE}/out/fits_mine.json", "w"), indent=1)
    print(f"done {len(rows)} tasks in {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
