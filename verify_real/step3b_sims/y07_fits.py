"""y07 (sims lens): cosmology bias of the own finder's catalogue with step 2's verified fit machinery
(analysis/realistic/step2/s2lib.py: SmoothModelS, LnSigma, bin_tables, counts_table, Fitter3, run_fit, configs;
imported read-only as a library).  Nothing from step 3b's code.
Analyst's model: step 2's count model with sigma_eff = rbar(theta) sigma_Y (rbar = mean r after the iteration over the
own realistic tiles, linear in ln theta between the 15 scales, constant beyond) and the OWN kernel K (y05) in place of
Phi(q' - e): the kernel tables go through s2lib.bin_tables (scatter convolved, rebuilt for every sigma_lnY) and
SmoothModel.counts_table (linear in ln theta500 between the 25 nodes).
Data (Asimov) = mu(truth) x R(z, q) for R = N_det/N_exp, R_p = N_det/N_iso (y06), and R_s = mu_{K + dK}(truth) /
mu_K(truth) with the pooled dK of y06 (model route).  ML fits (run_fit: Fisher scoring + L-BFGS-B polish) for configs
a, d, e (SO40, full sky; q_cut 5; q_cut 6 for R_s); linearised biases F^-1 J^T (R - 1) for configs a-e with
tile-bootstrap errors (R, R_p: 2000 resamples; R_s: 200 N-route resamples added to the model-route value; R: also the
joint kernel+data resamples); the self check (model fitted to its own expectation) and the amplitude / z tilt / q shape
split of p06b.  Output: out/fits_<METHOD>.json.   Usage: python y07_fits.py METHOD NPROC
"""
import os, sys, time, json
HERE = os.path.dirname(os.path.abspath(__file__))
os.environ.setdefault("NUMBA_CACHE_DIR", os.path.join(HERE, "out", "numba_cache"))
for v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "NUMBA_NUM_THREADS"):
    os.environ[v] = "1"
sys.dont_write_bytecode = True
import numpy as np
ROOT = "/Users/boris/GitHub/sz-poisson-blending"
sys.path.insert(0, f"{ROOT}/analysis/realistic/step2")
sys.path.insert(0, HERE)
import s2lib as S                                   # noqa: E402  (step 2's fit machinery, read-only)
from s2lib import CL                                 # noqa: E402

TH_K = 0.3 * 2.0 ** ((np.arange(25) - 1) / 3.0)
THETAS = np.geomspace(0.5, 15.0, 15)
SURV = {"SO40": 0.4, "full": 1.0}
_W = {}


class KModel(S.SmoothModelS):
    """step 2's SmoothModelS with a measured kernel table K (25 theta nodes, q' grid QG, EDGES) in place of Phi."""

    def __init__(self, qcut, lnsig, K, QG, EDGES, grid):
        super().__init__(qcut, lnsig, grid=grid)
        self.K, self.QG, self.E = K, QG, EDGES
        self.lth = np.log(TH_K)
        self._tc = {}

    def tables(self, slnY):
        key = round(float(slnY), 12)
        if key not in self._tc:
            if len(self._tc) > 40:
                self._tc.clear()
            self._tc[key] = S.bin_tables(self, self.K, self.QG, self.E, float(slnY))
        return self._tc[key]

    def counts(self, th, fsky=1.0):
        return self.counts_table(th, self.tables(th[4]), self.lth, fsky)


def dK_grid(Ssum, nsum, QC, QG):
    with np.errstate(invalid="ignore", divide="ignore"):
        dk = np.where(nsum[:, None] > 0, Ssum / np.maximum(nsum[:, None], 1), 0.0).reshape(TH_K.size, QC.size - 1, -1)
    cq = np.clip(np.searchsorted(QC, QG, side="right") - 1, 0, QC.size - 2)
    out = dk[:, cq, :]
    out[:, QG < QC[0], :] = 0.0
    return out


def setup(meth, mode, qcut):
    key = (meth, mode, qcut)
    if key in _W:
        return _W[key]
    if "G" not in _W:
        _W["G"] = CL.HMFGrid()
    kz = np.load(os.path.join(HERE, "out", f"kernel_{meth}_{mode}.npz"))
    dv = np.load(os.path.join(HERE, "out", f"datavec_{meth}_{mode}.npz"))
    K, QG, E = kz["K"], kz["QG"], kz["EDGES"]
    s = f"q{qcut:.0f}"
    lnsig = S.LnSigma(S.sz_model(mode), THETAS, dv["rbar"])
    M = KModel(qcut, lnsig, K, QG, E, _W["G"])
    mu = M.counts(S.FID)
    QC = dv["QC"]
    data = {"R": mu * dv[f"R_{s}"], "Rp": mu * dv[f"Rp_{s}"], "Rs_N": mu * dv[f"Rs_N_{s}"], "self": mu.copy()}
    for m in ("closest1", "fof10", "keepnr"):
        data[f"Rp_{m}"] = mu * dv[f"Rp_{m}_{s}"]
    for m in ("none", "closest1", "fof10"):
        dk = dK_grid(dv[f"S_{m}_{s}"], dv[f"ncell_{s}"], QC, QG)
        Ms = KModel(qcut, lnsig, K + dk, QG, E, _W["G"])
        data["Rs" if m == "none" else f"Rs_{m}"] = Ms.counts(S.FID)
    _W[key] = (M, mu, data)
    return _W[key]


def analytic_model(meth, mode, qcut, k, zero=True):
    """the analytic analyst: Phi(sqrt(q'^2 + k) - e), zero below the credit limit q' = 2 (zero=True), same sigma_eff."""
    key = ("an", meth, mode, qcut, k, zero)
    if key in _W:
        return _W[key]
    M, mu, data = setup(meth, mode, qcut)
    from scipy.special import ndtr
    QG, E = M.QG, M.E
    Ka = ndtr(np.sqrt(QG[:, None] ** 2 + k) - E[None, :]) * ((QG[:, None] >= 2.0) if zero else 1.0)
    Ka = np.broadcast_to(Ka, (TH_K.size,) + Ka.shape).copy()
    Ma = KModel(qcut, M.sigY, Ka, QG, E, _W["G"])
    _W[key] = Ma
    return Ma


def job(task):
    meth, mode, qcut, est, cn, sv = task
    t0 = time.time()
    M, mu, data = setup(meth, mode, qcut)
    if est.startswith("an"):                     # an<k>_cat (i: catalogue = mu x R_s) / an<k>_iso (ii: mu); anz: no zero
        zero = not est.startswith("anz")
        k = float(est[3 if not zero else 2:].split("_")[0])
        Ma = analytic_model(meth, mode, qcut, k, zero)
        d = np.clip(data["Rs"] if est.endswith("cat") else mu, 0, None)
        r = S.run_fit(Ma, d, cn, S.FID, SURV[sv], polish=True)
        r.update(meth=meth, mode=mode, qcut=qcut, est=est, survey=sv, sec=time.time() - t0)
        return r
    d = np.clip(data[est], 0, None)
    r = S.run_fit(M, d, cn, S.FID, SURV[sv], polish=True)
    r.update(meth=meth, mode=mode, qcut=qcut, est=est, survey=sv, sec=time.time() - t0)
    return r


def linearised(meth, mode, qcut, log):
    """linearised S8 (Om, s8) biases with bootstrap errors, every config and survey; p06b split for configs a, d."""
    M, mu, data = setup(meth, mode, qcut)
    dv = np.load(os.path.join(HERE, "out", f"datavec_{meth}_{mode}.npz"))
    s = f"q{qcut:.0f}"
    Rs = data["Rs"] / mu
    boots = {"R": dv[f"Rboot_{s}"], "Rp": dv[f"Rpboot_{s}"],
             "Rs": Rs[None] + (dv[f"Rs_Nboot_{s}"] - dv[f"Rs_N_{s}"][None])}
    for m in ("closest1", "fof10", "keepnr"):
        boots[f"Rp_{m}"] = dv[f"Rpboot_{m}_{s}"]
    if f"Rjboot_{s}" in dv.files:
        boots["Rjoint"] = dv[f"Rjboot_{s}"]
    cen = {k: data[k] / mu for k in data}
    rows = []
    for sv, fs in SURV.items():
        for cn in ("a", "b", "c", "d", "d3", "d5", "e"):
            free, pri = S.config(cn, S.FID)
            fi = [CL.PNAMES.index(p) for p in free]
            F_ = S.Fitter3(M, mu * fs, free, fsky=fs, priors=pri, truth=S.FID)
            xt = S.FID[fi]
            Ft = F_.fisher(xt); J = F_.jac(xt)
            sig, Cov, gS = S.derived_sigma3(Ft, fi, S.FID)
            lin = lambda R: np.linalg.solve(Ft, J.T @ (np.atleast_2d(R).reshape(-1, mu.size).T - 1.0))   # (nfree, B)
            cen2 = dict(cen)
            if "Rjoint" in boots:
                cen2["Rjoint"] = cen["R"]
            for est, R in cen2.items():
                x = lin(R)[:, 0]
                row = dict(meth=meth, mode=mode, qcut=qcut, survey=sv, config=cn, est=est,
                           lin_S8=float(gS @ x / sig["S8"]), lin_Om=float(x[0] / sig["Om"]), lin_s8=float(x[1] / sig["s8"]),
                           s_S8=sig["S8"])
                if est in boots:
                    X = lin(boots[est])
                    row["boot_S8"] = float(np.std(gS @ X / sig["S8"]))
                    row["boot_Om"] = float(np.std(X[0] / sig["Om"])); row["boot_s8"] = float(np.std(X[1] / sig["s8"]))
                if est == "R" and qcut == 5.0 and "R_joint_boot_q5" in dv.files:
                    pass
                if cn in ("a", "d", "e") and est in ("R", "Rp", "Rs"):
                    Dl = mu * (R - 1)
                    eps = Dl.sum() / mu.sum(); rz = Dl.sum(1) / mu.sum(1)
                    parts = {"amplitude": eps * mu, "z tilt": (rz[:, None] - eps) * mu}
                    parts["q shape"] = Dl - parts["amplitude"] - parts["z tilt"]
                    row["parts"] = {k: float(gS @ np.linalg.solve(Ft, J.T @ (v / mu).ravel()) / sig["S8"]) for k, v in parts.items()}
                    xx = np.linalg.solve(Ft, J.T @ (R - 1).ravel())
                    row["nuis"] = {CL.PNAMES[i]: float(xx[k] / sig[CL.PNAMES[i]]) for k, i in enumerate(fi)}
                rows.append(row)
    return rows


def main():
    meth = sys.argv[1]; nproc = int(sys.argv[2])
    sys.path.insert(0, HERE)
    import vlib3b as V
    log = V.Tee(os.path.join(HERE, f"y07_fits_{meth}.log"))
    t00 = time.time()
    modes = ("baseline", "goal")
    tasks = [(meth, m, 5.0, est, cn, sv) for m in modes for est in ("Rs", "Rp", "R") for cn in ("a", "d", "e") for sv in SURV]
    tasks += [(meth, m, 6.0, "Rs", cn, "SO40") for m in modes for cn in ("a", "d", "e")]
    tasks += [(meth, m, 5.0, "self", cn, "SO40") for m in modes for cn in ("a", "d")]
    if meth == "whitened":
        tasks += [(meth, m, 5.0, est, "d", "SO40") for m in modes for est in ("Rs_closest1", "Rs_fof10", "Rp_keepnr")]
        tasks += [(meth, m, 5.0, est, "a", "SO40") for m in modes for est in ("Rs_fof10",)]
        tasks += [(meth, m, 5.0, f"an{k}_{w}", cn, "SO40") for m in modes for k in (0, 2, 3) for w in ("cat", "iso")
                  for cn in ("a", "d", "e")]
        tasks += [(meth, m, 5.0, f"anz{k}_{w}", cn, "SO40") for m in modes for k in (2, 3) for w in ("cat", "iso")
                  for cn in ("a", "d", "e")]
    log(f"y07 [{meth}]: {len(tasks)} ML fits on {nproc} processes")
    from multiprocessing import get_context
    rows = []
    with get_context("spawn").Pool(nproc, maxtasksperchild=12) as p:
        for k, r in enumerate(p.imap_unordered(job, tasks, chunksize=1)):
            rows.append(r)
            log(f"  [{k + 1}/{len(tasks)}] {r['mode']:8s} q{r['qcut']:.0f} {r['est']:11s} {r['config']} {r['survey']:4s}: "
                f"S8 {r['b_S8']:+.3f} (lin {r['lin_S8']:+.3f}) Om {r['b_Om']:+.3f} s8 {r['b_s8']:+.3f}"
                + "".join(f" {p}:{r['b_' + p]:+.2f}" for p in r["free"][2:])
                + f" | chi2 {r['dev']:.2f} power {r['power']:.3f} | polish shift {r['polish_shift']:.3f} at bound "
                f"{r['at_bound']} [{r['sec']:.0f}s] [elapsed {time.time() - t00:.0f}s]")
    lin = []
    for m in modes:
        for qc in (5.0, 6.0):
            lin += linearised(meth, m, qc, log)
    json.dump(dict(ml=rows, lin=lin), open(os.path.join(HERE, "out", f"fits_{meth}.json"), "w"), indent=1)
    log(f"\nlinearised S8 biases (sigma) with tile-bootstrap errors, q_cut 5 [SO40 | full]:")
    for m in modes:
        for est in ("R", "Rjoint", "Rp", "Rs", "Rs_N", "Rp_closest1", "Rp_fof10", "Rp_keepnr", "Rs_closest1", "Rs_fof10"):
            line = []
            for cn in ("a", "b", "c", "d", "e"):
                v = [r for r in lin if r["mode"] == m and r["est"] == est and r["config"] == cn and r["qcut"] == 5.0]
                if not v:
                    continue
                a = {r["survey"]: r for r in v}
                line.append(f"{cn}: {a['SO40']['lin_S8']:+.2f}" + (f"+-{a['SO40']['boot_S8']:.2f}" if 'boot_S8' in a['SO40'] else "")
                            + f" | {a['full']['lin_S8']:+.2f}" + (f"+-{a['full']['boot_S8']:.2f}" if 'boot_S8' in a['full'] else ""))
            log(f"  {m:8s} {est:11s} " + "; ".join(line))
    log("\np06b split (linearised S8, SO40, q_cut 5): amplitude / z tilt / q shape = sum; nuisance shifts (sigma)")
    for m in modes:
        for est in ("Rs", "Rp", "R"):
            for cn in ("a", "d", "e"):
                v = [r for r in lin if r["mode"] == m and r["est"] == est and r["config"] == cn and r["qcut"] == 5.0
                     and r["survey"] == "SO40"][0]
                p = v["parts"]
                log(f"  {m:8s} {est:3s} {cn}: {p['amplitude']:+.2f} / {p['z tilt']:+.2f} / {p['q shape']:+.2f} = "
                    f"{v['lin_S8']:+.2f}; " + " ".join(f"{k} {x:+.2f}" for k, x in v["nuis"].items()))
    log(f"\ny07 done in {time.time() - t00:.0f} s")


if __name__ == "__main__":
    main()
