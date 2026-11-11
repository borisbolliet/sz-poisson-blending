"""Converged nonlinear Poisson fits replacing c03_analysis.py sections [7], [8] (and recomputing [5]).

Data (Asimov, full sky x fsky):  N_true(theta_t) = C(z,q; theta_t) x N_exact(theta_t)
   C from the stored cosmolib runs (grid / extra points), N_exact = replib.naive_exact (smooth HMF, exact binning).
Models fitted:
   ignore blending : M(theta) = N_exact(theta)
   fixed correction: M(theta) = C_fid(z,q) x N_exact(theta)
   emulator        : M(theta) = C_emu(z,q; theta) x N_exact(theta)  (quadratic in each param on the 3^4 grid)
   iterated        : fixed correction at C(theta_1), theta_1 = fixed-correction best fit (C(theta_1) from emulator,
                     standing in for one recomputation of the blending calculation at the first best fit)
Free parameters:  A = {sigma8, Om};  (1-b), sigma_lnY fixed at truth
                  B = {sigma8, Om, 1-b, sigma_lnY}, Gaussian priors s(1-b)=0.04, s(sigma_lnY)=0.05 centred on truth
                  C = {sigma8, Om, 1-b, sigma_lnY}, no priors
sigma used for 'bias/sigma': Fisher error of the correct (blended) model at the fiducial, for that case and fsky,
computed directly at each fsky (priors included), not by sqrt(fsky) rescaling.
usage: r02_fits.py  -> data_cosmo/r02_fits.json, log r02_fits.log
"""
import os, sys, time, json
os.environ["NUMBA_NUM_THREADS"] = "1"; os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"; os.environ["MKL_NUM_THREADS"] = "1"
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
from multiprocessing import Pool
import replib as R

FID = np.array(R.FIDP)
SP = R.SIG_P18
CASES = {"A": ([0, 1], {}), "B": ([0, 1, 2, 3], {2: 0.04, 3: 0.05}), "C": ([0, 1, 2, 3], {})}
STEP_C = (0.01, 0.01, 0.02, 0.03)          # stored FD half-steps of C (c02 extra)


def truth_points():
    pts = []
    for a in range(2):
        for k in (-3, -1, 1, 3):
            p = FID.copy(); p[a] += k * SP[a]; pts.append((f"{R.PN[a]} {k:+d} sig_P18", p))
    for k in (-3, 3):
        p = FID.copy(); p[0] += k * SP[0]; p[1] -= k * SP[1]; pts.append((f"s8 {k:+d}s, Om {-k:+d}s (P18)", p))
    for a in range(4):
        for v in (R.NODES[a][0], R.NODES[a][2]):
            p = FID.copy(); p[a] = v; pts.append((f"grid {R.PN[a]}={v}", p))
    return pts


def fisher_sigmas():
    """Fisher errors of the blended model at the fiducial, derivative d(C N)/dtheta."""
    N0 = R.naive(FID); C0 = R.C_FID
    dN = []
    for a in range(4):
        pp, pm = FID.copy(), FID.copy(); pp[a] += STEP_C[a]; pm[a] -= STEP_C[a]
        dC = (R.find(pp) - R.find(pm)) / (2 * STEP_C[a])
        dNn = (R.naive_exact(pp) - R.naive_exact(pm)) / (2 * STEP_C[a])
        dN.append(C0 * dNn + N0 * dC)
    dN = np.array(dN); Nb = C0 * N0
    out = {}
    for c, (free, pri) in CASES.items():
        for fs in (1.0, 0.4):
            F = R.fisher_from(dN[free], Nb, free, pri, fs)
            Ci = np.linalg.inv(F)
            out[(c, fs)] = dict(sig=np.sqrt(np.diag(Ci)), cov=Ci, F=F)
    return out, dN


def work(task):
    t0 = time.time()
    kind, name, tt, case, fs, start, mask = (task[k] for k in ("kind", "name", "truth", "case", "fs", "start", "mask"))
    tt = np.array(tt)
    free, pri = CASES[case]
    data = R.find(tt) * R.naive(tt) if kind != "emu" else R.find(tt) * R.naive(tt)
    if kind == "ignore":
        model = R.naive
    elif kind == "fixed":
        model = lambda th: R.C_FID * R.naive(th)
    elif kind == "emu":
        model = lambda th: R.emu_C(th) * R.naive(th)
    elif kind == "iter":
        C1 = R.emu_C(np.array(task["c_at"]))
        model = lambda th: C1 * R.naive(th)
    msk = None
    if mask == "noq50":
        msk = np.ones((8, 10), bool); msk[:, -1] = False
    th, c2, info = R.fit(model, data, tt, free, pri, fs, start=start, mask=msk, maxit=40)
    c2_truth = R.chi2_tot(model, data, tt, tt, free, pri, fs, msk)
    return dict(task, theta=th.tolist(), chi2=c2, chi2_truth=c2_truth, it=info["it"], dec=info["dec"],
                conv=bool(info["conv"]), sec=time.time() - t0)


def tkey(t):
    return json.dumps({k: t[k] for k in ("kind", "name", "case", "fs", "start_lab", "mask")}, sort_keys=True)


def run_pool(tasks):
    """run with 3 workers; every finished fit is appended to data_cosmo/r02_fits_partial.jsonl and reused on restart"""
    part = f"{R.D}/r02_fits_partial.jsonl"
    done = {}
    if os.path.exists(part):
        for line in open(part):
            r = json.loads(line); done[tkey(r)] = r
    todo = [t for t in tasks if tkey(t) not in done]
    print(f"  {len(tasks)} tasks, {len(tasks)-len(todo)} reused from {os.path.basename(part)}", flush=True)
    t0 = time.time()
    with Pool(3) as pool, open(part, "a") as fh:
        for k, r in enumerate(pool.imap_unordered(work, todo, chunksize=1)):
            fh.write(json.dumps(r) + "\n"); fh.flush(); done[tkey(r)] = r
            print(f"  [{k+1}/{len(todo)} {time.time()-t0:.0f}s] {r['kind']} {r['name']} {r['case']} fs={r['fs']} "
                  f"{r['start_lab']} it={r['it']} conv={r['conv']} {r['sec']:.0f}s", flush=True)
    return [done[tkey(t)] for t in tasks]


def main():
    t00 = time.time()
    SIG, dNb = fisher_sigmas()
    print("[F] Fisher 1-sigma (blended model at fiducial, replib exact binning; direct at each fsky)")
    for (c, fs), v in SIG.items():
        r = v["cov"][0, 1] / v["sig"][0] / v["sig"][1]
        print(f"   case {c} fsky={fs}: " + " ".join(f"s({R.PN[a]})={s:.4f}" for a, s in zip(CASES[c][0], v["sig"]))
              + f"  r(s8,Om)={r:+.2f}")
    tasks = []
    # ---- ignore blending, multi-start
    old = {("B", 1.0): [0.0018, -0.0085, 0.0124, 0.0064], ("C", 1.0): [0.0003, -0.0096, 0.0190, 0.0066],
           ("B", 0.4): [0.0028, -0.0079, 0.0085, 0.0046], ("C", 0.4): [0.0003, -0.0096, 0.0190, 0.0066],
           ("A", 1.0): [0.0050, -0.0065], ("A", 0.4): [0.0050, -0.0065]}
    for c, (free, pri) in CASES.items():
        for fs in (1.0, 0.4):
            sg = SIG[(c, fs)]["sig"]
            starts = {"truth": FID.copy()}
            s = FID.copy(); s[free] += np.array(old[(c, fs)]); starts["old c03"] = s
            for lab, sgn in (("+2sig", 1), ("-2sig", -1)):
                s = FID.copy(); s[free] += 2 * sgn * sg * np.array([1, -1, 1, -1][:len(free)]); starts[lab] = s
            for lab, s in starts.items():
                tasks.append(dict(kind="ignore", name="fiducial", truth=FID.tolist(), case=c, fs=fs,
                                  start=s.tolist(), start_lab=lab, mask=None))
        tasks.append(dict(kind="ignore", name="fiducial", truth=FID.tolist(), case=c, fs=1.0, start=FID.tolist(),
                          start_lab="truth", mask="noq50"))
    # ---- fixed correction
    for name, p in truth_points():
        for c, (free, pri) in CASES.items():
            for fs in (1.0, 0.4):
                tasks.append(dict(kind="fixed", name=name, truth=p.tolist(), case=c, fs=fs, start=p.tolist(),
                                  start_lab="truth", mask=None))
                if name.startswith("grid") and c in "BC" and fs == 1.0:
                    sg = SIG[(c, fs)]["sig"]
                    for lab, sgn in (("+2sig", 1), ("-2sig", -1)):
                        s = p.copy(); s[free] += 2 * sgn * sg * np.array([1, -1, 1, -1]);
                        tasks.append(dict(kind="fixed", name=name, truth=p.tolist(), case=c, fs=fs,
                                          start=s.tolist(), start_lab=lab, mask=None))
    # ---- emulator at the 10 random interior points
    for p in R.E["params"][-10:]:
        for c in "ABC":
            tasks.append(dict(kind="emu", name=str(tuple(np.round(p, 4))), truth=p.tolist(), case=c, fs=1.0,
                              start=p.tolist(), start_lab="truth", mask=None))
    print(f"\n{len(tasks)} fits, 3 workers", flush=True)
    res = run_pool(tasks)
    # ---- iterated correction (needs the fixed-correction best fits)
    it_tasks = []
    for r in res:
        if r["kind"] == "fixed" and r["start_lab"] == "truth" and r["fs"] == 1.0 and r["case"] in "AB" and (
                "+3" in r["name"] or "-3" in r["name"] or r["name"].startswith("grid")):
            it_tasks.append(dict(kind="iter", name=r["name"], truth=r["truth"], case=r["case"], fs=1.0,
                                 start=r["theta"], start_lab="fit1", mask=None, c_at=r["theta"]))
    res += run_pool(it_tasks)
    for r in res:
        free = CASES[r["case"]][0]
        sg = SIG[(r["case"], r["fs"])]["sig"]
        d = np.array(r["theta"]) - np.array(r["truth"])
        r["d"] = d[free].tolist(); r["b"] = (d[free] / sg).tolist()
    json.dump(dict(res=res, sig={f"{c}_{fs}": v["sig"].tolist() for (c, fs), v in SIG.items()}),
              open(f"{R.D}/r02_fits.json", "w"), indent=1)
    print(f"done {time.time()-t00:.0f}s; max iterations {max(r['it'] for r in res)}; "
          f"non-converged {sum(not r['conv'] for r in res)}; max Newton decrement {max(r['dec'] for r in res):.1e}")


if __name__ == "__main__":
    main()
