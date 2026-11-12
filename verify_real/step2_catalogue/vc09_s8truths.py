"""vc09: truths off the fiducial (S8 = 0.75, 0.87 at Om = 0.3111; sigma8 = 0.7365, 0.8543), SO baseline, fdom 1.
Direct route on fresh skies drawn from the sampler's table recomputed at that sigma8 (vc01 <seed> prod <s8>):
  (1) Delta_t / N at q > 5 and the redshift tilt vs the build's truth_S8_*.npz (claim C17);
  (2) r02's fixed-correction model M(theta) = C_fid(z, q) x mu(theta), C_fid = (mu(fid) + Delta_fid) / mu(fid), fitted
      with MY model and fitter to d_t = mu(theta_t) + Delta_t, and the naive model ('ignore') to the same data;
      (i) the build's Delta_fid and Delta_t (fitter check), (ii) MY Delta_fid (fiducial skies) and MY Delta_t (claim C16).
  MC errors: linearised map applied to the block values of Delta_t and of Delta_fid (independent skies).
"""
import os, sys, glob, json, time
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
os.environ["VC_THREADS"] = "1"
import vclib as V
import vcmodel as VM
import numpy as np

t0 = time.time()
D2 = f"{V.ROOT}/data_real/step2"
TR = {"S8=0.75": (0.7365, "S8_0.75_f1"), "S8=0.87": (0.8543, "S8_0.87_f1")}
ZC = [(0.01, 0.1), (0.1, 0.5), (0.5, 1.0), (1.0, 1.5), (1.5, 2.0), (2.0, 3.0)]
SURV = {"SO40": 0.4, "full": 1.0}
vb = [b[0] for b in V.BVARS].index("baseline_f1"); vs = [s[0] for s in V.SVARS].index("baseline")


def zsel(a, b):
    return (V.ZED[:-1] >= a - 1e-9) & (V.ZED[1:] <= b + 1e-9)


def load_mine(pattern, qed):
    files = sorted(glob.glob(pattern))
    HB = np.array([np.load(f)["HB"][:, vb] for f in files]); HS = np.array([np.load(f)["HS"][:, vs] for f in files])
    D = V.ccdf_to_bins(HB - HS, qed); N = V.ccdf_to_bins(HS, qed)
    ns = len(files)
    blocks = D.reshape(ns * V.NBLK, *D.shape[2:])
    return D.sum(1).mean(0), N.sum(1).mean(0), blocks, ns


class FixedCorr:
    def __init__(self, base, C):
        self.base, self.C = base, C

    def counts(self, th, fsky=1.0):
        return self.C * self.base.counts(th, fsky)


out = []
for qc, qed in ((5.0, V.QED5), (6.0, V.QED6)):
    tag = f"{qc:.0f}"
    base = VM.CountModel("baseline", qc)
    mu_f = base.counts(V.FID)
    dm = np.load(f"{HERE}/out/delta_mine.npz")
    Dfid_m, Bfid_m, nsf = dm[f"baseline_f1_q{tag}_D"], dm[f"baseline_f1_q{tag}_blocks"].astype(float), len(dm["seeds"])
    Dfid_b = np.load(f"{D2}/truth_baseline_f1.npz")[f"D_q{tag}"]
    C_m = (mu_f + Dfid_m) / mu_f; C_b = (mu_f + Dfid_b) / mu_f
    for tn, (s8, setup) in TR.items():
        truth = V.FID.copy(); truth[1] = s8
        Dt_m, Nt_m, Bt_m, nst = load_mine(f"{HERE}/out/skyS8_{s8}_*.npz", qed)
        if nst == 0:
            continue
        Dt_b = np.load(f"{D2}/truth_{setup}.npz")[f"D_q{tag}"]; Nn_b = np.load(f"{D2}/truth_{setup}.npz")[f"Nn_q{tag}"]
        mu_t = base.counts(truth)
        et = np.sqrt(nst * V.NBLK * Bt_m.sum((1, 2)).var(ddof=1)) / nst
        print(f"\n=== truth {tn} (sigma8 = {s8}), q_cut {tag}: {nst} skies ===")
        print(f"  Delta total mine {Dt_m.sum():+.1f} +- {et:.1f} = {Dt_m.sum()/Nt_m.sum():+.3%} +- {et/Nt_m.sum():.3%} | build "
              f"{Dt_b.sum():+.1f} = {Dt_b.sum()/Nn_b.sum():+.3%} | diff {(Dt_m.sum()-Dt_b.sum())/et:+.2f} sigma; my N_single {Nt_m.sum():,.0f}, "
              f"build Nn {Nn_b.sum():,.0f}, my model mu(truth) {mu_t.sum():,.0f}")
        if qc == 5.0:
            tm = [Dt_m[zsel(a, b)].sum() / Nt_m[zsel(a, b)].sum() for a, b in ZC]
            te = [np.sqrt(nst * V.NBLK * Bt_m[:, zsel(a, b)].sum((1, 2)).var(ddof=1)) / nst / Nt_m[zsel(a, b)].sum() for a, b in ZC]
            tb = [Dt_b[zsel(a, b)].sum() / Nn_b[zsel(a, b)].sum() for a, b in ZC]
            print("  z tilt mine : " + "  ".join(f"{100*x:+.2f}+-{100*e:.2f}%" for x, e in zip(tm, te)))
            print("  z tilt build: " + "  ".join(f"{100*x:+.2f}%" for x in tb))
        for sv, fs in SURV.items():
            for cfg in ("a", "b", "c", "d", "e"):
                row = dict(truth=tn, qcut=qc, survey=sv, config=cfg)
                for lab, Cfid, Dt, Bt, Bf in (("build", C_b, Dt_b, None, None), ("mine", C_m, Dt_m, Bt_m, Bfid_m)):
                    data = np.clip(mu_t + Dt, 0, None) * fs
                    for kind, mdl in (("ignore", base), ("fixed", FixedCorr(base, Cfid))):
                        Fo = VM.Fit(mdl, data, cfg, truth, fs)
                        dx, F, J, mu = Fo.linearised(np.array([0.001, 0.001, 0.003, 0.003, 0.003])[Fo.free])
                        sig, sS8, g, C = Fo.sigmas(F)
                        lin = float((g @ dx) / sS8)
                        if cfg in ("a", "d"):
                            xnm, _ = Fo.fit_nm(F)
                            xn, _ = Fo.newton_polish(xnm, 0.05 * sig)
                            th = Fo.full(xn)
                            ml = float((VM.S8_of(th[0], th[1]) - VM.S8_of(truth[0], truth[1])) / sS8)
                        else:
                            ml = None
                        row[f"{lab}_{kind}"] = dict(lin=lin, ml=ml, s_S8=sS8)
                        if lab == "mine":
                            A = np.linalg.solve(F, (J / mu[:, None]).T) * fs
                            a = (A.T @ g) / sS8                                  # S8 bias per unit full-sky data change
                            pt = Bt.reshape(len(Bt), -1) @ a
                            var = len(Bt) * pt.var(ddof=1) / nst ** 2
                            if kind == "fixed":                                  # model term from Delta_fid: C_fid mu(theta)
                                pf = (Bf.reshape(len(Bf), -1) * (mu_t / mu_f).ravel()[None, :]) @ a
                                var += len(Bf) * pf.var(ddof=1) / nsf ** 2
                            row[f"{lab}_{kind}"]["mc"] = float(np.sqrt(var))
                out.append(row)
                bi, bf = row["build_ignore"], row["build_fixed"]; mi, mf_ = row["mine_ignore"], row["mine_fixed"]
                fmt = lambda r: f"{r['ml']:+.2f}" if r["ml"] is not None else f"{r['lin']:+.2f}(lin)"
                print(f"  {sv:4s} {cfg}: ignore: build-vectors {fmt(bi)} | mine {fmt(mi)} +- {mi['mc']:.2f}   ||   fixed correction: "
                      f"build-vectors {fmt(bf)} | mine {fmt(mf_)} +- {mf_['mc']:.2f}", flush=True)
json.dump(out, open(f"{HERE}/out/s8truths_mine.json", "w"), indent=1)
fx = [r for r in out]
for lab in ("build", "mine"):
    va = [abs(r[f"{lab}_fixed"]["ml"] if r[f"{lab}_fixed"]["ml"] is not None else r[f"{lab}_fixed"]["lin"]) for r in fx if r["config"] == "a"]
    vb_ = [abs(r[f"{lab}_fixed"]["ml"] if r[f"{lab}_fixed"]["ml"] is not None else r[f"{lab}_fixed"]["lin"]) for r in fx if r["config"] != "a"]
    vi = [abs(r[f"{lab}_ignore"]["ml"] if r[f"{lab}_ignore"]["ml"] is not None else r[f"{lab}_ignore"]["lin"]) for r in fx]
    if va:
        print(f"[{lab} vectors] max |S8 bias| fixed correction: config a {max(va):.3f}, configs b-e {max(vb_):.3f}; ignoring blending {max(vi):.3f}")
print(f"total {time.time()-t0:.0f}s")
