"""Cosmology bias from ignoring blending: Asimov maximum-likelihood fits of the NAIVE model to BLENDED expectations.

Data vectors (full sky; scaled by fsky):
  cat        mu_naive(fid) + Delta_cat                              (c03: compound-Poisson P(D), mean removed, dominance)
  map        mu_naive(fid) + (H_map   - H_iso) * fullsky/area        (c04: 324 patches; all accepted detections)
  map_clean  mu_naive(fid) + (H_clean - H_iso) * fullsky/area        (fragments and noise-dominated peaks removed)
  map_raw    mu_naive(fid) + (H_map   - H_single) * fullsky/area     (also includes the multi-scale pipeline's own
                                                                      max-over-scales boost on isolated halos, i.e. a naive
                                                                      model NOT calibrated on injections; illustration only)
  self       mu_naive(fid)                                          (sanity check: must return the truth)
Fit configurations (free parameters; all others fixed at the truth):
  a: Om, s8                 b: + (1-b)            c: + (1-b), slnY
  d: + (1-b), alpha, slnY with Gaussian prior (1-b) ~ N(0.8, 0.032) [4%]   (d3/d5: 3% / 5% priors; e: no prior)
Errors: Fisher at the truth (marginal, incl. priors). Bias = (ML - truth)/sigma_Fisher.
Map routes: patch bootstrap (300 resamples of the 324 patches) of the bias, linearised estimator
  dtheta = F^-1 J^T (d - mu)/mu  (checked against the full ML on the central data vector).
Goodness of fit: Asimov deviance chi2_min = 2 sum[mu - d + d ln(d/mu)] -> expected chi2 ~ ndof + chi2_min;
  power = P(noncentral chi2(ndof, chi2_min) > 95% quantile of chi2(ndof)).
Output: data_cosmo/fits.npz, printed tables.
"""
import os, sys, time, json
os.environ.setdefault("OMP_NUM_THREADS", "3")
import numpy as np
from scipy.stats import chi2, ncx2

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import cosmolib as CL

CONFIGS = {
    "a": (["Om", "s8"], {}),
    "b": (["Om", "s8", "omb"], {}),
    "c": (["Om", "s8", "omb", "slnY"], {}),
    "d": (["Om", "s8", "omb", "alpha", "slnY"], {"omb": (0.8, 0.032)}),
    "d3": (["Om", "s8", "omb", "alpha", "slnY"], {"omb": (0.8, 0.024)}),
    "d5": (["Om", "s8", "omb", "alpha", "slnY"], {"omb": (0.8, 0.040)}),
    "e": (["Om", "s8", "omb", "alpha", "slnY"], {}),
}
SURVEYS = {"full": 1.0, "SO40": 0.4}
ONLY = os.environ.get("CONFIGS")
TAG = os.environ.get("TAG", "")
NBOOT = 300


def main():
    t0 = time.time()
    tc = np.load(f"{CL.DC}/truth_catalogue.npz")
    tm = np.load(f"{CL.DC}/truth_maps.npz")
    scale = float(tm["fullsky_deg2"] / tm["area_deg2"])
    rng = np.random.default_rng(2026)
    rows = []
    store = {}
    for qc in [float(v) for v in os.environ.get("QCUTS", "5,6").split(",")]:
        s = f"{qc:.0f}"
        model = CL.NaiveModel(qcut=qc)
        N0 = model.counts(CL.FID)
        Hm, Hc, Hi, Hs = tm["map" + s], tm["clean" + s], tm["iso" + s], tm["single" + s]
        data = {"self": N0, "cat": N0 + tc["D_q" + s],
                "map": N0 + (Hm - Hi).sum(0) * scale, "map_clean": N0 + (Hc - Hi).sum(0) * scale,
                "map_raw": N0 + (Hm - Hs).sum(0) * scale}
        boot_idx = rng.integers(0, Hm.shape[0], (NBOOT, Hm.shape[0]))
        boot = {"map": np.stack([N0 + (Hm[b] - Hi[b]).sum(0) * scale for b in boot_idx]),
                "map_clean": np.stack([N0 + (Hc[b] - Hi[b]).sum(0) * scale for b in boot_idx]),
                "map_raw": np.stack([N0 + (Hm[b] - Hs[b]).sum(0) * scale for b in boot_idx])}
        for dn, dv in data.items():
            if np.any(dv < 0):
                print(f"WARNING q{s} {dn}: {np.sum(dv<0)} negative bins, clipped to 0")
                data[dn] = np.clip(dv, 0, None)
        print(f"\n######## q_cut = {qc}: full-sky totals: naive {N0.sum():,.1f}; " +
              ", ".join(f"{k} {v.sum():,.1f} ({v.sum()/N0.sum()-1:+.4f})" for k, v in data.items() if k != "self"))
        for sv, fsky in SURVEYS.items():
            for cn, (free, pri) in CONFIGS.items():
                if ONLY and cn not in ONLY.split(","):
                    continue
                fi = [CL.PNAMES.index(p) for p in free]
                for dn, dv in data.items():
                    F_ = CL.Fitter(model, dv * fsky, free, fsky=fsky, priors=pri)
                    xt = CL.FID[fi]
                    Ft = F_.fisher(xt)
                    sig, C = CL.derived_sigma(Ft, fi)
                    x, L, nit = F_.fit(polish=(dn != 'self'))
                    th = F_.full(x)
                    dev = F_.deviance(x)
                    nb = dv.size; ndof = nb - len(free)
                    crit = chi2.ppf(0.95, ndof)
                    power = float(ncx2.sf(crit, ndof, dev)) if dev > 1e-9 else 0.05
                    p_typ = float(chi2.sf(ndof + dev, ndof))
                    # linearised bias at the truth
                    mu = F_.mu(xt).ravel(); J = F_.jac(xt)
                    g = J.T @ (((dv * fsky).ravel() - mu) / mu)
                    xlin = np.linalg.solve(Ft, g)
                    S8t = CL.S8_of(*CL.FID[:2]); S8f = CL.S8_of(th[0], th[1])
                    r = dict(qcut=qc, survey=sv, config=cn, data=dn, free=free, priors={k: list(v) for k, v in pri.items()},
                             nit=nit, dev=dev, polish_gain=getattr(F_, 'polish_gain', 0.0), polish_shift=getattr(F_, 'polish_shift', 0.0), at_bound=getattr(F_, 'at_bound', []), ndof=ndof, power=power, p_typ=p_typ, lnL=L)
                    for k, i in enumerate(fi):
                        p = CL.PNAMES[i]
                        r[f"d_{p}"] = th[i] - CL.FID[i]; r[f"s_{p}"] = sig[p]; r[f"b_{p}"] = (th[i] - CL.FID[i]) / sig[p]
                        r[f"lin_{p}"] = xlin[k] / sig[p]
                    r["d_S8"] = S8f - S8t; r["s_S8"] = sig["S8"]; r["b_S8"] = (S8f - S8t) / sig["S8"]
                    gS = np.zeros(len(fi)); gS[0] = 0.5 * S8t / CL.FID[0]; gS[1] = S8t / CL.FID[1]
                    r["lin_S8"] = (gS @ xlin) / sig["S8"]
                    if dn in boot:
                        bl = []
                        for bv in boot[dn]:
                            gb = J.T @ (((np.clip(bv, 0, None) * fsky).ravel() - mu) / mu)
                            xb = np.linalg.solve(Ft, gb)
                            bl.append(np.concatenate([xb[:2] / [sig["Om"], sig["s8"]], [(gS @ xb) / sig["S8"]],
                                                      [xb[k] / sig[CL.PNAMES[i]] for k, i in enumerate(fi) if i >= 2]]))
                        bl = np.array(bl)
                        r["boot_std"] = bl.std(0, ddof=1).tolist(); r["boot_mean"] = bl.mean(0).tolist()
                    rows.append(r)
                    if dn != "self" or cn in ("a", "e"):
                        nui = "  ".join(f"{p}:{r['d_'+p]:+.4f}({r['b_'+p]:+.2f}s)" for p in free[2:])
                        bs = (f" boot(Om,s8,S8)=±{r['boot_std'][0]:.2f},±{r['boot_std'][1]:.2f},±{r['boot_std'][2]:.2f}"
                              if "boot_std" in r else "")
                        print(f"q{s} {sv:4s} {cn:2s} {dn:9s}: dOm={r['d_Om']:+.4f} ({r['b_Om']:+.2f}s, lin {r['lin_Om']:+.2f})"
                              f"  ds8={r['d_s8']:+.4f} ({r['b_s8']:+.2f}s, lin {r['lin_s8']:+.2f})"
                              f"  dS8={r['d_S8']:+.4f} ({r['b_S8']:+.2f}s)  | sig(Om,s8,S8)=({r['s_Om']:.4f},{r['s_s8']:.4f},{r['s_S8']:.4f})"
                              f" | {nui} | chi2min={dev:.1f} ndof={ndof} power={power:.2f}{bs} [it {nit}; LBFGSB dlnL {r['polish_gain']:+.1e}, shift {r['polish_shift']:.2f}s, at bound {r['at_bound']}]", flush=True)
        print(f"[{time.time()-t0:.0f}s]")
    json.dump(rows, open(f"{CL.DC}/fits{TAG}.json", "w"), indent=1)
    # ---- sanity summary: self-fits
    self_rows = [r for r in rows if r["data"] == "self"]
    mx = max(abs(r[f"b_{p}"]) for r in self_rows for p in r["free"])
    print(f"\nSANITY: naive model fitted to its own expectation: max |bias| over {len(self_rows)} fits and all free "
          f"params = {mx:.2e} sigma; max chi2min = {max(r['dev'] for r in self_rows):.2e}")
    pg = [r for r in rows if r['data'] != 'self']
    print(f"L-BFGS-B polish after Fisher scoring: max lnL gain {max(r['polish_gain'] for r in pg):.2e}, "
          f"max shift {max(r['polish_shift'] for r in pg):.3f} sigma over {len(pg)} fits")
    print(f"total {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
