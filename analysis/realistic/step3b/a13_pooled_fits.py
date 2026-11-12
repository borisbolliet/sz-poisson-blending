"""Step 3b arbitration / a13: ML fits of the headline (production finder, merge none) with the pooled data vector of a10
(the build's 1,548 tiles + the sims verifier's 400 + at baseline the arbiter's 400 tiles made with the verifier's
generator and run through the build's pipeline).

As b09: data = mu_naive(truth) x R_s (Asimov), s2lib.run_fit (Fisher scoring + L-BFGS-B polish), configs a, b, c, d, d3,
d5, e; SO40 and full sky; q_cut 5 and 6; both noise levels.  Errors: linearised S8 bias over a10's tile-bootstrap
resamples.  Also: Om and s8 biases, goodness-of-fit power, config-d nuisance shifts, p06b's split (amplitude / z tilt /
q shape) and the pooled count change by coarse z and by q (model weights), with R_p for comparison.
Output: data_real/step3b/arbiter/fits_pooled.json.  Usage: python a13_pooled_fits.py NPROC
"""
import os, sys, json, time
NPROC = int(sys.argv[1]) if len(sys.argv) > 1 else 3
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import s3blib as B                                 # noqa: E402
B.threads(1)
import numpy as np                                 # noqa: E402
import s3bcosmo as SC                              # noqa: E402
import arblib as A                                 # noqa: E402
S2, CL = SC.S2, SC.CL
_W = {}


def get(mode, qc):
    key = (mode, qc)
    if key not in _W:
        _W.clear()
        M = SC.model("prod", mode, qc)
        dv = np.load(os.path.join(A.ARB, f"datavec_pooled_{mode}.npz"))
        _W[key] = (M, M.counts(SC.FID), dv[f"Rs_q{qc:.0f}"])
    return _W[key]


def job(task):
    mode, qc, sv, cn = task
    M, N0, R = get(mode, qc)
    r = S2.run_fit(M, np.clip(N0 * R, 0, None), cn, SC.FID, SC.SURVEYS[sv], polish=True)
    r.update(mode=mode, qcut=qc, survey=sv)
    return r


if __name__ == "__main__":
    from multiprocessing import get_context
    t00 = time.time()
    tasks = [(m, qc, sv, cn) for m in B.MODES for qc in SC.QCUTS for sv in SC.SURVEYS for cn in SC.CFGS]
    with get_context("spawn").Pool(NPROC) as p:
        rows = list(p.imap_unordered(job, tasks, chunksize=1))
    json.dump(rows, open(os.path.join(A.ARB, "fits_pooled.json"), "w"), indent=1, default=float)
    print(f"a13: {len(rows)} ML fits of the pooled headline [{time.time() - t00:.0f} s]; fits at a bound: "
          f"{sum(bool(r['at_bound']) for r in rows)}; max polish shift {max(r['polish_shift'] for r in rows):.3f}", flush=True)
    fit = {(r["mode"], r["qcut"], r["survey"], r["config"]): r for r in rows}
    for mode in B.MODES:
        dv = np.load(os.path.join(A.ARB, f"datavec_pooled_{mode}.npz"))
        print(f"==== {mode}: {int(dv['ntiles'])} tiles", flush=True)
        for qc in SC.QCUTS:
            s = f"q{qc:.0f}"
            M, N0, R = get(mode, qc)
            RB, Rp, RpB = dv[f"Rsboot_{s}"].astype(float), dv[f"Rp_{s}"], dv[f"Rpboot_{s}"].astype(float)
            mu = dv[f"mu_{s}"]
            if qc == 5.0:
                zed, qed = CL.default_bins(qc)
                zz = []
                for a, b in zip(A.ZC[:-1], A.ZC[1:]):
                    kz = (zed[:-1] >= a - 1e-9) & (zed[1:] <= b + 1e-9)
                    w = mu[kz]
                    zz.append((np.sum(w * (R[kz] - 1)) / w.sum(), (np.sum(w * (RB[:, kz] - 1), (1, 2)) / w.sum()).std()))
                qq = [(np.sum(mu[:, k] * (R[:, k] - 1)) / mu[:, k].sum(), (np.sum(mu[:, k] * (RB[:, :, k] - 1), 1) / mu[:, k].sum()).std())
                      for k in range(qed.size - 1) if mu[:, k].sum() > 0]
                tot = np.sum(mu * (R - 1)) / mu.sum(); etot = (np.sum(mu * (RB - 1), (1, 2)) / mu.sum()).std()
                print(f"  count change R_s - 1 at q > 5 (full-sky model weights): {A.fmt(tot, etot)}; by z (0.01-0.1, 0.1-0.5, 0.5-1, 1-1.5, "
                      f"1.5-2, 2-3): " + ", ".join(f"{100 * a:+.1f}+-{100 * b:.1f}" for a, b in zz) + "; by q (5-7.3, 7.3-10.6, 10.6-15.4, "
                      "15.4-22.4, 22.4-32.5, 32.5-47.3): " + ", ".join(f"{100 * a:+.1f}+-{100 * b:.1f}" for a, b in qq[:6]), flush=True)
            for sv in SC.SURVEYS:
                cells = []
                for cn in SC.CFGS:
                    L = SC.linear_parts(M, SC.FID, R, cn, SC.SURVEYS[sv])
                    bl = SC.lin_bias(L, R)[0][0]; bb = SC.lin_bias(L, RB)[0]; bp = SC.lin_bias(L, Rp)[0][0]; bpb = SC.lin_bias(L, RpB)[0]
                    r = fit[(mode, qc, sv, cn)]
                    r.update(lin_S8_check=float(bl), boot_S8=float(bb.std()), lin_S8_Rp=float(bp), boot_S8_Rp=float(bpb.std()))
                    cells.append(f"{cn} {r['b_S8']:+.2f}+-{bb.std():.2f} (lin {bl:+.2f}; R_p {bp:+.2f}+-{bpb.std():.2f})")
                print(f"  q_cut {qc:.0f} {sv:4s} S8 bias, ML (R_s) +- bootstrap: " + "; ".join(cells), flush=True)
                om = "; ".join(f"{cn}: Om {fit[(mode, qc, sv, cn)]['b_Om']:+.2f}, s8 {fit[(mode, qc, sv, cn)]['b_s8']:+.2f}" for cn in ("a", "d", "e"))
                pw = max(fit[(mode, qc, sv, cn)]["power"] for cn in SC.CFGS)
                rd = fit[(mode, qc, sv, "d")]
                nui = "/".join(f"{rd['b_' + p]:+.2f}" for p in ("omb", "alpha", "slnY"))
                print(f"      {om}; max power a-e {pw:.2f} (a {fit[(mode, qc, sv, 'a')]['power']:.2f}, d {rd['power']:.2f}); d: chi2 {rd['dev']:.1f}, "
                      f"(1-b)/alpha/slnY {nui}", flush=True)
                if qc == 5.0:
                    parts = []
                    for cn in ("a", "d", "e"):
                        L = SC.linear_parts(M, SC.FID, R, cn, SC.SURVEYS[sv])
                        p_ = SC.split_parts(L, R)
                        parts.append(f"{cn}: {p_['amplitude']:+.2f} {p_['z tilt']:+.2f} {p_['q shape']:+.2f} = {p_['sum']:+.2f}")
                    print(f"      split amplitude / z tilt / q shape (linearised): " + "; ".join(parts), flush=True)
    json.dump(rows, open(os.path.join(A.ARB, "fits_pooled.json"), "w"), indent=1, default=float)
    print(f"done [{time.time() - t00:.0f} s]; peak RSS {A.peak_rss_gb():.2f} GB", flush=True)
