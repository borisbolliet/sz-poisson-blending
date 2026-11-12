"""y11 (sims lens): extra checks on the own fits (y07) and data vectors (y06), production finder.
 (1) the analyst's model at the truth against the own tiles' N_exp (same kernel; tiles' area): total and per z bin;
 (2) goodness of fit with the noise of the own data vector removed: chi2_min(ML, y07) - sum_b mu_b fsky Var_boot(R_b)
     (the deviance a noiseless data vector of the same expectation would have), and the corresponding power;
 (3) precision: the own tile-bootstrap errors of the linearised S8 bias scaled to the build's 1,548 tiles;
 (4) injection-reference sensitivity: linearised S8 bias with R_s x (1 + 0.0026) (baseline) / (1 + 0.0085) (goal), the
     build's pooled t03 + a02 N_exp offsets.
Usage: python y11_extra.py"""
import os, sys, json
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import numpy as np
import y07_fits as Y
from y07_fits import S, CL
import vlib3b as V
from scipy.stats import chi2, ncx2

log = V.Tee(os.path.join(HERE, "y11_extra.log"))
FS = {"SO40": 0.4, "full": 1.0}
fits = json.load(open(os.path.join(HERE, "out", "fits_whitened.json")))
for mode in ("baseline", "goal"):
    M, mu, data = Y.setup("whitened", mode, 5.0)
    dv = np.load(os.path.join(HERE, "out", f"datavec_whitened_{mode}.npz"))
    nt = int(dv["ntiles"])
    area = nt * V.FSKY_TILE
    Ne = dv["Nexp_q5"]
    pred = mu * area
    zs = pred.sum(1); zt = Ne.sum(1)
    # error of the tiles' N_exp: tile bootstrap
    W = np.zeros((500, nt)); rng = np.random.default_rng(5)
    for b in range(500):
        W[b] = np.bincount(rng.integers(0, nt, nt), minlength=nt)
    Ne_t = dv["Ne_tile_q5"].sum(2)                                     # (nt, nz)
    zb = W @ Ne_t
    ez = zb.std(0)
    ok = zs > 1
    log(f"\n=== [{mode}] ({nt} tiles) ===")
    log(f"(1) model at the truth x tiles' area vs the tiles' N_exp (q > 5): {pred.sum():,.1f} vs {Ne.sum():,.1f} "
        f"({Ne.sum() / pred.sum() - 1:+.2%} +- {zb.sum(1).std() / pred.sum():.2%} tile bootstrap); per z bin chi2 "
        f"{np.sum(((zt - zs) / ez)[ok] ** 2):.1f} for {ok.sum()} bins")
    # (2) noise-subtracted goodness of fit
    for est, key in (("Rs", None), ("Rp", "Rpboot_q5"), ("R", "Rboot_q5")):
        if key is None:
            Rb = (data["Rs"] / mu)[None] + (dv["Rs_Nboot_q5"] - dv["Rs_N_q5"][None])
        else:
            Rb = dv[key]
        var = Rb.var(0)
        for sv, fs in FS.items():
            noise = float(np.sum(mu * fs * var))
            for cn in ("a", "d", "e"):
                r = [x for x in fits["ml"] if x["mode"] == mode and x["est"] == est and x["config"] == cn and x["survey"] == sv
                     and x["qcut"] == 5.0]
                if not r:
                    continue
                r = r[0]
                dev = r["dev"] - noise
                crit = chi2.ppf(0.95, r["ndof"])
                pw = float(ncx2.sf(crit, r["ndof"], max(dev, 1e-9))) if dev > 1e-9 else 0.05
                log(f"(2) {est:2s} {sv:4s} {cn}: chi2_min {r['dev']:.2f}, noise term {noise:.2f} -> {dev:+.2f} (power "
                    f"{pw:.2f}; as fitted {r['power']:.2f})")
    # (3) precision scaled to 1,548 tiles; (4) injection reference
    for r in fits["lin"]:
        if r["mode"] == mode and r["qcut"] == 5.0 and r["survey"] == "SO40" and r["config"] in ("d", "e") and r["est"] in ("R", "Rp", "Rs", "Rjoint"):
            log(f"(3) {r['est']:6s} {r['config']}: own bootstrap error {r['boot_S8']:.3f} ({nt} tiles) -> scaled to 1,548 tiles "
                f"{r['boot_S8'] * np.sqrt(nt / 1548.0):.3f}; to 324 tiles {r['boot_S8'] * np.sqrt(nt / 324.0):.3f}")
    dlt = 0.0026 if mode == "baseline" else 0.0085
    for sv, fs in (("SO40", 0.4),):
        for cn in ("a", "d", "e"):
            free, pri = S.config(cn, S.FID)
            fi = [CL.PNAMES.index(p) for p in free]
            F_ = S.Fitter3(M, mu * fs, free, fsky=fs, priors=pri, truth=S.FID)
            Ft = F_.fisher(S.FID[fi]); J = F_.jac(S.FID[fi])
            sig, Cov, gS = S.derived_sigma3(Ft, fi, S.FID)
            Rs = data["Rs"] / mu
            b0 = gS @ np.linalg.solve(Ft, J.T @ (Rs - 1).ravel()) / sig["S8"]
            b1 = gS @ np.linalg.solve(Ft, J.T @ (Rs * (1 + dlt) - 1).ravel()) / sig["S8"]
            log(f"(4) {sv} {cn}: linearised S8 with R_s {b0:+.2f} -> with the injection reference (N_exp x (1 - {dlt:.4f})) "
                f"{b1:+.2f}")
