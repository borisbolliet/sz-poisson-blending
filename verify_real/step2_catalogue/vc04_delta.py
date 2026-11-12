"""vc04: combine the direct-route skies (vc01) into Delta(z, q_obs) = N_blended - N_single per variant, with sky-to-sky
errors, and compare with the build's Delta (data_real/step2/truth_<setup>.npz D_q5 / D_q6, read as data) bin by bin
(pulls) and in total, redshift tilt and q shape; white also vs Q3's data_cosmo/truth_catalogue_v2.npz.

Errors: each sky is cut into 48 equal-area blocks (by the primary's position).  The error of the mean over n skies is
sqrt(n_blocks_total * var_block) / n_sky (blocks i.i.d.), cross-checked with the scatter of the n sky values.
Output: out/delta_mine.npz (mean Delta, block-level Delta for the bias errors, single counts).
"""
import os, sys, glob, time
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
os.environ["VC_THREADS"] = "1"
import vclib as V
import numpy as np

t0 = time.time()
D2 = f"{V.ROOT}/data_real/step2"
files = sorted(f for f in glob.glob(f"{HERE}/out/sky_*.npz") if not f.endswith("_test.npz"))
seeds = [int(os.path.basename(f)[4:-4]) for f in files]
HB = []; HS = []; info = []
for f in files:
    d = np.load(f)
    HB.append(d["HB"]); HS.append(d["HS"])
    info.append((int(d["N"]), float(d["Nexp"]), int(d["NP"]), float(d["tsum"]), float(d["tacc"]),
                 float(np.interp(0.0, np.log(np.load(f"{HERE}/out/expected_mean.npz")['TH']), d["num_real"][0] / np.load(f"{HERE}/out/expected_mean.npz")["num"][0]))))
    bnames = list(d["bnames"]); snames = list(d["snames"])
HB = np.array(HB); HS = np.array(HS)                      # (nsky, 48, nv, 16, 17)
ns = len(files)
print(f"{ns} skies: seeds {seeds}")
print("  halos per sky (expected 29,295,487): " + ", ".join(f"{i[0]:,}" for i in info))
print("  primaries per sky: " + ", ".join(f"{i[2]:,}" for i in info) + f"; neighbour-sum time {np.mean([i[3] for i in info]):.0f}s/sky")
print("  realised / expected mean kernel response (white; = total Y of the sky / expected): "
      + ", ".join(f"{i[5]:.4f}" for i in info) + f"  (mean {np.mean([i[5] for i in info]):.4f}, std {np.std([i[5] for i in info], ddof=1):.4f})")

res = {}
Q3 = np.load(f"{V.ROOT}/data_cosmo/truth_catalogue_v2.npz")
ZC = [(0.01, 0.1), (0.1, 0.5), (0.5, 1.0), (1.0, 1.5), (1.5, 2.0), (2.0, 3.0)]


def zsel(a, b):
    return (V.ZED[:-1] >= a - 1e-9) & (V.ZED[1:] <= b + 1e-9)


summary = []
for v, bn in enumerate(bnames):
    sn = V.B2S[bn]; s = snames.index(sn)
    for qc, qed in ((5.0, V.QED5), (6.0, V.QED6)):
        tag = f"{qc:.0f}"
        Db = V.ccdf_to_bins(HB[:, :, v] - HS[:, :, s], qed)                # (nsky, 48, 16, 9)
        Ns = V.ccdf_to_bins(HS[:, :, s], qed)
        Dm = Db.sum(1).mean(0)                                             # mean over skies of the sky total
        Nm = Ns.sum(1).mean(0)
        blocks = Db.reshape(ns * V.NBLK, *Db.shape[2:])
        var_blk = blocks.var(0, ddof=1)
        err = np.sqrt(ns * V.NBLK * var_blk) / ns                         # error of the mean of the sky totals
        sky_tot = Db.sum(1)                                                # (nsky, 16, 9)
        err_sky = sky_tot.std(0, ddof=1) / np.sqrt(ns) if ns > 1 else np.full_like(Dm, np.nan)
        tot_blk = blocks.sum((1, 2))
        err_tot = np.sqrt(ns * V.NBLK * tot_blk.var(ddof=1)) / ns
        err_tot_sky = sky_tot.sum((1, 2)).std(ddof=1) / np.sqrt(ns) if ns > 1 else np.nan
        res[f"{bn}_q{tag}"] = dict(D=Dm, err=err, Ns=Nm, blocks=blocks.astype(np.float32), err_sky=err_sky)
        name = bn if bn != "white_f1" else "white"
        if bn == "white_f1_muB":
            name = "white"
        Bf = f"{D2}/truth_{name}.npz"
        if not os.path.exists(Bf):
            continue
        B = np.load(Bf)
        DB, NnB = B[f"D_q{tag}"], B[f"Nn_q{tag}"]
        ok = err > 0
        pull = np.where(ok, (Dm - DB) / np.where(ok, err, 1), 0.0)
        good = NnB > 1.0                                                   # bins with > 1 naive cluster (full sky)
        chi2 = float(np.sum(pull[good] ** 2)); ndof = int(good.sum())
        tot_m, tot_b = Dm.sum(), DB.sum()
        print(f"\n=== [{bn}] q_cut {tag} (vs build truth_{name}.npz) ===")
        print(f"  Delta total: mine {tot_m:+9.1f} +- {err_tot:5.1f} (block error; sky scatter {err_tot_sky:5.1f}) = "
              f"{tot_m/Nm.sum():+.4%} +- {err_tot/Nm.sum():.4%} of my N_single {Nm.sum():,.0f} | build {tot_b:+9.1f} = {tot_b/NnB.sum():+.4%} "
              f"| diff {tot_m-tot_b:+.1f} = {(tot_m-tot_b)/err_tot:+.2f} sigma")
        print(f"  per-bin pulls (mine - build)/err over {ndof} bins with N > 1: chi2 = {chi2:.1f} (p = "
              f"{__import__('scipy.stats', fromlist=['chi2']).chi2.sf(chi2, ndof):.3f}); max |pull| {np.abs(pull[good]).max():.2f}; "
              f"mean pull {pull[good].mean():+.3f}")
        g10 = NnB > 10.0
        c10 = float(np.sum(pull[g10] ** 2))
        print(f"  same over the {int(g10.sum())} bins with N > 10: chi2 = {c10:.1f} (p = "
              f"{__import__('scipy.stats', fromlist=['chi2']).chi2.sf(c10, int(g10.sum())):.3f}); max |pull| {np.abs(pull[g10]).max():.2f}")
        # error as a fraction of the SO40 Poisson sigma (to compare with the build's tolerances)
        print(f"  MC error per bin in Poisson sigma at SO40 (0.4 err / sqrt(0.4 N)): median {np.median((0.4*err/np.sqrt(0.4*NnB))[good]):.3f}, "
              f"max {np.max((0.4*err/np.sqrt(0.4*NnB))[good]):.3f}; |mine - build| in the same units: max "
              f"{np.max((0.4*np.abs(Dm-DB)/np.sqrt(0.4*NnB))[good]):.3f}")
        if qc == 5.0:
            tm = [Dm[zsel(a, b)].sum() / Nm[zsel(a, b)].sum() for a, b in ZC]
            te = [np.sqrt(ns * V.NBLK * blocks[:, zsel(a, b)].sum((1, 2)).var(ddof=1)) / ns / Nm[zsel(a, b)].sum() for a, b in ZC]
            tb = [DB[zsel(a, b)].sum() / NnB[zsel(a, b)].sum() for a, b in ZC]
            print("  z tilt Delta/N  z = " + ", ".join(f"{a}-{b}" for a, b in ZC))
            print("     mine : " + "  ".join(f"{100*x:+.2f}+-{100*e:.2f}%" for x, e in zip(tm, te)))
            print("     build: " + "  ".join(f"{100*x:+.2f}%" for x in tb))
            qm = Dm.sum(0) / Nm.sum(0); qe = np.sqrt(ns * V.NBLK * blocks.sum(1).var(0, ddof=1)) / ns / Nm.sum(0)
            qb = DB.sum(0) / NnB.sum(0)
            print("  q shape Delta/N per q bin (edges " + " ".join(f"{e:.1f}" for e in qed) + ")")
            print("     mine : " + " ".join(f"{100*x:+.2f}+-{100*e:.2f}" for x, e in zip(qm, qe)))
            print("     build: " + " ".join(f"{100*x:+.2f}" for x in qb))
            summary.append((bn, tot_m / Nm.sum(), err_tot / Nm.sum(), tot_b / NnB.sum(), (tot_m - tot_b) / err_tot, chi2, ndof,
                            tm[0], te[0], tb[0], tm[-1], te[-1], tb[-1]))
        if bn.startswith("white"):
            DQ = Q3[f"D_q{tag}"]
            pq = (Dm - DQ) / np.where(ok, err, 1)
            print(f"  vs Q3 truth_catalogue_v2 (c11): Q3 total {DQ.sum():+.1f}; mine - Q3 = {tot_m-DQ.sum():+.1f} "
                  f"({(tot_m-DQ.sum())/err_tot:+.2f} sigma); per-bin chi2 {np.sum(pq[good]**2):.1f} / {ndof}")
        # single counts: realised mean vs the build's naive expectation (sanity of the population)
        print(f"  my single counts (mean of {ns} skies) / build Nn - 1: total {Nm.sum()/NnB.sum()-1:+.2e} "
              f"(expected scatter of the mean ~{1/np.sqrt(NnB.sum()*ns):.1e} Poisson)")

print("\n=== summary, q_cut 5 (Delta/N in %) ===")
print(f"{'variant':16s} {'mine':>16s} {'build':>8s} {'(m-b)/err':>9s} {'chi2/ndof':>10s} | {'z<0.1 mine':>16s} {'build':>7s} | {'z 2-3 mine':>16s} {'build':>7s}")
for r in summary:
    print(f"{r[0]:16s} {100*r[1]:+7.3f} +- {100*r[2]:.3f} {100*r[3]:+8.3f} {r[4]:+9.2f} {r[5]:6.1f}/{r[6]:3d} | "
          f"{100*r[7]:+7.2f} +- {100*r[8]:.2f} {100*r[9]:+7.2f} | {100*r[10]:+7.2f} +- {100*r[11]:.2f} {100*r[12]:+7.2f}")
np.savez(f"{HERE}/out/delta_mine.npz", seeds=np.array(seeds), **{f"{k}_{kk}": vv for k, v in res.items() for kk, vv in v.items()})
print(f"\nsaved out/delta_mine.npz; total {time.time()-t0:.0f}s")
