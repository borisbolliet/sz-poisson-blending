"""Step 1 / s12 (arbiter): per-object test of the last redshift bin (z = 2-3), where s11's direct counts on 12 fresh
skies came out below the P(D) expectation of s09 (goal, q > 5: +4.47 +- 0.14% vs +4.96% like-for-like; baseline
q > 4 similar).  Two possible sources are separated here:
   (a) the per-node P(D) probabilities (s09 tables) are biased for these primaries, or
   (b) the expected primary population H (HMF integral on (theta node, q node)) differs from the realised one,
   (c) or the s11 truncation (neighbour sums only for q_A > 1) was not accounted for well enough.
For the same 12 skies (seeds 41001-41012, redrawn bit-identically), every primary with z >= 2 and q_A > 0.5 (no
truncation) gets its direct neighbour sum (build code, as s06d/s11).  Per object we compare
   direct:  dom * Phi(q_A + delta - q_cut)                         (exact noise marginalisation)
   P(D):    P_det(theta_A, q_A) from the s09 node tables (fdom = 1), bilinear in (ln theta, q) -- the SAME primaries.
Sum over objects: direct - P(D) isolates (a); P(D)(realised primaries) vs P(D)(H) isolates (b).
Output: data_real/ztilt_direct/perobj_{seed}.npz and the summary in s12_ztilt_perobject.log.
Usage: s12_ztilt_perobject.py SEED | summary
"""
import os, sys, time, glob, resource
NTH = os.environ.get("S12_THREADS", "4")
for v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ[v] = "1"
os.environ["NUMBA_NUM_THREADS"] = NTH
import numpy as np
from scipy.special import ndtr

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import pdlib1 as pl                      # noqa: E402
import blendlib as bl                    # noqa: E402
from szsim import sampler                # noqa: E402

ROOT = pl.ROOT
OUTD = f"{ROOT}/data_real/ztilt_direct"
TIERS = [0.0, 2.0, 10.0, 1e9]
ZLO = 2.0
QC = np.array([4.0, 5.0, 6.0, 7.0])
MODES = ("baseline", "goal")


def rss_gb():
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e9


def one_sky(seed):
    t0 = time.time()
    tabs = {m: bl.RhoTable(np.load(f"{ROOT}/data_real/rho_mmf_{m}.npz")) for m in MODES}
    models = {m: pl.make_model(("so", m, 100.0)) for m in MODES}
    cat = sampler.draw_catalog(seed, fields=["ra", "dec", "z", "M500", "Y500", "theta500"])
    N = len(cat["ra"])
    cls = bl.mass_class(cat["M500"]); del cat["M500"]
    prim = {}
    for m in MODES:
        q = cat["Y500"] / models[m].sigma_Y(cat["theta500"])
        sel = np.nonzero((q > 0.5) & (cat["z"] >= ZLO))[0]
        prim[m] = dict(q=q[sel].astype(np.float64), z=cat["z"][sel], ra=cat["ra"][sel], dec=cat["dec"][sel],
                       th=cat["theta500"][sel], idx=sel.astype(np.int64))
        del q
    del cat["z"]
    th = cat["theta500"]
    tiers = {}
    for k in (2, 1):
        s = np.nonzero((th > TIERS[k]) & (th <= TIERS[k + 1]))[0]
        tiers[k] = bl.ZoneSet(cat["ra"][s], cat["dec"][s], cat["Y500"][s], th[s], cls[s], s.astype(np.int32))
    s = np.nonzero((th > TIERS[0]) & (th <= TIERS[1]))[0]
    sub = (cat["ra"][s], cat["dec"][s], cat["Y500"][s], th[s], cls[s], s.astype(np.int32))
    del cat, th, cls, s
    tiers[0] = bl.ZoneSet(*sub)
    del sub
    zsets = [tiers[0], tiers[1], tiers[2]]
    out = dict(seed=seed)
    for m in MODES:
        P = prim[m]; tp = time.time()
        D, xmax, npair = bl.neighbour_sums(zsets, tabs[m], P["ra"], P["dec"], P["th"], P["idx"])
        sA = models[m].sigma_Y(P["th"].astype(np.float64))
        out[f"{m}_q"] = P["q"]; out[f"{m}_th"] = P["th"]; out[f"{m}_z"] = P["z"]
        out[f"{m}_delta"] = D.sum(1) / sA
        out[f"{m}_xmax"] = xmax.max(1) / sA
        print(f"   [{m}] z >= {ZLO}, q > 0.5: {len(P['q']):,} primaries, mean pairs {npair.mean():,.0f}, {time.time()-tp:.0f}s; "
              f"rss {rss_gb():.2f} GB", flush=True)
    np.savez(f"{OUTD}/perobj_{seed}.npz", **out)
    print(f"[sky {seed}] N = {N:,}; saved; total {time.time()-t0:.0f}s; peak rss {rss_gb():.2f} GB", flush=True)


def interp_table(Tab, th_nodes, theta, q):
    """bilinear interpolation of a (n_theta, n_q) node table at (ln theta, q) (CIC in ln theta as primary_population)."""
    lth0 = np.log(th_nodes[0]); dlth = np.log(th_nodes[1] / th_nodes[0]); NT = len(th_nodes)
    fa = np.clip((np.log(theta) - lth0) / dlth, 0, NT - 1.000001); ia = fa.astype(int); wa = fa - ia
    Q = pl.QNODES
    k = np.clip(np.searchsorted(Q, q) - 1, 0, len(Q) - 2); wq = np.clip((q - Q[k]) / (Q[k + 1] - Q[k]), 0, 1)
    return ((1 - wa) * ((1 - wq) * Tab[ia, k] + wq * Tab[ia, k + 1]) + wa * ((1 - wq) * Tab[ia + 1, k] + wq * Tab[ia + 1, k + 1]))


def summary():
    files = sorted(glob.glob(f"{OUTD}/perobj_*.npz"))
    S = [np.load(f) for f in files]
    print(f"=== s12 summary: z = 2-3, all primaries q_A > 0.5, {len(S)} skies (seeds {[int(s['seed']) for s in S]}) ===")
    for m in MODES:
        T = np.load(f"{ROOT}/data_real/pd_tables_{m}.npz")
        th_nodes = T["th"]
        for ic, qc in enumerate(QC):
            Pdet = T["T1.0_det"][ic]
            Hb = T["Hz"][6]
            Ns_H = (Hb * ndtr(pl.QNODES[None, :] - qc)).sum(); Nd_H = (Hb * Pdet).sum()
            rows = []
            for s in S:
                q, th, d, xm = s[f"{m}_q"], s[f"{m}_th"].astype(np.float64), s[f"{m}_delta"], s[f"{m}_xmax"]
                dom = xm < q
                ps = ndtr(q - qc)
                direct = dom * ndtr(q + d - qc)
                pdr = interp_table(Pdet, th_nodes, th, q)
                rows.append([ps.sum(), direct.sum(), pdr.sum(),
                             *[((direct - ps)[(q > a) & (q <= b)]).sum() for a, b in ((0.5, 1), (1, 2), (2, 3), (3, 4), (4, 6), (6, 1e9))],
                             *[((pdr - ps)[(q > a) & (q <= b)]).sum() for a, b in ((0.5, 1), (1, 2), (2, 3), (3, 4), (4, 6), (6, 1e9))]])
            R = np.array(rows)
            Ns, Nd, Np = R[:, 0], R[:, 1], R[:, 2]
            dd = 100 * (Nd / Ns - 1); dp = 100 * (Np / Ns - 1)
            diff = Nd - Np
            n = len(S)
            print(f"[{m}] q>{qc:.0f}: N_single/sky {Ns.mean():.1f} (H: {Ns_H:.1f}) | Delta N/N: direct {dd.mean():+.2f} +- {dd.std(ddof=1)/np.sqrt(n):.2f}%; "
                  f"P(D) on the same realised primaries {dp.mean():+.2f} +- {dp.std(ddof=1)/np.sqrt(n):.2f}%; P(D) with H {100*(Nd_H/Ns_H-1):+.2f}% | "
                  f"per sky: direct - P(D) = {diff.mean():+.2f} +- {diff.std(ddof=1)/np.sqrt(n):.2f} objects")
            if qc == 5.0:
                lab = ["0.5-1", "1-2", "2-3", "3-4", "4-6", ">6"]
                print("      Delta N per sky by q_A: direct " + " ".join(f"{l}: {v:+.2f}" for l, v in zip(lab, R[:, 3:9].mean(0)))
                      + "\n                             P(D)   " + " ".join(f"{l}: {v:+.2f}" for l, v in zip(lab, R[:, 9:15].mean(0)))
                      + "\n             direct - P(D) +- err  " + " ".join(f"{l}: {a:+.2f}+-{e:.2f}" for l, a, e in
                                                                    zip(lab, (R[:, 3:9] - R[:, 9:15]).mean(0),
                                                                        (R[:, 3:9] - R[:, 9:15]).std(0, ddof=1) / np.sqrt(n))))


if __name__ == "__main__":
    if sys.argv[1] == "summary":
        summary()
    else:
        one_sky(int(sys.argv[1]))
