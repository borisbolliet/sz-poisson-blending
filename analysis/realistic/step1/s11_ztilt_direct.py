"""Step 1 / s11 (arbiter): redshift tilt Delta N/N(z) at q_cut = 4..7 counted DIRECTLY on fresh full skies, to arbitrate
between the 3-sky direct count of s06d (quoted in RESULTS; e.g. baseline z = 2-3: +5.46 +- 0.05%) and the P(D)
expectation (s09; verifier 2's maps: +4.84%).

Same method as s06d part 2 (build code: blendlib.neighbour_sums with the build's rho tables, support 5 thA + 5 thB + 150'
for the MMF; exact per-object noise marginalisation; dominance x_max < q_A; white control with the map mean removed
using this sky's ybar), on NEW skies (sampler seeds given on the command line, never used before), one process per sky
(memory).  To keep the cost down:
  * MMF (baseline, goal): neighbour sums only for primaries with z >= 1.5 and q_A > 1.0 (the disputed bins
    1.5-2 and 2-3).  Primaries with 0.5 < q_A <= 1.0 enter N_single and N_det with delta = 0; s09 bounds the effect of
    this truncation on Delta N/N from the P(D) (z_trunc_q1 in data_real/pd_tables_{run}.npz).
  * white: all primaries q_A > 0.5, all z bins (cheap: compact kernel).
Per sky the z-binned N_single and N_det are stored for 8 RA wedges (within-sky jackknife) in
data_real/ztilt_direct/sky_{seed}.npz.  'summary' combines all sky files: mean +- std/sqrt(n_sky) per bin, the
empirical per-sky scatter vs the s09 prediction, and the comparison with the P(D) expectation (s09), s06d and V1.
Usage: s11_ztilt_direct.py SEED  |  s11_ztilt_direct.py summary
"""
import os, sys, time, glob, resource
NTH = os.environ.get("S11_THREADS", "4")
for v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ[v] = "1"
os.environ["NUMBA_NUM_THREADS"] = NTH
import numpy as np
from scipy.special import ndtr

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import pdlib1 as pl                      # noqa: E402
import blendlib as bl                    # noqa: E402
from szsim import sampler, signal as sg  # noqa: E402

ROOT = pl.ROOT
OUTD = f"{ROOT}/data_real/ztilt_direct"
TIERS = [0.0, 2.0, 10.0, 1e9]
ZB = np.array([0.01, 0.1, 0.25, 0.5, 1.0, 1.5, 2.0, 3.0])
QC = np.array([4.0, 5.0, 6.0, 7.0])
NW = 8
Q_SUM_MMF, Z_SUM_MMF = 1.0, 1.5


def rss_gb():
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e9


def binned(q, delta, dom, z, ra):
    iz = np.clip(np.searchsorted(ZB, z, side="right") - 1, 0, len(ZB) - 2)
    iw = np.minimum((ra / (360.0 / NW)).astype(int), NW - 1)
    key = iz * NW + iw
    Ns = np.zeros((len(QC), len(ZB) - 1, NW)); Nd = np.zeros_like(Ns)
    for ic, qc in enumerate(QC):
        Ns[ic] = np.bincount(key, weights=ndtr(q - qc), minlength=(len(ZB) - 1) * NW).reshape(len(ZB) - 1, NW)
        Nd[ic] = np.bincount(key, weights=dom * ndtr(q + delta - qc), minlength=(len(ZB) - 1) * NW).reshape(len(ZB) - 1, NW)
    return Ns, Nd


def one_sky(seed):
    t0 = time.time()
    os.makedirs(OUTD, exist_ok=True)
    tabs = {m: bl.RhoTable(np.load(f"{ROOT}/data_real/rho_mmf_{m}.npz")) for m in ("baseline", "goal")}
    tabs["white"] = bl.load_rho_table()
    models = {m: pl.make_model(("so", m, 100.0)) for m in ("baseline", "goal")}
    models["white"] = sg.SZModel()
    cat = sampler.draw_catalog(seed, fields=["ra", "dec", "z", "M500", "Y500", "theta500"])
    N = len(cat["ra"])
    cls = bl.mass_class(cat["M500"]); del cat["M500"]
    ybar = float(cat["Y500"].astype(np.float64).sum()) / bl.ARCMIN2_FULLSKY
    print(f"[sky {seed}] N = {N:,} (expected {cat['meta']['N_expected']:,.0f}); ybar = {ybar:.4e}; drawn {time.time()-t0:.0f}s; "
          f"rss {rss_gb():.2f} GB", flush=True)
    # memory (first version peaked at 4.1 GB RSS): primaries keep the catalogue's float32 values (upcast to float64
    # exactly where used) and only the summed subset keeps dec/theta/index; the small zone tiers are built first and
    # the full catalogue is released before the large tier-0 zone set is sorted.  Same arrays, same zone sets.
    prim = {}
    for m in ("baseline", "goal", "white"):
        q = cat["Y500"] / models[m].sigma_Y(cat["theta500"])
        sel = np.nonzero(q > 0.5)[0]
        qs, zs = q[sel].astype(np.float64), cat["z"][sel]
        do = np.ones(len(sel), bool) if m == "white" else (qs > Q_SUM_MMF) & (zs >= Z_SUM_MMF)
        ii = np.nonzero(do)[0]
        prim[m] = dict(q=qs, z=zs, ra=cat["ra"][sel], ii=ii, dec_ii=cat["dec"][sel[ii]], th_ii=cat["theta500"][sel[ii]],
                       idx_ii=sel[ii].astype(np.int64))
        del q, sel
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
    print(f"[sky {seed}] zones built {time.time()-t0:.0f}s; rss {rss_gb():.2f} GB", flush=True)
    out = dict(seed=seed, N=N, ybar=ybar, zb=ZB, qc=QC, nw=NW)
    for m in ("white", "baseline", "goal"):
        P = prim[m]; tp = time.time()
        ii = P["ii"]
        D, xmax, npair = bl.neighbour_sums(zsets, tabs[m], P["ra"][ii], P["dec_ii"], P["th_ii"], P["idx_ii"])
        sA = models[m].sigma_Y(P["th_ii"].astype(np.float64))
        dcs = bl.dc_response(sA) / sA if m == "white" else np.zeros_like(sA)       # MMF: compensated, mu = 0
        delta = np.zeros(len(P["q"])); dom = np.ones(len(P["q"]))
        delta[ii] = D.sum(1) / sA - ybar * dcs
        dom[ii] = (xmax.max(1) / sA < P["q"][ii]).astype(float)
        Ns, Nd = binned(P["q"], delta, dom, P["z"], P["ra"])
        out[f"{m}_Ns"] = Ns; out[f"{m}_Nd"] = Nd
        out[f"{m}_nsum"] = len(ii); out[f"{m}_nprim"] = len(P["q"])
        r = Nd.sum(2)[1] / Ns.sum(2)[1] - 1
        print(f"   [{m}] primaries q>0.5: {len(P['q']):,}; neighbour sums for {len(ii):,} (mean pairs {npair.mean():,.0f}) "
              f"{time.time()-tp:.0f}s; q>5 Delta N/N per z bin: " + " ".join(f"{100*v:+.2f}" for v in r)
              + f"; rss {rss_gb():.2f} GB", flush=True)
        del D, xmax, npair
    np.savez(f"{OUTD}/sky_{seed}.npz", **out)
    print(f"[sky {seed}] saved; total {time.time()-t0:.0f}s; peak rss {rss_gb():.2f} GB", flush=True)


def summary():
    files = sorted(glob.glob(f"{OUTD}/sky_*.npz"))
    S = [np.load(f) for f in files]
    print(f"=== s11 summary: {len(S)} fresh skies (seeds {[int(s['seed']) for s in S]}) ===")
    zlab = [f"{a:g}-{b:g}" for a, b in zip(ZB[:-1], ZB[1:])]
    # reference values quoted elsewhere (q > 5): s06d 3 stored skies (RESULTS before arbitration) and V1's 2 fresh skies
    ref = {"white": dict(s06d=[-4.94, -2.96, -1.85, -0.79, +0.11, +1.48, +4.99],
                         v1=[[-5.29, -3.03, -1.90, -0.78, +0.41, +1.62, +4.39], [-4.71, -2.93, -1.94, -0.93, +0.10, +0.92, +4.54]]),
           "baseline": dict(s06d=[+0.58, +0.89, +0.96, +1.28, +2.01, +3.10, +5.46],
                            v1=[[+0.39, +0.95, +1.00, +1.42, +1.93, +3.30, +5.98], [+0.90, +0.92, +0.95, +1.19, +2.13, +2.65, +4.99]]),
           "goal": dict(s06d=[+0.64, +0.94, +1.01, +1.33, +2.23, +3.40, +4.83],
                        v1=[[+0.77, +1.01, +1.01, +1.49, +2.10, +3.14, +5.67], [+1.12, +0.94, +0.90, +1.21, +2.21, +3.00, +5.25]])}
    for m in ("white", "baseline", "goal"):
        if not os.path.exists(f"{ROOT}/data_real/pd_tables_{m}.npz"):
            print(f"\n[{m}] data_real/pd_tables_{m}.npz missing (run s09 first): skipped")
            continue
        T = np.load(f"{ROOT}/data_real/pd_tables_{m}.npz")
        bins = range(len(ZB) - 1) if m == "white" else (5, 6)
        print(f"\n[{m}] Delta N/N [%] per z bin at q > 5  (direct: mean over skies +- std/sqrt(n))")
        Ns = np.array([s[f"{m}_Ns"] for s in S]); Nd = np.array([s[f"{m}_Nd"] for s in S])     # (nsky, nqc, nzb, nw)
        r = Nd.sum(3) / Ns.sum(3) - 1                                                              # (nsky, nqc, nzb)
        # within-sky jackknife over the 8 RA wedges (per sky), averaged in quadrature over skies
        jk = []
        for k in range(NW):
            keep = [w for w in range(NW) if w != k]
            jk.append(Nd[..., keep].sum(3) / Ns[..., keep].sum(3) - 1)
        jk = np.array(jk)
        sig_jk = np.sqrt((NW - 1) / NW * ((jk - jk.mean(0)) ** 2).sum(0))                         # (nsky, nqc, nzb)
        ic = 1
        for b in bins:
            v = 100 * r[:, ic, b]
            pred_sd = 100 * T["z_sd"][ic, b]
            v1 = [x[b] for x in ref[m]["v1"]]
            # like-for-like P(D): the MMF direct sums skip delta for primaries with q_A <= 1 (their P(D) share is z_trunc_q1)
            pd_full = 100 * T["z_dNN"][ic, b]
            pd_ll = pd_full - (100 * T["z_trunc_q1"][ic, b] if m != "white" else 0.0)
            err = v.std(ddof=1) / np.sqrt(len(v))
            print(f"   z {zlab[b]:>8s}: direct {v.mean():+.2f} +- {err:.2f} | P(D) s09 {pd_full:+.2f} (like-for-like {pd_ll:+.2f}; pull "
                  f"{(v.mean()-pd_ll)/err:+.1f}) | per-sky std: empirical {v.std(ddof=1):.2f}, within-sky jackknife "
                  f"{100*np.sqrt(np.mean(sig_jk[:, ic, b]**2)):.2f}, predicted (s09) {pred_sd:.2f} | s06d {ref[m]['s06d'][b]:+.2f}, "
                  f"V1 skies {v1[0]:+.2f} / {v1[1]:+.2f}")
            # are the 8 RA wedges of a sky independent?  var(sky totals of Delta N) vs sum over wedges of the wedge variance
            dN_w = Nd[:, ic, b, :] - Ns[:, ic, b, :]
            var_tot = dN_w.sum(1).var(ddof=1)
            var_w = np.mean(NW * dN_w.var(1, ddof=1))
            print(f"            Delta N per sky: mean {dN_w.sum(1).mean():.2f}, std over skies {np.sqrt(var_tot):.2f}; "
                  f"sqrt(8 x wedge variance) {np.sqrt(var_w):.2f} (equal if wedges are independent)")
        for ic2, qc in enumerate(QC):
            if qc == 5.0:
                continue
            rr = 100 * r[:, ic2, :]
            print(f"   q>{qc:.0f}: direct " + " ".join(f"{rr[:, b].mean():+.2f}+-{rr[:, b].std(ddof=1)/np.sqrt(len(rr)):.2f}" for b in bins)
                  + " | P(D) " + " ".join(f"{100*T['z_dNN'][ic2, b]:+.2f}" for b in bins))
        if m != "white":
            print(f"   (truncation q_A > 1 for the MMF neighbour sums; P(D) effect on Delta N/N at q>5: "
                  + " ".join(f"{100*T['z_trunc_q1'][1, b]:+.4f}%" for b in bins) + ")")
        else:
            # the old filter is not compensated: each sky subtracts its own mean y, which fluctuates (0.54% predicted)
            x = 100 * (np.array([float(s["ybar"]) for s in S]) / 3.589536e-7 - 1)
            print(f"   sky mean y vs the population expectation 3.5895e-7: offsets {np.round(x, 2)} % (std {x.std(ddof=1):.2f}%; "
                  f"Poisson expectation 0.54%)")
            for b in bins:
                v = 100 * r[:, 1, b]
                fit = np.polyfit(x, v, 1)
                print(f"   z {zlab[b]:>8s}: corr(Delta N/N, mean y) {np.corrcoef(x, v)[0, 1]:+.2f}, slope {fit[0]:+.2f} pp per 1%; "
                      f"per-sky std {v.std(ddof=1):.2f} -> {(v - np.polyval(fit, x)).std(ddof=2):.2f} without the mean-y trend "
                      f"(predicted {100*T['z_sd'][1, b]:.2f})")


if __name__ == "__main__":
    if sys.argv[1] == "summary":
        summary()
    else:
        one_sky(int(sys.argv[1]))
