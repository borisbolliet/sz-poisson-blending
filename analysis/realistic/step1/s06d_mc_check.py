"""Step 1 / s06d: independent check of the MMF compound-Poisson P(D) by DIRECT neighbour sums on the stored
full-sky M500 > 1e13 catalogues (data/catalogs/cat_M1e13_seed{1,2,3}.npz, read only).

[arbiter notes, after the two verification reports; the code below is unchanged and is what produced the log]
 * Not kernel-independent: the direct sums use the same rho tables as the FFT P(D) (s06a).  A direct count with an
   independent kernel is verifier 1's (verify_real/step1_derivation/v08_*).
 * The z-bin errors of part 2 (std of the 3 per-sky values / sqrt 3) are unreliable with 3 skies: they are typically
   1-4x, and up to ~10x (baseline, z = 2-3), below the per-sky scatter / sqrt 3 measured on 12 fresh skies (s11) and
   predicted by s09.  The redshift tilt is now quoted from the P(D) expectation (s09) and checked directly (s11, s12).
 * Peak memory footprint 5.22 GB (max RSS 4.34 GB): above the 5 GB budget.  s11 does the same per-sky work at 3.2 GB.

Method 1 (s06b): rate histogram from the analytic population (blendlib.neighbour_population) + FFT compound
Poisson (pdlib1.PDs), untruncated.
Method 2 (here): at random sky positions (Slivnyak), delta = sum_B rho(thA, thB, s) Y_B / sigma_Y(thA) over the
REALISED halos of each sky (zone search blendlib.neighbour_sums, the same rho table; support 5 thA + 5 thB + 150').
Compared: mean, variance, quantiles and P(delta > t), P(delta < -t).  Errors on the MC side: scatter of
60 blocks (3 skies x 20 RA wedges of 18 deg; the kernel support is <= ~4 deg, so blocks are nearly independent).
theta_A = the s06b primary nodes closest to 1, 4 and 16 arcmin.  20,000 random points per sky and theta_A.

Part 2 (independent check of the FRACTIONS): at every real primary (q_A = Y500/sigma_Y(theta500) > 0.5 for that
instrument, Y500 incl. scatter) of each sky, D_A = sum over all other halos of rho Y (same zone search, self
excluded) and the brightest single contribution x_max; dominance = x_max < q_A; the instrument noise is then
marginalised EXACTLY per object (q_b = q_A + n + delta, mu = 0 for the compensated MMF):
   P(det_b) = dom Phi(q_A + delta - q_cut),  boosted = dom [Phi(q_A+delta-q_cut) - Phi(q_A-q_cut)] 1(delta>0),
   lost = dom [Phi(q_A-q_cut) - Phi(q_A+delta-q_cut)] 1(delta<=0),  B2(eps) = dom Phi(.) 1(delta > eps q_A), ...
and the fractions are counted directly (no P(D), no rate histogram, no HMF integral), per sky.
Control: the same for the OLD white-noise filter (data/blending/rho_table.npz), where the map mean is removed
explicitly: delta -> delta - ybar DC / sigma_A with ybar the mean Y per arcmin^2 of THIS sky (convention c=1).
Redshift split at q_cut = 5: Delta N / N = N_det,blended / N_single - 1 in z bins (the driver of the old S8 bias).
"""
import os, sys, time, resource
for v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[v] = "1"
os.environ["NUMBA_NUM_THREADS"] = "4"
import numpy as np
from scipy.special import ndtr

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import pdlib1 as pl                      # noqa: E402
import blendlib as bl                    # noqa: E402
from szsim import sampler, signal as sg  # noqa: E402

ROOT = pl.ROOT
CAT = f"{ROOT}/data/catalogs/cat_M1e13_seed{{}}.npz"
TIERS = [0.0, 2.0, 10.0, 1e9]
NPTS = 20000
NBLK = 20
T_POS = np.array([0.1, 0.2, 0.5, 1.0, 2.0, 5.0])
T_NEG = np.array([0.1, 0.2, 0.5, 1.0])
QL = np.array([0.02275, 0.15866, 0.5, 0.84134, 0.97725])
QC = [4.0, 5.0, 6.0, 7.0]
KEYS = ["N_single", "N_det_b", "B2_0.1", "B2_0.2", "B2neg_0.1", "B3", "lost", "sat", "meanbias"]
ZB = np.array([0.01, 0.1, 0.25, 0.5, 1.0, 1.5, 2.0, 3.0])


def direct_fractions(qA, delta, xmx, z=None):
    """noise-marginalised blending fractions of one sky (convention c=1: delta is already mean-subtracted;
    mu = 0 for the MMF).  With z: also N_single and N_det_b in the ZB redshift bins at q_cut = 5."""
    dom = xmx < qA
    out = {}
    if z is not None:
        iz = np.clip(np.searchsorted(ZB, z) - 1, 0, len(ZB) - 2)
        out["zNs"] = np.bincount(iz, weights=ndtr(qA - 5.0), minlength=len(ZB) - 1)
        out["zNd"] = np.bincount(iz, weights=dom * ndtr(qA + delta - 5.0), minlength=len(ZB) - 1)
    for qc in QC:
        ps = ndtr(qA - qc); pb = ndtr(qA + delta - qc)
        det = dom * pb
        Ns, Nd = ps.sum(), det.sum()
        out[qc] = dict(N_single=Ns, N_det_b=Nd, **{"B2_0.1": (det * (delta > 0.1 * qA)).sum() / Nd,
                                                   "B2_0.2": (det * (delta > 0.2 * qA)).sum() / Nd,
                                                   "B2neg_0.1": (det * (delta < -0.1 * qA)).sum() / Nd},
                       B3=(dom * np.clip(pb - ps, 0, None) * (delta > 0)).sum() / Nd,
                       lost=(dom * np.clip(ps - pb, 0, None) * (delta <= 0)).sum() / Ns,
                       sat=((~dom) * ps).sum() / Ns, meanbias=(det * delta / qA).sum() / Nd)
    return out


def rss_gb():
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e9


def main():
    t0 = time.time()
    modes = ["baseline", "goal"]
    tabs = {m: bl.RhoTable(np.load(f"{ROOT}/data_real/rho_mmf_{m}.npz")) for m in modes}
    models = {m: pl.make_model(("so", m, 100.0)) for m in modes}
    pmodes = modes + ["white"]                       # primaries part: also the old white filter as a control
    tabs["white"] = bl.load_rho_table(); models["white"] = sg.SZModel()
    TA = tabs["baseline"].th[::2]
    thAs = np.array([TA[int(np.argmin(np.abs(TA - s)))] for s in (1.0, 4.0, 16.0)])
    print(f"theta_A = {np.round(thAs, 4)}", flush=True)
    MC = {(m, i): [] for m in modes for i in range(len(thAs))}     # list of (delta, block) per sky
    PR = {}                                                        # (mode, floor) -> list over skies of fractions
    for seed in (1, 2, 3):
        ts = time.time()
        cat = sampler.load_catalog(CAT.format(seed), fields=["ra", "dec", "z", "M500", "Y500", "theta500"])
        N = len(cat["ra"])
        cls = bl.mass_class(cat["M500"]); del cat["M500"]
        ybar = np.array([cat["Y500"][cls == c].astype(np.float64).sum() for c in range(3)]) / bl.ARCMIN2_FULLSKY
        prim = {}
        for m in pmodes:
            qm = cat["Y500"] / models[m].sigma_Y(cat["theta500"])
            sel = np.nonzero(qm > 0.5)[0]
            prim[m] = dict(ra=cat["ra"][sel].astype(np.float64), dec=cat["dec"][sel].astype(np.float64),
                           th=cat["theta500"][sel].astype(np.float64), q=qm[sel].astype(np.float64), idx=sel.astype(np.int64),
                           z=cat["z"][sel].astype(np.float64))
            del qm
        del cat["z"]
        orig = np.arange(N, dtype=np.int32)
        zsets = []
        th = cat["theta500"]
        for lo, hi in zip(TIERS[:-1], TIERS[1:]):
            s = (th > lo) & (th <= hi)
            zsets.append(bl.ZoneSet(cat["ra"][s], cat["dec"][s], cat["Y500"][s], th[s], cls[s], orig[s]))
        del cat, th, cls, orig
        print(f"[sky {seed}] N = {N:,}; zones built {time.time()-ts:.0f}s; rss {rss_gb():.2f} GB", flush=True)
        rng = np.random.default_rng(777 + seed)
        for i, thA in enumerate(thAs):
            ra = 360 * rng.random(NPTS); dec = np.degrees(np.arcsin(2 * rng.random(NPTS) - 1))
            blk = np.minimum((ra / (360.0 / NBLK)).astype(int), NBLK - 1) + NBLK * (seed - 1)
            for m in modes:
                D, xmax, npair = bl.neighbour_sums(zsets, tabs[m], ra, dec, np.full(NPTS, thA), np.full(NPTS, -1))
                sA = models[m].sigma_Y(thA)
                MC[(m, i)].append((D.sum(1) / sA, blk))
            print(f"   thA={thA:.3f}': done ({time.time()-ts:.0f}s; mean pairs/point {npair.mean():,.0f})", flush=True)
        for m in pmodes:
            P = prim[m]; tp = time.time()
            D, xmax, npair = bl.neighbour_sums(zsets, tabs[m], P["ra"], P["dec"], P["th"], P["idx"])
            sA = models[m].sigma_Y(P["th"])
            dcs = bl.dc_response(sA) / sA if m == "white" else np.zeros_like(sA)   # MMF: compensated, mu = 0
            for fl, (dl, xm, yb) in {"1e13": (D.sum(1), xmax.max(1), ybar.sum()), "1e14": (D[:, 2], xmax[:, 2], ybar[2])}.items():
                PR.setdefault((m, fl), []).append(direct_fractions(P["q"], dl / sA - yb * dcs, xm / sA, P["z"]))
            print(f"   [{m}] primaries: {len(P['q']):,} (q>0.5), mean pairs {npair.mean():,.0f}, {time.time()-tp:.0f}s; "
                  f"rss {rss_gb():.2f} GB", flush=True)
            del D, xmax, npair
        del zsets, prim
    print(f"MC done {time.time()-t0:.0f}s, peak rss {rss_gb():.2f} GB", flush=True)

    pop = bl.neighbour_population(zsub=1, msub=6, ngh=16)
    pd_un = pl.PDs(N=2 ** 20, dd=0.001, x_small=0.002, off=60.0)
    g = pd_un.grid + 0.5 * pd_un.dd                       # cdf[j] = P(delta <= g_j + dd/2)
    out = {}
    for m in modes:
        for i, thA in enumerate(thAs):
            sA = models[m].sigma_Y(thA)
            R = pl.rate_signed(thA, sA, pop, tabs[m])
            mu, var = pl.moments(R, (0, 1, 2))
            W, X, X2, lam_cap = pl.signed_arrays(R, (0, 1, 2))
            p = pd_un.pdf(W, X, X2)
            cdf = np.cumsum(p) * np.exp(-lam_cap)
            q_an = np.interp(QL, cdf, g)
            ccdf_an = 1 - np.interp(T_POS, g, cdf)
            cdfn_an = np.interp(-T_NEG, g, cdf)
            d = np.concatenate([x[0] for x in MC[(m, i)]]); b = np.concatenate([x[1] for x in MC[(m, i)]])
            nb_ = b.max() + 1

            def blockstat(f):
                vals = np.array([f(d[b == k]) for k in range(nb_)])
                return f(d), vals.std(ddof=1) / np.sqrt(nb_)

            print(f"\n[{m}] thA = {thA:.3f}' ({len(d):,} random points, {nb_} blocks)")
            mm, em = blockstat(np.mean); vv, ev = blockstat(np.var)
            print(f"   mean   : analytic {mu:+.5f}   MC {mm:+.5f} +- {em:.5f}   (pull {(mm-mu)/em:+.2f})")
            print(f"   var    : analytic {var:.5f}   MC {vv:.5f} +- {ev:.5f}   (pull {(vv-var)/ev:+.2f}; heavy-tailed)")
            for qq, qa in zip(QL, q_an):
                v, e = blockstat(lambda x, qq=qq: np.quantile(x, qq))
                print(f"   q{qq*100:5.2f}% : analytic {qa:+.4f}   MC {v:+.4f} +- {e:.4f}   (pull {(v-qa)/e:+.2f})")
            for t, a in zip(T_POS, ccdf_an):
                v, e = blockstat(lambda x, t=t: np.mean(x > t))
                print(f"   P(delta > {t:4.1f}) : analytic {a:.5f}   MC {v:.5f} +- {e:.5f}   (pull {(v-a)/max(e,1e-9):+.2f})")
            for t, a in zip(T_NEG, cdfn_an):
                v, e = blockstat(lambda x, t=t: np.mean(x < -t))
                print(f"   P(delta < -{t:3.1f}) : analytic {a:.5f}   MC {v:.5f} +- {e:.5f}   (pull {(v-a)/max(e,1e-9):+.2f})")
            out[f"{m}_{i}_mc"] = d
    print("\n=== part 2: blending fractions counted directly on the 3 stored skies vs the FFT P(D) (s06b) ===")
    print("    (MC: mean of the 3 per-sky values +- their standard error; full sky each)")
    for (m, fl), lst in PR.items():
        d = np.load(f"{ROOT}/data_real/pd_fractions_{m}.npz")
        cols = list(d["cols"]); rows = d[f"rows_{fl}"]
        print(f"[{m}] neighbours M>{fl}")
        for qc in QC:
            row = rows[[r[0] for r in rows].index(qc)]
            parts = []
            for k in KEYS:
                v = np.array([x[qc][k] for x in lst]); mu_, e = v.mean(), v.std(ddof=1) / np.sqrt(len(v))
                a = row[cols.index(k)]
                if k.startswith("N_"):
                    parts.append(f"{k} {mu_:,.0f}+-{e:,.0f} (FFT {a:,.0f})")
                else:
                    parts.append(f"{k} {mu_:.4f}+-{e:.4f} (FFT {a:.4f})")
            print(f"   q>{qc:.0f}: " + "; ".join(parts))
        zr = np.array([x["zNd"] / x["zNs"] - 1 for x in lst])
        print(f"   q>5, Delta N/N in z bins {list(ZB)}: " + " ".join(f"{a:+.4f}+-{b:.4f}" for a, b in
                                                            zip(zr.mean(0), zr.std(0, ddof=1) / np.sqrt(len(zr)))))
        print(f"        N_single per z bin (mean over skies): " + " ".join(f"{v:,.0f}" for v in np.mean([x['zNs'] for x in lst], 0)))
    np.savez(f"{ROOT}/data_real/mc_randpts_check.npz", thAs=thAs, **out)
    print(f"total {time.time()-t0:.0f}s, peak rss {rss_gb():.2f} GB")


if __name__ == "__main__":
    main()
