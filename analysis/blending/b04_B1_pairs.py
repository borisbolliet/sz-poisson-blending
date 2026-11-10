"""B1 'resolved-pair blend': two DETECTED clusters (each q_obs > q_cut on its own) closer than theta_b.

Analytic (exact for a Poisson process with independent marks): detections are an independent
thinning of the halo Poisson process, hence Poisson with surface density Sigma_det = N_det/(4 pi).
By Slivnyak's theorem the other detections around a detected cluster are Poisson with the same
density, so the number k of other detections within theta_b of a detection is Poisson with mean
Sigma_det * A_cap(theta_b), A_cap = 2 pi (1 - cos theta_b):
    f_{>=1}(theta_b) = 1 - exp(-Sigma_det A),   f_{>=k} = Poisson CCDF.
B1-phys (size-aware merge criterion): two equal sources merge into one filtered-map peak when their
separation is < 2 s_infl, s_infl = inflection radius of the filtered profile rho(th, th, s).
For unequal sizes we use d_merge = s_infl(thA) + s_infl(thB).  Analytic:
    f = < 1 - exp( - int dSigma_det(thB) A_cap(s_infl(thA) + s_infl(thB)) ) >_{A in det}
MC: stored full skies (M>1e13, seeds 1-3) + floor-1e14 skies (stored 101-110 + drawn 1001-1040).
"""
import os, sys, time
os.environ.setdefault("OMP_NUM_THREADS", "2"); os.environ.setdefault("NUMBA_NUM_THREADS", "2")
import numpy as np
from scipy.spatial import cKDTree
from scipy.stats import poisson
from scipy.interpolate import CubicSpline
import healpy as hp
import resource


def rss_gb():
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e9

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import blendlib as bl
from szsim import sampler, signal as sg, cosmo_hmf as ch

QCUTS = [4.0, 5.0, 6.0, 7.0]
THB = np.array([0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 5.0])
KMAX = 3
CATS = "/Users/boris/GitHub/sz-poisson-blending/data/catalogs/"


def unitvec(ra, dec):
    ra = np.radians(ra.astype(np.float64)); dec = np.radians(dec.astype(np.float64))
    return np.column_stack([np.cos(dec) * np.cos(ra), np.cos(dec) * np.sin(ra), np.sin(dec)])


def chord(theta_arcmin):
    return 2 * np.sin(0.5 * np.asarray(theta_arcmin) * ch.ARCMIN)


def s_infl_table(tab):
    """Inflection radius of rho(th, th, s) for each table node."""
    out = np.zeros(tab.NT)
    for i, th in enumerate(tab.th):
        S = 5 * th + 5 * th + tab.R_BEAM
        s = S * tab.t ** 2
        cs = CubicSpline(s, tab.rho[i, i])
        sf = np.linspace(0, S * 0.6, 20000)
        d2 = cs(sf, 2)
        k = np.nonzero(d2 > 0)[0][0]
        out[i] = sf[k]
    return out


def jackknife_ratio(Nn, Nd):
    """Ratio sum(Nn)/sum(Nd) of per-region sums with delete-one-region jackknife error."""
    r = Nn.sum() / Nd.sum()
    ok = Nd > 0
    rj = (Nn.sum() - Nn[ok]) / (Nd.sum() - Nd[ok])
    n = ok.sum()
    return r, np.sqrt((n - 1) / n * ((rj - rj.mean()) ** 2).sum())


def main():
    t0 = time.time()
    m = sg.SZModel()
    tab = bl.load_rho_table()
    sinf = s_infl_table(tab)
    print("inflection radius s_infl(theta500) of the filtered profile [arcmin]:")
    for t in (0.3, 0.5, 1, 2, 3, 5, 10):
        print(f"   theta500={t:5.1f}'  s_infl={np.interp(np.log(t), np.log(tab.th), sinf):.3f}'")
    print(f"   point-source expectation sqrt(2) sigma_b = {np.sqrt(2)*m.sigma_beam_rad/ch.ARCMIN:.3f}'")
    s_of = lambda th: np.interp(np.log(th), np.log(tab.th), sinf)

    # ---- analytic: detection densities and theta distribution of detections
    T = ch.default_table()
    M = np.exp(T.lnM)
    wz = T._weights(T.z, T.zmin, T.zmax)
    analytic = {}
    for floor in (1e13, 1e14):
        wM = T._weights(T.lnM, np.log(floor), T.lnM[-1])
        for qc in QCUTS:
            P = np.array([m.p_det(M, zz, qc) for zz in T.z])
            W = (wz[:, None] * T.d2N * P) * wM[None, :]
            TH = np.array([m.theta500(M, zz) for zz in T.z])
            Ndet = W.sum()
            # theta histogram of detections (log bins)
            edges = np.geomspace(0.2, 80, 401)
            h, _ = np.histogram(TH.ravel(), bins=edges, weights=W.ravel())
            thc = np.sqrt(edges[1:] * edges[:-1])
            sig = Ndet / (4 * np.pi)  # per sr
            Acap = 2 * np.pi * (1 - np.cos(THB * ch.ARCMIN))
            mu = sig * Acap
            fk = np.array([[poisson.sf(k - 1, mm) for mm in mu] for k in range(1, KMAX + 1)])
            # B1-phys
            sA = s_of(thc)
            dmerge = sA[:, None] + sA[None, :]
            lamA = (h[None, :] / (4 * np.pi) * 2 * np.pi * (1 - np.cos(dmerge * ch.ARCMIN))).sum(1)
            fphys = (h * (1 - np.exp(-lamA))).sum() / h.sum()
            analytic[(floor, qc)] = dict(Ndet=Ndet, fk=fk, fphys=fphys, mu=mu)
            print(f"[analytic] floor {floor:.0e} q_cut={qc}: N_det={Ndet:,.1f}, Sigma_det={Ndet/(4*np.pi)*ch.ARCMIN**2*3600:.4f} /deg^2; "
                  f"f>=1(theta_b={list(THB)}) = {np.round(fk[0], 6)}; f_phys = {fphys:.6f}")
    print(f"analytic done {time.time()-t0:.1f}s, peak rss {rss_gb():.2f} GB", flush=True)

    # ---- MC
    results = {}
    sky_sets = {1e13: [("file", CATS + f"cat_M1e13_seed{s}.npz") for s in (1, 2, 3)],
                1e14: [("file", CATS + f"cat_M1e14_seed{s}.npz") for s in range(101, 111)]
                + [("draw", s) for s in range(1001, 1041)]}
    for floor, skies in sky_sets.items():
        nlab = 48 * len(skies)
        # per-region sums only (memory): Nn[region, k, theta_b], Nd[region], Nph[region]
        acc = {qc: dict(Nn=np.zeros((nlab, KMAX, len(THB))), Nd=np.zeros(nlab), Nph=np.zeros(nlab), nsky=0)
               for qc in QCUTS}
        for isky, (kind, src) in enumerate(skies):
            if kind == "file":
                cat = sampler.load_catalog(src, fields=["ra", "dec", "theta500", "q_obs"], qmin=min(QCUTS))
            else:
                cat = sampler.draw_catalog(src, Mmin=1e14, fields=("ra", "dec", "theta500", "q", "q_obs"))
                s = cat["q_obs"] >= min(QCUTS)
                cat = {k: cat[k][s] for k in ("ra", "dec", "theta500", "q_obs")}
            for qc in QCUTS:
                s = cat["q_obs"] > qc
                v = unitvec(cat["ra"][s], cat["dec"][s]); th = cat["theta500"][s].astype(np.float64)
                tree = cKDTree(v)
                counts = np.stack([tree.query_ball_point(v, chord(tb), return_length=True) - 1 for tb in THB])
                a = acc[qc]
                lab = hp.ang2pix(2, cat["ra"][s].astype(float), cat["dec"][s].astype(float), lonlat=True) + 48 * isky
                for k in range(KMAX):
                    for it in range(len(THB)):
                        a["Nn"][:, k, it] += np.bincount(lab, weights=(counts[it] >= k + 1), minlength=nlab)
                a["Nd"] += np.bincount(lab, minlength=nlab)
                # B1-phys pairs
                sA = s_of(th)
                rmax = 2 * sA.max()
                pairs = tree.query_pairs(chord(rmax), output_type="ndarray")
                if len(pairs):
                    d = 2 * np.arcsin(np.clip(np.linalg.norm(v[pairs[:, 0]] - v[pairs[:, 1]], axis=1) / 2, 0, 1)) / ch.ARCMIN
                    mrg = d < sA[pairs[:, 0]] + sA[pairs[:, 1]]
                    flag = np.zeros(s.sum()); flag[pairs[mrg, 0]] = 1; flag[pairs[mrg, 1]] = 1
                else:
                    flag = np.zeros(s.sum())
                a["Nph"] += np.bincount(lab, weights=flag, minlength=nlab)
                a["nsky"] += 1
            if isky % 10 == 0 or floor == 1e13:
                print(f"   floor {floor:.0e} sky {isky} done {time.time()-t0:.1f}s, peak rss {rss_gb():.2f} GB", flush=True)
        for qc in QCUTS:
            a = acc[qc]
            N_mean = a["Nd"].sum() / len(skies)
            an = analytic[(floor, qc)]
            print(f"\n[MC] floor {floor:.0e}, q_cut={qc}: {len(skies)} skies, <N_det> = {N_mean:,.1f} "
                  f"(analytic {an['Ndet']:,.1f})")
            for k in range(KMAX):
                row = []
                for it, tb in enumerate(THB):
                    r, e = jackknife_ratio(a["Nn"][:, k, it], a["Nd"])
                    nev = a["Nn"][:, k, it].sum()  # z-scores only for >= 20 MC events
                    row.append((tb, r, e, an["fk"][k, it], (r - an["fk"][k, it]) / e if (e > 0 and nev >= 20) else np.nan))
                    results[(floor, qc, k + 1, tb)] = row[-1]
                print(f"   P(>= {k+1} other det within theta_b): " + "; ".join(
                    f"{tb}': MC {r:.3e}±{e:.1e} an {f:.3e} ({z:+.2f}σ)" for tb, r, e, f, z in row))
            r, e = jackknife_ratio(a["Nph"], a["Nd"])
            results[(floor, qc, "phys")] = (r, e, an["fphys"])
            print(f"   B1-phys (sep < s_infl(A)+s_infl(B)): MC {r:.4e}±{e:.1e}  analytic {an['fphys']:.4e} "
                  f"({(r-an['fphys'])/e:+.2f}σ)")
    zs = np.array([v[4] for k, v in results.items() if len(k) == 4 and np.isfinite(v[4])])
    zp = np.array([(v[0] - v[2]) / v[1] for k, v in results.items() if len(k) == 3])
    print(f"\nall B1 tests: {len(zs)} z-scores, rms {np.sqrt((zs**2).mean()):.2f}, max|z| {np.abs(zs).max():.2f}; "
          f"B1-phys: {len(zp)} z-scores rms {np.sqrt((zp**2).mean()):.2f}, max|z| {np.abs(zp).max():.2f}")
    np.save(os.path.join(bl.DATA, "B1_results.npy"), np.array([(str(k), v) for k, v in results.items()], dtype=object),
            allow_pickle=True)
    np.savez(os.path.join(bl.DATA, "B1_analytic.npz"), THB=THB, QCUTS=QCUTS, sinf=sinf, th=tab.th,
             **{f"fk_{int(np.log10(f))}_{int(q)}": v["fk"] for (f, q), v in analytic.items()},
             **{f"fphys_{int(np.log10(f))}_{int(q)}": v["fphys"] for (f, q), v in analytic.items()},
             **{f"Ndet_{int(np.log10(f))}_{int(q)}": v["Ndet"] for (f, q), v in analytic.items()})
    # MC arrays for figures
    mc = {}
    for (floor, qc, k, tb), v in [(k, v) for k, v in results.items() if len(k) == 4]:
        mc[f"mc_{int(np.log10(floor))}_{int(qc)}_{k}_{tb}"] = np.array(v)
    for (floor, qc, _), v in [(k, v) for k, v in results.items() if len(k) == 3]:
        mc[f"mcphys_{int(np.log10(floor))}_{int(qc)}"] = np.array(v)
    np.savez(os.path.join(bl.DATA, "B1_mc.npz"), **mc)
    print(f"total {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
