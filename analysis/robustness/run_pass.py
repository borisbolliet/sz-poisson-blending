"""One full-sky realisation for one sensitivity PASS (see configs.py): draw M>1e13 halos, compute the
grouped neighbour sums for all primaries, and store per-jackknife-region counts for every config.

Definitions (same as analysis/blending, so the numbers are comparable):
  primary A: true mean SNR q_A > 0.5 (q_A = Y_A / sigma_Y(theta_A), Y scatter included)
  n_A = q_obs - q (unit Gaussian noise), single (isolated) detection: q_A + n_A > q_cut
  x_B = rho_AB Y_B / sigma_A,  delta_A = sum_B x_B (neighbours kept by floor/zmax)
  mu_A = sigma_A RATIO ybar_sky / N_w  (response to the sky-mean y of the kept neighbours)
  c=0 raw:      q_b = q_A + n_A + delta_A ;   c=1 monopole-subtracted: q_b = q_A + n_A + delta_A - mu_A
  dominant: max_B x_B < q_A ;  det_b = (q_b > q_cut) & dominant
  B1(theta, k): fraction of single detections with >= k other single detections within theta (Q1)
  B2(eps): fraction of det_b with (delta - c mu)/q_A > eps (Y over-estimated by > eps)
  B3: fraction of det_b with q_A + n_A < q_cut (detection owed to blending)
  sat: single detections that are not dominant; lost: single & dominant & q_b < q_cut
  overlap: neighbour physically overlapping A in 3D (comoving sep < (R500A + R500B)(1+z))
Usage: python run_pass.py PASS SEED
"""
import os, sys, time, json, resource
os.environ.setdefault("NUMBA_NUM_THREADS", "2"); os.environ.setdefault("OMP_NUM_THREADS", "2")
import numpy as np
import healpy as hp
from scipy.spatial import cKDTree

HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import blendlib as bl
import configs as C
from szsim import sampler, signal as sg, cosmo_hmf as ch

OUT = os.path.join(ch.DATA_DIR, "robustness"); os.makedirs(OUT, exist_ok=True)


def rss():
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e9


def main(pname, seed):
    P = C.PASSES[pname]
    t0 = time.time()
    model = sg.SZModel(beam_fwhm_arcmin=P["beam"], noise_y_arcmin=P["noise"], sigma_lnY=P["sig"])
    table = ch.default_table() if P["hmf"] == "tinker08" else ch.HMFTable(model=P["hmf"])
    rt = bl.RhoTable(P["beam"])
    cat = sampler.draw_catalog(seed, Mmin=1e13, model=model, table=table)
    bl.add_distances(cat, model)
    N = len(cat["ra"])
    print(f"[{pname} seed {seed}] N={N} drawn {time.time()-t0:.1f}s rss {rss():.2f} GB", flush=True)
    grp = (3 * np.digitize(cat["M500"], C.MCLASS_EDGES[1:3]) + np.digitize(cat["z"], C.ZCLASS_EDGES[1:3])).astype(np.int8)
    ysum = np.bincount(grp, weights=cat["Y500"].astype(np.float64), minlength=C.G)  # arcmin^2
    ybar_g = sg.RATIO_5R500 * ysum * bl.ARCMIN ** 2 / (4 * np.pi)                     # per group
    fmin = min(c["f"] for c in P["configs"].values())
    prim = np.nonzero(cat["q"] > C.QPRIM * fmin)[0]
    cand = {k: cat[k][prim].copy() for k in ("ra", "dec", "z", "M500", "theta500", "q", "q_obs", "chi", "Rcom")}
    for k in ("z", "M500", "q", "q_obs"):
        del cat[k]
    print(f"   primaries {len(prim)} (q > {C.QPRIM*fmin}); building grids", flush=True)
    t = time.time()
    ids = np.arange(N, dtype=np.int32)
    NG = bl.NeighbourGrids(cat, ids, grp, P["beam"], thA_typ=float(np.median(cand["theta500"])))
    del cat, grp, ids
    print(f"   grids {time.time()-t:.1f}s rss {rss():.2f} GB", flush=True)
    t = time.time()
    S, X, O = NG.run(cand, prim.astype(np.int32), rt, C.G)
    del NG
    print(f"   neighbour sums {time.time()-t:.1f}s rss {rss():.2f} GB", flush=True)

    sA = model.sigma_Y(cand["theta500"].astype(np.float64))
    q0 = cand["q"].astype(np.float64); n = (cand["q_obs"] - cand["q"]).astype(np.float64)
    lab = hp.ang2pix(C.NSIDE_JK, cand["ra"].astype(np.float64), cand["dec"].astype(np.float64), lonlat=True)
    NREG = hp.nside2npix(C.NSIDE_JK)
    xyz = np.column_stack(bl.unitvec(cand["ra"], cand["dec"]))
    res = {"meta": dict(pass_=pname, seed=seed, N=N, nprim=len(prim), ybar_g=ybar_g.tolist(),
                        beam=P["beam"], noise=P["noise"], sig=P["sig"], hmf=P["hmf"])}
    for cname, cf in P["configs"].items():
        f, gl, zmax = cf["f"], C.groups_for(cf["floor"], cf["zmax"]), cf["zmax"]
        q = q0 / f
        keep = (q > C.QPRIM) & (cand["z"] < zmax)
        D = np.zeros(len(q0)); Xm = np.zeros(len(q0)); Ov = np.zeros(len(q0)); has_ovl = np.zeros(len(q0), bool)
        for g in gl:
            D += S[:, g]; Ov += O[:, g]; np.maximum(Xm, X[:, g], out=Xm); has_ovl |= O[:, g] > 0
        D /= sA * f; Xm /= sA * f; Ov /= sA * f
        mu = sA * sg.RATIO_5R500 * ybar_g[gl].sum() / model.noise_y_arcmin ** 2 / f
        dom = Xm < q
        r = {}
        for qc in C.QCUTS:
            single = keep & (q + n > qc)
            key = f"{qc:g}"
            cnt = lambda m: np.bincount(lab[m], minlength=NREG)
            r[f"single_{key}"] = cnt(single)
            r[f"sat_{key}"] = cnt(single & ~dom)
            r[f"single_hasovl_{key}"] = cnt(single & has_ovl)
            for c in (0, 1):
                dp = D - c * mu
                qb = q + n + dp
                det = keep & (qb > qc) & dom
                r[f"detb{c}_{key}"] = cnt(det)
                r[f"b3_{c}_{key}"] = cnt(det & (q + n < qc))
                r[f"lost{c}_{key}"] = cnt(single & dom & (qb < qc))
                for e in C.EPS:
                    r[f"b2_{c}_{e:g}_{key}"] = cnt(det & (dp / q > e))
                    r[f"b2ovl_{c}_{e:g}_{key}"] = cnt(det & (Ov / q > e))
                    # B2 flags where the physically overlapping part alone exceeds eps
                if c == 1:
                    detb1 = det
            # B1 / Q1: pairs among single detections (and among det_b, c=1)
            for tag, sel in (("s", single), ("b", detb1)):
                idx = np.nonzero(sel)[0]
                tree = cKDTree(xyz[idx])
                for th in C.THETAS:
                    chord = 2 * np.sin(np.radians(th / 60) / 2)
                    nn = tree.query_ball_point(xyz[idx], chord, return_length=True) - 1
                    for k in range(1, C.KMAX + 1):
                        r[f"B1{tag}_{th:g}_{k}_{key}"] = cnt(idx[nn >= k])
                if tag == "s" and qc == 5.0 and f == 1.0:
                    # exclusion test: detected pairs (any separation < 150') that overlap in 3D
                    pairs = tree.query_pairs(2 * np.sin(np.radians(150 / 60) / 2), output_type="ndarray")
                    a, b = idx[pairs[:, 0]], idx[pairs[:, 1]]
                    sep = np.degrees(2 * np.arcsin(np.linalg.norm(xyz[a] - xyz[b], axis=1) / 2)) * 60
                    chia = cand["chi"][a].astype(np.float64); chib = cand["chi"][b].astype(np.float64)
                    rp = 0.5 * (chia + chib) * np.radians(sep / 60)
                    d3 = np.sqrt(rp ** 2 + (chia - chib) ** 2)
                    ovl = d3 < cand["Rcom"][a] + cand["Rcom"][b]
                    r["pairs_sep_ovl"] = np.array([np.sum(ovl & (sep < th)) for th in C.THETAS + [150.0]])
                    r["pairs_sep_all"] = np.array([np.sum(sep < th) for th in C.THETAS + [150.0]])
                    r["pairs_ovl_dz"] = np.abs(cand["z"][a] - cand["z"][b])[ovl].astype(np.float64)
        # mean blending moments over single detections at q_cut = 5 (for scaling plots)
        s5 = keep & (q + n > 5.0)
        r["mean_delta_single5"] = np.array([D[s5].mean(), mu[s5].mean(), Ov[s5].mean(), np.median(D[s5] - mu[s5])])
        # per-jackknife-region sums (added after verification) -> region-jackknife error on <delta - mu>
        r["sumD5_reg"] = np.bincount(lab[s5], weights=D[s5], minlength=NREG)
        r["summu5_reg"] = np.bincount(lab[s5], weights=mu[s5], minlength=NREG)
        r["n5_reg"] = np.bincount(lab[s5], minlength=NREG)
        r["n_prim"] = np.array([keep.sum()])
        res[cname] = r
    fn = os.path.join(OUT, f"pass_{pname}_seed{seed}.npz")
    flat = {"meta": np.array(json.dumps(res["meta"]))}
    for cname in P["configs"]:
        for k, v in res[cname].items():
            flat[f"{cname}|{k}"] = v
    np.savez(fn, **flat)
    print(f"[{pname} seed {seed}] saved {fn}; total {time.time()-t0:.0f}s peak rss {rss():.2f} GB", flush=True)


if __name__ == "__main__":
    main(sys.argv[1], int(sys.argv[2]))
