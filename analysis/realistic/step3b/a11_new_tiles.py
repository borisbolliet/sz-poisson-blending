"""Step 3b arbitration / a11: is the sims verifier's baseline paired loss a fluctuation or a property of its sky generator?

a02/a03 showed that the build's code reproduces the verifier's per-halo results on the verifier's own maps (same
credited halos and copies), so the remaining difference (R_p -1.71 +- 0.13% on its 400 tiles vs -1.36 +- 0.06% on the
build's 1,548) lies in the sky samples.  a02 also ran the build's pipeline on 400 NEW tiles made with the verifier's
generators (halos 9_310_000 + t, CMB 9_320_000 + t, noise 9_330_000 + t, t = 400..799; baseline).  Here the same
statistics as a01 / a03 for these new tiles, against the build's tiles and the verifier's first 400:
  R_p at q > 5 (total, step 3a's coarse z bins), per-tile N_iso - N_det, its slope with the number of bright halos,
  fof 10' removal of credited halos; and the pooled values (all tiles).
Usage: python a11_new_tiles.py
"""
import os, sys, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import arblib as A
A.threads(1)
import numpy as np

E = 5.0


def summary(label, D):
    nt = D["ntiles"]
    det, iso = D["qo_none"] > E, D["qiso"] > E
    nd = A.per_tile(D["t"], det.astype(float), nt); ni = A.per_tile(D["t"], iso.astype(float), nt)
    W = A.boot_w(nt, 2000, 31)
    v, e, _ = A.ratio_boot(nd, ni, W)
    zs = []
    for a, b in zip(A.ZC[:-1], A.ZC[1:]):
        m = (D["z"] >= a) & (D["z"] < b)
        x = A.per_tile(D["t"][m], det[m].astype(float), nt); y = A.per_tile(D["t"][m], iso[m].astype(float), nt)
        zs.append(A.ratio_boot(x, y, W)[:2])
    x = ni - nd
    bright = A.per_tile(D["t"], (D["qp"] > 20).astype(float), nt)
    Xm = np.stack([np.ones(nt), bright - bright.mean()], 1)
    coef, *_ = np.linalg.lstsq(Xm, x, rcond=None)
    res = x - Xm @ coef
    se = np.sqrt(np.linalg.inv(Xm.T @ Xm)[1, 1] * res.var(ddof=2))
    nf = A.per_tile(D["t"], (D["qo_fof10"] > E).astype(float), nt)
    rem, erem, _ = A.ratio_boot(nf, nd, W)
    print(f"  {label:34s} tiles {nt:4d}: R_p = {A.fmt(v, e)}; by z: " + ", ".join(f"{100 * a:+.1f}+-{100 * b:.1f}" for a, b in zs)
          + f"; N_iso - N_det per tile {x.mean():.2f} +- {x.std(ddof=1) / np.sqrt(nt):.2f} (std {x.std(ddof=1):.2f}); slope with bright "
          f"halos (q' > 20) {coef[1]:+.3f} +- {se:.3f}; fof 10' removes {-100 * rem:.2f} +- {100 * erem:.2f}% of the credited halos", flush=True)
    return dict(nd=nd, ni=ni, x=x, v=v, e=e, zs=zs)


def pool(parts):
    keys = ("t", "z", "th", "qp", "qiso", "qo_none", "qo_closest1", "qo_fof10")
    off, out = 0, {k: [] for k in keys}
    for D in parts:
        for k in keys:
            out[k].append(D[k] + off if k == "t" else D[k])
        off += D["ntiles"]
    out = {k: np.concatenate(v) for k, v in out.items()}
    out["ntiles"] = off
    return out


if __name__ == "__main__":
    t00 = time.time()
    for mode in ("baseline",):
        print(f"=================== {mode} ===================", flush=True)
        Bq, Bn, S = A.load_build(mode, "q2"), A.load_build(mode, "new"), A.load_sims(mode)
        X = A.load_xpipe(mode, 400, 800)
        r = {}
        for lab, D in (("build Q2 (324)", Bq), ("build new tiles (1,224)", Bn), ("sims verifier, its tiles 0-399", S),
                       ("arbiter: verifier's generator, 400-799", X)):
            r[lab] = summary(lab, D)
        Ball = A.load_build(mode, "all")
        r["build all"] = summary("build all (1,548)", Ball)
        r["sims gen all"] = summary("verifier's generator, all 800", pool([S, X]))
        r["pooled"] = summary("pooled: build + verifier's generator", pool([Ball, S, X]))
        a, b = r["arbiter: verifier's generator, 400-799"], r["build all"]
        d = a["x"].mean() - b["x"].mean(); e = np.sqrt(a["x"].var(ddof=1) / a["x"].size + b["x"].var(ddof=1) / b["x"].size)
        print(f"  new 400 tiles - build (per tile N_iso - N_det): {d:+.2f} +- {e:.2f} ({d / e:+.2f} sigma); R_p new - build "
              f"{100 * (a['v'] - b['v']):+.2f} +- {100 * np.hypot(a['e'], b['e']):.2f} pp", flush=True)
        a = r["sims gen all"]
        d = a["x"].mean() - b["x"].mean(); e = np.sqrt(a["x"].var(ddof=1) / a["x"].size + b["x"].var(ddof=1) / b["x"].size)
        print(f"  verifier's generator (800 tiles) - build (1,548): per tile {d:+.2f} +- {e:.2f} ({d / e:+.2f} sigma); R_p "
              f"{100 * (a['v'] - b['v']):+.2f} +- {100 * np.hypot(a['e'], b['e']):.2f} pp; z < 0.1: {100 * (a['zs'][0][0] - b['zs'][0][0]):+.1f} +- "
              f"{100 * np.hypot(a['zs'][0][1], b['zs'][0][1]):.1f} pp", flush=True)
    print(f"done [{time.time() - t00:.0f} s]; peak RSS {A.peak_rss_gb():.2f} GB", flush=True)
