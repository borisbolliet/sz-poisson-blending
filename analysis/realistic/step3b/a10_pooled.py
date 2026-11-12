"""Step 3b arbitration / a10: the headline data vector and its linearised S8 biases with every production tile available.

a02 / a03 showed that the build's code and the sims verifier's code give the same credited halos and isolated copies on
the same maps, so the verifier's 400 tiles and the arbiter's 400 new tiles (the verifier's generators, the build's
pipeline; baseline only) are further independent production tiles.  Here R_p and R_s (b07's tents, s3bcosmo) for the
production finder, merge none, on:
  build    the build's 1,548 tiles (must reproduce b07);
  sims     the verifier's 400 tiles;  arb  the arbiter's 400 tiles (baseline);
  pooled   all of them (baseline 2,348 tiles; goal 1,948).
Printed per set: count change at q > 5 (total, coarse z bins, first q bin) and the linearised S8 bias (configs a, d, e;
SO40 and full sky; q_cut 5 and 6) with tile-bootstrap errors (NBOOT resamples).  Output: data_real/step3b/arbiter/
datavec_pooled_MODE.npz (R_p, R_s and resamples of the pooled set).
Usage: python a10_pooled.py [NBOOT]
"""
import os, sys, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import arblib as A
A.threads(1)
import numpy as np
import s3blib as B
import s3bcosmo as SC
import b07_datavec as D7
CL = SC.CL
NBOOT = int(sys.argv[1]) if len(sys.argv) > 1 else 1000
NE = B.EDGES.size


def combine(parts):
    keys = ("t", "z", "th", "qp", "qiso", "qo_none")
    off, out = 0, {k: [] for k in keys}
    for D in parts:
        for k in keys:
            out[k].append(D[k] + off if k == "t" else D[k])
        off += D["ntiles"]
    out = {k: np.concatenate(v) for k, v in out.items()}
    out["ntiles"] = off
    return out


def datavec(D, models, nboot, seed):
    nt = D["ntiles"]
    pid = D["t"].astype(np.int64)
    d = (D["qo_none"][:, None] > B.EDGES[None, :]).astype(np.float32) - (D["qiso"][:, None] > B.EDGES[None, :]).astype(np.float32)
    St, Wt = D7.cell_sums(pid, D["th"], D["qp"], d, None, nt=nt)
    Wtot = Wt.sum(0, dtype=np.float64)
    dK = np.where(Wtot[:, None] > 0, St.sum(0, dtype=np.float64) / np.where(Wtot > 0, Wtot, 1.0)[:, None], 0.0).reshape(B.TH_K.size, SC.QDK.size, NE)
    rng = np.random.default_rng(seed)
    WB = np.stack([np.bincount(rng.integers(0, nt, nt), minlength=nt) for _ in range(nboot)]).astype(np.float32)
    out = {}
    Sflat = St.reshape(nt, -1)
    RsB = {qc: np.empty((nboot, 16, 9)) for qc in SC.QCUTS}
    for a in range(0, nboot, 100):
        wb = WB[a:a + 100]
        Sb = (wb @ Sflat).reshape(-1, St.shape[1], NE); Wb = wb @ Wt
        dKb = np.where(Wb[..., None] > 0, Sb / np.where(Wb > 0, Wb, 1.0)[..., None], 0.0).reshape(-1, B.TH_K.size, SC.QDK.size, NE)
        for qc in SC.QCUTS:
            mu, Icell, zed, qed, sel = models[qc]
            RsB[qc][a:a + 100] = SC.rs_from_dk(Icell, mu, dKb, sel)
    del Sflat, St, Wt
    for qc in SC.QCUTS:
        mu, Icell, zed, qed, sel = models[qc]
        Nd = D7.per_tile_hist(pid, D["z"], D["qo_none"], zed, qed, nt=nt)
        Ni = D7.per_tile_hist(pid, D["z"], D["qiso"], zed, qed, nt=nt)
        Rp = D7.ratio(Nd.sum(0), Ni.sum(0))
        RpB = np.array([D7.ratio(np.tensordot(w, Nd, 1), np.tensordot(w, Ni, 1)) for w in WB])
        Rs = SC.rs_from_dk(Icell, mu, dK, sel)
        out[qc] = dict(Nd=Nd, Ni=Ni, Rp=Rp, RpB=RpB, Rs=Rs, RsB=RsB[qc], mu=mu, WB=WB)
    return out


def report(label, dv, Ls, mode):
    o = dv[5.0]
    Nd, Ni, WB = o["Nd"], o["Ni"], o["WB"]
    a, b = Nd.sum((1, 2)), Ni.sum((1, 2))
    tot = a.sum() / b.sum() - 1; etot = ((WB @ a) / (WB @ b) - 1).std()
    w_ = o["mu"]
    Tm = np.sum(w_ * (o["Rs"] - 1)) / w_.sum(); eTm = (np.sum(w_ * (o["RsB"] - 1), (1, 2)) / w_.sum()).std()
    zed = CL.default_bins(5.0)[0]
    kz = zed[1:] <= 0.1 + 1e-9
    az, bz = Nd[:, kz].sum((1, 2)), Ni[:, kz].sum((1, 2))
    lowz = az.sum() / bz.sum() - 1; elowz = ((WB @ az) / (WB @ bz) - 1).std()
    rs_low = np.sum(w_[kz] * (o["Rs"][kz] - 1)) / w_[kz].sum(); ers_low = (np.sum(w_[kz] * (o["RsB"][:, kz] - 1), (1, 2)) / w_[kz].sum()).std()
    q1 = np.sum(w_[:, 0] * (o["Rs"][:, 0] - 1)) / w_[:, 0].sum(); eq1 = (np.sum(w_[:, 0] * (o["RsB"][:, :, 0] - 1), 1) / w_[:, 0].sum()).std()
    print(f"  {label:44s} tiles {WB.shape[1]:5d}: q > 5 R_p {A.fmt(tot, etot)}, R_s {A.fmt(Tm, eTm)}; z < 0.1: R_p {100 * lowz:+.1f} +- "
          f"{100 * elowz:.1f}%, R_s {100 * rs_low:+.1f} +- {100 * ers_low:.1f}%; q 5-7.3: R_s {100 * q1:+.2f} +- {100 * eq1:.2f}%", flush=True)
    for qc in SC.QCUTS:
        cells = []
        for sv in ("SO40", "full"):
            for cn in ("a", "d", "e"):
                L = Ls[(qc, sv, cn)]
                bs = SC.lin_bias(L, dv[qc]["Rs"])[0][0]; bsb = SC.lin_bias(L, dv[qc]["RsB"])[0]
                bp = SC.lin_bias(L, dv[qc]["Rp"])[0][0]; bpb = SC.lin_bias(L, dv[qc]["RpB"])[0]
                cells.append(f"{sv} {cn} {bs:+.2f}+-{bsb.std():.2f} ({bp:+.2f}+-{bpb.std():.2f})")
        print(f"      q_cut {qc:.0f}, linearised S8 bias R_s (R_p): " + "; ".join(cells), flush=True)


if __name__ == "__main__":
    t00 = time.time()
    for mode in ("baseline", "goal"):
        print(f"=================== {mode} ===================", flush=True)
        models, Ls = {}, {}
        for qc in SC.QCUTS:
            M = SC.model("prod", mode, qc)
            mu = M.counts(SC.FID)
            zed, qed = CL.default_bins(qc)
            sel = [int(np.argmin(np.abs(B.EDGES - e))) for e in qed[:-1]]
            models[qc] = (mu, SC.cell_integrals(M, SC.FID), zed, qed, sel)
            Rs_b = np.load(os.path.join(B.OUT, f"datavec_prod_none_{mode}.npz"))[f"Rs_q{qc:.0f}"]
            for sv, fs in SC.SURVEYS.items():
                for cn in ("a", "d", "e"):
                    Ls[(qc, sv, cn)] = SC.linear_parts(M, SC.FID, Rs_b, cn, fs)
            del M
        Bd = A.load_build(mode, "all"); S = A.load_sims(mode)
        sets = [("build (1,548 tiles; b07)", Bd), ("sims verifier's 400 tiles", S)]
        if mode == "baseline":
            X = A.load_xpipe(mode, 400, 800)
            sets.append(("arbiter's 400 tiles (verifier's generator)", X))
            sets.append(("verifier's generator, 800 tiles", combine([S, X])))
            sets.append(("pooled: build + verifier's generator", combine([Bd, S, X])))
        else:
            sets.append(("pooled: build + verifier's tiles", combine([Bd, S])))
        for k, (lab, D) in enumerate(sets):
            dv = datavec(D, models, NBOOT, 100 + k)
            if k == 0:
                ref = np.load(os.path.join(B.OUT, f"datavec_prod_none_{mode}.npz"))
                mu5 = models[5.0][0]
                print(f"  check: rebuilt R_s / R_p of the build's tiles vs b07: max |dR_s| {np.abs(dv[5.0]['Rs'] - ref['Rs_q5'])[mu5 > 10].max():.1e}, "
                      f"max |dR_p| {np.abs(dv[5.0]['Rp'] - ref['Rp_q5']).max():.1e}", flush=True)
            report(lab, dv, Ls, mode)
            if lab.startswith("pooled"):
                np.savez(os.path.join(A.ARB, f"datavec_pooled_{mode}.npz"), ntiles=D["ntiles"],
                         **{f"{k_}_q{qc:.0f}": dv[qc][k_] for qc in SC.QCUTS for k_ in ("Rp", "Rs", "mu")},
                         **{f"{k_}boot_q{qc:.0f}": dv[qc][k_ + 'B'].astype(np.float32) for qc in SC.QCUTS for k_ in ("Rp", "Rs")})
            del dv
        del Bd, S, sets
    print(f"done [{time.time() - t00:.0f} s]; peak RSS {A.peak_rss_gb():.2f} GB", flush=True)
