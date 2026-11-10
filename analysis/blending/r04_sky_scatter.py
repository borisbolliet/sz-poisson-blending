"""[repair after verification] Primary-level MC-vs-analytic tests (same definitions as b05 part (b)) with
TWO error estimates per statistic:
  e_jk  : delete-one-region jackknife over 48 HEALPix nside=2 regions x 8 skies (as in b05)
  e_sky : between-sky scatter, std over the 8 per-sky values / sqrt(8)
The verifier pointed out that statistics sensitive to the monopole (lost, N_det_b/N_single, B2neg) carry a
sky-level component (the realised Y-sum of a sky fluctuates by ~0.5%) that the region jackknife cannot see.
We report z_jk, z_sky and z = (MC - an) / max(e_jk, e_sky).
Memory-light: each sky is reduced to per-region numerators/denominators before the next is loaded.
"""
import os, sys, glob
os.environ.setdefault("OMP_NUM_THREADS", "2")
import numpy as np
import healpy as hp

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import blendlib as bl
from szsim import signal as sg

m = sg.SZModel()
A = dict(np.load(os.path.join(bl.DATA, "analytic.npz")))
FLOORS = list(A["FLOORS"]); QCUTS = A["QCUTS"]; EPS = A["EPS"]; DMIN = A["DMIN"]; H = A["H"]
files = sorted(glob.glob(os.path.join(bl.DATA, "mc_sky*.npz")), key=lambda f: int(f.split("sky")[1][:-4]))
nsky = len(files); NREG = 48; nlab = NREG * nsky
print(f"MC skies: {nsky}")


def an_frac(fi, c, ic, arr, den):
    Nn = (H * arr).sum()
    Nd = (H * A["Pdet"][fi, c, ic]).sum() if den == "det" else (H * A["Phi_single"][ic][None, :]).sum()
    return Nn / Nd


NUM, DEN, AN = {}, {}, {}
CNT_S = {}; CNT_D = {}


def acc(key, flag, den, lab, isky, ra, dname):
    if key not in NUM:
        NUM[key] = np.zeros(nlab); DEN[key] = np.zeros(nlab); AN[key] = (ra, dname)
    NUM[key] += np.bincount(lab, weights=flag.astype(float), minlength=nlab)
    DEN[key] += np.bincount(lab, weights=den.astype(float), minlength=nlab)


for isky, f in enumerate(files):
    d = np.load(f)
    th = d["p_theta500"].astype(np.float64); sig = m.sigma_Y(th)
    q = d["p_q"].astype(np.float64); qo = d["p_q_obs"].astype(np.float64)
    Dn = d["D"] / sig[:, None]; xm = d["xmax"] / sig[:, None]
    lt = np.log(th)
    MU = np.column_stack([np.interp(lt, np.log(A["th"]), A["mu_exact"][:, c]) for c in range(3)])
    lab = hp.ang2pix(2, d["p_ra"].astype(float), d["p_dec"].astype(float), lonlat=True) + NREG * isky
    del d
    for fi, fl in enumerate(FLOORS):
        cl = list(bl.FLOORS[fl])
        dl = Dn[:, cl].sum(1); xmx = xm[:, cl].max(1); mu = MU[:, cl].sum(1); dom = q > xmx
        for c in (0, 1):
            dpr = dl - c * mu; qb = qo + dpr
            for ic, qc in enumerate(QCUTS):
                single = qo > qc; det = (qb > qc) & dom
                key0 = (fl, c, qc)
                CNT_S.setdefault(key0, np.zeros(nsky))[isky] = single.sum()
                CNT_D.setdefault(key0, np.zeros(nsky))[isky] = det.sum()
                acc(key0 + ("ratio",), det, single, lab, isky, (H * A["Pdet"][fi, c, ic]).sum() / (H * A["Phi_single"][ic][None, :]).sum(), "single")
                acc(key0 + ("B3",), det & (qo < qc), det, lab, isky, an_frac(fi, c, ic, A["Pb3"][fi, c, ic], "det"), "det")
                acc(key0 + ("lost",), single & dom & (qb < qc), single, lab, isky, an_frac(fi, c, ic, A["Plost"][fi, c, ic], "single"), "single")
                acc(key0 + ("sat",), single & ~dom, single, lab, isky, an_frac(fi, c, ic, A["Psat"][fi, ic], "single"), "single")
                for idm, dm in enumerate(DMIN):
                    acc(key0 + (f"B3d_{dm}",), det & (qo < qc) & (dpr > dm), det, lab, isky,
                        an_frac(fi, c, ic, A["Pb3d"][fi, c, ic, :, :, idm], "det"), "det")
                for ie, e in enumerate(EPS):
                    acc(key0 + (f"B2_{e}",), det & (dpr / q > e), det, lab, isky, an_frac(fi, c, ic, A["Pb2"][fi, c, ic, :, :, ie], "det"), "det")
                    if c == 1:
                        acc(key0 + (f"B2neg_{e}",), det & (dpr / q < -e), det, lab, isky, an_frac(fi, c, ic, A["Pb2n"][fi, c, ic, :, :, ie], "det"), "det")
    print(f"  sky {isky+1} reduced", flush=True)

rows = []
for key in NUM:
    Nn, Nd = NUM[key], DEN[key]; ra = AN[key][0]
    r = Nn.sum() / Nd.sum()
    ok = Nd > 0
    rj = (Nn.sum() - Nn[ok]) / (Nd.sum() - Nd[ok]); n = ok.sum()
    e_jk = np.sqrt((n - 1) / n * ((rj - rj.mean()) ** 2).sum())
    rs = Nn.reshape(nsky, NREG).sum(1) / Nd.reshape(nsky, NREG).sum(1)
    e_sky = rs.std(ddof=1) / np.sqrt(nsky)
    nev = Nn.sum()
    rows.append((key, r, e_jk, e_sky, ra, nev))

print("\nper-statistic results (MC pooled over 8 skies | analytic): value e_jk e_sky  z_jk z_sky z_comb")
zj, zs, zc, names, ratios = [], [], [], [], []
for key, r, ej, es, ra, nev in rows:
    if nev < 20 or ej == 0:
        continue
    z1 = (r - ra) / ej; z2 = (r - ra) / es if es > 0 else np.nan; z3 = (r - ra) / max(ej, es)
    zj.append(z1); zs.append(z2); zc.append(z3); names.append(key[3]); ratios.append(es / ej)
zj, zs, zc, ratios = map(np.array, (zj, zs, zc, ratios)); names = np.array(names)
# count tests: N_single, N_det_b with between-sky errors
zN = []
for key0 in CNT_S:
    fl, c, qc = key0; fi = FLOORS.index(fl); ic = list(QCUTS).index(qc)
    for nm, cnt, an in (("Ns", CNT_S[key0], (H * A["Phi_single"][ic][None, :]).sum()), ("Nd", CNT_D[key0], (H * A["Pdet"][fi, c, ic]).sum())):
        e = max(cnt.std(ddof=1), np.sqrt(an)) / np.sqrt(nsky)
        zN.append((cnt.mean() - an) / e)
zN = np.array(zN)

for key, r, ej, es, ra, nev in rows:
    fl, c, qc, nm = key
    if fl == "1e13" and c == 1 and nm in ("ratio", "lost", "B2neg_0.1", "B2_0.1", "B2_0.2", "B2_0.5", "B2_1.0", "B3", "B3d_1.0", "sat"):
        print(f"  floor {fl} c={c} q_cut={qc:.0f} {nm:10s}: MC {r:.5f} e_jk {ej:.5f} e_sky {es:.5f} | an {ra:.5f}  "
              f"z_jk {(r-ra)/ej:+.2f} z_sky {(r-ra)/es:+.2f} z {(r-ra)/max(ej,es):+.2f}")
print(f"\nall fraction tests (>=20 events): n={len(zc)}; rms z_jk {np.sqrt(np.mean(zj**2)):.2f} (max {np.abs(zj).max():.2f}); "
      f"rms z_sky {np.sqrt(np.nanmean(zs**2)):.2f} (max {np.nanmax(np.abs(zs)):.2f}); rms z_comb {np.sqrt(np.mean(zc**2)):.2f} (max {np.abs(zc).max():.2f}); "
      f"frac |z_comb|>3: {(np.abs(zc)>3).mean():.3f}")
print(f"count tests N_single/N_det_b (between-sky errors): n={len(zN)} rms {np.sqrt(np.mean(zN**2)):.2f} max {np.abs(zN).max():.2f}")
zall = np.concatenate([zc, zN])
print(f"ALL primary-level tests: n={len(zall)} rms {np.sqrt(np.mean(zall**2)):.2f} max|z| {np.abs(zall).max():.2f}; "
      f"frac |z|>2 {(np.abs(zall)>2).mean():.3f}, |z|>3 {(np.abs(zall)>3).mean():.3f}")
for nm in ["ratio", "lost", "sat", "B3", "B3d", "B2_", "B2neg"]:
    s = np.array([n.startswith(nm) for n in names])
    print(f"   {nm:6s}: n={s.sum():3d} median e_sky/e_jk {np.median(ratios[s]):.2f} (max {ratios[s].max():.2f}); "
          f"rms z_jk {np.sqrt(np.mean(zj[s]**2)):.2f}, rms z_comb {np.sqrt(np.mean(zc[s]**2)):.2f}, max|z_comb| {np.abs(zc[s]).max():.2f}")
np.save(os.path.join(bl.DATA, "r04_rows.npy"), np.array(rows, dtype=object), allow_pickle=True)
