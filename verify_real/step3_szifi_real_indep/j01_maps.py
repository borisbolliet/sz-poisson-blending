"""j01 (project .venv): the verifier's inputs for SZiFi, made from the tiles' seeds.

Lens 'indep'.  Nothing from analysis/realistic/step3_szifi_real/ is imported.  The tiles come from the project's own
simulator (szsim.finder through analysis/realistic/step3/s3lib.py), exactly as step 3a built them:
  realistic tile pid  = painted tSZ + CMB (seed 300000 + pid) + noise (seed 400000 + pid);
  noise-only tile pid = CMB + noise of the same seeds (no halos).
What this script does:
  1. picks 6 new tiles from pid 1-99 that are not in the a01 set (multiples of 11), with a fixed random seed;
  2. regenerates them (realistic and noise-only, baseline and goal) and the 3 tiles of the build used for the
     reproducibility test, reruns the production finder and checks it against the stored step-3a outputs
     (tiles/tile_PID.npz and noise_only/noise_PID.npz) bit for bit;
  3. writes each map for SZiFi as float32 (1200, 1200, 6) in microK, array axis 0 = finder y, axis 1 = finder x
     (out/maps/KIND_MODE_PID.npy);
  4. runs step 3a's emulation of SZiFi's iteration (szsim.finder.masked_cov method 'szifi': defaults, and l_sep 12000)
     on the new tiles, for paired comparisons; checks the defaults run against the arbiter's stored tiles_szifi/;
  5. stores step 1's sigma_Y at the 15 scales, the instrument, and the analytic CMB + noise covariance on every
     in-band |k| shell (SZiFi's units: microK^2 sr per mode) for the verifier's own analytic sigma of SZiFi's filter.
Output: out/j01_ref.npz, out/maps/*.npy.  Usage: .venv/bin/python -I -B j01_maps.py
"""
import os, sys, time, resource
HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "out")
os.environ["NUMBA_CACHE_DIR"] = os.path.join(OUT, "numba_cache_proj")
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "NUMBA_NUM_THREADS"):
    os.environ[_v] = "1"
sys.dont_write_bytecode = True
ROOT = "/Users/boris/GitHub/sz-poisson-blending"
for _p in (os.path.join(ROOT, "analysis", "realistic", "step3"), os.path.join(ROOT, "analysis", "q2_maps"), ROOT):
    if _p not in sys.path:
        sys.path.insert(0, _p)
import numpy as np
from scipy import fft as sfft
import s3lib as S                                    # noqa: E402  (project: step 3a's tile helpers)
from szsim import finder as F, instrument as ins     # noqa: E402

T0 = time.time()
N = F.N
TH = F.THETAS15
MODES = ("baseline", "goal")
D3 = F.STEP3_DATA

# ---------------------------------------------------------------- 1. tile choice
ELIG = [p for p in range(1, 100) if p % 11 != 0]
NEW = sorted(int(p) for p in np.random.default_rng(20261112).choice(ELIG, 6, replace=False))
# three of the build's runs, for the bit-level reproducibility test: (config, mode, kind, pid)
REPRO = [("S1", "baseline", "real", 11), ("S2", "goal", "noise", 22), ("S3it0", "goal", "real", 33)]
print(f"j01: new tiles (pid 1-99, not multiples of 11, rng seed 20261112): {NEW}")
print(f"     reproducibility runs of the build: {REPRO}")


def peak_gb():
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e9


def load(path, keys=None):
    with np.load(path) as d:
        return {k: d[k] for k in (keys or d.files)}


def write_map(X, kind, mode, pid):
    """irfft2 of each channel (float64, microK), stacked on the last axis, cast to float32."""
    m = np.stack([sfft.irfft2(X[i], s=(N, N)) for i in range(X.shape[0])], axis=-1).astype(np.float32)
    fn = os.path.join(OUT, "maps", f"{kind}_{mode}_{pid:03d}.npy")
    np.save(fn, m)
    rt = np.stack([sfft.rfft2(m[:, :, i].astype(np.float64)) for i in range(6)])
    return fn, float(np.abs(rt - X).max() / np.abs(X).max())


def same_run(its, st, pre):
    """Production-finder rerun vs stored arrays: identical catalogues and sigma?"""
    out = []
    for k, it in enumerate(its):
        d = it["det"]
        q = st[f"{pre}it{k}_q"]
        same = (d["q"].size == q.size and np.array_equal(d["q"], q) and np.array_equal(d["iy"], st[f"{pre}it{k}_iy"])
                and np.array_equal(d["ix"], st[f"{pre}it{k}_ix"]) and np.array_equal(d["j"], st[f"{pre}it{k}_j"]))
        ds = float(np.abs(it["filt"].sigma_arcmin2 / st[f"{pre}it{k}_sig"] - 1).max())
        out.append((d["q"].size, q.size, same, ds))
    return out


def counts(q):
    return np.array([(q > t).sum() for t in (4.0, 4.5, 5.0)])


G = F.grid()
TAB = F.build_paint_tables()        # reads data_real/step3/paint_tables.npz (never rewritten)
R = {"new_pids": np.array(NEW), "thetas": TH}
EMUL = (("szifi", dict(lsep=3000.0, blend=True, n_inpaint=100)), ("szifi_l12000", dict(lsep=12000.0, blend=True, n_inpaint=100)))

need_maps = {(k, m, p) for p in NEW for k in ("real", "noise") for m in MODES}
need_maps |= {(k, m, p) for (_, m, k, p) in REPRO}
pids_all = sorted({p for (_, _, p) in need_maps})
print(f"\n(1) regenerate {len(need_maps)} tile maps from their seeds; production finder rerun vs stored step-3a outputs")
nbad = 0
for pid in pids_all:
    t1 = time.time()
    st_real = load(os.path.join(D3, "tiles", f"tile_{pid:03d}.npz"),
                   [f"{m}_it{k}_{v}" for m in MODES for k in (0, 1) for v in ("q", "iy", "ix", "j", "sig")])
    st_noise = load(os.path.join(D3, "noise_only", f"noise_{pid:03d}.npz"),
                    [f"{m}_whitened_it{k}_{v}" for m in MODES for k in (0, 1) for v in ("q", "iy", "ix", "j", "sig")])
    st_emul = load(os.path.join(D3, "tiles_szifi", f"tile_{pid:03d}.npz"),
                   [f"{m}_it{k}_{v}" for m in MODES for k in (0, 1) for v in ("q", "iy", "ix", "j", "sig", "fsky")]) \
        if pid < 100 else None
    d = S.truth(pid)
    Xt = F.tsz_channels(F.paint_y(d["y"], d["x"], d["Y500"], d["theta500"], TAB, G), G)
    for mode in MODES:
        Xc = F.sim_cmb(S.SEED_CMB + pid, G)
        Xn = F.sim_noise(S.SEED_NOISE + pid, mode, G)
        for kind in ("real", "noise"):
            if (kind, mode, pid) not in need_maps:
                continue
            X = (Xt + Xc + Xn) if kind == "real" else (Xc + Xn)
            its = F.run_finder(X, G)
            chk = same_run(its, st_real if kind == "real" else st_noise, f"{mode}_" if kind == "real" else f"{mode}_whitened_")
            ok = all(c[2] and c[3] == 0.0 for c in chk)
            nbad += (not ok)
            fn, rt = write_map(X, kind, mode, pid)
            line = (f"  {kind:5s} {pid:3d} {mode:8s}: production finder it0 {chk[0][0]} det (stored {chk[0][1]}), it1 {chk[1][0]} "
                    f"(stored {chk[1][1]}); identical catalogues and sigma: {ok} (max|sig/stored-1| {max(c[3] for c in chk):.1e});"
                    f" map round trip {rt:.1e}")
            pre = f"{kind}_{mode}_{pid:03d}_"
            R[pre + "prod_sig0"] = its[0]["filt"].sigma_arcmin2
            R[pre + "prod_sig1"] = its[1]["filt"].sigma_arcmin2
            if pid in NEW:
                for name, kw in EMUL:
                    e = F.run_finder(X, G, mask_method="szifi", mask_kw=kw)
                    p2 = pre + name + "_"
                    for k in (0, 1):
                        dd = e[k]["det"]
                        R[p2 + f"it{k}_sig"] = e[k]["filt"].sigma_arcmin2
                        R[p2 + f"it{k}_C"] = e[k]["Cbin"]
                        for v in ("j", "iy", "ix", "q"):
                            R[p2 + f"it{k}_{v}"] = dd[v]
                    R[p2 + "fsky"] = e[1]["fsky"]
                    if name == "szifi" and kind == "real" and st_emul is not None:
                        same_e = all(np.array_equal(e[k]["det"][v], st_emul[f"{mode}_it{k}_{v}"]) for k in (0, 1)
                                     for v in ("q", "iy", "ix", "j"))
                        dse = max(float(np.abs(e[k]["filt"].sigma_arcmin2 / st_emul[f"{mode}_it{k}_sig"] - 1).max()) for k in (0, 1))
                        line += f"; emulation 'szifi' = stored tiles_szifi: {same_e} (sig {dse:.1e}, f {e[1]['fsky']:.4f} vs {float(st_emul[mode + '_it1_fsky']):.4f})"
                    del e
            print(line, flush=True)
            del X, its
        del Xc, Xn
    del Xt, d
    print(f"     tile {pid}: {time.time() - t1:.0f} s; peak RSS so far {peak_gb():.2f} GB", flush=True)
print(f"  tile maps whose production-finder rerun differs from the stored outputs: {nbad}")

# ---------------------------------------------------------------- 2. references
print("\n(2) step 1's sigma_Y [arcmin^2] at the 15 scales, instrument, analytic covariance per in-band shell")
for mode in MODES:
    R[f"sigY_{mode}"] = np.asarray(S.sigma_Y(mode, TH), float)
    print(f"  sigma_Y {mode:8s}: " + " ".join(f"{v:.4e}" for v in R[f"sigY_{mode}"]))
R["freqs_GHz"] = np.asarray(ins.SO_LAT["freqs"], float)
R["fwhm_arcmin"] = np.asarray(ins.SO_LAT["fwhm"], float)
R["tsz_sed_uK"] = np.asarray(ins.tsz_f_uK(ins.SO_LAT["freqs"]), float)
for mode in MODES:
    R[f"white_{mode}"] = np.asarray(ins.SO_LAT["white"][mode], float)
print("  freqs [GHz]", R["freqs_GHz"], "; FWHM [']", R["fwhm_arcmin"], "; tSZ SED [uK]", np.round(R["tsz_sed_uK"], 1))
# Analytic covariance of our skies (beam-convolved lensed CMB + white noise + atmosphere) on every distinct in-band
# |k| = 36 sqrt(n2) shell.  szsim gives it in 'rfft units' (E|rfft2|^2 = Npx C / Omega_pix); SZiFi's get_fft has
# |X|^2 = C (microK^2 sr), so divide by Npx / Omega_pix.
R["shell_n2"] = G.shell_n2.astype(np.int64)
for mode in MODES:
    C = G.analytic_cov_bands(G.shell_L, mode) / (G.Npx / G.Om)
    R[f"shellC_{mode}"] = C
    print(f"  analytic covariance {mode}: {G.shell_n2.size} shells, l = {G.shell_L[0]:.1f} - {G.shell_L[-1]:.1f}")
np.savez(os.path.join(OUT, "j01_ref.npz"), **R)
print(f"\ndone [{time.time() - T0:.0f} s]; peak RSS {peak_gb():.2f} GB")
