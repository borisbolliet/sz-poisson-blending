"""c02: the maps the config verifier feeds to SZiFi (project .venv: szsim needs colossus).

(A) Regenerate step 3a's tiles from their seeds exactly as verify_real/step3_szifi/vsz_lib.py does (through
    analysis/realistic/step3/s3lib.py and szsim.finder): realistic tiles 0 and 33 (tSZ + CMB 300000+pid + noise
    400000+pid) and noise-only tiles 0 and 33 (CMB + noise), both noise levels.  Check: the production finder rerun on
    each regenerated tile gives the stored catalogues and sigma exactly (data_real/step3/tiles/, noise_only/).
(B) Inject isolated clusters into the two noise-only tiles (own painting, vc_model.paint_tsz_rfft: step 1's Arnaud
    template normalised to the spherical Y500, the tiles' beams and band, Fourier painting at continuous positions).
    Per tile and noise level: 49 clusters on a 7 x 7 grid (pitch 144 px = 72'), jittered by integers in [-10, 10] px
    plus 1/2 px on both axes (so every cluster sits at a half-integer index position, where SZiFi's sampled S/N peak is
    expected exactly one half pixel later, at an integer pixel); theta500 on 13 of SZiFi's 15 scales (0.5'-9.23'), the
    4 largest only on the 16 even-even grid nodes (>= 134' apart), so profiles (5 theta500) never overlap; >= 79' from
    the tile edges; target q uniform in [6, 30] for SZiFi's own filter with the analytic covariance (vc_model).
(C) Write (1200, 1200, 6) float32 maps in uK, axis 0 = y (pixel iy), axis 1 = x (pixel ix), to out/maps/, and the
    injection truth to out/inject_truth.npz.  Also cross-check our analytic sigma of SZiFi's filter with the build's
    z04 values (data_real/step3_szifi_real/analytic_sigma.npz; read only).
Usage: /Users/boris/GitHub/sz-poisson-blending/.venv/bin/python -I -B c02_maps.py
"""
import os, sys, time, resource
sys.dont_write_bytecode = True
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = "/Users/boris/GitHub/sz-poisson-blending"
os.environ["NUMBA_CACHE_DIR"] = os.path.join(HERE, "out", "numba_cache_proj")
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "NUMBA_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_v] = "1"
sys.path.insert(0, os.path.join(ROOT, "analysis", "realistic", "step3"))
sys.path.insert(0, HERE)
import numpy as np                                   # noqa: E402
from scipy import fft as sfft                        # noqa: E402
import s3lib as S                                    # noqa: E402  (the project's tile seeds and truth tables)
from szsim import finder as F                        # noqa: E402  (the project's simulator, only to regenerate)
import vc_model as M                                 # noqa: E402  (own numerics)

t0 = time.time()
MAPD = os.path.join(HERE, "out", "maps")
os.makedirs(MAPD, exist_ok=True)
G = F.grid()
TAB = F.build_paint_tables()          # loads data_real/step3/paint_tables.npz (exists, never rewritten)
PIDS = (0, 33)


def load_npz(path):
    with np.load(path) as d:
        return {k: d[k] for k in d.files}


def write_map(X, fn):
    m = np.empty((F.N, F.N, X.shape[0]), np.float32)
    worst = 0.
    for i in range(X.shape[0]):
        r = sfft.irfft2(X[i], s=(F.N, F.N))
        m[:, :, i] = r
        worst = max(worst, float(np.abs(m[:, :, i].astype(np.float64) - r).max() / r.std()))
    np.save(fn, m)
    return worst


def check(its, get, lab):
    msg, ok = [], True
    for k, it in enumerate(its):
        d = it["det"]
        q = get(f"it{k}_q")
        same = q.size == d["q"].size and np.array_equal(q, d["q"]) and np.array_equal(get(f"it{k}_iy"), d["iy"]) \
            and np.array_equal(get(f"it{k}_ix"), d["ix"]) and np.array_equal(get(f"it{k}_j"), d["j"])
        ds = float(np.abs(it["filt"].sigma_arcmin2 / get(f"it{k}_sig") - 1).max())
        ok &= bool(same) and ds == 0.
        msg.append(f"it{k} {d['q'].size} det (stored {q.size}) identical {bool(same)}, max|sig/stored-1| {ds:.0e}")
    print(f"  {lab}: " + "; ".join(msg), flush=True)
    return ok


# ------------------------------------------------------------------ (A) realistic tiles
n_ok = n_all = 0
f32 = 0.
print("(A) regeneration from the seeds, checked with the production finder against the stored outputs")
for pid in PIDS:
    st = load_npz(os.path.join(F.STEP3_DATA, "tiles", f"tile_{pid:03d}.npz"))
    d = S.truth(pid)
    Xt = F.tsz_channels(F.paint_y(d["y"], d["x"], d["Y500"], d["theta500"], TAB, G), G)
    for mode in S.MODES:
        X = Xt + F.sim_cmb(S.SEED_CMB + pid, G) + F.sim_noise(S.SEED_NOISE + pid, mode, G)
        n_ok += check(F.run_finder(X, G), lambda k: st[f"{mode}_{k}"], f"real  {pid:3d} {mode:8s}")
        n_all += 1
        f32 = max(f32, write_map(X, os.path.join(MAPD, f"real_{mode}_{pid:03d}.npy")))
        del X
    del Xt, d, st
NOISE = {}
for pid in PIDS:
    st = load_npz(os.path.join(F.STEP3_DATA, "noise_only", f"noise_{pid:03d}.npz"))
    for mode in S.MODES:
        X = F.sim_cmb(S.SEED_CMB + pid, G) + F.sim_noise(S.SEED_NOISE + pid, mode, G)
        n_ok += check(F.run_finder(X, G), lambda k: st[f"{mode}_whitened_{k}"], f"noise {pid:3d} {mode:8s}")
        n_all += 1
        NOISE[(pid, mode)] = X
print(f"  -> {n_ok} of {n_all} regenerated tiles reproduce the stored production catalogues and sigma exactly")

# ------------------------------------------------------------------ (B) injections
print("\n(B) injections: our analytic sigma of SZiFi's own filter (bins of width 72, analytic C at the bin centres)")
lf = (np.arange(300) + 1) * 4 * np.pi / (M.N * M.DX)
l0 = np.arange(300) * 4 * np.pi / (M.N * M.DX)
ellc = 0.5 * (l0 + lf)[1:]
sed = M.tsz_sed_uK(M.FREQS_GHZ)
SIGP, KAPP = {}, {}
ref = np.load(os.path.join(ROOT, "data_real", "step3_szifi_real", "analytic_sigma.npz"))
for mode in S.MODES:
    sp = M.analytic_cov(ellc, mode)
    SIGP[mode], KAPP[mode] = M.filter_quantities(ellc, sp, sed, M.FWHM)
    print(f"  {mode:8s} sigma_pred [y0]  " + " ".join(f"{x:.4e}" for x in SIGP[mode]))
    print(f"  {mode:8s} / build z04 (exact per-mode analytic C, SZiFi's own code) - 1: "
          + " ".join(f"{x:+.1e}" for x in SIGP[mode] / ref[f"{mode}_S0"] - 1))
    print(f"  {mode:8s} kappa (painted cluster / SZiFi template response)  " + " ".join(f"{x:.4f}" for x in KAPP[mode]))

TH = M.THETAS
SMALL = list(range(0, 9))           # 0.5' .. 3.49'
LARGE = [9, 10, 11, 12]             # 4.45' .. 9.23'
base = 168 + 144 * np.arange(7)
truth = {}
for t_i, pid in enumerate(PIDS):
    for m_i, mode in enumerate(S.MODES):
        rng = np.random.default_rng(20261112 + 100 * pid + m_i)
        gy, gx = np.meshgrid(np.arange(7), np.arange(7), indexing="ij")
        gy, gx = gy.ravel(), gx.ravel()
        large = (gy % 2 == 0) & (gx % 2 == 0)
        jt = np.empty(49, np.int64)
        jt[large] = rng.permutation(np.repeat(LARGE, 4))
        nsm = int((~large).sum())
        jt[~large] = rng.permutation(np.resize(np.array(SMALL), nsm))
        py = base[gy] + rng.integers(-10, 11, 49) + 0.5
        px = base[gx] + rng.integers(-10, 11, 49) + 0.5
        qt = rng.uniform(6., 30., 49)
        th = TH[jt]
        Y500 = qt * SIGP[mode][jt] * th ** 2 / (M.F_NORM * KAPP[mode][jt])
        y0 = Y500 * M.F_NORM / th ** 2            # what SZiFi's y0 estimates (its template is normalised to y(1e-8 R500))
        Xi = M.paint_tsz_rfft(py, px, Y500, th)
        X = NOISE[(pid, mode)] + Xi
        w = write_map(X, os.path.join(MAPD, f"inj_{mode}_{pid:03d}.npy"))
        f32 = max(f32, w)
        key = f"{mode}_{pid:03d}"
        truth.update({f"{key}_py": py, f"{key}_px": px, f"{key}_jtheta": jt, f"{key}_theta": th, f"{key}_Y500": Y500,
                      f"{key}_y0": y0, f"{key}_qtarget": qt})
        dmin = np.sqrt((py[:, None] - py[None, :]) ** 2 + (px[:, None] - px[None, :]) ** 2) * M.PIXA
        np.fill_diagonal(dmin, np.inf)
        print(f"  injected {mode:8s} tile {pid:3d}: 49 clusters, scales used {np.bincount(jt, minlength=15).tolist()}, "
              f"min separation {dmin.min():.1f}', min distance to the edges "
              f"{min(py.min(), px.min(), M.N - 1 - py.max(), M.N - 1 - px.max()) * M.PIXA:.1f}', "
              f"Y500 {Y500.min():.2e}..{Y500.max():.2e} arcmin^2", flush=True)
        del X, Xi
truth.update(sigma_pred_baseline=SIGP["baseline"], sigma_pred_goal=SIGP["goal"], kappa_pred_baseline=KAPP["baseline"],
             kappa_pred_goal=KAPP["goal"], thetas=TH, F_NORM=M.F_NORM, F_CENTRE=M.F_CENTRE)
np.savez(os.path.join(HERE, "out", "inject_truth.npz"), **truth)
print(f"\nfloat32 maps: max |map32 - map64| / map std = {f32:.1e}")
print("maps written: " + ", ".join(sorted(os.listdir(MAPD))))
print(f"done [{time.time() - t0:.0f} s]; peak RSS {resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e9:.2f} GB")
