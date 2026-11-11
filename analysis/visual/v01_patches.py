"""Pictures of one simulated 10 x 10 deg patch.

Part A (idealised setup of Q2/Q3, analysis/q2_maps/q2lib.py): the same patch, noise seed and multi-scale matched
filter as the map pipeline (patch PID, halos M500 > 1e13, 1.4' beam, white y-noise 4e-6 arcmin).
  - true y map, observed map (y + noise), max-over-scales SNR map with the detections q > 5 (merging radius 2');
  - each detection is classified with the pipeline's exact decomposition (data/q2_maps/decomp/decomp_PID.npz):
    clean / neighbours > 10% of the signal / detected only thanks to neighbours / no real cluster / fragment;
  - zooms on blended detections with their main contributors (exact per-halo responses, q2lib.Truth).
  Check: the recomputed candidates must equal the stored pipeline candidates (data/q2_maps/patch_out).
Part B (realistic SO LAT, baseline noise, as analysis/realistic/step1/s04_mc_maps.py): the same halos seen by the six
SO channels (lensed CMB + white noise + atmosphere with the 0.9 in-tube correlation + beams + tSZ SED), 0.5' pixels,
and the multi-frequency matched filter (MMF, explicit per-mode 6x6 inversion, l = 100-20000) at the 12 pipeline
scales; detections = peaks of the max-over-scales MMF SNR map, q > 5, merged within 2'.
  Check: discrete-grid MMF sigma vs the library sigma_Y (szsim.instrument.MMFModel).
Outputs: figures/visual/patch_idealised.png, figures/visual/patch_so.png; log: v01_patches.log
"""
import os, sys, time
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "NUMBA_NUM_THREADS"):
    os.environ[_v] = "2"
import numpy as np

ROOT = "/Users/boris/GitHub/sz-poisson-blending"
sys.path.insert(0, f"{ROOT}/analysis/q2_maps")
import q2lib as L                                              # noqa: E402
from scipy import fft as sfft                                  # noqa: E402
from scipy.special import j0                                   # noqa: E402
import matplotlib                                              # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt                                # noqa: E402
from matplotlib.colors import LogNorm                          # noqa: E402
from matplotlib.patches import Circle                          # noqa: E402
from szsim import signal as sg, instrument as ins              # noqa: E402

PID = int(os.environ.get("PID", "0"))
QC = 5.0
R_MERGE = 2.0
FIG = f"{ROOT}/figures/visual"
os.makedirs(FIG, exist_ok=True)
t0 = time.time()
N, PIX, BOX = L.NPIX, L.PIX, L.LBOX
EXT = [0, BOX / 60, 0, BOX / 60]                               # degrees


def pool_max(a, b):
    """Display only: maximum over b x b pixel blocks (keeps compact S/N peaks visible after down-sampling)."""
    n = a.shape[0] // b
    return a[: n * b, : n * b].reshape(n, b, n, b).max(axis=(1, 3))


BLK = 2.0                                                      # display block [arcmin]

# ============================================================ Part A: idealised pipeline patch
tab = L.build_tables()
d = L.load_patch(PID)
psis, sig = L.filter_bank()
print(f"patch {PID}: {d['x'].size:,} halos M500 > 1e13 in 100 deg^2; max theta500 {d['theta500'].max():.1f}'")
ymap = np.zeros((N, N))
L.paint_map(ymap, d["x"], d["y"], d["Y500"], d["theta500"], tab["paint"], tab["rP"], L.LNTH0, L.DLNTH, PIX)
rng = np.random.default_rng(10_000 + PID)                      # the pipeline's noise seed for this patch
nmap = rng.standard_normal((N, N)) * (L.MODEL.noise_y_arcmin / PIX)
Fy = sfft.rfft2(ymap.astype(np.float32)); Fn = sfft.rfft2(nmap.astype(np.float32))
Qb = np.full((N, N), -np.inf, np.float32); kb = np.zeros((N, N), np.int8)
Sb = np.zeros((N, N), np.float32); Nb = np.zeros((N, N), np.float32)
for k in range(len(L.THETA_K)):
    ps = psis[k].astype(np.float32)
    S = sfft.irfft2(Fy * ps, s=(N, N)); Nn = sfft.irfft2(Fn * ps, s=(N, N))
    L.update_best(Qb, kb, Sb, Nb, S, Nn, sig[k], k)
del S, Nn
iy, ix = L.local_max(Qb, np.float32(3.5))
ref = np.load(f"{L.DATA}/patch_out/patch_{PID:03d}.npz")
c_ref, v_ref = ref["cand_V13"], ref["candv_V13"]
same = (len(iy) == len(c_ref)) and np.array_equal(np.sort(iy * N + ix), np.sort(c_ref[:, 0] * N + c_ref[:, 1]))
o1 = np.argsort(iy * N + ix); o2 = np.argsort(c_ref[:, 0] * N + c_ref[:, 1])
dq = np.abs(Qb[iy, ix][o1] - v_ref[o2, 0]).max() if same else np.nan
print(f"[check] recomputed candidates (Q >= 3.5): {len(iy)} vs stored {len(c_ref)}; identical pixels: {same}; "
      f"max |dQ| = {dq:.2e}")

# pipeline decomposition of every candidate: [iy, ix, k, Q, S, N, ip, cp, c2, ip2, Ssum, nc10, dp, yk]
rec = np.load(f"{L.DATA}/decomp/decomp_{PID:03d}.npz")["cand_V13"]
riy, rix, rk = rec[:, 0].astype(int), rec[:, 1].astype(int), rec[:, 2].astype(int)
Q, S, Nr, ip, cp = rec[:, 3], rec[:, 4], rec[:, 5], rec[:, 6].astype(int), rec[:, 7]
xy = np.stack([(rix + 0.5) * PIX, (riy + 0.5) * PIX], 1)
th = L.THETA_K[rk]; sk = sig[rk]
acc = L.merge_greedy(xy[:, 0].copy(), xy[:, 1].copy(), th.astype(np.float64), Q.copy(), R_MERGE, 0.0, BOX)
acc = acc[Q[acc] >= QC]
acc = acc[np.argsort(-Q[acc])]
f_cont = (S - cp) / S
cls = {}
seen = set()
for j in acc:
    if ip[j] < 0 or cp[j] / sk[j] < 2.0:
        c = "none"
    elif ip[j] in seen:
        c = "fragment"
    elif (cp[j] + Nr[j]) / sk[j] < QC:
        c = "needs"
    elif f_cont[j] > 0.1:
        c = "boosted"
    else:
        c = "clean"
    seen.add(ip[j])
    cls[j] = c
names = {"clean": "neighbours < 10% of signal", "boosted": "neighbours > 10% of signal",
         "needs": "detected only thanks to neighbours", "none": "no real cluster (noise/confusion)",
         "fragment": "fragment of a brighter detection"}
cols = {"clean": "white", "boosted": "orange", "needs": "red", "none": "magenta", "fragment": "grey"}
counts = {c: sum(v == c for v in cls.values()) for c in names}
print(f"detections q > {QC:.0f} (merge {R_MERGE}'): {len(acc)}  " + ", ".join(f"{names[c]}: {n}" for c, n in counts.items()))

truth = L.Truth(d, tab)


def contributors(j):
    I, C = truth.contributions(xy[j, 0], xy[j, 1], rk[j])
    o = np.argsort(-C)
    return I[o], C[o]


def pick(c, qmax=12.0):
    js = [j for j in acc if cls[j] == c and Q[j] < qmax]
    return js[0] if js else None


examples = []
for c in ("boosted", "needs", "none"):
    j = pick(c)
    if j is not None:
        examples.append(j)
for j in acc:                                                   # fill up with further blended detections
    if len(examples) >= 3:
        break
    if cls[j] in ("boosted", "needs") and Q[j] < 12.0 and j not in examples:
        examples.append(j)

fig, axs = plt.subplots(2, 3, figsize=(16, 11.2))
ax = axs[0, 0]
im = ax.imshow(np.clip(ymap, 1e-8, None), origin="lower", extent=EXT, cmap="inferno", norm=LogNorm(1e-7, 1e-4))
ax.set_title(f"(a) true SZ sky: y from {d['x'].size:,} halos, M500 > 1e13\n(1.4' beam, no noise)", fontsize=11)
plt.colorbar(im, ax=ax, fraction=0.046, label="Compton y")
ax = axs[0, 1]
im = ax.imshow(ymap + nmap, origin="lower", extent=EXT, cmap="RdBu_r", vmin=-4e-5, vmax=4e-5)
ax.set_title("(b) what the telescope sees: y + white noise\n(4e-6 arcmin, 0.25' pixels)", fontsize=11)
plt.colorbar(im, ax=ax, fraction=0.046, label="Compton y")
ax = axs[0, 2]
im = ax.imshow(pool_max(Qb, int(BLK / PIX)), origin="lower", extent=EXT, cmap="viridis", vmin=0, vmax=10)
for j in acc:
    ax.add_patch(Circle((xy[j, 0] / 60, xy[j, 1] / 60), 0.13, fill=False, ec=cols[cls[j]], lw=1.3))
for e, j in enumerate(examples):
    ax.annotate("ABC"[e], (xy[j, 0] / 60, xy[j, 1] / 60), xytext=(6, 6), textcoords="offset points", color="w",
                fontsize=11, weight="bold")
ax.set_title(f"(c) matched-filter S/N (best of 12 scales; max per 2' block);\n{len(acc)} detections with q > {QC:.0f}",
             fontsize=11)
plt.colorbar(im, ax=ax, fraction=0.046, label="signal-to-noise")
hand = [plt.Line2D([], [], ls="", marker="o", mfc="none", mec=cols[c], mew=1.5, ms=9,
                   label=f"{names[c]} ({counts[c]})") for c in names if counts[c] > 0]
ax.legend(handles=hand, loc="lower left", fontsize=8, facecolor="0.25", labelcolor="w", framealpha=0.8)
for a in axs[0]:
    a.set_xlabel("deg"); a.set_ylabel("deg")

HW = 15.0                                                       # zoom half-width [arcmin]
for e, j in enumerate(examples):
    ax = axs[1, e]
    x0, y0 = xy[j]
    I, C = contributors(j)
    P = I[0]
    i0, i1 = int((y0 - HW) / PIX), int((y0 + HW) / PIX)
    k0, k1 = int((x0 - HW) / PIX), int((x0 + HW) / PIX)
    sub = np.roll(np.roll(ymap, -i0, 0), -k0, 1)[: i1 - i0, : k1 - k0]
    ax.imshow(np.clip(sub, 1e-8, None), origin="lower", extent=[-HW, HW, -HW, HW], cmap="inferno",
              norm=LogNorm(1e-7, 1e-4))
    others = S[j] - C[0]
    lines = []
    for n_, (ii, cc) in enumerate(zip(I[:6], C[:6])):
        dx = (d["x"][ii] - x0 + BOX / 2) % BOX - BOX / 2; dy = (d["y"][ii] - y0 + BOX / 2) % BOX - BOX / 2
        main = n_ == 0
        ax.add_patch(Circle((dx, dy), max(d["theta500"][ii], 0.3), fill=False, ec="cyan" if main else "w",
                            lw=2 if main else 1, ls="-" if main else "--"))
        if n_ < 4:
            lab = "main" if main else f"#{n_}"
            if not main:
                ax.annotate(f"{n_}", (dx + max(d["theta500"][ii], 0.3) * 0.75, dy + max(d["theta500"][ii], 0.3) * 0.75),
                            color="w", fontsize=10, weight="bold")
            lines.append(f"{lab}: z={d['z'][ii]:.2f}, M={d['M500'][ii]:.1e}, "
                         f"{'own S/N ' + format(cc / sk[j], '.1f') if main else format(cc / C[0] * 100, '.0f') + '% of main'}")
    ax.set_xlim(-HW, HW); ax.set_ylim(-HW, HW)
    ax.set_xlabel("arcmin"); ax.set_ylabel("arcmin")
    nsig = int(np.sum(C > 0.01 * C[0]))
    ax.set_title(f"({'ABC'[e]}) {names[cls[j]]}\nmeasured q = {Q[j]:.1f}; main halo alone gives {cp[j] / sk[j]:.1f} + "
                 f"noise {Nr[j] / sk[j]:+.1f}", fontsize=10.5, color=cols[cls[j]] if cls[j] != "clean" else "k")
    ax.text(0.02, 0.02, f"other halos add {others / C[0] * 100:+.0f}% ({nsig - 1} halos > 1% of main)\n" + "\n".join(lines),
            transform=ax.transAxes, fontsize=7.6, color="w", va="bottom",
            bbox=dict(fc="k", alpha=0.55, ec="none"))
    print(f"example {'ABC'[e]}: cls={cls[j]} Q={Q[j]:.2f} own={cp[j] / sk[j]:.2f} noise={Nr[j] / sk[j]:+.2f} "
          f"others/main={others / C[0]:+.3f} scale={L.THETA_K[rk[j]]:.2f}'; " + " | ".join(lines))
fig.suptitle(f"Idealised setup (Q2/Q3 map pipeline), patch {PID}: 10 x 10 deg", fontsize=13)
fig.tight_layout()
fig.savefig(f"{FIG}/patch_idealised.png", dpi=110)
plt.close(fig)
print(f"wrote {FIG}/patch_idealised.png  [{time.time() - t0:.0f}s]", flush=True)
xy_ideal = xy[acc]
del Fy, Fn, nmap

# ============================================================ Part B: the same halos seen by SO LAT (baseline)
MODE = "baseline"
NB, PB = 1200, 0.5                                             # 0.5' pixels, Nyquist l = 21600
AR = ins.ARCMIN
FWP = 0.8                                                      # painting beam [arcmin], below every SO beam
sbp = FWP / np.sqrt(8 * np.log(2))
nth_need = int(np.searchsorted(L.TH_NODES, d["theta500"].max())) + 3
rPp = 5.0 * L.TH_NODES + 6.0 * sbp
paint = np.zeros((L.NTH, L.NR))
for i in range(min(nth_need, L.NTH)):
    dl = 0.25 / (rPp[i] * AR)
    lg = np.arange(0.0, 9.0 / (sbp * AR) + dl, dl)
    w = np.full_like(lg, dl); w[0] *= 0.5; w[-1] *= 0.5
    vp = w * lg * sg.template_hankel(lg * L.TH_NODES[i] * AR) * np.exp(-0.5 * (lg * sbp * AR) ** 2) / (2 * np.pi)
    rp = np.linspace(0, rPp[i], L.NR) * AR
    for a in range(0, L.NR, 256):
        paint[i, a:a + 256] = j0(np.outer(rp[a:a + 256], lg)) @ vp * AR ** 2
yfine = np.zeros((N, N))
L.paint_map(yfine, d["x"], d["y"], d["Y500"], d["theta500"], paint, rPp, L.LNTH0, L.DLNTH, PIX)
print(f"[B] painted y with a {FWP}' beam: sum check {yfine.sum() * PIX ** 2 / (d['Y500'].sum() * sg.RATIO_5R500):.4f} "
      f"(should be ~1)  [{time.time() - t0:.0f}s]", flush=True)
Ff = np.fft.rfft2(yfine)
rows = np.r_[0:NB // 2, N - NB // 2:N]
Ysig = Ff[rows][:, : NB // 2 + 1] / 4.0                        # Fourier crop to the 0.5' grid (sum convention)
del Ff, yfine

kx = np.fft.fftfreq(NB, d=PB * AR) * 2 * np.pi
ky = np.fft.rfftfreq(NB, d=PB * AR) * 2 * np.pi
Lm = np.hypot(kx[:, None], ky[None, :])
Om = (PB * AR) ** 2
Npx = NB * NB
fr, fw = ins.SO_LAT["freqs"], ins.SO_LAT["fwhm"]
f_uK = ins.tsz_f_uK(fr)
Bl = np.exp(-0.5 * (Lm[None] * (fw[:, None, None] * AR / np.sqrt(8 * np.log(2)))) ** 2)
Bp = np.exp(-0.5 * (Lm * sbp * AR) ** 2)
Ccmb = ins.cmb_cl_TT(Lm.ravel()).reshape(Lm.shape)
with np.errstate(divide="ignore"):
    shp = np.where(Lm > 0, (Lm / ins.SO_LAT["ell_knee"]) ** ins.SO_LAT["alpha_knee"], 0.0)
red = ins.SO_LAT["N_red"] * 4 * np.pi * ins.SO_LAT["fsky"] / ins.so_t_eff()
wl = ins.SO_LAT["white"][MODE]
rr = ins.SO_LAT["atm_corr"]
rng = np.random.default_rng(31_000 + PID)
Xs = (Bl / Bp[None]) * f_uK[:, None, None] * Ysig[None]        # tSZ in each channel [uK]
X = Xs.copy()
g = np.fft.rfft2(rng.standard_normal((NB, NB))) * np.sqrt(np.where(Lm > 0, Ccmb, 0.0) / Om)
X += Bl * g[None]
for i in range(6):
    X[i] += np.fft.rfft2(rng.standard_normal((NB, NB)) * wl[i] / PB)
for (i, j) in ins.SO_LAT["tubes"]:
    xa = np.fft.rfft2(rng.standard_normal((NB, NB))) / np.sqrt(Om)
    xb = np.fft.rfft2(rng.standard_normal((NB, NB))) / np.sqrt(Om)
    X[i] += np.sqrt(red[i] * shp) * xa
    X[j] += np.sqrt(red[j] * shp) * (rr * xa + np.sqrt(1 - rr * rr) * xb)

C = np.zeros(Lm.shape + (6, 6))
for i in range(6):
    C[..., i, i] += (wl[i] * AR) ** 2 + red[i] * shp
    for j in range(6):
        C[..., i, j] += Bl[i] * Bl[j] * Ccmb
for (i, j) in ins.SO_LAT["tubes"]:
    C[..., i, j] += rr * np.sqrt(red[i] * red[j]) * shp
    C[..., j, i] += rr * np.sqrt(red[i] * red[j]) * shp
C *= Npx / Om
sel = (Lm >= 100.0) & (Lm <= 20000.0)
a_ = np.moveaxis(Bl * f_uK[:, None, None], 0, -1)
Cia = np.zeros(Lm.shape + (6,))
Cia[sel] = np.linalg.solve(C[sel], a_[sel][..., None])[..., 0]
del C
aCa = np.einsum("...i,...i->...", a_, Cia)
wcol = np.full(Lm.shape[1], 2.0); wcol[0] = 1.0; wcol[-1] = 1.0
CX = np.einsum("xyi,ixy->xy", Cia, X)                          # a^T C^-1 X per mode
CXs = np.einsum("xyi,ixy->xy", Cia, Xs)                        # signal only
mlib = ins.MMFModel(ins.Instrument.so_lat(MODE, lmin=100.0, lmax=20000.0))
QB = np.full((NB, NB), -np.inf); KB = np.zeros((NB, NB), int); SB = np.zeros((NB, NB))
sigB = []
for k, tk in enumerate(L.THETA_K):
    tau = sg.template_hankel(Lm * tk * AR) / Om
    norm = np.sum(wcol[None, :] * tau ** 2 * aCa)
    s_k = np.sqrt(1.0 / norm)
    sigB.append(s_k / AR ** 2)
    Yh = np.fft.irfft2(tau * CX / norm, s=(NB, NB)) * Npx
    Ys = np.fft.irfft2(tau * CXs / norm, s=(NB, NB)) * Npx
    q = Yh / s_k
    b = q > QB
    QB[b] = q[b]; KB[b] = k; SB[b] = Ys[b] / s_k
sigB = np.array(sigB)
lib = np.array([mlib.sigma_Y_exact(t)[0] for t in L.THETA_K])
print("[B check] MMF sigma on the grid / library sigma_Y - 1 at theta = " +
      ", ".join(f"{t:.2f}': {s / l - 1:+.4f}" for t, s, l in zip(L.THETA_K, sigB, lib)))
print("[B] sigma_Y MMF baseline / idealised at the 12 scales: " + " ".join(f"{s / s0:.2f}" for s, s0 in zip(sigB, sig)))
jy, jx = L.local_max(QB.astype(np.float32), np.float32(3.5))
xyB = np.stack([jx * PB + 0.125, jy * PB + 0.125], 1)          # coarse sample J = fine sample 2J
QBp = QB[jy, jx]
thB = L.THETA_K[KB[jy, jx]]
accB = L.merge_greedy(xyB[:, 0].copy(), xyB[:, 1].copy(), thB.astype(np.float64), QBp.copy(), R_MERGE, 0.0, BOX)
accB = accB[QBp[accB] >= QC]
accB = accB[np.argsort(-QBp[accB])]


def nearest(pts, ref_pts):
    dd = (pts[:, None, :] - ref_pts[None, :, :] + BOX / 2) % BOX - BOX / 2
    return np.sqrt((dd ** 2).sum(-1)).min(1) if len(ref_pts) else np.full(len(pts), np.inf)


both = nearest(xyB[accB], xy_ideal) < 2.0
both_i = nearest(xy_ideal, xyB[accB]) < 2.0
print(f"[B] SO baseline MMF detections q > {QC:.0f}: {len(accB)} (idealised filter: {len(xy_ideal)}); "
      f"{both.sum()} of them within 2' of an idealised detection")
nsel = SB[jy[accB], jx[accB]]
print(f"[B] noise-free signal S/N at the SO detections: median {np.median(nsel):.2f}; "
      f"{np.sum(nsel < 2)} below 2 (noise peaks)")

# S/N of the brightest halos in both filters (max within ~1' of the halo)
print("[B] brightest halos (catalogue q = idealised mean S/N): measured S/N, idealised filter vs SO baseline MMF")
for i in np.argsort(-d["q"])[:8]:
    jx, jy = int(d["x"][i] / PB), int(d["y"][i] / PB)
    kx_, ky_ = int(d["x"][i] / PIX), int(d["y"][i] / PIX)
    q_mmf = QB[max(jy - 2, 0):jy + 3, max(jx - 2, 0):jx + 3].max()
    q_id = Qb[max(ky_ - 4, 0):ky_ + 5, max(kx_ - 4, 0):kx_ + 5].max()
    print(f"    q={d['q'][i]:5.1f} theta500={d['theta500'][i]:5.2f}' z={d['z'][i]:.2f} M500={d['M500'][i]:.1e}: "
          f"idealised {q_id:5.1f}  SO MMF {q_mmf:5.1f} (best scale {L.THETA_K[KB[jy, jx]]:.2f}')")

# channel maps around the brightest cluster (by catalogue q), high-passed at l > 300 for display
ib = int(np.argmax(d["q"]))
cx, cy = d["x"][ib], d["y"][ib]
hp = (Lm > 300.0) & (Lm < 20000.0)
HWB = 60.0                                                     # half-width [arcmin] of the channel zooms
fig = plt.figure(figsize=(18, 9.6))
gs = fig.add_gridspec(2, 6, height_ratios=[1, 1.45])
for i in range(6):
    m = np.fft.irfft2(X[i] * hp, s=(NB, NB))
    r0, c0 = int((cy - HWB) / PB), int((cx - HWB) / PB)
    sub = np.roll(np.roll(m, -r0, 0), -c0, 1)[: int(2 * HWB / PB), : int(2 * HWB / PB)]
    ax = fig.add_subplot(gs[0, i])
    im = ax.imshow(sub, origin="lower", extent=[-HWB / 60, HWB / 60, -HWB / 60, HWB / 60], cmap="RdBu_r",
                   vmin=-250, vmax=250)
    ax.set_title(f"{fr[i]:.0f} GHz\nbeam {fw[i]}', noise {wl[i]} uK'", fontsize=10)
    ax.set_xticks([-1, 0, 1]); ax.set_yticks([-1, 0, 1])
    if i == 0:
        ax.set_ylabel("deg")
    print(f"[B] channel {fr[i]:.0f} GHz: high-passed map rms {m.std():.1f} uK")
cb = fig.colorbar(im, ax=fig.axes[:6], fraction=0.012, pad=0.01)
cb.set_label("uK (CMB), l > 300")
ax = fig.add_subplot(gs[1, 0:2])
sub = np.roll(np.roll(ymap, -int((cy - HWB) / PIX), 0), -int((cx - HWB) / PIX), 1)[: int(2 * HWB / PIX), : int(2 * HWB / PIX)]
ax.imshow(np.clip(sub, 1e-8, None), origin="lower", extent=[-HWB / 60, HWB / 60, -HWB / 60, HWB / 60], cmap="inferno",
          norm=LogNorm(1e-7, 1e-4))
ax.set_title(f"true y in the same 2 x 2 deg (brightest cluster at centre:\nz = {d['z'][ib]:.2f}, M500 = {d['M500'][ib]:.1e})",
             fontsize=10)
ax.set_xlabel("deg"); ax.set_ylabel("deg")
panels = [(pool_max(QB, int(BLK / PB)), xyB[accB], both, "cyan",
           f"SO baseline multi-frequency filter S/N: {len(accB)} detections q > 5\n"
           f"white: also found by the idealised filter ({both.sum()}); cyan: only here ({(~both).sum()})"),
          (pool_max(Qb, int(BLK / PIX)), xy_ideal, both_i, "orange",
           f"idealised filter S/N (as in (c) of the other figure): {len(xy_ideal)} detections q > 5\n"
           f"white: also found by SO ({both_i.sum()}); orange: only here ({(~both_i).sum()})")]
for col, (Qmap, pts, ok, cnot, ttl) in enumerate(panels):
    ax = fig.add_subplot(gs[1, 2 + 2 * col: 4 + 2 * col])
    imq = ax.imshow(Qmap, origin="lower", extent=EXT, cmap="viridis", vmin=0, vmax=10)
    for p, o_ in zip(pts, ok):
        ax.add_patch(Circle((p[0] / 60, p[1] / 60), 0.13, fill=False, ec="w" if o_ else cnot, lw=1.1))
    ax.add_patch(plt.Rectangle(((cx - HWB) / 60, (cy - HWB) / 60), 2 * HWB / 60, 2 * HWB / 60, fill=False, ec="orange",
                               lw=1.2, ls="--"))
    ax.set_title(ttl, fontsize=9.5)
    ax.set_xlabel("deg")
fig.colorbar(imq, ax=fig.axes[-1], fraction=0.046, label="S/N (max per 2' block)")
fig.suptitle(f"The same halos seen by SO LAT (baseline noise, CMB, atmosphere), patch {PID}. Top: the six channels "
             f"around the brightest cluster (dashed box below). Clusters are dark below 217 GHz and bright above.",
             fontsize=11.5)
fig.savefig(f"{FIG}/patch_so.png", dpi=105, bbox_inches="tight")
print(f"wrote {FIG}/patch_so.png  [{time.time() - t0:.0f}s]")
