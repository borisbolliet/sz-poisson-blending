"""Step 3 / t06: figures.
  figures/realistic/step3_tile.png        one tile (pid 0, baseline): finder S/N map with the q > 5 detections and their
                                          classes, and zooms on blended detections;
  figures/realistic/step3_r_theta.png     r(theta) = sigma_finder / sigma_Y before and after the iteration against step 2's
                                          variants (i), (ii), (iii); since the arbitration also SZiFi's own iteration with
                                          its defaults (t04 MASK szifi, 100 tiles) and with l_sep = 12000 (a01, 30 tiles);
  figures/realistic/step3_selection.png   the finder's selection of isolated injected clusters (t03 + a02 pooled, with the
                                          credit limit q_inj = 2 r_inj; t03 alone before the arbitration);
  figures/realistic/step3_classes.png     class fractions of the q > 5 detections against q_obs.
"""
import os, sys, time
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "NUMBA_NUM_THREADS"):
    os.environ[_v] = "2"
import numpy as np
from scipy.special import ndtr
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt                     # noqa: E402
from matplotlib.patches import Circle               # noqa: E402
from matplotlib.colors import LogNorm               # noqa: E402
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import s3lib as S                                   # noqa: E402
F = S.F
t0 = time.time()
FIG = f"{S.ROOT}/figures/realistic"
os.makedirs(FIG, exist_ok=True)
COL = dict(clean="white", b10="orange", needs="red", none="magenta", frag="grey", merge="cyan")
NAMES = dict(clean="neighbours < 10% of own signal", b10="neighbours > 10%", needs="detected only thanks to neighbours",
             none="no real cluster (c_p < 2 sigma)", frag="fragment of a brighter detection", merge="merge of two q > 5 halos")

# ------------------------------------------------------------------ 1. one tile
PID, MODE = 0, "baseline"
G = F.grid()
tables = F.build_paint_tables()
d = S.truth(PID)
r = dict(np.load(os.path.join(S.OUT, "tiles", f"tile_{PID:03d}.npz")))
Xt = F.tsz_channels(F.paint_y(d["y"], d["x"], d["Y500"], d["theta500"], tables, G), G)
X = Xt + F.sim_cmb(S.SEED_CMB + PID, G) + F.sim_noise(S.SEED_NOISE + PID, MODE, G)
its = F.run_finder(X, G, keep_cube=True)
qc = its[-1]["q"]
det = its[-1]["det"]
same = np.array_equal(det["q"], r[f"{MODE}_it1_q"]) and np.array_equal(det["iy"], r[f"{MODE}_it1_iy"])
print(f"[tile] pid {PID} {MODE}: rerun catalogue identical to the stored one: {same}")
qmax = qc.max(0)
del qc, X
# y map for display (painted at 0.5' through the Fourier crop)
ymap = np.fft.irfft2(F.paint_y(d["y"], d["x"], d["Y500"], d["theta500"], tables, G), s=(F.N, F.N))
sig = r[f"{MODE}_it1_sig"]; j = r[f"{MODE}_it1_j"]; q = r[f"{MODE}_it1_q"]
ex = r[f"{MODE}_d_top_ex"]; ti = r[f"{MODE}_d_top_idx"]
cp = ex[:, 0]; sd = sig[j]
delta = r[f"{MODE}_d_S_tsz"] - cp
qa = r[f"{MODE}_d_q_alone"]
sel = np.flatnonzero(q > 5)
order = sel[np.argsort(-q[sel])]
seen = set(); cls = {}
for k in order:
    if cp[k] / sd[k] < 2:
        cls[k] = "none"
    elif ti[k, 0] in seen:
        cls[k] = "frag"
    elif qa[k] < 5:
        cls[k] = "needs"
    elif (ti[k, 1] >= 0 and d["q_" + MODE][ti[k, 1]] > 5 and np.nan_to_num(ex[k, 1]) > 0.1 * cp[k]):
        cls[k] = "merge"
    elif delta[k] > 0.1 * cp[k]:
        cls[k] = "b10"
    else:
        cls[k] = "clean"
    if cp[k] / sd[k] >= 2:
        seen.add(ti[k, 0])
counts = {c: sum(v == c for v in cls.values()) for c in NAMES}
print("[tile] classes of the q > 5 detections: " + ", ".join(f"{NAMES[c]}: {n}" for c, n in counts.items()))


def pool_max(a, b):
    n = a.shape[0] // b
    return a[: n * b, : n * b].reshape(n, b, n, b).max(axis=(1, 3))


fig, axs = plt.subplots(2, 3, figsize=(16, 11))
EXT = [0, 10, 0, 10]
ax = axs[0, 0]
im = ax.imshow(np.clip(ymap, 1e-8, None), origin="lower", extent=EXT, cmap="inferno", norm=LogNorm(1e-7, 1e-4))
ax.set_title(f"(a) true SZ sky (y, painting beam 0.8'): {d['x'].size:,} halos", fontsize=11)
plt.colorbar(im, ax=ax, fraction=0.046, label="Compton y")
ax = axs[0, 1]
im = ax.imshow(pool_max(qmax, 4), origin="lower", extent=EXT, cmap="viridis", vmin=0, vmax=10)
for k, c in cls.items():
    ax.add_patch(Circle((det["x"][k] / 60, det["y"][k] / 60), 0.13, fill=False, ec=COL[c], lw=1.3))
ax.set_title(f"(b) finder S/N (best of 15 scales, max per 2' block)\n{sel.size} detections q > 5 (no merging)", fontsize=11)
plt.colorbar(im, ax=ax, fraction=0.046, label="S/N")
hand = [plt.Line2D([], [], ls="", marker="o", mfc="none", mec=COL[c], mew=1.5, ms=9, label=f"{NAMES[c]} ({counts[c]})")
        for c in NAMES if counts[c] > 0]
ax.legend(handles=hand, loc="lower left", fontsize=7.5, facecolor="0.25", labelcolor="w", framealpha=0.8)
ax = axs[0, 2]
th = F.THETAS15
for lab, key, c in (("before iteration", "it0_sig", "C0"), ("after iteration", "it1_sig", "C1")):
    ax.plot(th, r[f"{MODE}_{key}"] / S.sigma_Y(MODE, th), "o-", color=c, label=f"finder, {lab}")
ax.plot(th, r[f"{MODE}_sig_cn_est"] / S.sigma_Y(MODE, th), "s--", color="0.4", ms=4, label="same filter on CMB + noise only")
ax.axhline(1, color="k", lw=0.8)
ax.set_xscale("log"); ax.set_xlabel("filter scale theta [arcmin]"); ax.set_ylabel("sigma_finder / sigma_Y")
ax.set_title("(c) this tile: noise level the S/N is divided by", fontsize=11)
ax.legend(fontsize=8)
blended = []
for c in ("b10", "needs", "merge", "none"):                      # one example per class where available
    ks = [k for k in order if cls[k] == c and q[k] < 12]
    if ks and len(blended) < 3:
        blended.append(ks[0])
for k in order:                                                  # fill up with further blended detections
    if len(blended) >= 3:
        break
    if cls[k] in ("b10", "needs") and k not in blended:
        blended.append(k)
HW = 12.0
for e, k in enumerate(blended):
    ax = axs[1, e]
    y0, x0 = det["y"][k], det["x"][k]
    i0, k0 = int((y0 - HW) / F.PIX), int((x0 - HW) / F.PIX)
    sub = np.roll(np.roll(ymap, -i0, 0), -k0, 1)[: int(2 * HW / F.PIX), : int(2 * HW / F.PIX)]
    ax.imshow(np.clip(sub, 1e-8, None), origin="lower", extent=[-HW, HW, -HW, HW], cmap="inferno", norm=LogNorm(1e-7, 1e-4))
    lines = []
    for n_, h in enumerate(ti[k, :3]):
        if h < 0:
            continue
        dy = (d["y"][h] - y0 + 300) % 600 - 300; dx = (d["x"][h] - x0 + 300) % 600 - 300
        ax.add_patch(Circle((dx, dy), max(d["theta500"][h], 0.3), fill=False, ec="cyan" if n_ == 0 else "w", lw=2 if n_ == 0 else 1,
                            ls="-" if n_ == 0 else "--"))
        lines.append(f"#{n_}: z={d['z'][h]:.2f} M={d['M500'][h]:.1e} q_true={d['q_' + MODE][h]:.1f} gives {ex[k, n_] / sd[k]:.2f} sigma")
    ax.plot(0, 0, "+", color="lime", ms=12, mew=2)
    ax.set_xlim(-HW, HW); ax.set_ylim(-HW, HW); ax.set_xlabel("arcmin"); ax.set_ylabel("arcmin")
    ax.set_title(f"{NAMES[cls[k]]}: q_obs = {q[k]:.1f} at scale {th[j[k]]:.2f}'\nprimary alone (search) {qa[k]:.1f}; "
                 f"neighbours add {100 * delta[k] / cp[k]:+.0f}% of its signal", fontsize=9.5)
    ax.text(0.02, 0.02, "\n".join(lines), transform=ax.transAxes, fontsize=7.5, color="w", va="bottom",
            bbox=dict(fc="k", alpha=0.55, ec="none"))
fig.suptitle(f"SZiFi-like finder on one SO tile (baseline noise, CMB, atmosphere; all halos M500 > 1e13), tile {PID}: 10 x 10 deg. "
             f"Green +: detection; cyan: largest contributor; white dashed: next two.", fontsize=11.5)
fig.tight_layout()
fig.savefig(f"{FIG}/step3_tile.png", dpi=100)
plt.close(fig)
print(f"wrote {FIG}/step3_tile.png [{time.time() - t0:.0f} s]")

# ------------------------------------------------------------------ 2. r(theta)
s2 = dict(np.load(f"{S.ROOT}/data_real/step2/inputs.npz"))
fig, axs = plt.subplots(1, 2, figsize=(13, 4.8), sharey=True)
for ax, mode in zip(axs, S.MODES):
    a = dict(np.load(os.path.join(S.OUT, f"t05_tiles_{mode}.npz")))
    th = a["thetas"]
    for key, lab, c in (("r0", "finder, before iteration (iteration 0)", "C0"), ("r1", "finder, after iteration (production: whitened mask)", "C1"),
                        ("r_lit", f"literal binary f_sky iteration (not used; off the scale: {a['r_lit'].mean(0).min():.2f}-{a['r_lit'].mean(0).max():.2f})", "C7")):
        m, s = a[key].mean(0), a[key].std(0)
        ax.plot(th, m, "o-", color=c, label=lab, ms=4)
        ax.fill_between(th, m - s, m + s, color=c, alpha=0.25)
    fz = os.path.join(S.OUT, f"t05_tiles_szifi_{mode}.npz")
    if os.path.exists(fz):
        z_ = dict(np.load(fz))
        m, s_ = z_["r1"].mean(0), z_["r1"].std(0)
        ax.plot(z_["thetas"], m, "^-", color="C3", ms=4, label=f"SZiFi's own iteration, defaults (l_sep 3000; {z_['r1'].shape[0]} tiles)")
        ax.fill_between(z_["thetas"], m - s_, m + s_, color="C3", alpha=0.25)
    pa = sorted(f for f in os.listdir(os.path.join(S.OUT, "arbiter")) if f.startswith("a01_real_"))
    if pa:
        r12 = np.array([np.load(os.path.join(S.OUT, "arbiter", f))[f"{mode}_szifi_l12000_sig1"] for f in pa]) / S.sigma_Y(mode, F.THETAS15)
        ax.plot(F.THETAS15, r12.mean(0), "v:", color="C2", ms=4, label=f"SZiFi's iteration with l_sep = 12000 ({len(pa)} tiles)")
    k3 = np.load(f"{S.ROOT}/data_real/step2/kernel_{mode}_effiii.npz")
    ax.plot(s2[mode + "_th"], s2[mode + "_r_i"], "k--", label="step 2 (i): sigma_Y (+) sub-threshold confusion")
    ax.plot(s2[mode + "_th"], s2[mode + "_r_ii"], "k:", label="step 2 (ii): robust width")
    ax.plot(k3["th"], k3["r"], "k-.", label="step 2 (iii): plain std")
    ax.axhline(1, color="0.5", lw=0.8)
    ax.set_xscale("log"); ax.set_xlim(0.4, 20); ax.set_ylim(0.95, 1.35)
    ax.set_xlabel("filter scale theta [arcmin]"); ax.set_title(f"SO {mode}: {a['r1'].shape[0]} tiles (band: tile-to-tile std)")
axs[0].set_ylabel("r = sigma_finder / sigma_Y")
axs[0].legend(fontsize=8, loc="upper left")
fig.tight_layout()
fig.savefig(f"{FIG}/step3_r_theta.png", dpi=110)
plt.close(fig)
print(f"wrote {FIG}/step3_r_theta.png")

# ------------------------------------------------------------------ 3. selection
_sa = os.path.join(S.OUT, "selection_arbiter.npz")
POOLED = os.path.exists(_sa)
sel = dict(np.load(_sa if POOLED else os.path.join(S.OUT, "selection.npz")))
fig, axs = plt.subplots(1, 2, figsize=(13, 4.8), sharey=True)
qe = sel["qedges"]; qm = 0.5 * (qe[1:] + qe[:-1])
qq = np.linspace(1.5, 10, 300)
for ax, mode in zip(axs, S.MODES):
    for it_, (thv, c) in enumerate(zip(sel["theta_inj"], plt.cm.viridis(np.linspace(0, 0.9, len(sel["theta_inj"]))))):
        if POOLED:
            k_, n_ = sel[mode + "_tab_k"][it_], np.maximum(sel[mode + "_tab_n"][it_], 1)
            p_ = k_ / n_
            ax.errorbar(qm, p_, np.sqrt(np.maximum(p_ * (1 - p_), 1.0 / n_) / n_), fmt="o", ms=3, color=c, label=f"theta500 = {thv}'")
            qmin = 2.0 * np.interp(np.log(thv), np.log(F.THETAS15), sel[mode + "_rinj"])
        else:
            ax.errorbar(qm, sel[mode + "_comp"][it_], sel[mode + "_comp_err"][it_], fmt="o", ms=3, color=c, label=f"theta500 = {thv}'")
            qmin = 0.0
        b, mu, ls = sel[mode + "_fit"][it_]
        ax.plot(qq, np.where(qq < qmin, 0.0, ndtr((np.sqrt(np.maximum(qq ** 2 + b, 0.0)) - 5 - mu) / np.exp(ls))), "-", color=c, lw=0.8)
    ax.plot(qq, ndtr(qq - 5), "k--", lw=1.5, label="N(q_true, 1) (steps 1-2)")
    ax.plot(qq, ndtr(np.sqrt(qq ** 2 + 3) - 5), "k:", lw=1.5, label="N(sqrt(q_true^2 + 3), 1)")
    ax.set_xlabel("q_true of the injected cluster (Y500 / sigma_Y)")
    ax.set_title(f"SO {mode}: isolated clusters, P(q_obs > 5)" + (" (t03 + a02 pooled)" if POOLED else ""))
    ax.grid(alpha=0.3)
axs[0].set_ylabel("completeness")
axs[0].legend(fontsize=8, loc="lower right")
fig.tight_layout()
fig.savefig(f"{FIG}/step3_selection.png", dpi=110)
plt.close(fig)
print(f"wrote {FIG}/step3_selection.png")

# ------------------------------------------------------------------ 4. class fractions vs q
edges = np.array([5, 5.5, 6, 7, 8, 10, 13, 20, 1000])
fig, axs = plt.subplots(1, 2, figsize=(13, 4.8), sharey=True)
for ax, mode in zip(axs, S.MODES):
    a = dict(np.load(os.path.join(S.OUT, f"t05_tiles_{mode}.npz")))
    qd = a["det_q"]
    xm = np.sqrt(edges[:-1] * np.minimum(edges[1:], 40))
    for key, lab, c, base in (("b10", "Y boosted > 10% (of main)", "orange", "main"), ("b20", "Y boosted > 20% (of main)", "saddlebrown", "main"),
                              ("needs", "detected only thanks to neighbours (of main)", "red", "main"),
                              ("none", "no real cluster (of all)", "magenta", "all"), ("frag", "fragments (of all)", "grey", "all"),
                              ("merge", "merges (of main)", "cyan", "main"), ("wrong", "wrong size, factor > 2 (of main)", "C0", "main")):
        f, e = [], []
        for lo, hi in zip(edges[:-1], edges[1:]):
            m = (qd > lo) & (qd <= hi)
            den = m & a["det_main"] if base == "main" else m
            n = den.sum()
            k = (a["det_" + key] & den).sum()
            f.append(k / max(n, 1)); e.append(np.sqrt(k * (1 - k / max(n, 1))) / max(n, 1))
        ax.errorbar(xm, 100 * np.array(f), 100 * np.array(e), fmt="o-", ms=3, color=c, label=lab)
    ax.set_xscale("log"); ax.set_xlabel("q_obs of the detection"); ax.set_title(f"SO {mode}: {qd.size} detections q > 5")
    ax.set_xticks([5, 6, 7, 8, 10, 13, 20, 30]); ax.set_xticklabels(["5", "6", "7", "8", "10", "13", "20", "30"])
    ax.minorticks_off()
    ax.grid(alpha=0.3)
axs[0].set_ylabel("fraction [%]")
axs[0].legend(fontsize=8)
fig.tight_layout()
fig.savefig(f"{FIG}/step3_classes.png", dpi=110)
plt.close(fig)
print(f"wrote {FIG}/step3_classes.png [{time.time() - t0:.0f} s]")
