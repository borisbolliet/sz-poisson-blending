"""Step 2 / p08: figures.
  figures/realistic/step2_bias_bars.png          bias/sigma in Om, s8, S8 per fit configuration (q > 5; SO40 and full sky):
                                                 old white filter, SO baseline, SO goal (whiskers: dominance rule fdom 0.5-2),
                                                 low-pressure population
  figures/realistic/step2_bias_bars_mapsigma.png same for the analyst's SNR normalisation: sigma_Y vs map-sigma (i), (ii)
  figures/realistic/step2_delta_zq.png           Delta / N_naive [%] in the (z, q_obs) bins (q > 5)
Style of figures/cosmo/bias_bars_v2.png (c17).  Palette: categorical slots 1-3 (validated all-pairs, light mode) plus a
neutral grey for the old-filter reference; marker shapes as the secondary encoding.
"""
import os, sys, json
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import s2lib as S                                  # noqa: E402
from s2lib import CL                                # noqa: E402

rows = json.load(open(f"{S.D2}/fits.json"))
plt.rcParams.update({"font.size": 10, "axes.edgecolor": "#52514e", "axes.labelcolor": "#0b0b0b",
                     "xtick.color": "#52514e", "ytick.color": "#52514e", "axes.spines.top": False, "axes.spines.right": False})
BLUE, ORANGE, AQUA, GREY = "#2a78d6", "#eb6834", "#1baf7a", "#52514e"
CFG = ["a", "b", "c", "d", "e"]
CFG_LAB = {"a": "a: Ωm, σ8", "b": "b: + (1−b)", "c": "c: + (1−b), σlnY", "d": "d: + (1−b), α, σlnY\n   (1−b) prior 4%",
           "e": "e: + (1−b), α, σlnY\n   no prior"}
PL = {"Om": r"$\Omega_m$", "s8": r"$\sigma_8$", "S8": r"$S_8$"}


def get(setup, cn, sv, qc=5.0):
    for r in rows:
        if r["setup"] == setup and r["config"] == cn and r["survey"] == sv and r["qcut"] == qc and r["route"] == "cat" \
                and r.get("start", "truth") == "truth":
            return r


def bars(series, fn, title):
    fig, axs = plt.subplots(2, 3, figsize=(13, 9.6), sharey=True)
    n = len(series)
    offs = np.linspace(-0.32, 0.32, n) if n > 1 else [0.0]
    for irow, sv in enumerate(["SO40", "full"]):
        for ip, p in enumerate(["Om", "s8", "S8"]):
            ax = axs[irow, ip]
            ax.axvspan(-1, 1, color="#e9e8e4", zorder=0)
            ax.axvline(0, color="#52514e", lw=0.8)
            for k, sr in enumerate(series):
                xs, ys, lo, hi = [], [], [], []
                for j, cn in enumerate(CFG):
                    r = get(sr["setup"], cn, sv)
                    xs.append(r[f"b_{p}"]); ys.append(j + offs[k])
                    if "range" in sr:
                        v = [get(s2, cn, sv)[f"b_{p}"] for s2 in sr["range"]]
                        lo.append(max(xs[-1] - min(v), 0)); hi.append(max(max(v) - xs[-1], 0))
                kw = dict(fmt=sr["mk"], ms=7, color=sr["col"], mec="white" if sr.get("fill", True) else sr["col"],
                          mfc=sr["col"] if sr.get("fill", True) else "white", mew=1.2, elinewidth=1.6, capsize=0, zorder=3,
                          label=sr["lab"] if (irow == 0 and ip == 0) else None)
                if "range" in sr:
                    ax.errorbar(xs, ys, xerr=[lo, hi], **kw)
                else:
                    ax.errorbar(xs, ys, **kw)
            ax.set_yticks(range(len(CFG))); ax.set_yticklabels([CFG_LAB[c] for c in CFG])
            ax.invert_yaxis()
            ax.set_xlabel(f"bias in {PL[p]}  [σ]")
            ax.grid(axis="x", color="#e9e8e4", lw=0.6)
            if irow == 0:
                ax.set_title(PL[p])
    import textwrap
    fig.legend(loc="upper center", ncol=2 if n <= 4 else 4, frameon=False, bbox_to_anchor=(0.5, 0.995), fontsize=9)
    fig.text(0.5, 0.005, "\n".join(textwrap.wrap(title, 175)), ha="center", va="bottom", fontsize=8.5, color="#52514e")
    fig.tight_layout(rect=(0, 0.045, 1, 0.885), h_pad=3.2)
    for irow, sv in enumerate(["SO40", "full"]):
        y1 = axs[irow, 0].get_position().y1
        fig.text(0.01, y1 + 0.032, "SO-like 16,500 deg² (f_sky = 0.4)" if sv == "SO40" else "full sky (41,253 deg²)",
                 fontsize=11, fontweight="bold", va="bottom")
    fig.savefig(fn, dpi=150); plt.close(fig)
    print("wrote", fn)


bars([dict(setup="white", lab="old white filter (Q3 setup, step-2 kernel)", col=GREY, mk="D"),
      dict(setup="baseline_f1", lab="SO baseline MMF (bar: attribution fdom 0.5–2)", col=BLUE, mk="o",
           range=["baseline_f0.5", "baseline_f1", "baseline_f2"]),
      dict(setup="goal_f1", lab="SO goal MMF (bar: fdom 0.5–2)", col=ORANGE, mk="s", range=["goal_f0.5", "goal_f1", "goal_f2"]),
      dict(setup="lowP_f1", lab="SO baseline, low-pressure population (1−b = 0.6)", col=AQUA, mk="^")],
     f"{S.FIG2}/step2_bias_bars.png",
     "Naive (no-blending) model fitted to the blended expectation (Asimov), q_obs > 5; σ = Fisher at the truth. Grey band: |bias| < 1σ. "
     "Markers: dominance rule x_B < q_A; bars: x_B < 0.5 q_A to 2 q_A.")

bars([dict(setup="baseline_f1", lab="baseline, q = Ŷ/σ_Y", col=BLUE, mk="o"),
      dict(setup="baseline_effi", lab="baseline, map σ (i): Gaussian sub-threshold", col=BLUE, mk="s", fill=False),
      dict(setup="baseline_effii", lab="baseline, map σ (ii): robust width", col=BLUE, mk="^", fill=False),
      dict(setup="goal_f1", lab="goal, q = Ŷ/σ_Y", col=ORANGE, mk="o"),
      dict(setup="goal_effi", lab="goal, map σ (i)", col=ORANGE, mk="s", fill=False),
      dict(setup="goal_effii", lab="goal, map σ (ii)", col=ORANGE, mk="^", fill=False),
      dict(setup="lowP_f1", lab="low pressure, q = Ŷ/σ_Y", col=AQUA, mk="o"),
      dict(setup="lowP_effi", lab="low pressure, map σ (i)", col=AQUA, mk="s", fill=False)],
     f"{S.FIG2}/step2_bias_bars_mapsigma.png",
     "As step2_bias_bars.png, fdom = 1. Map σ: the SNR (data and naive model) uses σ_eff(θ) measured once on the fiducial sky; "
     "(i) σ_eff² = σ_Y² + sub-threshold confusion variance, (ii) robust 16–84% half-width of noise + confusion.")

# ---------------- Delta / N maps
panels = [("white", "old white filter (Q3 setup)"), ("baseline_f1", "SO baseline MMF"), ("goal_f1", "SO goal MMF"),
          ("baseline_effi", "SO baseline, map σ (i)"), ("baseline_effii", "SO baseline, map σ (ii)"),
          ("lowP_f1", "SO baseline, low pressure (1−b = 0.6)")]
zed, qed = CL.default_bins(5.0)
vals = {}
for s, _ in panels:
    t = np.load(f"{S.D2}/truth_{s}.npz")
    vals[s] = 100 * t["D_q5"] / np.where(t["Nn_q5"] > 0, t["Nn_q5"], np.nan)
vmax = 8.0
fig, axs = plt.subplots(2, 3, figsize=(16, 12.5), constrained_layout=True)
zl = [f"{a:g}-{b:g}" for a, b in zip(zed[:-1], zed[1:])]
ql = [f"{a:.3g}-{b:.3g}" if np.isfinite(b) else f">{a:.3g}" for a, b in zip(qed[:-1], qed[1:])]
for k, (ax, (s, lab)) in enumerate(zip(axs.ravel(), panels)):
    t = np.load(f"{S.D2}/truth_{s}.npz")
    v = np.where(t["Nn_q5"] >= 1.0, vals[s], np.nan)          # bins with < 1 naive cluster (full sky) not shown
    ax.imshow(np.isnan(v).astype(float), origin="lower", aspect="auto", cmap="Greys", vmin=0, vmax=8)
    im = ax.imshow(np.ma.masked_invalid(v), origin="lower", aspect="auto", cmap="RdBu_r", vmin=-vmax, vmax=vmax)
    for i in range(v.shape[0]):
        for j in range(v.shape[1]):
            if np.isfinite(v[i, j]):
                ax.text(j, i, f"{v[i, j]:.1f}", ha="center", va="center", fontsize=6.5,
                        color="#ffffff" if abs(v[i, j]) > 0.65 * vmax else "#0b0b0b")
    ax.set_title(f"{lab}\nΔN/N total {100*t['D_q5'].sum()/t['Nn_q5'].sum():+.2f}%", fontsize=10)
    ax.set_xticks(range(len(ql))); ax.set_xticklabels(ql, rotation=45, ha="right", fontsize=7.5)
    ax.set_yticks(range(len(zl))); ax.set_yticklabels(zl if k % 3 == 0 else [], fontsize=7.5)
    ax.set_xlabel("q_obs bin")
    if k % 3 == 0:
        ax.set_ylabel("z bin")
fig.colorbar(im, ax=axs, shrink=0.55, label="blended / naive − 1  [%]  (q_obs > 5; colour clipped at ±8%; grey: < 1 naive cluster on the full sky)")
fig.savefig(f"{S.FIG2}/step2_delta_zq.png", dpi=150); plt.close(fig)
print("wrote", f"{S.FIG2}/step2_delta_zq.png")
