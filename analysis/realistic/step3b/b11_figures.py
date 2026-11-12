"""Step 3b / b11: figures.
  figures/realistic/step3b_R_zq.png                 R_s(z, q) - 1 [%] (the pooled paired blending ratio, q > 5) for the
                                                    production finder (merge none, fof 10'), SZiFi defaults and l_sep 12000,
                                                    both noise levels; numbers in each bin
  figures/realistic/step3b_bias_bars_MODE.png       bias / sigma in Om, s8, S8 per fit configuration (q > 5; SO40 and full
                                                    sky), style of step2_bias_bars.png: step 2's default (sigma_Y, fdom 1),
                                                    the step-3a preview range (step 2's rows fdom 0.5-1 x map sigma (i)),
                                                    and the step-3b fits (R_s; whiskers: tile-bootstrap error) for the
                                                    production finder (merge none and fof 10'), SZiFi defaults, l_sep 12000
Palette: the three categorical slots that validate all-pairs (dataviz reference: #2a78d6, #eb6834, #1baf7a) carry the finder
variant; marker fill carries the merge rule (open = fof 10'); step 2 and step 3a references are neutral grey.
"""
import os, sys, json
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import s3blib as B                                 # noqa: E402
import numpy as np                                 # noqa: E402
import matplotlib                                  # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt                    # noqa: E402
import textwrap                                    # noqa: E402
sys.path.insert(0, f"{B.ROOT}/analysis/cosmo/bias")
import cosmolib as CL                              # noqa: E402

plt.rcParams.update({"font.size": 10, "axes.edgecolor": "#52514e", "axes.labelcolor": "#0b0b0b", "xtick.color": "#52514e",
                     "ytick.color": "#52514e", "axes.spines.top": False, "axes.spines.right": False})
BLUE, ORANGE, AQUA, GREY, LIGHT = "#2a78d6", "#eb6834", "#1baf7a", "#52514e", "#b9b8b2"
CFG = ["a", "b", "c", "d", "e"]
CFG_LAB = {"a": "a: Ωm, σ8", "b": "b: + (1−b)", "c": "c: + (1−b), σlnY", "d": "d: + (1−b), α, σlnY\n   (1−b) prior 4%",
           "e": "e: + (1−b), α, σlnY\n   no prior"}
PL = {"Om": r"$\Omega_m$", "s8": r"$\sigma_8$", "S8": r"$S_8$"}
D2 = f"{B.ROOT}/data_real/step2"
S2ROWS = json.load(open(f"{D2}/fits.json")) + json.load(open(f"{D2}/fits_arbiter_fdom_eff.json"))
FITS = json.load(open(os.path.join(B.OUT, "fits.json")))
LIN = json.load(open(os.path.join(B.OUT, "lin.json")))


def s2get(setup, cn, sv):
    for r in S2ROWS:
        if r["setup"] == setup and r["config"] == cn and r.get("survey", "SO40" if abs(r["fsky"] - 0.4) < 1e-6 else "full") == sv \
                and r.get("qcut", 5.0) == 5.0 and r["route"] == "cat" and r.get("start", "truth") == "truth":
            return r


def s3get(setup, mode, cn, sv):
    r = [x for x in FITS if x["setup"] == setup and x["mode"] == mode and x["config"] == cn and x["survey"] == sv
         and x["qcut"] == 5.0 and x["route"] == "cat" and x["estimator"] == "Rs"][0]
    l_ = [x for x in LIN if x["setup"] == setup and x["mode"] == mode and x["config"] == cn and x["survey"] == sv
          and x["qcut"] == 5.0 and x["estimator"] == "Rs"][0]
    return r, l_


def bars(mode):
    fig, axs = plt.subplots(2, 3, figsize=(13, 9.8), sharey=True)
    series = [dict(kind="s2", setup=f"{mode}_f1", lab="step 2 default: q = Ŷ/σ_Y, attribution fdom 1", col=GREY, mk="D", fill=True),
              dict(kind="s3", setup="prod_none", lab="step 3b, production finder (whitened mask)", col=BLUE, mk="o", fill=True),
              dict(kind="s3", setup="prod_fof10", lab="step 3b, production finder, merged with fof 10′ (off-scale: value at the edge)",
                   col=BLUE, mk="o", fill=False, clip=True),
              dict(kind="s3", setup="szifi3000_none", lab="step 3b, SZiFi defaults (ℓ_sep 3000, blend)", col=ORANGE, mk="s", fill=True),
              dict(kind="s3", setup="szifi12000_none", lab="step 3b, SZiFi SO setting (ℓ_sep 12000)", col=AQUA, mk="^", fill=True)]
    n = len(series) + 1
    offs = np.linspace(-0.34, 0.34, n)
    for irow, sv in enumerate(["SO40", "full"]):
        for ip, p in enumerate(["Om", "s8", "S8"]):
            ax = axs[irow, ip]
            ax.axvspan(-1, 1, color="#e9e8e4", zorder=0)
            ax.axvline(0, color=GREY, lw=0.8)
            vals = [-1.0, 1.0]
            pts = []
            # step-3a preview range: step 2's rows fdom 0.5-1 x map sigma (i)
            for j, cn in enumerate(CFG):
                v = [s2get(s, cn, sv)[f"b_{p}"] for s in (f"{mode}_effi", f"{mode}_f0.5_effi")]
                vals += v
                ax.plot([min(v), max(v)], [j + offs[0]] * 2, color=LIGHT, lw=5, solid_capstyle="butt", zorder=2,
                        label="step 3a preview: step 2's fdom 0.5–1 × map σ (i)" if (irow == 0 and ip == 0 and j == 0) else None)
            for k, sr in enumerate(series):
                xs, ys, err = [], [], []
                for j, cn in enumerate(CFG):
                    if sr["kind"] == "s2":
                        xs.append(s2get(sr["setup"], cn, sv)[f"b_{p}"]); err.append(0.0)
                    else:
                        r, l_ = s3get(sr["setup"], mode, cn, sv)
                        xs.append(r[f"b_{p}"]); err.append(l_[f"boot_{p}"])
                    ys.append(j + offs[k + 1])
                if not sr.get("clip"):
                    vals += [x - e for x, e in zip(xs, err)] + [x + e for x, e in zip(xs, err)]
                pts.append((sr, xs, ys, err))
            lo, hi = min(vals), max(vals)
            pad = 0.12 * (hi - lo)
            lo, hi = lo - pad, hi + pad
            for sr, xs, ys, err in pts:
                kw = dict(ms=7, color=sr["col"], mfc=sr["col"] if sr["fill"] else "white", mec="white" if sr["fill"] else sr["col"],
                          mew=1.3, elinewidth=1.4, capsize=0, zorder=3)
                if sr.get("clip"):
                    xin = [x if lo <= x <= hi else np.nan for x in xs]
                    ax.errorbar(xin, ys, fmt=sr["mk"], label=sr["lab"] if (irow == 0 and ip == 0) else None, **kw)
                    for x, y in zip(xs, ys):
                        if x < lo or x > hi:
                            xe = lo + 0.01 * (hi - lo) if x < lo else hi - 0.01 * (hi - lo)
                            ax.plot([xe], [y], marker="<" if x < lo else ">", ms=7, color=sr["col"], mfc="white", mew=1.3, zorder=3)
                            ax.text(xe + (0.025 if x < lo else -0.025) * (hi - lo), y, f"{x:+.1f}", ha="left" if x < lo else "right",
                                    va="center", fontsize=7.5, color="#0b0b0b")
                else:
                    ax.errorbar(xs, ys, xerr=err, fmt=sr["mk"], label=sr["lab"] if (irow == 0 and ip == 0) else None, **kw)
            ax.set_xlim(lo, hi)
            ax.set_yticks(range(len(CFG))); ax.set_yticklabels([CFG_LAB[c] for c in CFG])
            ax.invert_yaxis()
            ax.set_xlabel(f"bias in {PL[p]}  [σ]")
            ax.grid(axis="x", color="#e9e8e4", lw=0.6)
            if irow == 0:
                ax.set_title(PL[p])
    fig.legend(loc="upper center", ncol=2, frameon=False, bbox_to_anchor=(0.5, 0.997), fontsize=9)
    title = (f"SO {mode}. Naive (no-blending) model fitted to the finder's own catalogue (Asimov), q_obs > 5; σ = Fisher at the truth. "
             "Step 3b: the analyst's model uses the finder's own isolated-cluster selection and noise level; data = model × the "
             "measured blending ratio R_s; whiskers: tile-bootstrap error (production finder: 1,548 tiles; SZiFi variants: 324 tiles). "
             "Grey band: |bias| < 1σ. Open circles beyond the axis are drawn at the edge with their value.")
    fig.text(0.5, 0.005, "\n".join(textwrap.wrap(title, 175)), ha="center", va="bottom", fontsize=8.5, color=GREY)
    fig.tight_layout(rect=(0, 0.055, 1, 0.875), h_pad=3.2)
    for irow, sv in enumerate(["SO40", "full"]):
        y1 = axs[irow, 0].get_position().y1
        fig.text(0.01, y1 + 0.032, "SO-like 16,500 deg² (f_sky = 0.4)" if sv == "SO40" else "full sky (41,253 deg²)",
                 fontsize=11, fontweight="bold", va="bottom")
    fn = os.path.join(B.FIG, f"step3b_bias_bars_{mode}.png")
    fig.savefig(fn, dpi=150); plt.close(fig)
    print("wrote", fn)


def rmaps():
    panels = [("prod_none", "production finder"), ("prod_fof10", "production, fof 10′ merging"),
              ("szifi3000_none", "SZiFi defaults (ℓ_sep 3000)"), ("szifi12000_none", "SZiFi ℓ_sep 12000")]
    zed, qed = CL.default_bins(5.0)
    vmax = 8.0
    fig, axs = plt.subplots(2, 4, figsize=(21, 12.5), constrained_layout=True)
    zl = [f"{a:g}-{b:g}" for a, b in zip(zed[:-1], zed[1:])]
    ql = [f"{a:.3g}-{b:.3g}" if np.isfinite(b) else f">{a:.3g}" for a, b in zip(qed[:-1], qed[1:])]
    im = None
    for irow, mode in enumerate(B.MODES):
        for icol, (s, lab) in enumerate(panels):
            ax = axs[irow, icol]
            d = np.load(os.path.join(B.OUT, f"datavec_{s}_{mode}.npz"))
            mu = d["mu_q5"]
            v = np.where(mu >= 1.0, 100 * (d["Rs_q5"] - 1), np.nan)
            ax.imshow(np.isnan(v).astype(float), origin="lower", aspect="auto", cmap="Greys", vmin=0, vmax=8)
            im = ax.imshow(np.ma.masked_invalid(v), origin="lower", aspect="auto", cmap="RdBu_r", vmin=-vmax, vmax=vmax)
            for i in range(v.shape[0]):
                for j in range(v.shape[1]):
                    if np.isfinite(v[i, j]):
                        ax.text(j, i, f"{v[i, j]:.1f}", ha="center", va="center", fontsize=6.3,
                                color="#ffffff" if abs(v[i, j]) > 0.65 * vmax else "#0b0b0b")
            tot = 100 * (np.sum(mu * d["Rs_q5"]) / mu.sum() - 1)
            totd = 100 * (d["Ndet_q5"].sum() / d["Nexp_q5"].sum() - 1)
            ax.set_title(f"{lab}, SO {mode}\nq > 5: R_s {tot:+.2f}%, R {totd:+.2f}%", fontsize=10)
            ax.set_xticks(range(len(ql))); ax.set_xticklabels(ql, rotation=45, ha="right", fontsize=7.5)
            ax.set_yticks(range(len(zl))); ax.set_yticklabels(zl if icol == 0 else [], fontsize=7.5)
            ax.set_xlabel("q_obs bin")
            if icol == 0:
                ax.set_ylabel("z bin")
    fig.colorbar(im, ax=axs, shrink=0.55, label="R_s − 1 [%]: blended / isolated-cluster expectation − 1, q_obs > 5 "
                 "(colour clipped at ±8%; grey: < 1 model cluster on the full sky)")
    fn = os.path.join(B.FIG, "step3b_R_zq.png")
    fig.savefig(fn, dpi=150); plt.close(fig)
    print("wrote", fn)


if __name__ == "__main__":
    rmaps()
    for mode in B.MODES:
        bars(mode)
