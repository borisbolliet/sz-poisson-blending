"""Aggregate the sensitivity passes (data/robustness/pass_*.npz), compare with analytic predictions,
print tables and make the scaling figures.

Analytic predictions used:
  * Sigma_det = N_det / 4pi from the selection integral of each config (SZModel.expected_detections with
    the config's beam, noise, scatter, HMF table and zmax);
  * Q1/B1: the other single detections around a detection are Poisson with mean lambda = Sigma_det A_cap(theta)
    (independent thinning + Slivnyak), P(>=k) = Poisson CCDF;
  * Campbell/Slivnyak: E[delta_A | A detected in isolation] = mu_A = sigma_A RATIO ybar / N_w;
  * exclusion: Poisson pairs that overlap in 3D (analytic in clustering_limber.py, q_cut = 5, baseline).
Errors: delete-one-region jackknife (HEALPix nside=2, 48 regions per sky, all skies of a pass).
[revised after verification] N_det and Sigma_det in the Q1/B1 test are restricted to the population
run_pass.py follows (primaries with true q > 0.5): N_det - N_lowq; per-region <delta> sums allow a
jackknife error on the mean boost (despali16 extra skies).
"""
NREG_JK = 48
import os, sys, glob, json
os.environ.setdefault("OMP_NUM_THREADS", "2")
import numpy as np
from scipy.stats import poisson

HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import configs as C
import blendlib as bl
from szsim import signal as sg, cosmo_hmf as ch

DD = os.path.join(ch.DATA_DIR, "robustness")
FIG = "/Users/boris/GitHub/sz-poisson-blending/figures/robustness"; os.makedirs(FIG, exist_ok=True)
ARC = ch.ARCMIN


def jk_ratio(num, den):
    """num, den: (nreg_total,) counts. Ratio of sums with delete-one jackknife error."""
    Nn, Nd = num.sum(), den.sum()
    ok = den > 0
    rj = (Nn - num[ok]) / (Nd - den[ok])
    n = ok.sum()
    return Nn / Nd, np.sqrt((n - 1) / n * np.sum((rj - rj.mean()) ** 2))


# ---------------------------------------------------------------- analytic N_det per config (cached)
ndet_lowq = bl.ndet_lowq   # moved to blendlib (added after verification)


an_fn = os.path.join(DD, "analytic_counts.json")
AN = json.load(open(an_fn)) if os.path.exists(an_fn) else {}
tables = {}
for pname, P in C.PASSES.items():
    if P["hmf"] not in tables:
        tables[P["hmf"]] = ch.default_table() if P["hmf"] == "tinker08" else ch.HMFTable(model=P["hmf"])
    for cname, cf in P["configs"].items():
        if cname in AN:
            continue
        m = sg.SZModel(beam_fwhm_arcmin=P["beam"], noise_y_arcmin=P["noise"] * cf["f"], sigma_lnY=P["sig"])
        nd = m.expected_detections(np.array(C.QCUTS), table=tables[P["hmf"]], zmax=cf["zmax"])
        # analytic mean y of the kept neighbours (ensemble), for mu
        yb = bl.mean_ybar(sg.SZModel(sigma_lnY=P["sig"]), tables[P["hmf"]], cf["floor"], zmax=cf["zmax"])
        AN[cname] = dict(Ndet={f"{q:g}": float(v) for q, v in zip(C.QCUTS, np.atleast_1d(nd))}, ybar=yb)
        print(f"analytic {cname}: N_det = {np.round(nd, 1)}, ybar_nb = {yb:.4e}", flush=True)
for pname, P in C.PASSES.items():
    for cname, cf in P["configs"].items():
        if "Nlowq" in AN[cname]:
            continue
        m = sg.SZModel(beam_fwhm_arcmin=P["beam"], noise_y_arcmin=P["noise"] * cf["f"], sigma_lnY=P["sig"])
        nl = ndet_lowq(m, C.QCUTS, tables[P["hmf"]], cf["zmax"])
        AN[cname]["Nlowq"] = {f"{q:g}": float(v) for q, v in zip(C.QCUTS, nl)}
        print(f"analytic {cname}: detections from q < {C.QPRIM} halos (not followed by run_pass) = {np.round(nl, 2)}", flush=True)
NPRIM = {c: AN[c]["Ndet"]["5"] - AN[c]["Nlowq"]["5"] for c in AN}
json.dump(AN, open(an_fn, "w"), indent=1)

# ---------------------------------------------------------------- load MC
results = {}
for pname, P in C.PASSES.items():
    files = sorted(glob.glob(os.path.join(DD, f"pass_{pname}_seed*.npz")))
    if not files:
        continue
    for cname in P["configs"]:
        acc = {}
        per_sky = {"mean": [], "nsingle5": [], "ybar": []}
        for fn in files:
            d = np.load(fn)
            meta = json.loads(str(d["meta"]))
            for k in d.files:
                if not k.startswith(cname + "|"):
                    continue
                kk = k.split("|", 1)[1]
                v = d[k]
                if kk in ("mean_delta_single5", "n_prim", "pairs_sep_ovl", "pairs_sep_all", "pairs_ovl_dz"):
                    acc.setdefault(kk, []).append(v)
                else:
                    acc.setdefault(kk, []).append(v)
            per_sky["ybar"].append(np.array(meta["ybar_g"]))
        R = {k: (np.concatenate(v) if k not in ("mean_delta_single5", "pairs_sep_ovl", "pairs_sep_all") else np.array(v))
             for k, v in acc.items()}
        R["nsky"] = len(files)
        R["ybar_g"] = np.array(per_sky["ybar"])
        results[cname] = R

print(f"\nMC skies per config: " + ", ".join(f"{c}: {r['nsky']}" for c, r in results.items()))

# ---------------------------------------------------------------- Q1 / B1 vs analytic
rows = []
zall = []
print("\n=== Q1 / B1: P(>=k other single detections within theta | single detection), q_cut = 5 ===")
for cname, R in results.items():
    Nall = AN[cname]["Ndet"]["5"]; Nan = NPRIM[cname]; Sig = Nan / (4 * np.pi)
    Nmc = R["single_5"].sum() / R["nsky"]
    eN = np.sqrt(Nan / R["nsky"])
    line = (f"{cname:14s} N_det(primaries q>0.5) MC {Nmc:9.1f} an {Nan:9.1f} ({(Nmc-Nan)/eN:+.2f}σ) [all halos an {Nall:9.1f}, "
            f"old comparison {(Nmc-Nall)/eN:+.2f}σ] Σ_det={Nan/(4*np.pi)*ARC**2*3600:.3f}/deg²")
    print(line)
    for th in C.THETAS:
        A = 2 * np.pi * (1 - np.cos(th * ARC))
        lam = Sig * A
        for k in range(1, C.KMAX + 1):
            num = R[f"B1s_{th:g}_{k}_5"]; den = R["single_5"]
            r, e = jk_ratio(num, den)
            pa = poisson.sf(k - 1, lam)
            if num.sum() >= 5:
                z = (r - pa) / e
                zall.append(z)
                rows.append(dict(config=cname, theta=th, k=k, mc=r, err=e, an=pa, z=z, lam=lam))
                if th in (1.0, 2.0, 5.0, 10.0) or k == 1:
                    print(f"     theta={th:5.1f}' k>={k}: MC {r:.4e} ± {e:.1e}, Poisson {pa:.4e} ({z:+.2f}σ)")
zall = np.array(zall)
print(f"B1 summary: {len(zall)} tests (with >=5 MC objects), rms z = {np.sqrt(np.mean(zall**2)):.2f}, max|z| = {np.max(np.abs(zall)):.2f}")

# ---------------------------------------------------------------- B1 on the blended catalogue (c=1) vs Poisson
print("\n=== B1 on the blended (monopole-subtracted, dominant) catalogue vs Poisson with its own Sigma ===")
for cname in ["base", "noise_x0.5", "noise_x2", "beam_5", "planck_like"]:
    if cname not in results:
        continue
    R = results[cname]
    Nb = R["detb1_5"].sum() / R["nsky"]; Sig = Nb / (4 * np.pi)
    for th in (0.5, 1.0, 2.0, 5.0, 10.0):
        lam = Sig * 2 * np.pi * (1 - np.cos(th * ARC))
        r, e = jk_ratio(R[f"B1b_{th:g}_1_5"], R["detb1_5"])
        print(f"   {cname:12s} theta={th:4.1f}': blended-catalogue P(>=1) = {r:.4e} ± {e:.1e}; Poisson(Σ_b) {1-np.exp(-lam):.4e} "
              f"ratio {r/(1-np.exp(-lam)):.3f} ({(r-(1-np.exp(-lam)))/max(e, 1e-30):+.2f}σ; n_obj={R[f'B1b_{th:g}_1_5'].sum()})")

# ---------------------------------------------------------------- Campbell/Slivnyak: mean delta of single detections
print("\n=== mean neighbour boost of single detections (q_cut=5): MC <delta> vs analytic mu (Campbell + Slivnyak) ===")
for cname, R in results.items():
    md = R["mean_delta_single5"]  # (nsky, 4): mean D, mean mu(sky ybar), mean overlap part, median(D-mu)
    m, e = md[:, 0].mean(), md[:, 0].std(ddof=1) / np.sqrt(len(md))
    mu_sky = md[:, 1].mean()
    P = [p for p in C.PASSES.values() if cname in p["configs"]][0]
    gl = C.groups_for(P["configs"][cname]["floor"], P["configs"][cname]["zmax"])
    ysky = R["ybar_g"][:, gl].sum(1)
    mu_an = np.mean(md[:, 1] * AN[cname]["ybar"] / ysky)
    R["mu_an"] = mu_an; R["delta_mc"] = (m, e)
    print(f"   {cname:14s} [{len(md)} skies] <delta> MC {m:.4f} ± {e:.4f} | mu (sky ybar) {mu_sky:.4f} ({(m-mu_sky)/e:+.2f}σ) | "
          f"mu (ensemble ybar) {mu_an:.4f} ({(m-mu_an)/e:+.2f}σ) | median(delta-mu) {md[:,3].mean():+.4f} | "
          f"overlap part {md[:,2].mean():.2e}")
    res_sky = md[:, 0] - md[:, 1]
    print(f"      per-sky (delta - mu_sky): rms {res_sky.std(ddof=1):.4f}; rel. offset (MC/mu_an - 1) = {m/mu_an-1:+.4f}; "
          f"sky ybar / ensemble ybar = {ysky.mean()/AN[cname]['ybar']:.4f} ± {ysky.std(ddof=1)/np.sqrt(len(ysky))/AN[cname]['ybar']:.4f}")
    R["res_sky"] = res_sky; R["mu_rel"] = (m / mu_an - 1, e / mu_an)
    if "sumD5_reg" in R:   # region jackknife on the skies that store per-region sums (added after verification)
        nsk = len(R["sumD5_reg"]) // NREG_JK
        sD, sM, nn = R["sumD5_reg"], R["summu5_reg"], R["n5_reg"]
        resid = lambda a, b, c: (a.sum() - b.sum()) / c.sum()
        r0 = resid(sD, sM, nn)
        ok = nn > 0
        rj = np.array([(sD.sum() - sD[i] - sM.sum() + sM[i]) / (nn.sum() - nn[i]) for i in np.nonzero(ok)[0]])
        nj = len(rj); ej = np.sqrt((nj - 1) / nj * np.sum((rj - rj.mean()) ** 2))
        md_new = md[-nsk:]
        e_sky = (md_new[:, 0] - md_new[:, 1]).std(ddof=1) / np.sqrt(nsk)
        print(f"      region jackknife ({nsk} skies x {NREG_JK} regions): <delta - mu_sky> = {r0:+.4f} ± {ej:.4f} ({r0/ej:+.2f}σ); "
              f"same skies, per-sky error {e_sky:.4f} ({r0/e_sky:+.2f}σ)")

# ---------------------------------------------------------------- blending fractions table
print("\n=== Catalogue blending fractions (fractions of det_b; c=0 raw map, c=1 map mean removed) ===")
hdr = "config          qcut  Ndet_b(c1)/Nsingle   B2(0.1)c1         B2(0.2)c1         B3 c1             B2(0.1)c0         B3 c0             sat/single        B2ovl(0.1)c1"
print(hdr)
TAB = {}
for cname, R in results.items():
    for qc in C.QCUTS:
        key = f"{qc:g}"
        out = {}
        out["ratio"] = jk_ratio(R[f"detb1_{key}"], R[f"single_{key}"])
        out["B2_0.1_c1"] = jk_ratio(R[f"b2_1_0.1_{key}"], R[f"detb1_{key}"])
        out["B2_0.2_c1"] = jk_ratio(R[f"b2_1_0.2_{key}"], R[f"detb1_{key}"])
        out["B2_0.05_c1"] = jk_ratio(R[f"b2_1_0.05_{key}"], R[f"detb1_{key}"])
        out["B2_0.5_c1"] = jk_ratio(R[f"b2_1_0.5_{key}"], R[f"detb1_{key}"])
        out["B3_c1"] = jk_ratio(R[f"b3_1_{key}"], R[f"detb1_{key}"])
        out["B2_0.1_c0"] = jk_ratio(R[f"b2_0_0.1_{key}"], R[f"detb0_{key}"])
        out["B3_c0"] = jk_ratio(R[f"b3_0_{key}"], R[f"detb0_{key}"])
        out["sat"] = jk_ratio(R[f"sat_{key}"], R[f"single_{key}"])
        out["lost_c1"] = jk_ratio(R[f"lost1_{key}"], R[f"single_{key}"])
        out["B2ovl_0.1_c1"] = jk_ratio(R[f"b2ovl_1_0.1_{key}"], R[f"detb1_{key}"])
        out["hasovl"] = jk_ratio(R[f"single_hasovl_{key}"], R[f"single_{key}"])
        TAB[(cname, qc)] = out
        f = lambda t: f"{t[0]:.4f}±{t[1]:.4f}"
        print(f"{cname:14s} {qc:4.0f}  {f(out['ratio'])}    {f(out['B2_0.1_c1'])}   {f(out['B2_0.2_c1'])}   {f(out['B3_c1'])}   "
              f"{f(out['B2_0.1_c0'])}   {f(out['B3_c0'])}   {f(out['sat'])}   {f(out['B2ovl_0.1_c1'])}")

print("\n=== sensitivity of q_cut=5 answers relative to base (difference in units of combined error) ===")
b = TAB[("base", 5.0)]
for cname in results:
    if cname == "base":
        continue
    o = TAB[(cname, 5.0)]
    s = []
    for k in ("B2_0.1_c1", "B3_c1", "B2_0.1_c0", "B3_c0"):
        d = o[k][0] - b[k][0]; e = np.hypot(o[k][1], b[k][1])
        s.append(f"{k}: {o[k][0]/b[k][0]:.3f}x ({d/e:+.1f}σ)")
    print(f"   {cname:14s} " + "; ".join(s))

# ---------------------------------------------------------------- exclusion: overlapping Poisson pairs (MC vs analytic)
L = np.load(os.path.join(DD, "limber.npz"))
print("\n=== Halo exclusion: detected (q>5) Poisson pairs that overlap in 3D (baseline config) ===")
R = results["base"]
po = R["pairs_sep_ovl"].sum(0); pa = R["pairs_sep_all"].sum(0); ns = R["nsky"]
for i, th in enumerate(list(C.THETAS) + [150.0]):
    an_o = L["Novl"][i]; an_a = L["Npairs"][i]
    lam_p = 2 * an_a / AN["base"]["Ndet"]["5"]           # mean number of neighbours within theta
    var_p = AN["base"]["Ndet"]["5"] * (lam_p / 2 + lam_p ** 2)  # Poisson-process U-statistic variance per sky
    print(f"   sep < {th:6.1f}': pairs MC {pa[i]/ns:10.1f} vs Poisson {an_a:10.1f} ({(pa[i]/ns-an_a)/np.sqrt(var_p/ns):+.2f}σ, Var = N lam/2 + N lam^2);"
          f" overlapping MC {po[i]} in {ns} skies = {po[i]/ns:.3f}/sky vs analytic {an_o:.3f}/sky "
          f"(Poisson p(MC>=obs|an) = {poisson.sf(po[i]-1, an_o*ns):.3f}, p(<=) = {poisson.cdf(po[i], an_o*ns):.3f}); fraction an {an_o/an_a:.2e}")
dz = np.concatenate(R["pairs_ovl_dz"]) if isinstance(R["pairs_ovl_dz"], list) else R["pairs_ovl_dz"]
r, e = jk_ratio(R["single_hasovl_5"], R["single_5"])
print(f"   P(single detection has >=1 M>1e13 halo overlapping in 3D): MC {r:.3e} ± {e:.1e}; analytic 1-exp(-E) = "
      f"{1-np.exp(-float(L['Eov'])):.3e} ({(r-(1-np.exp(-float(L['Eov']))))/e:+.2f}σ)")
for cname in ("base", "noise_x0.5", "floor_1e14"):
    o = TAB[(cname, 5.0)]
    print(f"   {cname}: fraction of blended detections whose Y is biased >10% by overlapping neighbours alone: "
          f"{o['B2ovl_0.1_c1'][0]:.2e} ± {o['B2ovl_0.1_c1'][1]:.1e} (vs B2(0.1) {o['B2_0.1_c1'][0]:.3f})")

# ---------------------------------------------------------------- analytic Sigma_det scalings (noise, beam)
sc_fn = os.path.join(DD, "sigma_det_scaling.json")
if os.path.exists(sc_fn):
    SC = json.load(open(sc_fn))
else:
    fgrid = [0.25, 0.35, 0.5, 0.7, 1.0, 1.4, 2.0, 2.8, 4.0]
    bgrid = [0.5, 1.0, 1.4, 2.0, 3.0, 5.0, 7.3, 10.0]
    Nf = [float(sg.SZModel(noise_y_arcmin=4e-6 * f).expected_detections(5.0)) for f in fgrid]
    Nb = [float(sg.SZModel(beam_fwhm_arcmin=b).expected_detections(5.0)) for b in bgrid]
    SC = dict(fgrid=fgrid, Nf=Nf, bgrid=bgrid, Nb=Nb)
    json.dump(SC, open(sc_fn, "w"))
fg, Nf, bg, Nb = map(np.array, (SC["fgrid"], SC["Nf"], SC["bgrid"], SC["Nb"]))
print("\n=== analytic Sigma_det(q>5) scalings ===")
print("   noise factor f : " + ", ".join(f"{f:g}:{n/(4*np.pi)*ARC**2*3600:.3f}" for f, n in zip(fg, Nf)) + "  [/deg^2]")
print("   local slope dlnN/dln(noise): " + ", ".join(f"{np.sqrt(fg[i]*fg[i+1]):.2f}:{np.log(Nf[i+1]/Nf[i])/np.log(fg[i+1]/fg[i]):.2f}" for i in range(len(fg)-1)))
print("   beam FWHM      : " + ", ".join(f"{b:g}':{n/(4*np.pi)*ARC**2*3600:.3f}" for b, n in zip(bg, Nb)) + "  [/deg^2]")
print("   local slope dlnN/dln(beam): " + ", ".join(f"{np.sqrt(bg[i]*bg[i+1]):.2f}:{np.log(Nb[i+1]/Nb[i])/np.log(bg[i+1]/bg[i]):.2f}" for i in range(len(bg)-1)))

# ---------------------------------------------------------------- figures
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
cols = plt.cm.tab20(np.linspace(0, 1, 20))
names = list(results.keys())
fig, axs = plt.subplots(2, 2, figsize=(12, 10))
ax = axs[0, 0]
x = np.geomspace(1e-5, 5, 200)
ax.plot(x, 1 - np.exp(-x), "k-", lw=1, label=r"$1-e^{-\Sigma_{\rm det}A}$")
ax.plot(x, poisson.sf(1, x), "k--", lw=1, label=r"$P(\geq2)$ Poisson")
for i, cn in enumerate(names):
    for k, mk in ((1, "o"), (2, "s")):
        rr = [r for r in rows if r["config"] == cn and r["k"] == k]
        ax.errorbar([r["lam"] for r in rr], [r["mc"] for r in rr], [r["err"] for r in rr], fmt=mk, ms=4, color=cols[i % 20],
                    label=cn if k == 1 else None, mfc="none" if k == 2 else None)
ax.set_xscale("log"); ax.set_yscale("log"); ax.set_xlim(1e-4, 5); ax.set_ylim(1e-5, 1.2)
ax.set_xlabel(r"$\lambda=\Sigma_{\rm det}(q_{\rm cut}=5)\,A_{\rm cap}(\theta)$ (analytic, per config)")
ax.set_ylabel(r"MC $P(\geq k$ other detections within $\theta)$")
ax.set_title("Q1: all configs collapse on the Poisson law (o: k>=1, □: k>=2)", fontsize=10)
ax.legend(fontsize=6, ncol=2)
ax = axs[0, 1]
ax.loglog(fg, Nf / (4 * np.pi) * ARC ** 2 * 3600, "C0-", label="noise x f (beam 1.4')")
ax.loglog(bg / 1.4, Nb / (4 * np.pi) * ARC ** 2 * 3600, "C1-", label="beam FWHM/1.4' (noise 4e-6)")
for cn, xv, c in (("base", 1, "k"), ("noise_x0.5", 0.5, "C0"), ("noise_x2", 2, "C0"), ("beam_5", 5 / 1.4, "C1")):
    if cn in results:
        R = results[cn]; n = R["single_5"].sum() / R["nsky"]
        ax.errorbar(xv, n / (4 * np.pi) * ARC ** 2 * 3600, np.sqrt(n / R["nsky"]) / (4 * np.pi) * ARC ** 2 * 3600, fmt="o", color=c)
if "planck_like" in results:
    R = results["planck_like"]; n = R["single_5"].sum() / R["nsky"]
    ax.plot(7.3 / 1.4, n / (4 * np.pi) * ARC ** 2 * 3600, "r*", ms=10, label="Planck-like (7.3', 1.16e-5) MC")
from matplotlib.ticker import FixedLocator, NullLocator, FixedFormatter
ax.xaxis.set_major_locator(FixedLocator([0.25, 0.5, 1, 2, 4, 7]))
ax.xaxis.set_major_formatter(FixedFormatter(["0.25", "0.5", "1", "2", "4", "7"])); ax.xaxis.set_minor_locator(NullLocator())
ax.set_xlabel("noise or beam relative to baseline"); ax.set_ylabel(r"$\Sigma_{\rm det}(q>5)$ [deg$^{-2}$]")
ax.set_title(r"$\Sigma_{\rm det}$ scaling (lines analytic, points MC)", fontsize=10); ax.legend(fontsize=7)
ax = axs[1, 0]
xs = np.arange(len(names))
for j, (k, lab, mk) in enumerate((("B2_0.1_c1", "B2(ε=0.1), mean removed", "o"), ("B3_c1", "B3 (blend-induced), mean removed", "s"),
                                  ("B2_0.1_c0", "B2(ε=0.1), raw map", "^"), ("B3_c0", "B3, raw map", "v"))):
    v = np.array([TAB[(cn, 5.0)][k] for cn in names])
    ax.errorbar(xs + 0.1 * (j - 1.5), v[:, 0], v[:, 1], fmt=mk, label=lab)
ax.set_xticks(xs); ax.set_xticklabels(names, rotation=60, ha="right", fontsize=7)
ax.set_yscale("log"); ax.set_ylabel("fraction of blended-catalogue detections (q_cut=5)")
ax.set_title("Q2 blending fractions across configurations", fontsize=10); ax.legend(fontsize=7)
ax = axs[1, 1]
for i, cn in enumerate(names):
    R = results[cn]
    ax.errorbar(R["mu_an"], R["delta_mc"][0], R["delta_mc"][1], fmt="o", color=cols[i % 20], label=cn)
xx = np.linspace(0, 2.2, 10); ax.plot(xx, xx, "k-", lw=1)
ax.set_xlabel(r"analytic $\langle\mu_A\rangle=\langle\sigma_A\rangle\,{\rm RATIO}\,\bar y/N_w$ (Campbell)")
ax.set_ylabel(r"MC $\langle\delta_A\rangle$ (single detections, q>5)")
ax.set_title("mean neighbour boost scales with ybar σ_A/N_w", fontsize=10); ax.legend(fontsize=6)
fig.tight_layout(); fig.savefig(os.path.join(FIG, "robustness_scaling.png"), dpi=140); plt.close(fig)

# clustering figure
fig, axs = plt.subplots(1, 2, figsize=(12, 4.6))
ax = axs[0]
ax.loglog(L["TH"], L["w"], label="w(θ) detected q>5, linear Tinker10 bias (Limber)")
ax.loglog(L["TEX"], np.clip(L["wex"], 1e-4, None), label="with halo exclusion (ξ=-1 for r<R_A+R_B)")
ax.loglog(L["TH"], L["wx"], label="cross: detected x all halos M>1e13")
ax.loglog(L["TH"], L["wdy"], label="cross: detected x y-weighted halos (blending)")
ax.plot(L["TCHK"], L["we"], "kx", label="exact non-Limber (check)")
ax.set_xlabel("θ [arcmin]"); ax.set_ylabel("w(θ)"); ax.legend(fontsize=7); ax.set_xlim(0.05, 120)
ax = axs[1]
rows_l = json.loads(str(L["rows"]))
th = [r["theta"] for r in rows_l]
ax.semilogx(th, [r["one_plus_wbar"] for r in rows_l], "o-", label="1+w̄ (linear)")
ax.semilogx(th, [r["one_plus_wbar_ex"] for r in rows_l], "s-", label="1+w̄ (linear + exclusion)")
ax.semilogx(th, [r["P_clust"] / r["P_poisson"] for r in rows_l], "^--", label="P_clust(≥1)/P_Poisson(≥1)")
ax.semilogx(th, [r["one_plus_wbar_x"] for r in rows_l], "d-", label="1+w̄_x (det x M>1e13)")
ax.axhline(1, color="k", lw=0.5); ax.set_xlabel("aperture θ [arcmin]"); ax.legend(fontsize=7)
ax.set_title("clustering correction to the Poisson pair probability (q_cut=5)", fontsize=10)
fig.tight_layout(); fig.savefig(os.path.join(FIG, "clustering_correction.png"), dpi=140); plt.close(fig)
print(f"\nfigures: {FIG}/robustness_scaling.png, {FIG}/clustering_correction.png")
json.dump({f"{k[0]}|{k[1]:g}": {kk: list(map(float, vv)) for kk, vv in v.items()} for k, v in TAB.items()},
          open(os.path.join(DD, "blending_fractions_table.json"), "w"), indent=1)
json.dump(rows, open(os.path.join(DD, "B1_rows.json"), "w"), indent=1)
