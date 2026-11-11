"""Tables from data_cosmo/r02_fits.json (converged nonlinear fits) plus linear-Fisher comparisons and the
count residuals of a fixed correction recomputed with replib's exact binning.  Log: r03_report.log"""
import os, sys, json
os.environ["NUMBA_NUM_THREADS"] = "1"; os.environ["OMP_NUM_THREADS"] = "1"
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
from scipy import stats
import replib as R
import r02_fits as F

J = json.load(open(f"{R.D}/r02_fits.json"))
res = J["res"]
SIG = {k: np.array(v) for k, v in J["sig"].items()}
FID = F.FID
PN = R.PN


def sel(**kw):
    return [r for r in res if all(r.get(k) == v for k, v in kw.items())]


def fmt(v, w=6, p=2):
    return " ".join(f"{x:+{w}.{p}f}" for x in v)


print("=" * 110)
print("[1] IGNORE blending at the fiducial: data = C_fid x N(fid), model = N(theta).  Multi-start converged fits.")
print("    columns: start, iterations, Newton decrement, chi2_min (Poisson deviance + prior), d(theta), d/sigma")
for c in "ABC":
    for fs in (1.0, 0.4):
        rr = sel(kind="ignore", case=c, fs=fs, mask=None)
        free = F.CASES[c][0]
        print(f"  case {c} fsky={fs}: chi2 at truth (no fit) {rr[0]['chi2_truth']:.2f}; free = {[PN[a] for a in free]}")
        for r in rr:
            print(f"     start {r['start_lab']:8s} it={r['it']:2d} dec={r['dec']:.1e} chi2={r['chi2']:.4f}  d=[{fmt(r['d'],8,4)}]  b/sig=[{fmt(r['b'])}]")
        B = np.array([r["b"] for r in rr]); Dd = np.array([r["d"] for r in rr])
        print(f"     spread over starts: max|db/sig| = {np.abs(B - B[0]).max():.4f}, max |dtheta| = {np.abs(Dd - Dd[0]).max():.1e},"
              f" chi2 spread {np.ptp([r['chi2'] for r in rr]):.1e}")
print("  sqrt(fsky) rescaling test: (b at fsky=0.4, direct) vs 0.632 x (b at fsky=1)")
for c in "ABC":
    b1 = np.array(sel(kind="ignore", case=c, fs=1.0, start_lab="truth", mask=None)[0]["b"])
    b4 = np.array(sel(kind="ignore", case=c, fs=0.4, start_lab="truth", mask=None)[0]["b"])
    print(f"     case {c}: direct [{fmt(b4)}]   0.632 x fs1 [{fmt(np.sqrt(0.4)*b1)}]")
print("  q>50 column excluded (sensitivity to the disputed C(q>50)), fsky=1, start truth:")
for c in "ABC":
    r = sel(kind="ignore", case=c, mask="noq50")[0]
    r0 = sel(kind="ignore", case=c, fs=1.0, start_lab="truth", mask=None)[0]
    print(f"     case {c}: b/sig [{fmt(r['b'])}] (all bins [{fmt(r0['b'])}]); chi2_min {r['chi2']:.2f} (all bins {r0['chi2']:.2f})")
print("  how much of the misfit do the fits absorb, and would it be noticed (Asimov chi2 = noncentrality lambda)?")
for c in "ABC":
    r = sel(kind="ignore", case=c, fs=1.0, start_lab="truth", mask=None)[0]
    for fs in (1.0, 0.4):
        r = sel(kind="ignore", case=c, fs=fs, start_lab="truth", mask=None)[0]
        nf = len(F.CASES[c][0]); npri = len(F.CASES[c][1]); dof = 80 - nf + npri
        crit = stats.chi2.ppf(0.95, dof)
        pdet = stats.ncx2.sf(crit, dof, r["chi2"])
        print(f"     case {c} fsky={fs}: chi2(truth, no fit) {r['chi2_truth']:.1f} -> chi2_min {r['chi2']:.1f} "
              f"(absorbed {1 - r['chi2']/r['chi2_truth']:.0%}); P(goodness-of-fit test fails at 95%) = {pdet:.2f} (dof {dof})")
print("  absolute shifts (fsky=1): ")
for c in "ABC":
    r = sel(kind="ignore", case=c, fs=1.0, start_lab="truth", mask=None)[0]
    d = dict(zip([PN[a] for a in F.CASES[c][0]], r["d"]))
    print(f"     case {c}: " + ", ".join(f"d{k} = {v:+.4f}" for k, v in d.items()))
yb = float(R.G["ybias"][[tuple(np.round(q, 6)) for q in R.G["params"]].index(tuple(np.round(R.FIDP, 6)))])
print(f"  Y-bias of q>5 detections at the fiducial (c02 grid) = {yb:.4f}; equivalent (1-b) shift "
      f"0.8 x ((1+ybias)^(1/1.79) - 1) = {0.8 * ((1 + yb) ** (1 / 1.79) - 1):+.4f}")
# best-constrained combination: eigenvector of the case-C Fisher matrix
SIGF, dNb = F.fisher_sigmas()
for c in "BC":
    Fm = SIGF[(c, 1.0)]["F"]
    w, V = np.linalg.eigh(Fm)
    r = sel(kind="ignore", case=c, fs=1.0, start_lab="truth", mask=None)[0]
    d = np.array(r["d"])
    print(f"  case {c}: bias projected on Fisher eigen-directions (best constrained first): "
          + ", ".join(f"[{fmt(V[:, k], 5, 2)}]: {V[:, k] @ d * np.sqrt(w[k]):+.2f} sig" for k in np.argsort(-w)))

print("\n[2] FIXED fiducial correction: data = C(theta_t) N(theta_t), model = C_fid x N(theta). Converged nonlinear fits.")
print("    bias / sigma, order (sigma8, Om[, 1-b, sigma_lnY]); sigma = Fisher error of that case at that fsky")
print(f"{'truth':30s} {'A fs1':>16s} {'A fs.4':>16s} | {'B fs1':>30s} | {'B fs.4':>30s} | {'C fs1':>30s} | chi2min A/B/C fs1")
names = [n for n, _ in F.truth_points()]
for n in names:
    row = []
    for c, fs in (("A", 1.0), ("A", 0.4), ("B", 1.0), ("B", 0.4), ("C", 1.0)):
        r = sel(kind="fixed", name=n, case=c, fs=fs, start_lab="truth")[0]
        row.append(fmt(r["b"]))
    c2 = [sel(kind="fixed", name=n, case=c, fs=1.0, start_lab="truth")[0]["chi2"] for c in "ABC"]
    print(f"{n:30s} {row[0]:>16s} {row[1]:>16s} | {row[2]:>30s} | {row[3]:>30s} | {row[4]:>30s} | {c2[0]:.1f}/{c2[1]:.1f}/{c2[2]:.1f}")
print("  multi-start check at grid edges (cases B, C, fsky=1): max |b - b(start truth)| over 3 starts")
mx = 0
for n in names:
    if not n.startswith("grid"):
        continue
    for c in "BC":
        rr = sel(kind="fixed", name=n, case=c, fs=1.0)
        B = np.array([r["b"] for r in rr]); mx = max(mx, np.abs(B - B[0]).max())
print(f"     {mx:.4f}")
print("  max |bias/sigma| over Planck +-1 sigma points and over +-3 sigma points (single-parameter offsets + joint):")
for c in "ABC":
    for fs in (1.0, 0.4):
        m1 = max(np.abs(sel(kind="fixed", name=n, case=c, fs=fs)[0]["b"]).max() for n in names if "1 sig_P18" in n)
        m3 = max(np.abs(sel(kind="fixed", name=n, case=c, fs=fs)[0]["b"]).max() for n in names if "3 sig_P18" in n or "(P18)" in n)
        print(f"     case {c} fsky={fs}: +-1 sigma_P18: {m1:.2f};  +-3 sigma_P18: {m3:.2f}")

print("\n[3] linear Fisher bias vs converged nonlinear fit (fsky=1)")
print("    lin-old: c03 method, derivative of the BLENDED model at the fiducial; lin-model: derivative of the fitted model"
      " C_fid x dN at the truth point")
for n, p in F.truth_points():
    if not ("3 sig_P18" in n or n.startswith("grid")):
        continue
    Dd = R.find(p) * R.naive(p)
    Mt = R.C_FID * R.naive(p)
    out = []
    for c in "AB":
        free, pri = F.CASES[c]
        sg = SIG[f"{c}_1.0"]
        Fo = R.fisher_from(dNb[free], R.C_FID * R.naive(FID), free, pri, 1.0)
        m = R.naive(FID) > 1e-3
        bo = np.linalg.solve(Fo, np.einsum("ak,k->a", dNb[free][:, m], ((Dd - Mt) / (R.C_FID * R.naive(FID)))[m]))
        Jm = R.C_FID * R.jac(R.naive, np.array(p), free)
        Fm = R.fisher_from(Jm, Mt, free, pri, 1.0)
        mm = Mt > 1e-3
        bm = np.linalg.solve(Fm, np.einsum("ak,k->a", Jm[:, mm], ((Dd - Mt) / Mt)[mm]))
        nl = np.array(sel(kind="fixed", name=n, case=c, fs=1.0, start_lab="truth")[0]["b"])
        out.append(f"{c}: lin-old [{fmt(bo/sg)}] lin-model [{fmt(bm/sg)}] nonlin [{fmt(nl)}]")
    print(f"  {n:22s} " + "\n  " + " " * 22 + " ".join([out[0]]) + "\n  " + " " * 22 + out[1])

print("\n[4] EMULATOR (quadratic ln C on the 3^4 grid) at the 10 random interior points: converged nonlinear bias/sigma")
for c in "ABC":
    rr = sel(kind="emu", case=c)
    B = np.array([r["b"] for r in rr])
    print(f"  case {c}: max |b| = {np.abs(B).max():.4f}  (per parameter max: {fmt(np.abs(B).max(0), 6, 4)})")
print("\n[5] ITERATED correction: refit with C recomputed (here: emulated) at the first fixed-correction best fit (fsky=1)")
for r in sel(kind="iter"):
    r1 = sel(kind="fixed", name=r["name"], case=r["case"], fs=1.0, start_lab="truth")[0]
    print(f"  {r['name']:22s} case {r['case']}: fixed [{fmt(r1['b'])}] -> iterated [{fmt(r['b'])}]")

print("\n[6] count residuals with exact binning (data C(theta_t) N, model C_fid N or N); Pearson chi2 over 80 bins")


def rs(Nd, Nm, fs):
    Nd, Nm = Nd * fs, Nm * fs
    s = Nm > 0
    return (Nd.sum() - Nm.sum()) / Nm.sum(), (Nd.sum() - Nm.sum()) / np.sqrt(Nm.sum()), (((Nd - Nm) ** 2)[s] / Nm[s]).sum()


print(f"{'truth':30s} {'fixed dN/N':>10s} {'dN/sqrtN':>9s} {'chi2 fs1':>8s} {'chi2 fs.4':>9s} | {'C=1 dN/N':>9s} {'dN/sqrtN':>8s} {'chi2 fs1':>8s} {'fs.4':>6s}")
for n, p in F.truth_points() + [("fiducial", FID)]:
    Nn = R.naive(p); Dd = R.find(p) * Nn
    f1, f4 = rs(Dd, R.C_FID * Nn, 1.0), rs(Dd, R.C_FID * Nn, 0.4)
    n1, n4 = rs(Dd, Nn, 1.0), rs(Dd, Nn, 0.4)
    print(f"{n:30s} {f1[0]:+10.2e} {f1[1]:+9.2f} {f1[2]:8.2f} {f4[2]:9.2f} | {n1[0]:+9.4f} {n1[1]:+8.1f} {n1[2]:8.1f} {n4[2]:6.1f}")
print(f"  N_tot(fid) blended, exact binning, full sky: {(R.C_FID * R.naive(FID)).sum():.0f}")
