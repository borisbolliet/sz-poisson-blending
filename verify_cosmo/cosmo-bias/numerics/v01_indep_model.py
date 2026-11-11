"""Independent re-implementation of the naive binned count model (direct colossus each call, Gauss-Legendre in z,
Simpson in lnM, own Y-M / theta500 / selection), used to check
 (1) cosmology propagation in cosmolib.NaiveModel at off-fiducial points (all 5 params),
 (2) independent Fisher + linearised biases (configs a, b, c, d) from the stored Delta vectors (cat, map_clean),
 (3) a full Gauss-Newton ML for config a and b with the independent model.
Only sigma_Y(theta500) is taken from szsim (library, validated earlier).
"""
import os, sys, time
os.environ["OMP_NUM_THREADS"] = "2"
import numpy as np
from scipy.special import ndtr
from scipy.integrate import simpson
sys.path.insert(0, "/Users/boris/GitHub/sz-poisson-blending/analysis/cosmo/bias")
import cosmolib as CL
from szsim import signal as sg
from colossus.cosmology import cosmology
from colossus.lss import mass_function as mf

SZ = sg.SZModel(); SZ.sigma_Y(1.0)
ARC = np.pi / 180 / 60
CKMS = 299792.458
FID = dict(Om=0.3111, s8=0.8102, omb=0.8, alpha=1.79, slnY=0.173)
NAMES = ["Om", "s8", "omb", "alpha", "slnY"]
lnM = np.linspace(np.log(1e13), np.log(1e16), 241)
M = np.exp(lnM)
xg, wg = np.polynomial.hermite_e.hermegauss(60); wg = wg / wg.sum()
gl_x, gl_w = np.polynomial.legendre.leggauss(6)


def zedges(qc):
    return CL.default_bins(qc)


def counts(p, qc=5.0):
    c = cosmology.setCosmology("planck18", Om0=p["Om"], sigma8=p["s8"], persistence="")
    h = c.h
    zed, qed = zedges(qc)
    out = np.zeros((len(zed) - 1, len(qed) - 1))
    for b in range(len(zed) - 1):
        a_, b_ = zed[b], zed[b + 1]
        zs = 0.5 * (b_ - a_) * gl_x + 0.5 * (a_ + b_); ws = 0.5 * (b_ - a_) * gl_w
        for z, w in zip(zs, ws):
            dn = mf.massFunction(M * h, z, mdef="500c", model="tinker08", q_out="dndlnM") * h ** 3
            chi = c.comovingDistance(0.0, z) / h
            E = c.Ez(z)
            dVdz = 4 * np.pi * chi ** 2 * CKMS / (c.H0 * E)
            DA = chi / (1 + z)
            Y = 1e-4 * 10 ** (-0.19) * E ** (2 / 3) * (p["omb"] * M / 6e14) ** p["alpha"] / DA ** 2 / ARC ** 2
            rhoc = c.rho_c(z) * 1e9 * h ** 2
            R = (3 * M / (4 * np.pi * 500 * rhoc)) ** (1 / 3)
            th = R / DA / ARC
            q0 = Y / SZ.sigma_Y(th)
            q = q0[:, None] * np.exp(p["slnY"] * xg)[None, :]
            cdf = np.stack([ndtr(q - e) if np.isfinite(e) else np.zeros_like(q) for e in qed], -1)
            Pb = ((cdf[..., :-1] - cdf[..., 1:]) * wg[None, :, None]).sum(1)      # (nM, nqb)
            out[b] += w * simpson(dn[:, None] * dVdz * Pb, x=lnM, axis=0)
    return out


def pvec(x, free):
    p = dict(FID); p.update({k: v for k, v in zip(free, x)}); return p


STEPS = dict(Om=0.004, s8=0.004, omb=0.008, alpha=0.01, slnY=0.01)


def jac(free, qc, base=None):
    J = []
    for k in free:
        pp = dict(FID if base is None else base); pm = dict(pp)
        pp[k] += STEPS[k]; pm[k] -= STEPS[k]
        J.append((counts(pp, qc) - counts(pm, qc)) / (2 * STEPS[k]))
    return np.stack(J, -1)


if __name__ == "__main__":
    t0 = time.time()
    # ---- (1) cosmology propagation check
    model = CL.NaiveModel(5.0)
    mu_f = counts(FID)
    N0 = model.counts(CL.FID)
    print(f"[{time.time()-t0:.0f}s] fiducial total: indep {mu_f.sum():.1f} vs cosmolib {N0.sum():.1f} rel {mu_f.sum()/N0.sum()-1:+.2e}; "
          f"max per-bin rel (mu>10) {np.abs(mu_f/N0-1)[N0>10].max():.2e}")
    for pt in [dict(Om=0.28, s8=0.84, omb=0.8, alpha=1.79, slnY=0.173),
               dict(Om=0.35, s8=0.77, omb=0.75, alpha=1.70, slnY=0.25),
               dict(Om=0.3111, s8=0.8102, omb=0.86, alpha=1.85, slnY=0.05)]:
        a = counts(pt); th = np.array([pt[k] for k in NAMES]); b = model.counts(th)
        ra = a / mu_f; rb = b / N0      # ratio to fiducial (cancels common integration error)
        print(f"  {pt}: total indep {a.sum():.1f}, cosmolib {b.sum():.1f} (rel {b.sum()/a.sum()-1:+.2e}); "
              f"ratio-to-fid max|diff| over bins (mu>10) {np.abs(rb-ra)[N0>10].max():.2e}; per-z total rel diff "
              f"{np.round(b.sum(1)/a.sum(1)-1, 4)}")
    print(f"[{time.time()-t0:.0f}s]")
    # ---- (2) independent Fisher and linearised bias from stored Delta vectors
    tc = np.load(f"{CL.DC}/truth_catalogue.npz"); tm = np.load(f"{CL.DC}/truth_maps.npz")
    sc = float(tm["fullsky_deg2"] / tm["area_deg2"])
    D = {"cat": tc["D_q5"], "map_clean": (tm["clean5"] - tm["iso5"]).sum(0) * sc, "map": (tm["map5"] - tm["iso5"]).sum(0) * sc}
    Jall = jac(NAMES, 5.0)
    np.savez("/private/tmp/claude-501/-Users-boris-GitHub/0935154e-1cde-4eab-ab64-4515ca671a83/scratchpad/v01_jac.npz", J=Jall, mu=mu_f)
    print(f"[{time.time()-t0:.0f}s] jacobian done")
    cfg = {"a": (["Om", "s8"], {}), "b": (["Om", "s8", "omb"], {}), "c": (["Om", "s8", "omb", "slnY"], {}),
           "d": (NAMES, {"omb": 0.032}), "e": (NAMES, {})}
    S8t = CL.S8_of(FID["Om"], FID["s8"])
    for fsky in (0.4, 1.0):
        for cn, (free, pri) in cfg.items():
            idx = [NAMES.index(k) for k in free]
            J = Jall[..., idx].reshape(-1, len(idx)) * fsky; mu = mu_f.ravel() * fsky
            F = J.T @ (J / mu[:, None])
            for k, s in pri.items():
                F[free.index(k), free.index(k)] += 1 / s ** 2
            C = np.linalg.inv(F); sig = np.sqrt(np.diag(C))
            g = np.zeros(len(idx)); g[0] = 0.5 * S8t / FID["Om"]; g[1] = S8t / FID["s8"]; sS8 = np.sqrt(g @ C @ g)
            line = f"fsky {fsky} cfg {cn}: sigma(Om,s8,S8)=({sig[0]:.4f},{sig[1]:.4f},{sS8:.4f}) | "
            for dn, Dv in D.items():
                dx = C @ (J.T @ (Dv.ravel() * fsky / mu))
                line += f"{dn}: Om {dx[0]/sig[0]:+.2f} s8 {dx[1]/sig[1]:+.2f} S8 {(g@dx)/sS8:+.2f}; "
            print(line)
    print(f"[{time.time()-t0:.0f}s]")
    # ---- (3) Gauss-Newton ML, config a and b, SO40, cat and map_clean, independent model
    for cn in ("a", "b"):
        free = cfg[cn][0]
        for dn in ("cat", "map_clean"):
            fsky = 0.4
            d = (mu_f + D[dn]).ravel() * fsky
            x = np.array([FID[k] for k in free])
            for it in range(3):
                p = pvec(x, free)
                mu = counts(p).ravel() * fsky
                J = jac(free, 5.0, base=p).reshape(-1, len(free)) * fsky
                F = J.T @ (J / mu[:, None]); gr = J.T @ (d / mu - 1)
                dx = np.linalg.solve(F, gr); x = x + dx
                print(f"   GN {cn} {dn} it {it}: x={np.round(x,5)} step(sigma)={np.sqrt(dx@F@dx):.3f}", flush=True)
            # sigma at truth
            idx = [NAMES.index(k) for k in free]
            Jt = Jall[..., idx].reshape(-1, len(idx)) * fsky; mut = mu_f.ravel() * fsky
            C = np.linalg.inv(Jt.T @ (Jt / mut[:, None])); sig = np.sqrt(np.diag(C))
            g = np.zeros(len(idx)); g[0] = 0.5 * S8t / FID["Om"]; g[1] = S8t / FID["s8"]; sS8 = np.sqrt(g @ C @ g)
            dS8 = CL.S8_of(x[0], x[1]) - S8t
            print(f"ML indep cfg {cn} {dn} SO40: Om {(x[0]-FID['Om'])/sig[0]:+.2f} s8 {(x[1]-FID['s8'])/sig[1]:+.2f} "
                  f"S8 {dS8/sS8:+.2f}" + (f" omb {x[2]-0.8:+.4f}" if len(x) > 2 else "") + f"  [{time.time()-t0:.0f}s]", flush=True)
