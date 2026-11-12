"""vc11: which naive model has the right Fisher sigmas in the weakly constrained configs (c, e)?
My CountModel with the HMF taken from DIRECT colossus calls at every evaluated (Om, sigma8) (no grid, no spline; the
colossus Delta_m rounding is then sampled at the evaluated points), for the finite-difference Jacobian at the truth.
Compared with my spline-grid model (vcmodel) and with the build's Fisher sigmas (fits.json, read as data).
Steps: Om, s8 = 0.002 / 0.004 (colossus rounding noise averages over a wider step), other parameters as vc05."""
import os, sys, json
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
os.environ["VC_THREADS"] = "1"
import numpy as np
import vclib as V, vcmodel as VM
from colossus.cosmology import cosmology
from colossus.lss import mass_function as mf

C_KMS = 299792.458


class DirectModel(VM.CountModel):
    def nodes(self, th):
        om, s8, omb, alpha, slnY = th
        c = cosmology.setCosmology("planck18", Om0=float(om), sigma8=float(s8), persistence="")
        h = c.h; M = np.exp(self.LNM)
        chi = c.comovingDistance(0.0, self.ZN) / h
        dV = (C_KMS / c.H0) / c.Ez(self.ZN) * chi ** 2 * 4 * np.pi
        lnN = np.array([np.log(mf.massFunction(M * h, float(z), mdef="500c", model="tinker08", q_out="dndlnM") * h ** 3 * dV[i])
                        for i, z in enumerate(self.ZN)])
        lnE = np.log(c.Ez(self.ZN))[:, None]; lnD = np.log(chi / (1 + self.ZN))[:, None]
        rhoc0 = c.rho_c(0.0) * 1e9 * h ** 2
        lnM = self.LNM[None, :]
        lnY = VM.LOGY0 + (2.0 / 3.0) * lnE + alpha * (np.log(omb) + lnM - np.log(6e14)) - 2 * lnD - 2 * np.log(VM.ARC)
        lnR = (np.log(3.0) + lnM - np.log(4 * np.pi * 500 * rhoc0) - 2 * lnE) / 3.0
        th500 = np.exp(lnR - lnD) / VM.ARC
        return lnN, lnY - np.log(self.smod.sigma_Y(th500))


B = json.load(open(f"{V.ROOT}/data_real/step2/fits.json"))
Bc = {(r["setup"], r["qcut"], r["survey"], r["config"]): r for r in B if r["route"] == "cat" and r["start"] == "truth"}
for filt, setup in (("baseline", "baseline_f1"), ("white", "white")):
    for cfg in ("a", "b", "c", "d", "e"):
        for sv, fs in (("SO40", 0.4),):
            out = []
            for lab, mod, st in (("spline grid", VM.CountModel(filt, 5.0), [0.001, 0.001]),
                                 ("direct colossus, h 0.002", DirectModel(filt, 5.0), [0.002, 0.002]),
                                 ("direct colossus, h 0.004", DirectModel(filt, 5.0), [0.004, 0.004])):
                mu_t = mod.counts(V.FID)
                Fo = VM.Fit(mod, mu_t * fs, cfg, V.FID, fs)
                steps = np.array(st + [0.003, 0.003, 0.003])[Fo.free]
                F, J, mu = Fo.fisher(V.FID[Fo.free], steps)
                s, sS8, g, C = Fo.sigmas(F)
                out.append(f"{lab}: S8 {sS8:.5f} Om {s[0]:.5f}")
            b = Bc[(setup, 5.0, sv, cfg)]
            print(f"[{setup} {sv} {cfg}] build S8 {b['s_S8']:.5f} Om {b['s_Om']:.5f} | " + " | ".join(out), flush=True)
