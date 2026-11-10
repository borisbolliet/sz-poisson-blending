"""Own Poisson sampler from a (z, lnM) grid of full-sky d2N/dz/dlnM (cell-centred, uniform jitter)."""
import numpy as np
import own_model as om


class Sampler:
    def __init__(self, zmin=0.01, zmax=3.0, Mmin=1e13, Mmax=1e16, nz=900, nM=600, model="tinker08"):
        self.ze = np.linspace(zmin, zmax, nz + 1); self.le = np.linspace(np.log(Mmin), np.log(Mmax), nM + 1)
        zc = 0.5 * (self.ze[1:] + self.ze[:-1]); lc = 0.5 * (self.le[1:] + self.le[:-1])
        # Simpson-like: average of d2N at cell centre (good to O(h^2))
        d2N = np.array([om.dndlnM(np.exp(lc), zz, model) for zz in zc]) * (4 * np.pi * om.dVdzdO(zc))[:, None]
        self.mu_full = d2N * np.diff(self.ze)[:, None] * np.diff(self.le)[None, :]
        self.Ntot = self.mu_full.sum()

    def draw(self, rng, fsky):
        n = rng.poisson(self.mu_full * fsky)
        iz, im = np.nonzero(n); cnt = n[iz, im]
        iz = np.repeat(iz, cnt); im = np.repeat(im, cnt)
        z = self.ze[iz] + rng.random(len(iz)) * np.diff(self.ze)[iz]
        lnM = self.le[im] + rng.random(len(im)) * np.diff(self.le)[im]
        return z, np.exp(lnM)
