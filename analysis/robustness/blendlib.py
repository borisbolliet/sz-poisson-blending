"""Blending machinery for the robustness study (analysis/robustness).

* RhoTable: tabulated matched-filter cross response rho(theta_A, theta_B, sep) for one beam
  (rho is independent of the white-noise level; it only depends on the beam and the two sizes).
  rho_AB = C_AB(sep) / C_AA(0), C_AB = (1/2pi) int dl l B_l^2 g(l thA) g(l thB) J0(l sep).
  Same definition as szsim.SZModel.cross_response (validated in validate_rho_table.py).
* blend_sums: numba kernel. For every candidate A, sums over all neighbours B (B != A) of
  rho_AB * Y_B (raw, i.e. the matched filter applied to a map whose mean is NOT subtracted),
  the largest single term, the index of the largest contributor, and the part of the sum that
  comes from neighbours that physically overlap A in 3D (comoving separation < (R500A+R500B)(1+z)).
  Neighbour search: dec-strip / RA-cell grid on the sphere, per theta_B bin.

Conventions: angles in arcmin unless stated, Y500 in arcmin^2, sigma_Y in arcmin^2.
"""
import os
import numpy as np
import numba as nb
from scipy.special import j0

from szsim import cosmo_hmf as ch
from szsim import signal as sg

ARCMIN = ch.ARCMIN
DATA = ch.DATA_DIR
TH_GRID = np.geomspace(0.05, 200.0, 112)          # theta500 nodes [arcmin]
SEP_GRID = np.concatenate([np.linspace(0, 12, 241), np.geomspace(12, 1200, 241)[1:]])  # [arcmin]


class RhoTable:
    def __init__(self, beam_fwhm_arcmin, nl=40000, cache=True):
        self.beam = float(beam_fwhm_arcmin)
        fn = os.path.join(DATA, f"rho_table_beam{self.beam:.2f}.npz")
        if cache and os.path.exists(fn):
            d = np.load(fn)
            self.th, self.sep, self.rho, self.I2 = d["th"], d["sep"], d["rho"], d["I2"]
        else:
            self.th, self.sep = TH_GRID, SEP_GRID
            self.rho, self.I2 = self._build(nl)
            if cache:
                np.savez(fn, th=self.th, sep=self.sep, rho=self.rho, I2=self.I2)
        self.lth = np.log(self.th)
        self.rho32 = np.ascontiguousarray(self.rho, dtype=np.float32)

    def _build(self, nl):
        sb = self.beam * ARCMIN / np.sqrt(8 * np.log(2))
        ug, _ = sg.template_hankel()
        lmax = 9.0 / sb
        l = np.linspace(0, lmax, nl)
        wt = np.full(nl, l[1] - l[0]); wt[0] *= 0.5; wt[-1] *= 0.5
        B2 = np.exp(-(l * sb) ** 2)
        G = sg.template_hankel(np.outer(self.th * ARCMIN, l))           # (nth, nl)
        J = j0(np.outer(self.sep * ARCMIN, l))                           # (nsep, nl)
        nth, nsep = len(self.th), len(self.sep)
        C = np.empty((nth, nth, nsep))
        I2raw = np.empty((nth, nth))
        for i in range(nth):
            W = (l * B2 * wt * G[i])[None, :] * G                        # (nth, nl)
            C[i] = W @ J.T / (2 * np.pi)
            # int d^2sep C_AB^2 = (1/2pi) int dl l B^4 gA^2 gB^2  (Parseval)
            I2raw[i] = ((l * B2 * B2 * wt * G[i] ** 2)[None, :] * G ** 2).sum(1) / (2 * np.pi)
        diag = np.array([C[i, i, 0] for i in range(nth)])
        rho = C / diag[:, None, None]
        # I2[i,j] = int d^2sep rho_ij^2  [rad^2]
        I2 = I2raw / diag[:, None] ** 2
        # zero numerical ringing outside the support 5(thA+thB) + 4 FWHM
        TA, TB, S = np.meshgrid(self.th, self.th, self.sep, indexing="ij")
        rho[S > 5 * (TA + TB) + 4 * self.beam] = 0.0
        return rho, I2

    def __call__(self, thA, thB, sep):
        """Vectorised trilinear interpolation (python side, for tests)."""
        thA, thB, sep = np.broadcast_arrays(np.asarray(thA, float), np.asarray(thB, float), np.asarray(sep, float))
        out = np.empty(thA.shape, dtype=np.float64)
        _rho_vec(self.rho32, self.lth, self.sep, np.log(thA).ravel(), np.log(thB).ravel(), sep.ravel(),
                 self.beam, out.ravel())
        return out


@nb.njit(cache=True, inline="always")
def _rho_one(rho, lth, sepg, la, lb, s, beam):
    nth = lth.shape[0]
    ta = np.exp(la); tb = np.exp(lb)
    if s > 5.0 * (ta + tb) + 4.0 * beam:
        return 0.0
    dl = lth[1] - lth[0]
    fa = (la - lth[0]) / dl
    fb = (lb - lth[0]) / dl
    if fa < 0: fa = 0.0
    if fb < 0: fb = 0.0
    if fa > nth - 1.000001: fa = nth - 1.000001
    if fb > nth - 1.000001: fb = nth - 1.000001
    ia = int(fa); ib = int(fb)
    fa -= ia; fb -= ib
    # sep index: binary search
    ns = sepg.shape[0]
    if s >= sepg[ns - 1]:
        return 0.0
    lo = 0; hi = ns - 1
    while hi - lo > 1:
        mid = (lo + hi) >> 1
        if sepg[mid] <= s:
            lo = mid
        else:
            hi = mid
    fs = (s - sepg[lo]) / (sepg[lo + 1] - sepg[lo])
    c00 = rho[ia, ib, lo] * (1 - fs) + rho[ia, ib, lo + 1] * fs
    c01 = rho[ia, ib + 1, lo] * (1 - fs) + rho[ia, ib + 1, lo + 1] * fs
    c10 = rho[ia + 1, ib, lo] * (1 - fs) + rho[ia + 1, ib, lo + 1] * fs
    c11 = rho[ia + 1, ib + 1, lo] * (1 - fs) + rho[ia + 1, ib + 1, lo + 1] * fs
    return (c00 * (1 - fb) + c01 * fb) * (1 - fa) + (c10 * (1 - fb) + c11 * fb) * fa


@nb.njit(cache=True)
def _rho_vec(rho, lth, sepg, la, lb, s, beam, out):
    for i in range(la.shape[0]):
        out[i] = _rho_one(rho, lth, sepg, la[i], lb[i], s[i], beam)


def unitvec(ra_deg, dec_deg):
    ra = np.radians(np.asarray(ra_deg, dtype=np.float64)); dec = np.radians(np.asarray(dec_deg, dtype=np.float64))
    cd = np.cos(dec)
    return cd * np.cos(ra), cd * np.sin(ra), np.sin(dec)


class SkyGrid:
    """Dec-strip / RA-cell grid of points (cell size h_deg)."""

    def __init__(self, ra, dec, h_deg):
        self.h = float(h_deg)
        self.nstrip = int(np.ceil(180.0 / self.h))
        edges_lo = -90.0 + self.h * np.arange(self.nstrip)
        edges_hi = np.minimum(edges_lo + self.h, 90.0)
        maxabs = np.maximum(np.abs(edges_lo), np.abs(edges_hi))
        self.nra = np.maximum(1, np.floor(360.0 * np.cos(np.radians(np.minimum(maxabs, 90.0))) / self.h)).astype(np.int64)
        self.off = np.concatenate([[0], np.cumsum(self.nra)]).astype(np.int64)
        n = len(ra); cell = np.empty(n, np.int32)
        for a in range(0, n, 2_000_000):
            b = min(n, a + 2_000_000)
            s = np.clip(((np.asarray(dec[a:b], float) + 90.0) / self.h).astype(np.int64), 0, self.nstrip - 1)
            w = 360.0 / self.nra[s]
            c = np.minimum((np.mod(np.asarray(ra[a:b], float), 360.0) / w).astype(np.int64), self.nra[s] - 1)
            cell[a:b] = self.off[s] + c
        self.order = np.argsort(cell, kind="stable")
        ncell = int(self.off[-1])
        self.start = np.searchsorted(cell[self.order], np.arange(ncell + 1)).astype(np.int64)


@nb.njit(parallel=True, cache=True)
def _blend_kernel(cra, cdec, cx, cy, cz, clth, cid, cchi, cR,
                  nx, ny, nz, nY, nlth, nid, nchi, nR,
                  start, nra, off, h, nstrip, thBmax,
                  rho, lth, sepg, beam,
                  out_sum, out_max, out_argmax, out_ovl):
    ncand = cra.shape[0]
    for i in nb.prange(ncand):
        tA = np.exp(clth[i])
        rmax = (5.0 * (tA + thBmax) + 4.0 * beam) / 60.0  # deg
        dec = cdec[i]; ra = cra[i]
        s0 = int((dec - rmax + 90.0) / h); s1 = int((dec + rmax + 90.0) / h)
        if s0 < 0: s0 = 0
        if s1 > nstrip - 1: s1 = nstrip - 1
        chord2max = (2.0 * np.sin(np.radians(min(rmax, 180.0)) / 2.0)) ** 2
        acc = 0.0; mx = 0.0; amx = -1; ovl = 0.0
        for s in range(s0, s1 + 1):
            n_ra = nra[s]
            w = 360.0 / n_ra
            full = False
            if abs(dec) + rmax >= 89.0:
                full = True
            else:
                cosd = np.cos(np.radians(abs(dec) + rmax))
                dra = np.degrees(np.arcsin(min(1.0, np.sin(np.radians(rmax)) / cosd))) * 1.0001 + 1e-9
                if 2 * dra >= 360.0 - w:
                    full = True
            if full:
                c0 = 0; c1 = n_ra - 1
            else:
                c0 = int(np.floor((ra - dra) / w)); c1 = int(np.floor((ra + dra) / w))
                if c1 - c0 + 1 >= n_ra:
                    c0 = 0; c1 = n_ra - 1
            for cc in range(c0, c1 + 1):
                c = cc % n_ra
                cell = off[s] + c
                for k in range(start[cell], start[cell + 1]):
                    if nid[k] == cid[i]:
                        continue
                    dx = cx[i] - nx[k]; dy = cy[i] - ny[k]; dz = cz[i] - nz[k]
                    ch2 = dx * dx + dy * dy + dz * dz
                    if ch2 > chord2max:
                        continue
                    sep = 2.0 * np.arcsin(0.5 * np.sqrt(ch2)) * (180.0 / np.pi * 60.0)  # arcmin
                    r = _rho_one(rho, lth, sepg, clth[i], nlth[k], sep, beam)
                    if r == 0.0:
                        continue
                    t = r * nY[k]
                    acc += t
                    if t > mx:
                        mx = t; amx = nid[k]
                    # 3D overlap test (comoving): r_perp = chi * sep, r_par = dchi
                    chim = 0.5 * (cchi[i] + nchi[k])
                    rp = chim * sep * (np.pi / 180.0 / 60.0)
                    rl = cchi[i] - nchi[k]
                    rr = cR[i] + nR[k]
                    if rp * rp + rl * rl < rr * rr:
                        ovl += t
        out_sum[i] += acc
        out_ovl[i] += ovl
        if mx > out_max[i]:
            out_max[i] = mx; out_argmax[i] = amx


TH_BIN_EDGES = np.array([0.0, 0.75, 1.5, 3.0, 6.0, 12.0, 24.0, 48.0, np.inf])


def blend_sums(cand, nbr, rt, cand_id, nbr_id):
    """cand/nbr: dicts with ra, dec [deg], theta500 [arcmin], z; nbr also Y500 [arcmin^2];
    both also 'chi' [Mpc] and 'Rcom' [comoving R500, Mpc] for the overlap test.
    cand_id, nbr_id: integer ids (same id = same halo -> excluded).
    Returns dict(sum, max, argmax, ovl) of raw rho-weighted neighbour Y500 [arcmin^2]."""
    n = len(cand["ra"])
    out_sum = np.zeros(n); out_max = np.zeros(n); out_arg = -np.ones(n, dtype=np.int64); out_ovl = np.zeros(n)
    cx, cy, cz = unitvec(cand["ra"], cand["dec"])
    cra = np.asarray(cand["ra"], np.float64); cdec = np.asarray(cand["dec"], np.float64)
    clth = np.log(np.asarray(cand["theta500"], np.float64))
    cchi = np.asarray(cand["chi"], np.float64); cR = np.asarray(cand["Rcom"], np.float64)
    cid = np.asarray(cand_id, np.int64)
    thA_typ = np.median(cand["theta500"])
    thB_all = np.asarray(nbr["theta500"])
    for b in range(len(TH_BIN_EDGES) - 1):
        sel = np.nonzero((thB_all >= TH_BIN_EDGES[b]) & (thB_all < TH_BIN_EDGES[b + 1]))[0]
        if sel.size == 0:
            continue
        thBmax = float(thB_all[sel].max())
        h = max(3.0, 5 * (thA_typ + thBmax) + 4 * rt.beam) / 60.0  # deg
        h = min(h, 30.0)
        g = SkyGrid(nbr["ra"][sel], nbr["dec"][sel], h)
        o = sel[g.order]
        nx, ny, nz = unitvec(nbr["ra"][o], nbr["dec"][o])
        f32 = np.float32
        _blend_kernel(cra, cdec, cx, cy, cz, clth, cid, cchi, cR,
                      nx.astype(f32), ny.astype(f32), nz.astype(f32),
                      np.asarray(nbr["Y500"][o], f32), np.log(np.asarray(nbr["theta500"][o], f32)),
                      np.asarray(nbr_id[o], np.int64), np.asarray(nbr["chi"][o], f32),
                      np.asarray(nbr["Rcom"][o], f32),
                      g.start, g.nra, g.off, g.h, g.nstrip, thBmax,
                      rt.rho32, rt.lth, rt.sep, rt.beam,
                      out_sum, out_max, out_arg, out_ovl)
        del nx, ny, nz, o, g
    return dict(sum=out_sum, max=out_max, argmax=out_arg, ovl=out_ovl)


def add_distances(cat, model, chunk=2_000_000):
    """Add comoving distance chi [Mpc] and comoving R500 [Mpc] to a catalogue dict (float32, chunked)."""
    n = len(cat["z"])
    cat["chi"] = np.empty(n, np.float32); cat["Rcom"] = np.empty(n, np.float32)
    for a in range(0, n, chunk):
        b = min(n, a + chunk)
        z = cat["z"][a:b].astype(np.float64)
        cat["chi"][a:b] = ch.chi_Mpc(z)
        cat["Rcom"][a:b] = model.R500_Mpc(cat["M500"][a:b].astype(np.float64), z) * (1 + z)
    return cat


def mean_ybar(model, table, Mmin, Mmax=None, zmin=None, zmax=None):
    """Mean Compton y of the halo population (sum of Y5R500 per sr):
    ybar = int d2N/dz/dlnM /(4pi) * <Y500>[sr] * RATIO_5R500, <Y500> = Y_med exp(sigma^2/2)."""
    M = np.exp(table.lnM)
    Ymean = np.array([model.Y500_med(M, zz) for zz in table.z]) * np.exp(0.5 * model.sigma_lnY ** 2) * ARCMIN ** 2
    wM = table._mass_weights(Mmin, Mmax)
    wz = table._weights(table.z, zmin or table.zmin, zmax or table.zmax)
    return float(wz @ (table.d2N * Ymean) @ wM) / (4 * np.pi) * sg.RATIO_5R500


def campbell_moments(model, table, rt, thetaA, Mmin, Mmax=None, zmin=None, zmax=None):
    """Campbell's theorem for the Poisson shot sum dY_A = sum_B rho_AB Y_B at a random point:
    mean = sigma_A^2 RATIO ybar / N_w   (exact: int rho d^2sep = sigma_A^2 RATIO^2/N_w)
    var  = int dN/dOmega <Y_B^2> int rho_AB^2 d^2sep  (I2 table, interpolated in theta_A, theta_B).
    Returns mean, var in arcmin^2 and arcmin^4."""
    from scipy.interpolate import RegularGridInterpolator
    ybar = mean_ybar(model, table, Mmin, Mmax, zmin, zmax)
    sA = model.sigma_Y(thetaA)
    Nw = model.noise_y_arcmin ** 2  # y^2 arcmin^2
    mean = sA ** 2 * sg.RATIO_5R500 * ybar / Nw
    M = np.exp(table.lnM)
    I2i = RegularGridInterpolator((rt.lth, rt.lth), np.log(rt.I2), bounds_error=False, fill_value=None)
    wM = table._mass_weights(Mmin, Mmax)
    wz = table._weights(table.z, zmin or table.zmin, zmax or table.zmax)
    var = np.zeros(np.size(thetaA))
    for iz, zz in enumerate(table.z):
        if wz[iz] == 0:
            continue
        Y2 = (model.Y500_med(M, zz)) ** 2 * np.exp(2 * model.sigma_lnY ** 2)  # arcmin^4
        thB = model.theta500(M, zz)
        for j, tA in enumerate(np.atleast_1d(thetaA)):
            I2 = np.exp(I2i(np.column_stack([np.full_like(thB, np.log(tA)), np.log(thB)]))) / ARCMIN ** 2  # arcmin^2
            var[j] += wz[iz] * np.sum(wM * table.d2N[iz] * Y2 * I2)
    var /= (4 * np.pi / ARCMIN ** 2)  # per arcmin^2
    return mean, var


# ----------------------------------------------------------------------------------------------
# Grouped version: neighbours carry a group label g (mass class x z class); per-candidate outputs
# are per group, so that neighbour mass floors and zmax cuts can be applied after the fact.
# ----------------------------------------------------------------------------------------------
@nb.njit(parallel=True, cache=True)
def _blend_kernel_g(cra, cdec, cx, cy, cz, clth, cid, cchi, cR,
                    nx, ny, nz, nY, nlth, nid, nchi, nR, ngrp,
                    start, nra, off, h, nstrip, thBmax,
                    rho, lth, sepg, beam,
                    out_sum, out_max, out_ovl):
    ncand = cra.shape[0]
    G = out_sum.shape[1]
    for i in nb.prange(ncand):
        acc = np.zeros(G); mx = np.zeros(G); ovl = np.zeros(G)
        tA = np.exp(clth[i])
        rmax = (5.0 * (tA + thBmax) + 4.0 * beam) / 60.0  # deg
        dec = cdec[i]; ra = cra[i]
        s0 = int((dec - rmax + 90.0) / h); s1 = int((dec + rmax + 90.0) / h)
        if s0 < 0: s0 = 0
        if s1 > nstrip - 1: s1 = nstrip - 1
        chord2max = (2.0 * np.sin(np.radians(min(rmax, 180.0)) / 2.0)) ** 2
        for s in range(s0, s1 + 1):
            n_ra = nra[s]
            w = 360.0 / n_ra
            full = False
            dra = 0.0
            if abs(dec) + rmax >= 89.0:
                full = True
            else:
                cosd = np.cos(np.radians(abs(dec) + rmax))
                dra = np.degrees(np.arcsin(min(1.0, np.sin(np.radians(rmax)) / cosd))) * 1.0001 + 1e-9
                if 2 * dra >= 360.0 - w:
                    full = True
            if full:
                c0 = 0; c1 = n_ra - 1
            else:
                c0 = int(np.floor((ra - dra) / w)); c1 = int(np.floor((ra + dra) / w))
                if c1 - c0 + 1 >= n_ra:
                    c0 = 0; c1 = n_ra - 1
            for cc in range(c0, c1 + 1):
                c = cc % n_ra
                cell = off[s] + c
                for k in range(start[cell], start[cell + 1]):
                    if nid[k] == cid[i]:
                        continue
                    dx = cx[i] - nx[k]; dy = cy[i] - ny[k]; dz = cz[i] - nz[k]
                    ch2 = dx * dx + dy * dy + dz * dz
                    if ch2 > chord2max:
                        continue
                    sep = 2.0 * np.arcsin(0.5 * np.sqrt(ch2)) * (180.0 / np.pi * 60.0)  # arcmin
                    r = _rho_one(rho, lth, sepg, clth[i], nlth[k], sep, beam)
                    if r == 0.0:
                        continue
                    t = r * nY[k]
                    g = ngrp[k]
                    acc[g] += t
                    if t > mx[g]:
                        mx[g] = t
                    chim = 0.5 * (cchi[i] + nchi[k])
                    rp = chim * sep * (np.pi / 180.0 / 60.0)
                    rl = cchi[i] - nchi[k]
                    rr = cR[i] + nR[k]
                    if rp * rp + rl * rl < rr * rr:
                        ovl[g] += t
        for g in range(G):
            out_sum[i, g] += acc[g]
            out_ovl[i, g] += ovl[g]
            if mx[g] > out_max[i, g]:
                out_max[i, g] = mx[g]


class NeighbourGrids:
    """Neighbour population split into theta_B bins, each on its own dec-strip/RA-cell grid.
    Memory-lean: one global argsort of (bin, cell) and field-by-field gathering into float32 arrays.
    CONSUMES (deletes) the fields ra, dec, Y500, theta500, chi, Rcom of the dict `nbr`."""

    def __init__(self, nbr, nbr_id, groups, beam, thA_typ=1.0, chunk=2_000_000):
        f32 = np.float32
        th = nbr["theta500"]
        n = len(th)
        tb = (np.digitize(th, TH_BIN_EDGES[1:-1])).astype(np.int8)
        nbin = len(TH_BIN_EDGES) - 1
        params = []
        celloff = 0
        for bi in range(nbin):
            m = tb == bi
            if not m.any():
                params.append(None); continue
            thBmax = float(th[m].max())
            h = min(max(3.0, 5 * (thA_typ + thBmax) + 4 * beam) / 60.0, 30.0)
            nstrip = int(np.ceil(180.0 / h))
            lo = -90.0 + h * np.arange(nstrip); hi = np.minimum(lo + h, 90.0)
            maxabs = np.maximum(np.abs(lo), np.abs(hi))
            nra = np.maximum(1, np.floor(360.0 * np.cos(np.radians(np.minimum(maxabs, 90.0))) / h)).astype(np.int64)
            off = np.concatenate([[0], np.cumsum(nra)]).astype(np.int64)
            params.append(dict(h=h, nstrip=nstrip, nra=nra, off=off, thBmax=thBmax, celloff=celloff))
            celloff += int(off[-1])
        key = np.empty(n, np.int64)
        for a in range(0, n, chunk):
            b = min(n, a + chunk)
            dec = np.asarray(nbr["dec"][a:b], float); ra = np.asarray(nbr["ra"][a:b], float); tbc = tb[a:b]
            kk = np.empty(b - a, np.int64)
            for bi, P in enumerate(params):
                if P is None:
                    continue
                m = tbc == bi
                if not m.any():
                    continue
                s = np.clip(((dec[m] + 90.0) / P["h"]).astype(np.int64), 0, P["nstrip"] - 1)
                w = 360.0 / P["nra"][s]
                c = np.minimum((np.mod(ra[m], 360.0) / w).astype(np.int64), P["nra"][s] - 1)
                kk[m] = P["celloff"] + P["off"][s] + c
            key[a:b] = kk
        del tb
        order = np.argsort(key, kind="stable")
        key = key[order]
        start_all = np.searchsorted(key, np.arange(celloff + 1)).astype(np.int64)
        del key
        # gather fields (deleting the sources as we go)
        self.nY = np.asarray(nbr["Y500"], f32)[order]; del nbr["Y500"]
        self.nlth = np.log(np.asarray(nbr["theta500"], f32)[order]); del nbr["theta500"]
        self.nchi = np.asarray(nbr["chi"], f32)[order]; del nbr["chi"]
        self.nR = np.asarray(nbr["Rcom"], f32)[order]; del nbr["Rcom"]
        self.nid = np.asarray(nbr_id, np.int32)[order]
        self.ngrp = np.asarray(groups, np.int8)[order]
        ras = np.asarray(nbr["ra"], f32)[order]; del nbr["ra"]
        decs = np.asarray(nbr["dec"], f32)[order]; del nbr["dec"]
        del order
        self.nx = np.empty(n, f32); self.ny = np.empty(n, f32); self.nz = np.empty(n, f32)
        for a in range(0, n, chunk):
            b = min(n, a + chunk)
            self.nx[a:b], self.ny[a:b], self.nz[a:b] = unitvec(ras[a:b], decs[a:b])
        del ras, decs
        self.bins = []
        for P in params:
            if P is None:
                continue
            c0 = P["celloff"]; c1 = c0 + int(P["off"][-1])
            st = start_all[c0:c1 + 1]
            i0, i1 = int(st[0]), int(st[-1])
            d = dict(start=st - i0, nra=P["nra"], off=P["off"], h=P["h"], nstrip=P["nstrip"], thBmax=P["thBmax"])
            for k in ("nx", "ny", "nz", "nY", "nlth", "nid", "nchi", "nR", "ngrp"):
                d[k] = getattr(self, k)[i0:i1]
            self.bins.append(d)

    def run(self, cand, cand_id, rt, G, chunk=1_000_000):
        n = len(cand["ra"])
        out_sum = np.zeros((n, G), np.float32); out_max = np.zeros((n, G), np.float32)
        out_ovl = np.zeros((n, G), np.float32)
        for a in range(0, n, chunk):
            b = min(n, a + chunk)
            cx, cy, cz = unitvec(cand["ra"][a:b], cand["dec"][a:b])
            cra = np.asarray(cand["ra"][a:b], np.float64); cdec = np.asarray(cand["dec"][a:b], np.float64)
            clth = np.log(np.asarray(cand["theta500"][a:b], np.float64))
            cchi = np.asarray(cand["chi"][a:b], np.float64); cR = np.asarray(cand["Rcom"][a:b], np.float64)
            cid = np.asarray(cand_id[a:b], np.int32)
            s = np.zeros((b - a, G), np.float32); m = np.zeros((b - a, G), np.float32); o = np.zeros((b - a, G), np.float32)
            for d in self.bins:
                _blend_kernel_g(cra, cdec, cx, cy, cz, clth, cid, cchi, cR,
                                d["nx"], d["ny"], d["nz"], d["nY"], d["nlth"], d["nid"], d["nchi"], d["nR"], d["ngrp"],
                                d["start"], d["nra"], d["off"], d["h"], d["nstrip"], d["thBmax"],
                                rt.rho32, rt.lth, rt.sep, rt.beam, s, m, o)
            out_sum[a:b] = s; out_max[a:b] = m; out_ovl[a:b] = o
        return out_sum, out_max, out_ovl


from scipy.special import ndtr


def ndet_lowq(m, qcuts, table, zmax, qprim=0.5, zs=4, ms=2):
    """[added after verification] expected full-sky detections (q_obs > q_cut) coming from halos with true
    q < qprim. run_pass.py only follows primaries with q > qprim, so the MC 'single' counts must be compared
    with N_det - N_lowq (and Sigma_det for the B1 test must use the same restricted population)."""
    z = table.z[::zs]; lnM = table.lnM[::ms]; M = np.exp(lnM); d2N = table.d2N[::zs, ::ms]
    wz = table._weights(z, table.zmin, zmax); wM = table._weights(lnM, lnM[0], lnM[-1])
    if m.sigma_lnY > 0:
        x = np.linspace(-7, 7, 141); wx = np.exp(-0.5 * x * x); wx /= wx.sum()
    else:
        x = np.zeros(1); wx = np.ones(1)
    out = np.zeros(len(qcuts))
    for i, zz in enumerate(z):
        if wz[i] == 0:
            continue
        qq = m.q_mean(M, zz)[:, None] * np.exp(m.sigma_lnY * x)[None, :]
        low = qq < qprim
        for j, qc in enumerate(qcuts):
            P = ((ndtr(qq - qc) * low) * wx).sum(1)
            out[j] += wz[i] * (d2N[i] * P) @ wM
    return out
