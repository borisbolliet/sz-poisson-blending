"""Full-sky Poisson halo catalogue sampler.

Algorithm (exact sampling of the bilinearly-interpolated d2N/dz/dlnM table):
  1. The (z, lnM) table (cosmo_hmf.HMFTable) defines cells between grid nodes.  The
     expected count in a cell is the integral of the bilinear interpolant
     (= dz dlnM * mean of the 4 corners), so the sum over cells equals the 2D trapezoid
     integral used by HMFTable.N_total.
  2. N_cell ~ Poisson(lambda_cell) independently (=> a Poisson point process in (z, lnM)).
  3. Within a cell, (z, lnM) is drawn exactly from the bilinear density (marginal in z is
     linear, conditional in lnM is linear; inverse CDF in closed form).
  4. Positions are isotropic and independent: RA ~ U[0,360), sin(Dec) ~ U[-1,1].
  5. Y500 = Y500_med(M,z) * exp(sigma_lnY * N(0,1)); q = Y500 / sigma_Y(theta500);
     q_obs = q + N(0,1).
A mass floor Mmin (>= table Mmin) restricts the draw to M500 >= Mmin (the partial lowest
cell is handled exactly by linearly interpolating the table at Mmin).

Output: dict of float32 arrays (ra, dec [deg], z, M500 [Msun], Y500 [arcmin^2, incl. scatter],
theta500 [arcmin], q, q_obs) plus metadata.  Rows are ordered by z cell (not by position).
float32 RA/Dec resolution ~1e-5 deg (~0.1 arcsec) - fine for arcmin-scale apertures.
"""
import numpy as np

from . import cosmo_hmf as ch
from .signal import SZModel

FIELDS = ("ra", "dec", "z", "M500", "Y500", "theta500", "q", "q_obs")


def _lin_inv(u, p0, p1):
    """Inverse CDF on [0,1] for density linear between p0 (at 0) and p1 (at 1), p0,p1 >= 0."""
    return u * (p0 + p1) / (p0 + np.sqrt(p0 * p0 + u * (p1 * p1 - p0 * p0)))


def _cell_grid(table, Mmin, Mmax, zmin, zmax):
    """Restrict the table to [zmin,zmax] x [lnMmin,lnMmax] with exact clipped edge nodes."""
    def clip_axis(x, lo, hi):
        lo = max(lo, x[0]); hi = min(hi, x[-1])
        inner = x[(x > lo) & (x < hi)]
        return np.concatenate([[lo], inner, [hi]])

    zz = clip_axis(table.z, zmin, zmax)
    lm = clip_axis(table.lnM, np.log(Mmin), np.log(Mmax))
    # bilinear interpolation of the table at the (possibly clipped) nodes
    def interp_idx(x, xn):
        i = np.clip(np.searchsorted(xn, x, side="right") - 1, 0, len(xn) - 2)
        f = (x - xn[i]) / (xn[i + 1] - xn[i])
        return i, f
    iz, fz = interp_idx(zz, table.z)
    im, fm = interp_idx(lm, table.lnM)
    D = table.d2N
    d = ((1 - fz)[:, None] * ((1 - fm)[None, :] * D[iz][:, im] + fm[None, :] * D[iz][:, im + 1])
         + fz[:, None] * ((1 - fm)[None, :] * D[iz + 1][:, im] + fm[None, :] * D[iz + 1][:, im + 1]))
    return zz, lm, d


def expected_count(Mmin=1e13, Mmax=1e16, zmin=0.01, zmax=3.0, table=None):
    """Expected number of halos in the sampler's model (identical to the sum of cell lambdas)."""
    table = table or ch.default_table()
    zz, lm, d = _cell_grid(table, Mmin, Mmax, zmin, zmax)
    lam = 0.25 * (d[:-1, :-1] + d[1:, :-1] + d[:-1, 1:] + d[1:, 1:]) * np.diff(zz)[:, None] * np.diff(lm)[None, :]
    return float(lam.sum())


def draw_catalog(seed, Mmin=1e13, Mmax=1e16, zmin=0.01, zmax=3.0, model=None, table=None,
                 fields=FIELDS, chunk=2_000_000, dtype=np.float32):
    """Draw one full-sky Poisson realisation.

    Parameters
    ----------
    seed : int  -- numpy default_rng seed (fully reproducible for fixed table/chunk)
    Mmin, Mmax : M500c range in Msun (h-free); Mmin is the 'floor' option
    zmin, zmax : redshift range
    model : SZModel (default baseline) used for Y500, theta500, q, q_obs
    fields : subset of FIELDS to return (saves memory)
    chunk : max halos processed at once (memory control)
    Returns dict {field: array(dtype)} + 'meta' dict.
    """
    model = model or SZModel()
    table = table or ch.default_table()
    rng = np.random.default_rng(seed)
    zz, lm, d = _cell_grid(table, Mmin, Mmax, zmin, zmax)
    dz = np.diff(zz); dl = np.diff(lm)
    lam = 0.25 * (d[:-1, :-1] + d[1:, :-1] + d[:-1, 1:] + d[1:, 1:]) * dz[:, None] * dl[None, :]
    counts = rng.poisson(lam)
    ntot = int(counts.sum())
    out = {f: np.empty(ntot, dtype=dtype) for f in fields}
    need_sz = any(f in fields for f in ("Y500", "theta500", "q", "q_obs"))

    nz_cells, nm_cells = counts.shape
    flat_counts = counts.ravel()
    cum = np.concatenate([[0], np.cumsum(flat_counts)])
    # iterate over groups of cells so that each group has <= chunk halos
    start_cell = 0
    ncell = flat_counts.size
    while start_cell < ncell:
        end_cell = int(np.searchsorted(cum, cum[start_cell] + chunk, side="right")) - 1
        end_cell = max(end_cell, start_cell + 1)
        end_cell = min(end_cell, ncell)
        cidx = np.repeat(np.arange(start_cell, end_cell), flat_counts[start_cell:end_cell])
        a, b = cum[start_cell], cum[end_cell]
        n = b - a
        if n > 0:
            i = cidx // nm_cells; j = cidx % nm_cells
            f00 = d[i, j]; f10 = d[i + 1, j]; f01 = d[i, j + 1]; f11 = d[i + 1, j + 1]
            # z (x) marginal is linear with end values (f00+f01)/2, (f10+f11)/2
            x = _lin_inv(rng.random(n), f00 + f01, f10 + f11)
            # lnM | x linear between f(x,0) and f(x,1)
            y = _lin_inv(rng.random(n), f00 + (f10 - f00) * x, f01 + (f11 - f01) * x)
            z = zz[i] + x * dz[i]
            M = np.exp(lm[j] + y * dl[j])
            del f00, f10, f01, f11, x, y, i, j
            if "ra" in fields:
                out["ra"][a:b] = 360.0 * rng.random(n)
            else:
                rng.random(n)
            if "dec" in fields:
                out["dec"][a:b] = np.degrees(np.arcsin(2.0 * rng.random(n) - 1.0))
            else:
                rng.random(n)
            eY = rng.standard_normal(n)
            eq = rng.standard_normal(n)
            if "z" in fields:
                out["z"][a:b] = z
            if "M500" in fields:
                out["M500"][a:b] = M
            if need_sz:
                th = model.theta500(M, z)
                Y = model.Y500_med(M, z) * np.exp(model.sigma_lnY * eY)
                q = Y / model.sigma_Y(th)
                if "Y500" in fields:
                    out["Y500"][a:b] = Y
                if "theta500" in fields:
                    out["theta500"][a:b] = th
                if "q" in fields:
                    out["q"][a:b] = q
                if "q_obs" in fields:
                    out["q_obs"][a:b] = q + eq
        start_cell = end_cell
    out["meta"] = dict(seed=seed, Mmin=Mmin, Mmax=Mmax, zmin=zmin, zmax=zmax, N=ntot,
                       N_expected=float(lam.sum()),
                       model={k: getattr(model, k) for k in ("logA", "alpha", "beta_E", "one_minus_b", "M_pivot",
                                                             "sigma_lnY", "h70_factor", "beam_fwhm_arcmin",
                                                             "noise_y_arcmin")},
                       cosmology="planck18", hmf="tinker08 500c")
    return out


def save_catalog(cat, path):
    """Save a catalogue dict to an uncompressed .npz (meta stored as a 0-d object -> use JSON string)."""
    import json
    arrs = {k: v for k, v in cat.items() if k != "meta"}
    np.savez(path, meta=np.array(json.dumps(cat.get("meta", {}))), **arrs)


def load_catalog(path, fields=None, Mmin=None, qmin=None):
    """Load a catalogue .npz; optionally only some fields and/or apply cuts M500>=Mmin, q_obs>=qmin."""
    import json
    with np.load(path) as f:
        keys = [k for k in f.files if k != "meta"]
        fields = list(fields) if fields is not None else keys
        meta = json.loads(str(f["meta"])) if "meta" in f.files else {}
        sel = None
        if Mmin is not None:
            sel = f["M500"] >= Mmin
        if qmin is not None:
            s2 = f["q_obs"] >= qmin
            sel = s2 if sel is None else sel & s2
        out = {}
        for k in fields:
            a = f[k]
            out[k] = a[sel] if sel is not None else a
    out["meta"] = meta
    return out
