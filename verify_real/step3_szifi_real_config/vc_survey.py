"""The config verifier's own 'survey file' for SZiFi (szifi.data.input_data loads the file named in
params_szifi['survey_file'] and instantiates its class input_data_survey; the class name and the dictionary keys are
SZiFi's interface).  Written from scratch for the verifier; it does not import the build's z_survey.py.

params_data (one field per call):
  field_ids     : [fid]
  map_file      : (1200, 1200, 6) float32 .npy in uK, axis 0 = y (pixel iy), axis 1 = x (pixel ix)
  channels      : indices of the map channels to hand to SZiFi (default all six: 27, 39, 93, 145, 225, 280 GHz)
  noise_level   : 'baseline' or 'goal' (white levels, used by SZiFi only for 'orphics' inpainting)
  buffer_arcmin : peak-finding / selection masks are 0 for pixels whose centre lies closer than this to the
                  outermost pixel centres (0 = no buffer); all other masks are ones
  coupling_matrix_name : NaMaster workspace file for decouple_type 'master' (optional)
The experiment object is SZiFi's 'SObaseline_simple' with every attribute SZiFi reads replaced by ours (vc_model):
nu_eff, FWHM, noise_levels, tsz_sed (own SED formula) and n_freqs, for the selected channels in map order.
"""
import os
import sys
import numpy as np
from szifi import expt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import vc_model as M          # noqa: E402


def buffer_mask(n, buffer_arcmin, pix_arcmin=M.PIXA):
    d = np.minimum(np.arange(n), n - 1 - np.arange(n)) * pix_arcmin     # distance of the pixel centre to the edge row
    ok = d >= buffer_arcmin - 1e-9
    return np.outer(ok, ok).astype(np.float64)


class input_data_survey:

    def __init__(self, params_szifi, params_data):
        fid = params_data["field_ids"][0]
        ch = list(params_data.get("channels", range(6)))
        m = np.load(params_data["map_file"])
        assert m.dtype == np.float32 and m.shape == (M.N, M.N, 6), (m.dtype, m.shape)
        tmap = np.ascontiguousarray(m[:, :, ch])
        if np.dtype(params_szifi.get("map_dtype", np.float32)) == np.float64:
            tmap = tmap.astype(np.float64)                  # 'f64' variants only (float32 -> float64 is exact)
        ones = np.ones((M.N, M.N))
        sel = buffer_mask(M.N, float(params_data.get("buffer_arcmin", 0.)))
        D = {"params_data": params_data}
        D["nx"], D["ny"] = {fid: M.N}, {fid: M.N}
        D["dx_arcmin"], D["dy_arcmin"] = {fid: M.PIXA}, {fid: M.PIXA}
        D["t_obs"], D["t_noi"] = {fid: tmap}, {fid: tmap}
        for k in ("mask_map", "mask_ps", "mask_point"):
            D[k] = {fid: ones.copy()}
        for k in ("mask_peak_finding", "mask_peak_finding_no_tile", "mask_select", "mask_select_no_tile"):
            D[k] = {fid: sel.copy()}
        D["mask_select_buffer"] = {fid: 0}
        D["coupling_matrix_name"] = {fid: params_data.get("coupling_matrix_name")}
        e = expt.experiment(experiment_name="SObaseline_simple", params_szifi=params_szifi)
        e.nu_eff = M.FREQS_GHZ[ch] * 1e9
        e.FWHM = M.FWHM[ch].copy()
        e.noise_levels = M.WHITE[params_data["noise_level"]][ch].copy()
        e.tsz_sed = M.tsz_sed_uK(M.FREQS_GHZ[ch])
        e.n_freqs = len(ch)
        D["experiment"] = e
        self.data = D
