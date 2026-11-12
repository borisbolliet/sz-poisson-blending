"""The verifier's own SZiFi 'survey file' (lens 'indep').  SZiFi's szifi.input_data executes this file and builds
input_data_survey(params_szifi=..., params_data=...); its .data dict is what cluster_finder reads.  Written from
SZiFi's interface (the keys that mmf.cluster_finder.find_clusters reads); no SZiFi code is copied.

One field per run.  params_data must hold, besides 'field_ids':
  'j_map'      : path of a float32 (1200, 1200, 6) map in microK (array axis 0 = the finder's y, axis 1 = its x);
  'j_buffer'   : edge buffer in pixels, left out of the peak-finding and selection masks (16 = 8');
  'j_fwhm', 'j_white', 'j_sed', 'j_nu': beams [arcmin], white levels [microK arcmin], tSZ SED [microK], nu [Hz];
  'j_cm_name'  : coupling-matrix file (only read or written when decouple_type is 'master').
All other masks are ones: no point sources, no apodisation, power spectra over the whole tile.
"""
import numpy as np
from szifi import maps, expt

N_PIX = 1200
DX_ARCMIN = 0.5


class input_data_survey:

    def __init__(self, params_szifi=None, params_data=None):
        self.data = {"params_data": params_data, "nside_tile": 8}
        for key in ("nx", "ny", "dx_arcmin", "dy_arcmin", "pix", "t_obs", "t_noi", "mask_map", "mask_point",
                    "mask_select", "mask_select_no_tile", "mask_select_buffer", "mask_ps", "mask_peak_finding",
                    "mask_peak_finding_no_tile", "mask_tile", "coupling_matrix_name"):
            self.data[key] = {}
        dx = DX_ARCMIN / 60.0 / 180.0 * np.pi
        b = int(params_data["j_buffer"])
        for fid in params_data["field_ids"]:
            tmap = np.load(params_data["j_map"])
            assert tmap.shape == (N_PIX, N_PIX, 6) and tmap.dtype == np.float32
            ones = np.ones((N_PIX, N_PIX))
            inner = np.zeros((N_PIX, N_PIX))
            inner[b:N_PIX - b, b:N_PIX - b] = 1.0
            self.data["nx"][fid] = N_PIX
            self.data["ny"][fid] = N_PIX
            self.data["dx_arcmin"][fid] = DX_ARCMIN
            self.data["dy_arcmin"][fid] = DX_ARCMIN
            self.data["pix"][fid] = maps.pixel(N_PIX, dx)
            self.data["t_obs"][fid] = tmap
            self.data["t_noi"][fid] = tmap
            self.data["mask_map"][fid] = ones.copy()
            self.data["mask_ps"][fid] = ones.copy()
            self.data["mask_point"][fid] = ones.copy()
            self.data["mask_tile"][fid] = ones.copy()
            self.data["mask_peak_finding_no_tile"][fid] = inner.copy()
            self.data["mask_peak_finding"][fid] = inner.copy()
            self.data["mask_select_no_tile"][fid] = inner.copy()
            self.data["mask_select"][fid] = inner.copy()
            self.data["mask_select_buffer"][fid] = 0
            self.data["coupling_matrix_name"][fid] = params_data["j_cm_name"]
        ex = expt.experiment(experiment_name="SOgoal_simple", params_szifi=params_szifi)
        ex.nu_eff = np.asarray(params_data["j_nu"], float)
        ex.FWHM = np.asarray(params_data["j_fwhm"], float)
        ex.noise_levels = np.asarray(params_data["j_white"], float)
        ex.tsz_sed = np.asarray(params_data["j_sed"], float)
        ex.n_freqs = len(ex.nu_eff)
        self.data["experiment"] = ex
