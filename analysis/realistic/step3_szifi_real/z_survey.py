"""Our own SZiFi 'survey file' for step 3a's simulated SO tiles (loaded by szifi.data.input_data from
params_szifi['survey_file']; it must define class input_data_survey).  Written for this project; the dictionary keys
are the ones SZiFi's cluster_finder reads.

One field per call: params_data['field_ids'] = [fid], with
  params_data['map_files'][fid]   : (1200, 1200, 6) float32 .npy in microK (z01_maps.py), axis 0 = y, axis 1 = x;
  params_data['noise_level']      : 'baseline' or 'goal' (white noise levels for the experiment object);
  params_data['instrument_file']  : data_real/step3_szifi_real/instrument.npz (z00_prep.py);
  params_data['buffer_arcmin']    : edge buffer of the peak-finding and selection masks (0 = no buffer);
  params_data['coupling_matrix_name'] (optional): NaMaster workspace file for decouple_type 'master'.
Tiles: 10 x 10 deg, 0.5' pixels, periodic (SZiFi treats them as ordinary flat tiles).  Masks: power spectrum, map
apodisation and point sources = ones (no apodisation, no point sources); peak finding and selection = ones except the
edge buffer (pixels whose centre lies closer than buffer_arcmin to the outermost pixel centres), for both SZiFi's
'field' and 'tile' versions.  Experiment: SZiFi's 'SOgoal_simple' object with nu_eff, FWHM, noise_levels and tsz_sed
replaced by ours (szsim.instrument), n_freqs recomputed.
"""
import numpy as np
from szifi import maps, expt

N_PIX = 1200
DX_ARCMIN = 0.5


def edge_buffer_mask(n, buffer_arcmin, dx_arcmin=DX_ARCMIN):
    """1 where the pixel centre is at least buffer_arcmin from the outermost pixel centres on both axes, else 0."""
    i = np.arange(n)
    d = np.minimum(i, n - 1 - i) * dx_arcmin
    keep = d >= buffer_arcmin - 1e-9
    return (keep[:, None] & keep[None, :]).astype(np.float64)


class input_data_survey:

    def __init__(self, params_szifi=None, params_data=None):
        self.data = {"params_data": params_data}
        keys = ("nx", "ny", "dx_arcmin", "dy_arcmin", "pix", "t_obs", "t_noi", "mask_map", "mask_point", "mask_select",
                "mask_select_no_tile", "mask_select_buffer", "mask_ps", "mask_peak_finding",
                "mask_peak_finding_no_tile", "mask_tile", "coupling_matrix_name")
        for k in keys:
            self.data[k] = {}
        inst = np.load(params_data["instrument_file"])
        mode = params_data["noise_level"]
        buf = float(params_data.get("buffer_arcmin", 0.0))
        for fid in params_data["field_ids"]:
            tmap = np.load(params_data["map_files"][fid])
            if tmap.shape != (N_PIX, N_PIX, 6) or tmap.dtype != np.float32:
                raise ValueError(f"unexpected map {tmap.shape} {tmap.dtype}")
            dx = DX_ARCMIN / 60. / 180. * np.pi
            self.data["nx"][fid] = N_PIX
            self.data["ny"][fid] = N_PIX
            self.data["dx_arcmin"][fid] = DX_ARCMIN
            self.data["dy_arcmin"][fid] = DX_ARCMIN
            self.data["pix"][fid] = maps.pixel(N_PIX, dx)
            self.data["t_obs"][fid] = tmap
            self.data["t_noi"][fid] = tmap            # noise estimated from the data themselves, as SZiFi's surveys do
            ones = np.ones((N_PIX, N_PIX))
            sel = edge_buffer_mask(N_PIX, buf)
            self.data["mask_map"][fid] = ones
            self.data["mask_ps"][fid] = ones
            self.data["mask_point"][fid] = ones
            self.data["mask_tile"][fid] = ones
            self.data["mask_peak_finding_no_tile"][fid] = sel
            self.data["mask_peak_finding"][fid] = sel.copy()
            self.data["mask_select_no_tile"][fid] = sel.copy()
            self.data["mask_select"][fid] = sel.copy()
            self.data["mask_select_buffer"][fid] = 0
            self.data["coupling_matrix_name"][fid] = params_data.get("coupling_matrix_name")
        e = expt.experiment(experiment_name="SOgoal_simple", params_szifi=params_szifi)
        e.nu_eff = np.asarray(inst["nu_eff_Hz"], float)
        e.FWHM = np.asarray(inst["fwhm_arcmin"], float)
        e.noise_levels = np.asarray(inst[f"white_{mode}"], float)
        e.tsz_sed = np.asarray(inst["tsz_sed_uK"], float)
        e.n_freqs = len(e.nu_eff)
        self.data["experiment"] = e
