"""c05 (test a): which branches of SZiFi's code (commit 73626fb, read in mmf.py, spec.py, maps.py, model.py, cat.py,
expt.py, data.py) each of the build's configurations takes, from the parameter dictionaries stored with the build's
runs; and the size of the convention differences between the build's tiles and SZiFi's filter.

(1) Branch table per configuration (S0, S1, S2, S3, S3i, S3it0), decided by our own reading of SZiFi's code (no SZiFi
    code is run or copied here; the conditions are paraphrased).
(2) Conventions, quantified with our own filter code (vc_model) and the analytic covariance of the tiles (bins of
    width 72 at their centres):
    (a) beams: SZiFi's template uses exp(-sigma^2 l(l+1)/2); the tiles were made with exp(-sigma^2 l^2/2):
        kappa and sigma with SZiFi's beams vs with the tiles' beams;
    (b) template: SZiFi samples its template at the pixel centres (centre between four pixels) and normalises it to
        the unconvolved central y; kappa = SZiFi's y0 for a true unit-y0 cluster (sigma in y0 units scales alike, so
        q is not biased by kappa);
    (c) l range [100, 20000] inside the grid's l range (max 30547): share of the grid's modes outside it.
Usage: /Users/boris/GitHub/szifi-venv/bin/python -I -B c05_trace.py
"""
import os, sys, json, time
sys.dont_write_bytecode = True
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import numpy as np                                   # noqa: E402
import vc_model as M                                 # noqa: E402

BR = "/Users/boris/GitHub/sz-poisson-blending/data_real/step3_szifi_real/runs"
t0 = time.time()
RUNS = {"S0": "S0_baseline_real_000", "S1": "S1_baseline_real_000", "S2": "S2_baseline_real_000",
        "S3": "smokeS3_baseline_real_000", "S3i": "S3i_baseline_real_000", "S3it0": "S3it0_baseline_real_033"}
print("(1) branches taken (our reading of SZiFi 73626fb), from the build's stored parameter dictionaries")
for cfg, run in RUNS.items():
    with np.load(os.path.join(BR, run + ".npz")) as d:
        p = json.loads(str(d["params_json"]))
        buf = float(d["buffer_arcmin"])
    b = []
    lmax1d = p["powspec_lmax1d"]
    b.append(f"spectra grid: {'556x556 (powspec_lmax1d 10000: maps degraded by Fourier cropping; bins of 72 up to l = 10008)' if lmax1d else '1200x1200 (bins of 72 up to l = 21600; first bin dropped)'}")
    b.append(f"l range {p['lrange']}: maps and templates zeroed where l < {p['lrange'][0]} or l > {p['lrange'][1]}"
             + ("; bins above l = 20016 and grid corners up to l = 30547 hold only round-off" if p["lrange"][1] < 21600 and not lmax1d else ""))
    dec = p["decouple_type"]
    b.append(f"decoupling '{dec}'" + (": NaMaster workspace read from / written to the survey's coupling_matrix_name"
                                      if dec == "master" and p["coupling_matrix_needed"] and not p["compute_coupling_matrix"] else
                                      " (coupled pseudo-spectra of an all-ones mask = binned periodogram)"))
    if p["cov_type"] == "isotropic":
        b.append(f"covariance 'isotropic': each mode takes the binned spectrum of the bin with the nearest centre ({p['interp_type']}, extrapolated)")
    else:
        b.append(f"covariance '{p['cov_type']}': isotropic spectra x Gaussian-smoothed ({p['cov_kernel_shape']} pixels) ratio of each 2D cross-periodogram to them")
    if p["inpaint"]:
        b.append(f"inpainting on: '{p['inpaint_type']}'" + (f" ({p['n_inpaint']} passes)" if p["inpaint_type"] == "diffusive" else " (orphics gapfill + white noise of the experiment's levels in the holes)")
                 + f"; at noise iteration 0 the masks are all ones, so the maps pass unchanged; from iteration 1 the cluster holes are inpainted, "
                 f"cosine-blended to the original map, l-filtered, and the spectra of bins with l > {p['lsep']} divided by the unmasked fraction")
    else:
        b.append("inpainting off: the cluster mask built after iteration 0 never reaches the spectra (only inpainting and the"
                 " l_sep correction read it), so iteration 1 re-estimates the iteration-0 covariance")
    b.append(f"iteration: iterative {p['iterative']}, max_it {p['max_it']}; masked: q > {p['q_th_noise']} detections of the "
             f"('field', 'field') catalogue, discs of radius max({p['min_radius_mask_arcmin']}', min({p['max_radius_mask_arcmin']}, {p['mask_radius']} theta_est))")
    b.append(f"tilemask_mode '{p['tilemask_mode']}', tile_type '{p['tile_type']}': reported catalogue ('tile', 'tile'); with the survey's identical"
             f" field and tile masks (edge buffer {buf}') the masking catalogue equals the reported one")
    b.append(f"template: profile '{'arnaud'}' (z = 0.2, M500 from theta500, cosmology {p['cosmology']}: shape independent of both),"
             f" sampled at pixel centres around index N/2 - 1/2, beam '{p['beam']}' = exp(-sigma^2 l(l+1)/2) per channel FWHM, normalised to"
             f" y at r = 1e-8 R500 (norm_type '{p['norm_type']}'), save_and_load_template {p['save_and_load_template']}")
    b.append(f"filter: mmf_type '{p['mmf_type']}' -> cmmf_type set to 'standard_mmf'; a_matrix = experiment tsz_sed (stored: "
             f"{np.ravel(p['a_matrix']).round(1).tolist()} uK per unit y); S/N by zero-padded 'same' convolution of float32 maps "
             f"(map_dtype {p['map_dtype']}); apod_type '{p['apod_type']}' multiplies the map by mask_map (ones)")
    b.append(f"detection '{p['detection_method']}' with 15 > 3 scales: peaks per scale strictly above their 4 neighbours (0 outside the"
             f" peak-finding mask), q_j > q_(j-1) and q_j >= q_(j+1); sigma_vec[0] never stored (stays 0)")
    b.append(f"get_lonlat {p['get_lonlat']}, integrate_bandpass {p['integrate_bandpass']}, snr_weighting {p.get('snr_weighting')}"
             f" (default dictionary spells it 'snr_weigthing': {p.get('snr_weigthing')}), freqs {p['freqs']}")
    print(f"\n  {cfg} ({run}):")
    for x in b:
        print(f"    - {x}")

print("\n(2) conventions, analytic covariance of the tiles (our own model), SZiFi's bins at their centres")
lf = (np.arange(300) + 1) * 4 * np.pi / (M.N * M.DX)
l0 = np.arange(300) * 4 * np.pi / (M.N * M.DX)
ellc = 0.5 * (l0 + lf)[1:]
SED = M.tsz_sed_uK(M.FREQS_GHZ)
TH = np.geomspace(0.5, 15.0, 15)
for mode in ("baseline", "goal"):
    sp = M.analytic_cov(ellc, mode)
    s_sz, k_sz = M.filter_quantities(ellc, sp, SED, M.FWHM, thetas=TH)
    # SZiFi's filter if its beams were the tiles' l^2 beams: emulate by painting with l(l+1) beams instead (same ratio)
    Bs_save = M.beam_paint
    M.beam_paint = M.beam_szifi
    s_b, k_b = M.filter_quantities(ellc, sp, SED, M.FWHM, thetas=TH)
    M.beam_paint = Bs_save
    print(f"  {mode}: theta [']                      " + " ".join(f"{x:7.3f}" for x in TH))
    print(f"  {mode}: (b) kappa = SZiFi y0 / true y0     " + " ".join(f"{x:7.4f}" for x in k_sz))
    print(f"  {mode}: (a) kappa if sky beams were l(l+1)  " + " ".join(f"{x:7.4f}" for x in k_b))
    print(f"  {mode}: (a) beam-convention effect on y0 and q (ratio - 1): "
          + " ".join(f"{x:+.1e}" for x in k_sz / k_b - 1))
ell = M.szifi_ell()
print(f"  (c) modes with l > 20000: {(ell > 20000).mean():.4f} of the 1200^2 grid; with l > 21600 (beyond the last bin): "
      f"{(ell > 21600).mean():.4f}; with l < 100: {(ell < 100).mean():.2e}")
print(f"\ndone [{time.time() - t0:.0f} s]")
