"""v01: code-reading audit of the build's claims about which SZiFi choices it copies.

SZiFi's source files are opened as TEXT only (nothing from the clone is imported or executed).  For each item the
script prints the relevant SZiFi lines (file:line) and an automatic check that the quoted pattern is present.
The verdict for each item is written in the log after the quotes.
"""
import os, re, sys
import numpy as np

SZ = "/Users/boris/GitHub/szifi"
SRC = {f: open(os.path.join(SZ, f)).read().splitlines() for f in
       ("szifi/params.py", "szifi/spec.py", "szifi/mmf.py", "szifi/maps.py", "szifi/model.py", "szifi/cat.py",
        "test_files/run_szifi_planck.py", "test_files/run_szifi_so_car.py", "surveys/data_planck.py")}
NOK = 0


def show(f, pattern, n_after=0, max_hits=3):
    global NOK
    hits = [i for i, l in enumerate(SRC[f]) if re.search(pattern, l)]
    if not hits:
        print(f"   !! pattern not found in {f}: {pattern}")
        NOK += 1
        return
    for i in hits[:max_hits]:
        for k in range(i, min(i + 1 + n_after, len(SRC[f]))):
            print(f"   {f}:{k + 1}: {SRC[f][k].strip()}")


def item(title):
    print(f"\n--- {title}")


print("SZiFi clone: /Users/boris/GitHub/szifi (git HEAD 73626fb, 2026-08-28).  Files read as text only.")

item("1. Defaults (params.py): q_th 4, q_th_noise 4, mask_radius 3, max_it 1, bin_fac 4, cov_type isotropic, "
     "interp nearest, maxima_lomem; also inpaint True (diffusive, 100), lsep 3000, decouple master, lrange [100,2500], "
     "beam 'real', 2 default scales, float32 maps")
for p in (r'"q_th":', r'"q_th_noise":', r'"mask_radius":', r'"max_it"', r'"powspec_bin_fac":', r'"cov_type":',
          r'"interp_type":', r'"detection_method":', r'"inpaint":', r'"inpaint_type":', r'"n_inpaint":', r'"lsep":',
          r'"decouple_type":', r'"lrange":', r'"beam":', r'"theta_500_vec_arcmin":', r'"map_dtype":',
          r'"min_radius_mask_arcmin":'):
    show("szifi/params.py", p, max_hits=1)

item("2. Covariance bins: width bin_fac*pi/(nx*dx); l0 <= l < lf (custom path); first bin dropped; nearest "
     "interpolation over bin centres with scipy interp1d")
show("szifi/spec.py", r"l0_bins = np.arange", 1, 1)
show("szifi/spec.py", r"indices = np.where\(\(ell_map < bins_high", 0, 1)
show("szifi/spec.py", r"ell_vec_eff = 0.5\*\(bins_low\+bins_high\)", 0, 1)
show("szifi/spec.py", r"ell = ell\[1:\]", 1, 1)
show("szifi/spec.py", r"interpolate.interp1d\(ell,cov_tensor", 0, 1)
w = 4 * np.pi / (1200 * 0.5 / 60. / 180. * np.pi)
print(f"   => bin width for nx = 1200, dx = 0.5': {w:.6f} (build: 72)")

item("3. The spectra are taken of the l-FILTERED noise map (modes outside lrange are zero) and averaged over ALL "
     "modes of a bin, so the two bins that straddle lmin and lmax are diluted (the build averages over in-band modes "
     "only, stated in RESULTS section 0.5)")
show("szifi/mmf.py", r"self.t_noi = maps.filter_tmap\(self.t_noi", 0, 1)
show("szifi/mmf.py", r"self.cspec.get_cross_spec\(self.pix", 3, 1)

item("4. The cluster mask reaches the covariance ONLY through inpainting: get_cross_spec uses mask_point only inside "
     "'if inpaint_flag == True' (correct_for_inpainting); the power-spectrum mask is ps.mask = mask_ps, not the "
     "cluster mask.  So with inpaint=False SZiFi's 'iteration' re-estimates exactly the iteration-0 covariance")
show("szifi/mmf.py", r"self.mask_point = mask_point_0\*mask_cluster", 0, 1)
show("szifi/mmf.py", r'if self.params_szifi\["inpaint"\] == True:', 12, 2)
show("szifi/spec.py", r"if inpaint_flag == True:", 2, 1)
show("szifi/spec.py", r"self.mask = mask", 0, 1)

item("5. correct_for_inpainting: divide by the unmasked fraction only for ell > lsep (default 3000)")
show("szifi/spec.py", r"def correct_for_inpainting", 10, 1)

item("6. 'fsky' decoupling divides the coupled spectrum by f*<m^2>^2/<m^4> of the PS mask (= f^2 for a binary mask)")
show("szifi/spec.py", r'elif decouple_type == "fsky":', 7, 1)

item("7. After inpainting SZiFi blends the inpainted holes back towards the ORIGINAL map (cosine taper in the "
     "distance to the hole edge; edge = original, centre = inpainted).  Not mentioned by the build")
show("szifi/maps.py", r"def circular_cosine_blend_adaptive", 0, 1)
show("szifi/maps.py", r"blend_core_radius = inner_zero_frac", 10, 1)
show("szifi/maps.py", r"radius = np.sqrt\(area / np.pi\)", 0, 1)
show("szifi/maps.py", r"r = distance_transform_edt\(region_box\)", 0, 1)

item("8. Diffusive inpainting: n_inpaint passes over the hole pixels in np.where order, each set to the mean of its "
     "in-map 8 neighbours (in place)")
show("szifi/maps.py", r"def diffusive_inpaint", 14, 1)

item("9. get_cluster_mask: masks detections with q_opt > q_th_noise; radius theta_500 * mask_radius (clipped to "
     "[min, max]); pixels with distance <= r (ps_mask)")
show("szifi/mmf.py", r'if catalogue.catalogue\["q_opt"\]\[j\] > q_th_noise:', 0, 1)
show("szifi/mmf.py", r"masking_radius_arcmin = max\(min_masking_radius_arcmin", 1, 1)
show("szifi/maps.py", r"indices = np.where\(distances <= self.r_source\)", 0, 1)

item("10. S/N maps (get_mmf_q_map): filter = IFFT(C^-1 a T) (no conjugate), y = sum_i fftconvolve(map, filter, "
     "mode='same') [LINEAR, zero padded, not periodic], norm = sum(tem*filter) dx dy, std = 1/sqrt(norm)")
show("szifi/mmf.py", r"filter_fft = get_tem_conv_fft\(pix,tem_fft", 1, 1)
show("szifi/mmf.py", r'y = sg.fftconvolve\(tmap\[:,:,i\],filter\[:,:,i\],mode="same"\)', 0, 1)
show("szifi/mmf.py", r"norm = np.sum\(tem_norm\*filter\)", 0, 1)
show("szifi/mmf.py", r"std = 1./np.sqrt\(norm\)", 0, 1)
ut =open(os.path.join(SZ, "szifi/utils.py")).read().splitlines()
for i, l in enumerate(ut):
    if "def get_inv_cov_conjugate" in l:
        print(f"   szifi/utils.py:{i + 1}: {l.strip()}  /  {ut[i + 2].strip()}")

item("11. Template: real-space Arnaud profile (truncated at 5 R500 in 3D) tabulated on 1000 radii to 5 theta500, "
     "np.interp at pixel-centre distances from the tile centre (index nx/2 - 1/2); Gaussian beam exp(-fwhm^2 "
     "l(l+1)/(16 ln 2)); normalised to the central y")
show("szifi/mmf.py", r"theta_cart = \[\(0.5\*self.pix.nx\)\*self.pix.dx", 1, 1)
show("szifi/model.py", r"theta_vec = np.linspace\(0.,theta_max,1000\)", 1, 1)
show("szifi/model.py", r"p_cal_map = np.interp\(theta_map,theta_vec,p_cal_vec,right=0.\)", 0, 1)
show("szifi/maps.py", r"x_coord = \(np.arange\(0,self.nx\)-self.nx\*0.5\+0.5\)\*self.dx \+ theta_x", 0, 1)
show("szifi/model.py", r"self.R_truncation = 5. #in units of R_500", 0, 1)
show("szifi/maps.py", r"return np.exp\(-\(fwhm_arcmin\*np.pi/180./60.\)\*\*2/\(16.\*np.log\(2.\)\)\*ell\*\(ell\+1.\)\)", 0, 1)
show("szifi/mmf.py", r't_tem = t_tem/nfw.get_y_norm\(self.params\["norm_type"\]\)', 0, 1)

item("12. Peak finding 'maxima_lomem' (n_theta > 3): strict maxima along both map axes with -inf padding; scale axis "
     "argmax of [q_(j-1), q_j, q_(j+1)] == 1 with q_(-1) = 0 (q_j > q_(j-1), q_j >= q_(j+1)); last scale argmax of "
     "[q_(n-2), q_(n-1)] == 1; catalogue order scale then row-major (np.argwhere); detection pixel coordinate "
     "(i + 1/2) dx")
show("szifi/mmf.py", r'detect_peaks_maxima_lomem = \(self.params\["detection_method"\] == "maxima_lomem"\) and \(n_theta > 3\)', 0, 1)
show("szifi/mmf.py", r"self.q_tensor = np.zeros\(\(self.pix.nx,self.pix.ny,3\)", 0, 1)
show("szifi/mmf.py", r"zmax = np.argmax\(self.q_tensor, axis=2\) == 1", 0, 1)
show("szifi/mmf.py", r"zmax = np.argmax\(self.q_tensor\[:,:,1:\], axis=2\) == 1", 0, 1)
show("szifi/mmf.py", r"coords = np.where\(np.logical_and\(mask_peak_finding, \(qmap > q_th\)\)\)", 0, 1)
show("szifi/mmf.py", r"r = np.ones\(\(r_input.shape\[0\]\+2, r_input.shape\[1\]\+2\)\) \* -np.inf", 0, 1)
show("szifi/mmf.py", r"xmax = np.argmax\(\[r\[2:, 1:-1\], r\[:-2, 1:-1\], r\[1:-1, 1:-1\]\], axis=0\) == 2", 1, 1)
show("szifi/maps.py", r"x_coord = \(np.arange\(0,self.nx\)\+0.5\)\*self.dx", 0, 1)

item("13. Merging (cat.merge_detections): 'closest' = anchor is the first remaining entry, keep the highest q with "
     "distance < radius of the anchor, drop that group; 'fof' = sklearn single linkage with distance_threshold = radius, "
     "keep the highest q.  Planck script: q > 5 cut first, then fof at 10'")
show("szifi/cat.py", r"dist = sphere.get_angdist\(catalogue_compare.catalogue\[\"lon\"\]\[0\]", 0, 1)
show("szifi/cat.py", r"indices = np.where\(dist < radius_arcmin/180./60.\*np.pi\)\[0\]", 0, 1)
show("szifi/cat.py", r'index_max = indices\[np.argmax\(catalogue_compare.catalogue\["q_opt"\]\[indices\]\)\]', 0, 1)
show("szifi/cat.py", r"indices_subtract = np.setdiff1d", 0, 1)
show("szifi/cat.py", r'clust = AgglomerativeClustering\(distance_threshold=radius_arcmin', 2, 1)
show("test_files/run_szifi_planck.py", r"q_th_final = 5.", 0, 1)
show("test_files/run_szifi_planck.py", r'radius_arcmin = 10.', 0, 1)
show("test_files/run_szifi_planck.py", r'mode="fof"\)', 0, 1)

item("14. SZiFi's own SO example (run_szifi_so_car.py) does NOT use the defaults the build copies: lsep 12000, "
     "orphics inpainting, cov_type mixed_gaussian, q_th 4.5, min mask radius 1', NaMaster 'master' decoupling")
for p in (r'"lsep": 12000', r'"inpaint_type": "orphics"', r'"cov_type": "mixed_gaussian"', r'"q_th": 4.5',
          r'"min_radius_mask_arcmin": 1.', r'"decouple_type": "master"'):
    show("test_files/run_szifi_so_car.py", p, 0, 1)

item("15. Planck survey file: t_noi = t_obs (the covariance is estimated from the data themselves), nx = 1024 "
     "(even: the 'same'-mode half-pixel offset applies to Planck tiles too)")
show("surveys/data_planck.py", r'self.data\["t_noi"\]\[field_id\] = tmap', 0, 1)
show("surveys/data_planck.py", r"self.nx = 1024", 0, 1)

print(f"\npatterns not found: {NOK}")
