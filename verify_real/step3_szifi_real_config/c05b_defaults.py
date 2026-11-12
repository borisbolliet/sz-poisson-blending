"""c05b (claim E2): can SZiFi's own default parameter dictionary drive cluster_finder?  SZiFi's params.py spells the
key 'snr_weigthing'; mmf.cluster_finder.find_clusters reads 'snr_weighting'.  A tiny 16 x 16 field (zeros, ones masks,
SZiFi's own 'SOgoal_simple' experiment) is handed to cluster_finder with (a) the default dictionary unchanged except
the paths SZiFi would open, and (b) the same plus 'snr_weighting' = False; we report the first exception raised.
Usage: /Users/boris/GitHub/szifi-venv/bin/python -I -B c05b_defaults.py
"""
import os, sys, copy, io, contextlib, traceback
sys.dont_write_bytecode = True
HERE = os.path.dirname(os.path.abspath(__file__))
os.environ["NUMBA_CACHE_DIR"] = os.path.join(HERE, "out", "numba_cache")
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "NUMBA_NUM_THREADS"):
    os.environ[_v] = "1"
sys.path.insert(0, "/Users/boris/GitHub/szifi")
import numpy as np                                    # noqa: E402
import szifi                                          # noqa: E402
from szifi import expt                                # noqa: E402

print(f"default dictionary has 'snr_weigthing': {'snr_weigthing' in szifi.params_szifi_default}; "
      f"'snr_weighting': {'snr_weighting' in szifi.params_szifi_default}")


class Fake:
    def __init__(self, p):
        n, fid = 16, 0
        z = np.zeros((n, n, 6), np.float32)
        o = np.ones((n, n))
        self.data = {"params_data": {"field_ids": [fid]}, "experiment": expt.experiment("SOgoal_simple", params_szifi=p),
                     "t_obs": {fid: z}, "t_noi": {fid: z.copy()}, "nx": {fid: n}, "dx_arcmin": {fid: 0.5},
                     "coupling_matrix_name": {fid: None}, "mask_select_buffer": {fid: 0}}
        for k in ("mask_point", "mask_ps", "mask_select", "mask_select_no_tile", "mask_map", "mask_peak_finding_no_tile",
                  "mask_peak_finding"):
            self.data[k] = {fid: o.copy()}


for lab, extra in (("(a) defaults as shipped", {}), ("(b) defaults + snr_weighting False", {"snr_weighting": False})):
    p = copy.deepcopy(szifi.params_szifi_default)
    p.update(extra)
    try:
        with contextlib.redirect_stdout(io.StringIO()):
            cf = szifi.cluster_finder(params_szifi=p, data_file=Fake(p), rank=0)
            cf.find_clusters()
        print(f"{lab}: ran without an exception")
    except Exception as e:                            # report the first exception and where it was raised
        tb = traceback.extract_tb(e.__traceback__)[-1]
        print(f"{lab}: {type(e).__name__} {e!s:.80} at {os.path.relpath(tb.filename, '/Users/boris/GitHub/szifi')}:{tb.lineno}")
