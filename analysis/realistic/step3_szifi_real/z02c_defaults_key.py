"""z02c: does SZiFi run with its own default dictionary's spelling of the SNR-weighting key?  SZiFi's
params_szifi_default holds 'snr_weigthing'; cluster_finder.find_clusters reads 'snr_weighting'.  Runs S1's parameters
without our 'snr_weighting' entry on one tile and reports what happens (it fails at once, before any filtering, if the
key is missing).
Usage: szifi-venv/bin/python -I -B z02c_defaults_key.py
"""
import os, sys, io, contextlib, traceback
sys.dont_write_bytecode = True
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import z_szlib as Z                                   # noqa: E402

p = Z.make_params("S1")
del p["snr_weighting"]
print(f"'snr_weigthing' in SZiFi's defaults: {'snr_weigthing' in Z.szifi.params_szifi_default}; "
      f"'snr_weighting' in SZiFi's defaults: {'snr_weighting' in Z.szifi.params_szifi_default}")
fid = "baseline_real_000"
pdata = {"data_set": "test", "field_ids": [fid], "map_files": {fid: os.path.join(Z.MAPD, f"{fid}.npy")},
         "noise_level": "baseline", "instrument_file": Z.INSTRUMENT, "buffer_arcmin": 8.0}
try:
    with contextlib.redirect_stdout(io.StringIO()):
        data = Z.szifi.input_data(params_szifi=p, params_data=pdata)
        cf = Z.szifi.cluster_finder(params_szifi=p, params_model=Z.params_model(), data_file=data, rank=0)
        cf.find_clusters()
    print("cluster_finder ran without 'snr_weighting'")
except Exception as e:
    tb = traceback.extract_tb(e.__traceback__)[-1]
    print(f"cluster_finder.find_clusters fails without 'snr_weighting': {type(e).__name__}: {e} "
          f"(raised in {os.path.basename(tb.filename)}, line {tb.lineno}, function {tb.name})")
