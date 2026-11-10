import sys, os, runpy
sys.path.insert(0, "/Users/boris/GitHub/sz-poisson-blending/analysis/q1_aperture")
import q1_common
q1_common.FIG = os.path.join(os.path.dirname(os.path.abspath(__file__)), "figs_rerun")
os.makedirs(q1_common.FIG, exist_ok=True)
q1_common.DATA_OUT_JSON_PATCH = True
# avoid overwriting summary json: patch json.dump into redirect
import json
_dump = json.dump
def dump(obj, fp, *a, **k):
    name = getattr(fp, "name", "")
    if "q1_fullsky_summary" in name:
        fp.close(); fp = open(os.path.join(q1_common.FIG, "summary_rerun.json"), "w")
    return _dump(obj, fp, *a, **k)
json.dump = dump
runpy.run_path(sys.argv[1], run_name="__main__")
