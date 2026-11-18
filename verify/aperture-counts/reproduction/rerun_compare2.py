import os  # portable-paths
_ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", ".."))  # portable-paths
import sys, os, runpy
sys.path.insert(0, (_ROOT + "/analysis/q1_aperture"))
import q1_common
q1_common.FIG = os.path.join(os.path.dirname(os.path.abspath(__file__)), "figs_rerun")
runpy.run_path(sys.argv[1], run_name="__main__")
