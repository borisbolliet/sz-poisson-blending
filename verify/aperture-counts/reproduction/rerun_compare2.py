import sys, os, runpy
sys.path.insert(0, "/Users/boris/GitHub/sz-poisson-blending/analysis/q1_aperture")
import q1_common
q1_common.FIG = os.path.join(os.path.dirname(os.path.abspath(__file__)), "figs_rerun")
runpy.run_path(sys.argv[1], run_name="__main__")
