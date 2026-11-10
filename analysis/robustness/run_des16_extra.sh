#!/bin/bash
# Repair run (after verification): 10 more despali16 skies to resolve the +4 sigma <delta> vs mu offset (claim C4).
export OMP_NUM_THREADS=2 NUMBA_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 VECLIB_MAXIMUM_THREADS=2
PY=/Users/boris/GitHub/sz-poisson-blending/.venv/bin/python
cd /Users/boris/GitHub/sz-poisson-blending/analysis/robustness
for s in 2006 2007 2008 2009 2010 2011 2012 2013 2014 2015; do $PY run_pass.py des16 $s; done
echo ALLDONE
