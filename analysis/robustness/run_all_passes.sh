#!/bin/bash
# Runs every sensitivity pass sequentially (2 threads), one process per sky.
export OMP_NUM_THREADS=2 NUMBA_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 VECLIB_MAXIMUM_THREADS=2
PY=/Users/boris/GitHub/sz-poisson-blending/.venv/bin/python
cd /Users/boris/GitHub/sz-poisson-blending/analysis/robustness
for s in 2002 2003 2004 2005 2006 2007 2008; do $PY run_pass.py fam $s; done
for p in beam5 planck scat0 scat03 boc16 des16; do
  for s in 2001 2002 2003 2004 2005; do $PY run_pass.py $p $s; done
done
echo ALLDONE
