#!/bin/bash
# runs vc01_sky.py for the given seeds sequentially (3 numba threads), one log per sky
cd "$(dirname "$0")"
for s in "$@"; do
  VC_THREADS=3 /usr/bin/time -l ../../.venv/bin/python vc01_sky.py $s > vc01_sky_$s.log 2>&1
done
