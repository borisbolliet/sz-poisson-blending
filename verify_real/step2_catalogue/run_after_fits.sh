#!/bin/bash
# waits for the fits (out/fits_mine.json), then: vc10 random points (3 threads), then the S8-truth skies (3 threads)
cd /Users/boris/GitHub/sz-poisson-blending/verify_real/step2_catalogue
until [ -f out/fits_mine.json ]; do sleep 20; done
VC_THREADS=3 /usr/bin/time -l ../../.venv/bin/python vc10_randpts.py > vc10_randpts.log 2>&1
for s8 in 0.7365 0.8543; do
  for k in 1 2 3; do
    seed=$([ "$s8" = "0.7365" ] && echo $((2000+k)) || echo $((3000+k)))
    VC_THREADS=3 /usr/bin/time -l ../../.venv/bin/python vc01_sky.py $seed prod $s8 > vc01_skyS8_${s8}_$seed.log 2>&1
  done
done
