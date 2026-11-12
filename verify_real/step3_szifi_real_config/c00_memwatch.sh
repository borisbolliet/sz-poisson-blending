#!/bin/bash
# c00_memwatch: samples every 5 s the summed RSS of this verifier's python processes (command lines containing
# step3_szifi_real_config or c03_run.py) and appends 'time  n_proc  sum_RSS_GB' to the log given as $1.
# Stops when the file out/STOP_memwatch exists.
LOG="$1"
D=/Users/boris/GitHub/sz-poisson-blending/verify_real/step3_szifi_real_config
while [ ! -f "$D/out/STOP_memwatch" ]; do
  ps -axo rss=,command= | grep -E "c03_run.py|c0[0-9]_[a-z_]*\.py" | grep -v grep | awk -v t="$(date +%H:%M:%S)" '{s+=$1; n+=1} END {printf "%s %d %.3f\n", t, n, s/1e6}' >> "$LOG"
  sleep 5
done
