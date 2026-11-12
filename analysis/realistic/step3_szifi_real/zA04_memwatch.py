"""zA04 (arbiter): every 2 s, the resident memory (RSS, from ps) of the arbiter's running Python scripts (zA*.py in this
folder) and their sum; one line per minute, and at the end the maximum of the sum and when it occurred.  Stops when no
such process has run for 3 minutes or when data_real/step3_szifi_real/arbiter/out/STOP_memwatch exists.
Usage: python3 -I -B zA04_memwatch.py
"""
import os, re, subprocess, time

STOP = "/Users/boris/GitHub/sz-poisson-blending/data_real/step3_szifi_real/arbiter/out/STOP_memwatch"
PAT = re.compile(r"(zA[0-9]{2}[a-z_0-9]*\.py)")
me = os.getpid()
t_start = time.time()
last_seen = time.time()
best = (0.0, "", [])
t_line = 0.0
print(f"zA04_memwatch started {time.strftime('%H:%M:%S')}", flush=True)
while True:
    if os.path.exists(STOP):
        print("STOP file found", flush=True)
        break
    out = subprocess.run(["ps", "-axo", "pid=,rss=,command="], capture_output=True, text=True).stdout.splitlines()
    procs = []
    for l in out:
        parts = l.split(None, 2)
        if len(parts) < 3:
            continue
        pid, rss, cmd = int(parts[0]), int(parts[1]) / 1e6, parts[2]
        if pid == me or "zA04_memwatch" in cmd or "/usr/bin/time" in cmd:
            continue
        m = PAT.search(cmd)
        if m and "python" in cmd.lower():
            procs.append((pid, rss, cmd[m.start():][:60]))
    tot = sum(p[1] for p in procs)
    now = time.strftime("%H:%M:%S")
    if procs:
        last_seen = time.time()
    if tot > best[0]:
        best = (tot, now, [(p[0], round(p[1], 2), p[2]) for p in procs])
    if time.time() - t_line >= 60:
        t_line = time.time()
        print(f"{now}: {len(procs)} processes, sum RSS {tot:.2f} GB "
              + "; ".join(f"{p[2][:40]} {p[1]:.2f}" for p in procs), flush=True)
    if time.time() - last_seen > 180:
        print("no arbiter process for 3 minutes", flush=True)
        break
    time.sleep(2)
print(f"maximum summed RSS {best[0]:.2f} GB at {best[1]}: {best[2]}", flush=True)
print(f"zA04_memwatch stopped {time.strftime('%H:%M:%S')} after {time.time() - t_start:.0f} s", flush=True)
