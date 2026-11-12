"""z05_memwatch: every MEMWATCH_SLEEP seconds (default 10), the resident memory (RSS, from ps) of this build's running scripts (z0*/z1* in this folder,
SZiFi and project venvs), their sum, and the system's free / compressed memory (vm_stat).  Prints a line per minute
and, at the end, the maximum of the sum.  Stops when no such process has run for 2 minutes, or when the file
data_real/step3_szifi_real/out/STOP_memwatch exists.
Usage: python3 -I -B z05_memwatch.py
"""
import os, re, subprocess, time

HERE = os.path.dirname(os.path.abspath(__file__))
SLEEP = float(os.environ.get("MEMWATCH_SLEEP", "10"))
STOP = "/Users/boris/GitHub/sz-poisson-blending/data_real/step3_szifi_real/out/STOP_memwatch"
PAT = re.compile(r"step3_szifi_real|z0[0-9][a-z_0-9]*\.py|z1[0-9][a-z_0-9]*\.py")
me = os.getpid()


def sample():
    out = subprocess.run(["ps", "-axo", "pid=,rss=,command="], capture_output=True, text=True).stdout.splitlines()
    procs = []
    for l in out:
        parts = l.split(None, 2)
        if len(parts) < 3:
            continue
        pid, rss, cmd = int(parts[0]), int(parts[1]) / 1e6, parts[2]
        if pid == me or "z05_memwatch" in cmd or "/usr/bin/time" in cmd:
            continue
        if "python" in cmd.lower() and PAT.search(cmd):
            m = re.search(r"(z[0-9]{2}[a-z_0-9]*\.py)(.*)", cmd)
            procs.append((pid, rss, (m.group(1) + m.group(2)[:40]) if m else cmd[-60:]))
    vm = subprocess.run(["vm_stat"], capture_output=True, text=True).stdout
    pg = 16384
    get = lambda k: int(re.search(rf"{k}:\s+(\d+)", vm).group(1)) * pg / 1e9 if re.search(rf"{k}:\s+(\d+)", vm) else float("nan")
    return procs, get("Pages free"), get("Pages occupied by compressor")


t0 = time.time()
mx = (0.0, "")
last_seen = time.time()
last_print = 0.0
print("z05_memwatch: summed RSS of this build's processes (GB); system free and compressor memory (GB)", flush=True)
while not os.path.exists(STOP):
    procs, free, comp = sample()
    tot = sum(p[1] for p in procs)
    if procs:
        last_seen = time.time()
    if tot > mx[0]:
        mx = (tot, time.strftime("%H:%M:%S") + " " + "; ".join(f"{p[2]} {p[1]:.2f}" for p in procs))
    if time.time() - last_print >= 60:
        print(f"  {time.strftime('%H:%M:%S')} n={len(procs)} sum RSS {tot:.2f} GB; free {free:.2f}, compressor {comp:.2f}; "
              + "; ".join(f"{p[2].split()[0]}{(' ' + p[2].split()[1].split('/')[-1]) if len(p[2].split()) > 1 else ''} {p[1]:.2f}" for p in procs), flush=True)
        last_print = time.time()
    if time.time() - last_seen > 120:
        break
    time.sleep(SLEEP)
print(f"maximum summed RSS {mx[0]:.2f} GB at {mx[1]}", flush=True)
print(f"done [{(time.time() - t0) / 3600:.2f} h]", flush=True)
