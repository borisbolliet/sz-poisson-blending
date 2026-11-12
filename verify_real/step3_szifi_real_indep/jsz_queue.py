"""Run SZiFi jobs one after the other (one SZiFi process at a time: the verifier's cap is 2 processes and 2 GB).

Jobs are lines 'CFG MODE KIND PID [norelease]' in out/queue.txt; the file is re-read after every job, so jobs can be
appended while the queue runs.  Finished jobs are listed in out/queue_done.txt.  A file out/queue_stop stops the queue
after the current job.  Each job runs as its own process under /usr/bin/time -l with MallocLargeCache=0 (macOS: freed
large blocks are returned to the system, which keeps SZiFi's peak RSS lower).  Each job's own output and the
/usr/bin/time report go to out/stdout/TAG.log; this queue prints one line per job.
Usage: python -I -B jsz_queue.py
"""
import os, sys, time, subprocess, re
HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "out")
PY = "/Users/boris/GitHub/szifi-venv/bin/python"
QF, DF, SF = os.path.join(OUT, "queue.txt"), os.path.join(OUT, "queue_done.txt"), os.path.join(OUT, "queue_stop")


def read(fn):
    if not os.path.exists(fn):
        return []
    with open(fn) as f:
        return [l.strip() for l in f if l.strip() and not l.startswith("#")]


env = dict(os.environ)
env["MallocLargeCache"] = "0"
print(f"jsz_queue: started {time.strftime('%H:%M:%S')}; MallocLargeCache={env['MallocLargeCache']}", flush=True)
while True:
    if os.path.exists(SF):
        print(f"stop file found {time.strftime('%H:%M:%S')}", flush=True)
        break
    done = set(read(DF))
    todo = [j for j in read(QF) if j not in done]
    if not todo:
        print(f"queue empty {time.strftime('%H:%M:%S')}", flush=True)
        break
    job = todo[0]
    a = job.split()
    tag = f"{a[0]}_{a[1]}_{a[2]}_{int(a[3]):03d}" + ("_norelease" if len(a) > 4 else "")
    lf = os.path.join(OUT, "stdout", tag + ".log")
    t0 = time.time()
    with open(lf, "w") as fo:
        rc = subprocess.call(["/usr/bin/time", "-l", PY, "-I", "-B", os.path.join(HERE, "jsz_run.py")] + a,
                             stdout=fo, stderr=subprocess.STDOUT, env=env, cwd=OUT)
    txt = open(lf).read()
    m = re.search(r"(\d+)\s+maximum resident set size", txt)
    rss = int(m.group(1)) / 1e9 if m else float("nan")
    summ = [l for l in txt.splitlines() if l.startswith(tag.replace("_norelease", ""))]
    print(f"{time.strftime('%H:%M:%S')} {tag}: exit {rc}; wall {time.time() - t0:.0f} s; max RSS {rss:.2f} GB"
          + (f" | {summ[-1]}" if summ else ""), flush=True)
    with open(DF, "a") as f:
        f.write(job + "\n")
    if rc != 0:
        print("  (job failed; see " + lf + ")", flush=True)
print(f"jsz_queue: finished {time.strftime('%H:%M:%S')}", flush=True)
