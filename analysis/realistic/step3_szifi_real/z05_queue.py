"""z05_queue: a worker that runs SZiFi itself (z_szlib.run_tile) on jobs taken in order from a queue file, claiming each
job atomically (a claim file created with O_EXCL), so that several workers share one queue.  Stops when the queue is
empty or when the file out/STOP_<queue name> exists (checked before each job).  Output per run:
data_real/step3_szifi_real/runs/CFG_MODE_KIND_PID.npz (existing outputs are skipped).
Queue file: one job per line, 'CFG MODE KIND PID BUFFER_ARCMIN'.
Usage: szifi-venv/bin/python -I -B z05_queue.py QUEUE_FILE WORKER_NAME
"""
import os, sys, time
sys.dont_write_bytecode = True
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import z_szlib as Z                                   # noqa: E402
import numpy as np                                    # noqa: E402

qfile, worker = sys.argv[1], sys.argv[2]
qname = os.path.splitext(os.path.basename(qfile))[0]
claims = os.path.join(Z.OUT, "out", "claims", qname)
os.makedirs(claims, exist_ok=True)
stop = os.path.join(Z.OUT, "out", f"STOP_{qname}")
jobs = [l.split() for l in open(qfile) if l.strip() and not l.startswith("#")]
t0 = time.time()
print(f"z05_queue: worker {worker}, queue {qname} ({len(jobs)} jobs); SZiFi {Z.szifi.__file__}; "
      f"MallocLargeCache={os.environ.get('MallocLargeCache', '(default)')}", flush=True)
printed = set()
n_done = 0
for cfg, mode, kind, pid, buf in jobs:
    pid, buf = int(pid), float(buf)
    if os.path.exists(stop):
        print(f"  STOP file found: {stop}", flush=True)
        break
    tag = f"{cfg}_{mode}_{kind}_{pid:03d}"
    try:
        fd = os.open(os.path.join(claims, tag), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        os.write(fd, worker.encode()); os.close(fd)
    except FileExistsError:
        continue
    if os.path.exists(os.path.join(Z.RUND, f"{tag}.npz")):
        print(f"  {tag}: exists, skipped", flush=True)
        continue
    r = Z.run_tile(cfg, mode, kind, pid, buf, save=True, params_fh=None if cfg in printed else sys.stdout)
    printed.add(cfg)
    n_done += 1
    msg = []
    for it in range(r["n_it"]):
        q = r[f"it{it}_q"]
        msg.append(f"it{it} q>4/4.5/5 {(q > 4).sum()}/{(q > 4.5).sum()}/{(q > 5).sum()}")
    if r["n_it"] == 2:
        rr = r["it1_sig"] / r["it0_sig"]
        msg.append(f"sigma it1/it0 at 0.5'/4.45'/15' {rr[0]:.4f}/{rr[9]:.4f}/{rr[14]:.4f}")
    msg.append(f"f {np.round(r['rec_fsky'], 4).tolist()}")
    print(f"  {tag} (buffer {buf}'): {r['t_run']:5.1f} s, peak RSS {r['peak_rss_gb']:.2f} GB; " + "; ".join(msg)
          + f"  [{time.strftime('%H:%M:%S')}, elapsed {time.time() - t0:.0f} s]", flush=True)
print(f"z05_queue {worker} done: {n_done} runs [{time.time() - t0:.0f} s]; process peak RSS {Z.peak_rss_gb():.2f} GB", flush=True)
