"""zA03 (arbiter): a worker that runs SZiFi itself (zA_lib.run_variant) on jobs taken in order from a queue file,
claiming each job atomically (O_EXCL claim files), so that two workers share one queue.  Stops when the queue is empty
or when data_real/step3_szifi_real/arbiter/out/STOP_<queue name> exists (checked before each job).
Queue file: one job per line, 'VARIANT MODE KIND PID'.  Output: data_real/step3_szifi_real/arbiter/runs/.
Usage: MallocLargeCache=0 szifi-venv/bin/python -I -B zA03_queue.py QUEUE_FILE WORKER_NAME
"""
import os, sys, time
sys.dont_write_bytecode = True
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import zA_lib as A                                    # noqa: E402
import numpy as np                                    # noqa: E402

qfile, worker = sys.argv[1], sys.argv[2]
qname = os.path.splitext(os.path.basename(qfile))[0]
claims = os.path.join(A.AOUT, "out", "claims", qname)
os.makedirs(claims, exist_ok=True)
stop = os.path.join(A.AOUT, "out", f"STOP_{qname}")
jobs = [l.split() for l in open(qfile) if l.strip() and not l.startswith("#")]
t0 = time.time()
print(f"zA03_queue: worker {worker}, queue {qname} ({len(jobs)} jobs); SZiFi {A.Z.szifi.__file__}; "
      f"MallocLargeCache={os.environ.get('MallocLargeCache', '(default)')}", flush=True)
n_done = 0
for var, mode, kind, pid in jobs:
    pid = int(pid)
    if os.path.exists(stop):
        print(f"  STOP file found: {stop}", flush=True)
        break
    tag = f"{var}_{mode}_{kind}_{pid:03d}"
    try:
        fd = os.open(os.path.join(claims, tag), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        os.write(fd, worker.encode()); os.close(fd)
    except FileExistsError:
        continue
    if os.path.exists(os.path.join(A.Z.RUND, f"{tag}.npz")):
        print(f"  {tag}: exists, skipped", flush=True)
        continue
    r = A.run_variant(var, mode, kind, pid)
    n_done += 1
    msg = []
    for it in range(r["n_it"]):
        q = r[f"it{it}_q"]
        msg.append(f"it{it} q>4/5 {(q > 4).sum()}/{(q > 5).sum()}")
    if r["n_it"] == 2:
        rr = r["it1_sig"] / r["it0_sig"]
        msg.append(f"sigma it1/it0 at 0.5'/4.45'/15' {rr[0]:.4f}/{rr[9]:.4f}/{rr[14]:.4f}")
    msg.append(f"f {np.round(r['rec_fsky'], 4).tolist()}, edge cover {r['edge_cov']:.3f}, n_oob {r['n_oob']}, "
               f"periodic calls {r['n_periodic_inpaint']}/{r['n_periodic_blend']}")
    print(f"  {tag}: {r['t_run']:5.1f} s, peak RSS {r['peak_rss_gb']:.2f} GB; " + "; ".join(msg)
          + f"  [{time.strftime('%H:%M:%S')}, elapsed {time.time() - t0:.0f} s]", flush=True)
print(f"zA03_queue {worker} done: {n_done} runs [{time.time() - t0:.0f} s]; process peak RSS {A.Z.peak_rss_gb():.2f} GB",
      flush=True)
