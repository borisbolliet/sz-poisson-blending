"""Run the map pipeline on all 108 patches with 4 single-threaded worker processes (thread cap 4)."""
import os, sys, time
os.environ.setdefault("OMP_NUM_THREADS", "1"); os.environ.setdefault("NUMBA_NUM_THREADS", "1")
from multiprocessing import Pool
import q2lib as L

def work(pid):
    if os.path.exists(f"{L.DATA}/patch_out/patch_{pid:03d}.npz"):
        return pid, 0.0, 0.0
    return L.process_patch(pid)

if __name__ == "__main__":
    pids = list(range(int(sys.argv[1]), int(sys.argv[2])))
    nproc = int(sys.argv[3]) if len(sys.argv) > 3 else 4
    L.build_tables()
    t0 = time.time()
    with Pool(nproc) as p:
        for pid, tt, tp in p.imap_unordered(work, pids):
            print(f"patch {pid:3d} done: {tt:6.1f}s (paint {tp:5.1f}s)   elapsed {time.time()-t0:7.1f}s", flush=True)
