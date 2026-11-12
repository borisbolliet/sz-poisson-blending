"""zA07 (arbiter): (1) is the SZiFi clone untouched?  (2) were SZiFi source lines copied into this repository?

(1) 'git -C /Users/boris/GitHub/szifi status --short' (read-only) and a search for __pycache__ folders in the clone.
(2) Every non-blank line of SZiFi's Python files (szifi/*.py, surveys/*.py, test_files/*.py), stripped of leading and
    trailing white space, with at least 30 characters and at least 3 word tokens, is looked up in every text file of
    this repository (.py, .md, .log, .txt, .sh, .json, .yaml, .toml, .cfg; .git, .venv and the data folders' binary
    files are not read).  Printed: the number of such SZiFi lines, the number of repository files read, and every
    repository line that equals one of them (file and line number).
Usage: python3 -I -B zA07_copycheck.py
"""
import os, re, subprocess, time

t0 = time.time()
SZ = "/Users/boris/GitHub/szifi"
REPO = "/Users/boris/GitHub/sz-poisson-blending"
st = subprocess.run(["git", "-C", SZ, "status", "--short"], capture_output=True, text=True)
print(f"(1) git -C {SZ} status --short: exit {st.returncode}, output {st.stdout!r}")
pyc = [os.path.join(d, x) for d, ds, fs in os.walk(SZ) if ".git" not in d for x in ds if x == "__pycache__"]
print(f"    __pycache__ folders in the clone: {len(pyc)} {pyc}")

src = set()
n_files = 0
for sub in ("szifi", "surveys", "test_files"):
    base = os.path.join(SZ, sub)
    for d, ds, fs in os.walk(base):
        for f in fs:
            if f.endswith(".py"):
                n_files += 1
                for line in open(os.path.join(d, f), errors="replace"):
                    s = line.strip()
                    if len(s) >= 30 and len(re.findall(r"[A-Za-z_]\w*", s)) >= 3:
                        src.add(s)
print(f"(2) SZiFi: {n_files} Python files, {len(src)} distinct lines of >= 30 characters and >= 3 words")
EXT = (".py", ".md", ".log", ".txt", ".sh", ".json", ".yaml", ".yml", ".toml", ".cfg")
SKIP = {".git", ".venv", "__pycache__", "node_modules"}
n_read, hits = 0, []
for d, ds, fs in os.walk(REPO):
    ds[:] = [x for x in ds if x not in SKIP]
    for f in fs:
        if not f.endswith(EXT):
            continue
        p = os.path.join(d, f)
        try:
            if os.path.getsize(p) > 50e6:
                continue
            with open(p, errors="replace") as fh:
                n_read += 1
                for k, line in enumerate(fh, 1):
                    if line.strip() in src:
                        hits.append((os.path.relpath(p, REPO), k, line.strip()[:100]))
        except OSError:
            continue
print(f"    repository text files read: {n_read}; lines equal to a SZiFi source line: {len(hits)}")
for h in hits:
    print(f"      {h[0]}:{h[1]}: {h[2]}")
print(f"done [{time.time() - t0:.0f} s]")
