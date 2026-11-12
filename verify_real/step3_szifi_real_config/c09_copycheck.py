"""c09 (test e): is the SZiFi clone unchanged, and was any SZiFi source copied into the build's or the verifier's files?

(1) git -C /Users/boris/GitHub/szifi status --short / HEAD; __pycache__ or .pyc anywhere in the clone outside .git;
    files in the clone modified after the clone's last commit checkout time is not used (git status is the test).
(2) Verbatim-copy search.  SZiFi source = every .py file of the clone (szifi/, surveys/, test_files/, setup.py).
    Targets = every text file (.py, .log, .md, .txt, .sh) of the build (analysis/realistic/step3_szifi_real/) and of this
    verifier folder (top level; out/ holds only SZiFi's own stdout captures, listed separately).
    (a) line test: every SZiFi line with >= 30 non-blank characters after collapsing white space, that is not a
        pure import / comment-free trivial line, searched as a substring of each target file (white space collapsed);
    (b) shingle test: every run of 15 consecutive SZiFi tokens (identifier / number / operator tokens) searched in
        each target's token stream; reported as the longest common token run per target.
Usage: /Users/boris/GitHub/szifi-venv/bin/python -I -B c09_copycheck.py
"""
import os, re, sys, subprocess
sys.dont_write_bytecode = True
HERE = os.path.dirname(os.path.abspath(__file__))
SZIFI = "/Users/boris/GitHub/szifi"
BUILD = "/Users/boris/GitHub/sz-poisson-blending/analysis/realistic/step3_szifi_real"


def sh(cmd):
    return subprocess.run(cmd, capture_output=True, text=True).stdout


st = sh(["git", "-C", SZIFI, "status", "--short"])
print(f"(1) clone HEAD {sh(['git', '-C', SZIFI, 'log', '-1', '--format=%h']).strip()}; 'git status --short' printed "
      f"{len(st)} characters{': ' + st if st else ''}")
untracked_ignored = sh(["git", "-C", SZIFI, "status", "--short", "--ignored"])
print(f"    'git status --short --ignored' printed {len(untracked_ignored)} characters{': ' + untracked_ignored if untracked_ignored else ''}")
pc = []
for root, dirs, files in os.walk(SZIFI):
    if ".git" in root.split(os.sep):
        continue
    pc += [os.path.join(root, d) for d in dirs if d == "__pycache__"] + [os.path.join(root, f) for f in files if f.endswith(".pyc")]
print(f"    __pycache__ / .pyc in the clone: {pc}")

src = []
for root, dirs, files in os.walk(SZIFI):
    if ".git" in root.split(os.sep):
        continue
    src += [os.path.join(root, f) for f in files if f.endswith(".py")]
src.sort()
TOK = re.compile(r"[A-Za-z_][A-Za-z0-9_]*|\d+\.?\d*(?:[eE][-+]?\d+)?|==|!=|<=|>=|\*\*|[-+*/%=<>()\[\]{},.:]")


def collapse(s):
    return re.sub(r"\s+", " ", s).strip()


lines = []
toks_src = []
for f in src:
    txt = open(f, errors="replace").read()
    for ln in txt.splitlines():
        c = collapse(ln)
        if len(c.replace(" ", "")) >= 30 and not c.startswith(("import ", "from ", "#")):
            lines.append((os.path.relpath(f, SZIFI), c))
    toks_src.append(TOK.findall(txt))
print(f"\nSZiFi source: {len(src)} .py files, {len(lines)} candidate lines (>= 30 non-blank characters)")
K = 15
shingles = set()
for tk in toks_src:
    for a in range(len(tk) - K + 1):
        shingles.add(tuple(tk[a:a + K]))
print(f"  {len(shingles)} distinct {K}-token shingles")


def targets(folder, top_only):
    out = []
    for root, dirs, files in os.walk(folder):
        if top_only and root != folder:
            continue
        out += [os.path.join(root, f) for f in files if f.endswith((".py", ".log", ".md", ".txt", ".sh"))]
    return sorted(out)


def longest_run(tk):
    """Longest run of consecutive target tokens all of whose K-shingles occur in SZiFi (lower bound K - 1 + n)."""
    best = cur = 0
    for a in range(len(tk) - K + 1):
        if tuple(tk[a:a + K]) in shingles:
            cur += 1
            best = max(best, cur)
        else:
            cur = 0
    return best + K - 1 if best else 0


for label, folder, top in (("build", BUILD, False), ("verifier", HERE, True),
                           ("verifier out/ (SZiFi stdout captures etc.)", os.path.join(HERE, "out"), False)):
    tf = targets(folder, top)
    n_line_hits, worst = 0, []
    for f in tf:
        txt = collapse(open(f, errors="replace").read())
        hits = [l for l in lines if l[1] in txt]
        n_line_hits += len(hits)
        lr = longest_run(TOK.findall(open(f, errors="replace").read()))
        worst.append((lr, os.path.relpath(f, folder), hits[:3]))
    worst.sort(reverse=True)
    print(f"\n(2) {label}: {len(tf)} text files; SZiFi lines found verbatim: {n_line_hits}")
    for lr, f, h in worst[:6]:
        print(f"    longest common run of SZiFi tokens: {lr:3d} tokens  in {f}" + (f"; line hits: {h}" if h else ""))
