"""Step 3 / t07: compute used, collected from the logs (each ends with the output of /usr/bin/time -l; pooled scripts
also print the peak resident memory of their workers).  The concurrent memory of a pooled run is bounded by
(number of workers) x (largest worker peak) + the parent."""
import os, re, sys
HERE = os.path.dirname(os.path.abspath(__file__))
LOGS = [("t00_paint_tables", "paint tables (once)", 6), ("t01_check_a", "check (a), decomposition exactness", 1),
        ("t02_noise_only", "check (b): 100 noise-only tiles x 2 levels x 2 mask methods", 2),
        ("t03_injections", "check (c): 270 injection tiles x 2 levels", 2),
        ("t04_tiles", "realistic tiles: 324 x 2 levels, finder + decomposition", 4),
        ("t04_tiles_s12", "realistic tiles 0-39, 12 scales", 4), ("t04_tiles_s20", "realistic tiles 0-39, 20 scales", 4),
        ("t05_analysis_build_original", "science numbers (build's version)", 1),
        ("t05b_scales_build_original", "12 / 15 / 20 scales (build's version)", 1),
        ("t06_figures_build_original", "figures (build's version)", 1),
        ("t08_extra_checks", "tile statistics, isolated-search window", 1)]
# the arbiter's runs (2026-11-12), reported separately
ARB = [("a01_szifi_iteration", "SZiFi's own iteration vs the build's: 30 + 30 + 12 tiles x 2 levels x 4 variants", 4),
       ("a02_injections_hi", "second injection set: 1080 tiles x 2 levels", 5),
       ("t04_tiles_szifi", "realistic tiles 0-99 x 2 levels with SZiFi's iteration (t04 MASK szifi)", 4),
       ("t05_analysis", "science numbers, rerun after the fix (324 tiles)", 1),
       ("t05_analysis_n100", "science numbers, tiles 0-99 (production finder)", 1),
       ("t05_analysis_szifi", "science numbers, tiles 0-99 (SZiFi's iteration)", 1),
       ("a03_selection_check", "isolated search vs credited detection on 360 a02 tiles x 2 levels; N1 calibrations", 4),
       ("t05b_scales", "12 / 15 / 20 scales, rerun", 1), ("a04_checks", "D3, R5, W1, tie-rule tests", 3),
       ("a05_virtual", "virtual clusters on 60 noise-only tiles x 2 levels", 4),
       ("t06_figures", "figures, rerun", 1)]


def table(LOGS, label):
  tot_w = tot_c = 0.0
  print(f"{'script':28s} {'what':80s} {'wall [s]':>9s} {'CPU [s]':>9s} {'proc':>5s} {'parent RSS':>11s} {'worker peak':>12s} {'bound':>8s}")
  for name, what, nproc in LOGS:
      fn = os.path.join(HERE, name + ".log")
      if not os.path.exists(fn):
          print(f"{name:28s} {what:80s} (no log)")
          continue
      s = open(fn).read()
      m = re.search(r"([\d.]+) real\s+([\d.]+) user\s+([\d.]+) sys", s)
      rss = re.search(r"(\d+)\s+maximum resident set size", s)
      if not m:
          print(f"{name:28s} {what:80s} (no timing in log)")
          continue
      wall, user, sysc = map(float, m.groups())
      prss = int(rss.group(1)) / 1e9 if rss else float("nan")
      wk = [float(v) for v in re.findall(r"worker peak RSS ([\d.]+) GB", s)]
      wmax = max(wk) if wk else float("nan")
      bound = (nproc * wmax + 0.1) if wk else prss
      tot_w += wall; tot_c += user + sysc
      print(f"{name:28s} {what:80s} {wall:9.0f} {user + sysc:9.0f} {nproc:5d} {prss:9.2f} GB {wmax:9.2f} GB {bound:6.1f} GB")
  print(f"{'total':28s} {label:80s} {tot_w:9.0f} {tot_c:9.0f}   = {tot_w / 3600:.2f} h wall, {tot_c / 3600:.2f} CPU-hours")


print("Build:")
table(LOGS, "(sequential sum; t02 and t03 ran while t04 ran)")
print("\nArbiter:")
table(ARB, "(sequential sum; some single-process runs overlapped pooled ones)")
