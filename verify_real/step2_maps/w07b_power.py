"""w07b: goodness-of-fit power of the map-route ML fits (w04_fits.json): maxima over the build's own MMF setups
(fdom 0.5/1/2, map-sigma i/ii; baseline, goal; q_cut 5) and the analyst variants (sigma_eff = plain std, clipped std,
robust half-width measured on the simulated maps, w03).  The central Delta_map still carries its bootstrap noise, so
chi2_min and power are upper bounds of the Asimov values."""
import os, json
HERE = os.path.dirname(os.path.abspath(__file__))
fits = json.load(open(os.path.join(HERE, "out", "w04_fits.json")))
out = open(os.path.join(HERE, "w07b_power.log"), "w")
def log(s):
    print(s); out.write(s + "\n")
build_setups = {'baseline_f1', 'baseline_f0.5', 'baseline_f2', 'goal_f1', 'goal_f0.5', 'goal_f2', 'baseline_effi',
                'baseline_effii', 'goal_effi', 'goal_effii'}
pw = {}
for r in fits:
    if r['setup'] in build_setups and r['qcut'] == 5.0:
        k = (r['survey'], 'a' if r['config'] == 'a' else 'b-e')
        if r['power'] > pw.get(k, (0, ''))[0]:
            pw[k] = (r['power'], r['setup'] + ' ' + r['config'])
log("[w07b] max goodness-of-fit power of the map-route ML fits over the build's MMF setups (q_cut 5):")
for k, v in sorted(pw.items()):
    log(f"   {k[0]:4s} config {k[1]:3s}: {v[0]:.3f} ({v[1]})")
for r in fits:
    if r['setup'].endswith(('_std', '_clip', '_rob')) and r['qcut'] == 5.0:
        log(f"   analyst variant {r['setup']:14s} {r['survey']:4s} {r['config']}: S8 {r['b_S8']:+.2f} Om {r['b_Om']:+.2f} "
            f"s8 {r['b_s8']:+.2f} chi2min {r['dev']:.1f} power {r['power']:.3f}")
out.close()
