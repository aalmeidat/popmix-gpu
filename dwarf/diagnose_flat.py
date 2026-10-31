"""Is the flat 'halo' a real Sculptor excess or a foreground mismatch?

Counts 'Sculptor-like' stars per deg^2 -- astrometry consistent with the systemic (parallax, PM) at
chi^2 < thr (3 dof) AND colour-magnitude inside the RGB/HB locus of high-probability central members --
in radial annuli of the target and over the whole control field. If the outer target annuli match
the control density, the flat component is absorbing ordinary foreground (a model problem); if they
sit clearly above it, there is a genuine excess (extended Sculptor stars or a foreground gradient).
"""
import json, os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import yaml
from astropy.table import Table
from fit_dwarf import load_field, ell_radius

cfg = yaml.safe_load(open(sys.argv[1]))
base = os.path.dirname(os.path.abspath(sys.argv[1]))
res = os.path.join(base, f"results_{cfg['name']}")
fit = json.load(open(os.path.join(res, "fit.json")))
m1 = fit["models"]["1"]
d, c = cfg["dwarf"], cfg["control"]
tgt = load_field(os.path.join(base, f"{cfg['name']}_target.fits"), (d["ra"], d["dec"]), d["radius_deg"], cfg["cuts"])
ctl = load_field(os.path.join(base, f"{cfg['name']}_control.fits"), (c["ra"], c["dec"]), c["radius_deg"], cfg["cuts"])
mem = Table.read(os.path.join(res, "members.fits"))
s_int = d["sigma_v_kms"] / (4.74 * d["distance_kpc"])
mean = np.array([m1["plx_mas"], m1["pmra"], m1["pmdec"]])


def chi2(f):
    T = f["C"] + np.diag([0, s_int ** 2, s_int ** 2])[None]
    dd = f["A"] - mean[None]
    return np.einsum("ni,nij,nj->n", dd, np.linalg.inv(T), dd)


# CMD locus from central, high-probability members of the 1-component fit
r_t = ell_radius(tgt["xi"], tgt["eta"], m1["ellipticity"], m1["pa_deg"])
core = (np.asarray(mem["p_dwarf_1comp"]) > 0.9) & (r_t < 2 * m1["a_arcmin"] / 60)
H, xe, ye = np.histogram2d(tgt["BR"][core], tgt["G"][core], bins=[50, 60],
                           range=[cfg["cuts"]["color"], [cfg["cuts"]["gmin"], cfg["cuts"]["gmax"]]])
from scipy.ndimage import gaussian_filter
H = gaussian_filter(H, 1.0)
thr_cmd = np.quantile(H[H > 0], 0.5)
locus = H > thr_cmd


def in_locus(f):
    i = np.clip(np.searchsorted(xe, f["BR"]) - 1, 0, len(xe) - 2)
    j = np.clip(np.searchsorted(ye, f["G"]) - 1, 0, len(ye) - 2)
    return locus[i, j]


for thr in (7.8, 11.3):  # 95% and 99% for 3 dof
    lt = (chi2(tgt) < thr) & in_locus(tgt)
    lc = (chi2(ctl) < thr) & in_locus(ctl)
    dens_c = lc.sum() / (np.pi * c["radius_deg"] ** 2)
    print(f"\nchi2 < {thr}:  control field: {lc.sum()} stars -> {dens_c:.2f} per deg^2")
    R = np.hypot(tgt["xi"], tgt["eta"])
    for lo, hi in [(0, 0.5), (0.5, 1.0), (1.0, 1.5), (1.5, 2.0), (2.0, 2.5), (2.5, 3.0), (3.0, 3.5), (3.5, 4.0)]:
        s = (R >= lo) & (R < hi)
        area = np.pi * (hi ** 2 - lo ** 2)
        n = (lt & s).sum()
        print(f"  target {lo:.1f}-{hi:.1f} deg: {n:6d} stars -> {n / area:8.2f} per deg^2"
              f"  ({n / area / dens_c:6.2f} x control)  +/- {np.sqrt(max(n, 1)) / area / dens_c:.2f}")
print("\ncore members used for the CMD locus:", int(core.sum()), "; locus cells:", int(locus.sum()))
