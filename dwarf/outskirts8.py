"""Model-independent outskirts test on the wide cone: density of 'dwarf-like' stars in annuli and
quadrants, relative to the outer annulus (fit_radius..radius), plus RR Lyrae at the dwarf's distance.
dwarf-like = astrometry within chi2 < 7.8 (3 dof) of the systemic (parallax, PM) AND inside the CMD
locus of core members (r < 0.25 deg, same astrometric cut). Uses the same cuts/masks as fit_dwarf2."""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, yaml
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from astropy.table import Table
from astropy.coordinates import SkyCoord
import astropy.units as u
from scipy.ndimage import gaussian_filter
from fit_dwarf2 import load_field

cfg = yaml.safe_load(open(sys.argv[1])); base = os.path.dirname(os.path.abspath(sys.argv[1]))
out = os.path.join(base, f"results_{cfg['name']}"); os.makedirs(out, exist_ok=True)
d, fg = cfg["dwarf"], cfg["foreground"]
path = os.path.join(base, f"{cfg['name']}_target.fits")
T = load_field(path, (d["ra"], d["dec"]), 0.0, d["radius_deg"], cfg["cuts"], cfg["model"].get("logg_column", "x"))
s_int = d["sigma_v_kms"] / (4.74 * d["distance_kpc"])
mean = np.array([1 / d["distance_kpc"], d["pmra"], d["pmdec"]])
Tm = T["C"] + np.diag([0, s_int ** 2, s_int ** 2])[None]
dd = T["A"] - mean[None]
chi2 = np.einsum("ni,nij,nj->n", dd, np.linalg.inv(Tm), dd)
R = np.hypot(T["xi"], T["eta"])
core = (R < 0.25) & (chi2 < 7.8)
cr, gr = cfg["cuts"]["color"], [cfg["cuts"]["gmin"], cfg["cuts"]["gmax"]]
H, xe, ye = np.histogram2d(T["BR"][core], T["G"][core], bins=[50, 60], range=[cr, gr])
H = gaussian_filter(H, 1.0); locus = H > np.quantile(H[H > 0], 0.5)
i = np.clip(np.searchsorted(xe, T["BR"]) - 1, 0, 49); j = np.clip(np.searchsorted(ye, T["G"]) - 1, 0, 59)
like = (chi2 < 7.8) & locus[i, j]

# area per annulus/quadrant from uniform points, honoring masks
rng = np.random.default_rng(0); n = 2_000_000
rr = d["radius_deg"] * np.sqrt(rng.random(n)); ph = 2 * np.pi * rng.random(n)
mx, my = rr * np.cos(ph), rr * np.sin(ph); keep = np.ones(n, bool)
c0 = SkyCoord(d["ra"] * u.deg, d["dec"] * u.deg)
for (ra, de, rad) in cfg["cuts"].get("masks", []) or []:
    dl, db = c0.spherical_offsets_to(SkyCoord(ra * u.deg, de * u.deg)); keep &= np.hypot(mx - dl.deg, my - db.deg) > rad
A_tot = np.pi * d["radius_deg"] ** 2
area = lambda sel: sel[keep].sum() / n * A_tot
mR = np.hypot(mx, my)
ref_sel = (R >= fg["fit_radius_deg"])
ref = like[ref_sel].sum() / area(mR >= fg["fit_radius_deg"])
print(f"dwarf-like stars: core locus from {core.sum()} stars; reference density ({fg['fit_radius_deg']}-{d['radius_deg']} deg) = {ref:.2f}/deg^2")
edges = np.array([0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0, 5.5, 6.25, 7.0, 8.0])
rows = []
for lo, hi in zip(edges[:-1], edges[1:]):
    s = (R >= lo) & (R < hi); a = area((mR >= lo) & (mR < hi)); k = like[s].sum()
    rows.append((lo, hi, k, k / a / ref, np.sqrt(max(k, 1)) / a / ref))
    print(f"  {lo:4.2f}-{hi:4.2f} deg: {k:6d} -> {k / a / ref:6.2f} x reference  (+/- {np.sqrt(max(k, 1)) / a / ref:.2f})")
# quadrants for 1.5-5.5 deg (E,N,W,S as PA ranges)
pa = np.degrees(np.arctan2(T["xi"], T["eta"])) % 360; mpa = np.degrees(np.arctan2(mx, my)) % 360
print("quadrants, 1.5-5.5 deg (PA east of north):")
quad = []
for name, lo, hi in [("N", 315, 45), ("E", 45, 135), ("S", 135, 225), ("W", 225, 315)]:
    inq = lambda p: ((p >= lo) | (p < hi)) if lo > hi else ((p >= lo) & (p < hi))
    s = (R >= 1.5) & (R < fg["fit_radius_deg"]) & inq(pa); a = area((mR >= 1.5) & (mR < fg["fit_radius_deg"]) & inq(mpa))
    k = like[s].sum(); quad.append((name, k / a / ref, np.sqrt(k) / a / ref))
    print(f"  {name}: {k:5d} -> {k / a / ref:5.2f} x reference (+/- {np.sqrt(k) / a / ref:.2f})")

# RR Lyrae at the dwarf distance (no CMD/RUWE cuts: mean G window only)
t = Table.read(path); t.rename_columns(t.colnames, [x.lower() for x in t.colnames])
cc = SkyCoord(t["ra"], t["dec"], unit="deg"); dl, db = c0.spherical_offsets_to(cc); r_all = np.hypot(dl.deg, db.deg)
g = np.asarray(t["int_average_g"].filled(np.nan) if hasattr(t["int_average_g"], "filled") else t["int_average_g"], float)
isrr = np.isin(np.asarray(t["best_classification"]).astype(str), ["RRab", "RRc", "RRd"]) & (g > 19.6) & (g < 20.6)
for (ra, de, rad) in cfg["cuts"].get("masks", []) or []:
    isrr &= SkyCoord(ra * u.deg, de * u.deg).separation(cc).deg > rad
print(f"RR Lyrae with 19.6 < <G> < 20.6: {isrr.sum()} total;", " ".join(
    f"[{lo},{hi}): {(isrr & (r_all >= lo) & (r_all < hi)).sum()}" for lo, hi in [(0, 1), (1, 2), (2, 3), (3, 4), (4, 5.5), (5.5, 8)]))

fig, ax = plt.subplots(1, 2, figsize=(11, 4.2))
mid = [0.5 * (r[0] + r[1]) for r in rows]
ax[0].errorbar(mid, [r[3] for r in rows], yerr=[r[4] for r in rows], fmt="o", ms=4, color="#0b0b0b")
ax[0].axhline(1, color="#9a9994", lw=1); ax[0].axvline(fg["fit_radius_deg"], color="#9a9994", lw=1, ls=":")
ax[0].set_yscale("log"); ax[0].set_xlabel("radius (deg)"); ax[0].set_ylabel("dwarf-like density / outer-annulus density")
ax[0].set_title("Sculptor-like stars vs radius (dotted: start of the reference annulus)", fontsize=9)
ax[1].scatter(dl.deg[isrr], db.deg[isrr], s=6, c="#2a78d6", lw=0)
for rad in (1, 2, 4, 5.5, 8):
    ax[1].add_patch(plt.Circle((0, 0), rad, fill=False, color="#c9c8c3", lw=0.8))
ax[1].set_xlim(8.2, -8.2); ax[1].set_ylim(-8.2, 8.2); ax[1].set_aspect("equal")
ax[1].set_xlabel("ξ (deg)  ← east"); ax[1].set_ylabel("η (deg)")
ax[1].set_title(f"RR Lyrae at the dwarf's distance: {isrr.sum()} (rings at 1, 2, 4, 5.5, 8 deg)", fontsize=9)
fig.tight_layout()
for ext in ("png", "pdf"):
    fig.savefig(os.path.join(out, f"outskirts8.{ext}"), dpi=160, bbox_inches="tight")
