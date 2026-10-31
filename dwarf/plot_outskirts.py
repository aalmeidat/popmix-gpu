"""Model-independent outskirts checks for a dwarf: RR Lyrae at the dwarf's distance, and the density of
'dwarf-like' stars (astrometry + CMD locus) in annuli relative to the control field."""
import json, os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, yaml
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from astropy.table import Table
from astropy.coordinates import SkyCoord
import astropy.units as u
from scipy.ndimage import gaussian_filter
from fit_dwarf import load_field, ell_radius

cfg = yaml.safe_load(open(sys.argv[1])); base = os.path.dirname(os.path.abspath(sys.argv[1]))
res = os.path.join(base, f"results_{cfg['name']}")
d, c = cfg["dwarf"], cfg["control"]
m1 = json.load(open(os.path.join(res, "fit.json")))["models"]["1"]
gwin = (19.6, 20.6)  # RR Lyrae mean-G window at the dwarf's distance (Sculptor: median <G> = 20.07)

def rrl(path, ra, dec):
    t = Table.read(path); t.rename_columns(t.colnames, [x.lower() for x in t.colnames])
    c0 = SkyCoord(ra * u.deg, dec * u.deg); cc = SkyCoord(t["ra"], t["dec"], unit="deg")
    dl, db = c0.spherical_offsets_to(cc)
    g = np.asarray(t["int_average_g"].filled(np.nan) if hasattr(t["int_average_g"], "filled") else t["int_average_g"], float)
    s = np.isin(np.asarray(t["best_classification"]).astype(str), ["RRab", "RRc", "RRd"]) & (g > gwin[0]) & (g < gwin[1])
    return dl.deg[s], db.deg[s]

xr, yr = rrl(os.path.join(base, f"{cfg['name']}_target.fits"), d["ra"], d["dec"])
xc, yc = rrl(os.path.join(base, f"{cfg['name']}_control.fits"), c["ra"], c["dec"])
rr = np.hypot(xr, yr)

# dwarf-like star density ratio (same construction as diagnose_flat.py, chi2 < 7.8)
tgt = load_field(os.path.join(base, f"{cfg['name']}_target.fits"), (d["ra"], d["dec"]), d["radius_deg"], cfg["cuts"])
ctl = load_field(os.path.join(base, f"{cfg['name']}_control.fits"), (c["ra"], c["dec"]), c["radius_deg"], cfg["cuts"])
mem = Table.read(os.path.join(res, "members.fits"))
s_int = d["sigma_v_kms"] / (4.74 * d["distance_kpc"]); mean = np.array([m1["plx_mas"], m1["pmra"], m1["pmdec"]])
def chi2(f):
    T = f["C"] + np.diag([0, s_int ** 2, s_int ** 2])[None]; dd = f["A"] - mean[None]
    return np.einsum("ni,nij,nj->n", dd, np.linalg.inv(T), dd)
core = (np.asarray(mem["p_dwarf_1comp"]) > 0.9) & (ell_radius(tgt["xi"], tgt["eta"], m1["ellipticity"], m1["pa_deg"]) < 2 * m1["a_arcmin"] / 60)
H, xe, ye = np.histogram2d(tgt["BR"][core], tgt["G"][core], bins=[50, 60], range=[cfg["cuts"]["color"], [cfg["cuts"]["gmin"], cfg["cuts"]["gmax"]]])
H = gaussian_filter(H, 1.0); locus = H > np.quantile(H[H > 0], 0.5)
def inl(f):
    i = np.clip(np.searchsorted(xe, f["BR"]) - 1, 0, len(xe) - 2); j = np.clip(np.searchsorted(ye, f["G"]) - 1, 0, len(ye) - 2)
    return locus[i, j]
lt, lc = (chi2(tgt) < 7.8) & inl(tgt), (chi2(ctl) < 7.8) & inl(ctl)
dc = lc.sum() / (np.pi * c["radius_deg"] ** 2)
R = np.hypot(tgt["xi"], tgt["eta"]); edges = np.arange(0, d["radius_deg"] + 0.01, 0.25)
ratio, err = [], []
for lo, hi in zip(edges[:-1], edges[1:]):
    n = (lt & (R >= lo) & (R < hi)).sum(); a = np.pi * (hi ** 2 - lo ** 2)
    ratio.append(n / a / dc); err.append(np.sqrt(max(n, 1)) / a / dc)
mid = 0.5 * (edges[:-1] + edges[1:])

fig, ax = plt.subplots(1, 3, figsize=(14, 4.3))
a0 = ax[0]
a0.scatter(xr, yr, s=6, c="#2a78d6", lw=0)
for rad in (1, 2, 3, 4):
    a0.add_patch(plt.Circle((0, 0), rad, fill=False, color="#c9c8c3", lw=0.8))
a0.set_xlim(4.1, -4.1); a0.set_ylim(-4.1, 4.1); a0.set_aspect("equal")
a0.set_xlabel("ξ (deg)  ← east"); a0.set_ylabel("η (deg)")
a0.set_title(f"Gaia DR3 RR Lyrae, {gwin[0]} < <G> < {gwin[1]}: {len(rr)} in target, {len(xc)} in control", fontsize=9)
a1 = ax[1]
a1.errorbar(mid, ratio, yerr=err, fmt="o", ms=4, color="#0b0b0b")
a1.axhline(1, color="#9a9994", lw=1)
a1.set_yscale("log"); a1.set_xlabel("radius (deg)"); a1.set_ylabel("dwarf-like stars per deg$^2$ / control field")
a1.set_title("astrometry + CMD-locus selection, target vs control", fontsize=9)
a2 = ax[2]
Rg = np.linspace(0.3, d["radius_deg"], 80); ap = m1["a_arcmin"] / 60
a2.step(np.sort(rr)[::-1], np.arange(1, len(rr) + 1), where="post", color="#2a78d6", lw=2, label="RR Lyrae, observed N(> R)")
a2.plot(Rg, len(rr) * ap ** 2 / (Rg ** 2 + ap ** 2), color="#eb6834", lw=1.5, ls="--", label=f"single Plummer, a = {m1['a_arcmin']:.1f}' (circular approx.)")
a2.set_yscale("log"); a2.set_xlim(0.3, d["radius_deg"]); a2.set_ylim(0.5, 300)
a2.set_xlabel("radius R (deg)"); a2.set_ylabel("N(> R)"); a2.legend(fontsize=8, frameon=False)
a2.set_title("RR Lyrae beyond R vs a Plummer profile", fontsize=9)
fig.tight_layout()
for ext in ("png", "pdf"):
    fig.savefig(os.path.join(res, f"outskirts.{ext}"), dpi=160, bbox_inches="tight")
print("RRL target", len(rr), "beyond 1 deg", (rr > 1).sum(), "beyond 1.5 deg", (rr > 1.5).sum(), "; control", len(xc))
