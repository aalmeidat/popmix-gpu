"""Illustrative mock Gaia catalogs (target + control) for testing fit_dwarf.py.

A Sculptor-like dwarf (elliptical Plummer main body plus an optional extended Plummer "halo"), with an
RGB/HB color-magnitude locus, parallax ~1/84 kpc, a systemic PM, and Gaia-like G-dependent errors,
on a uniform foreground with broad astrometry, a two-population CMD, and XP-like log g for G < 19.
All numbers are made up to look plausible; they are NOT Sculptor measurements.
Usage: python dwarf/make_mock_dwarf.py --config dwarf/sculptor.yaml --halo_frac 0.08
"""
import argparse, os
import numpy as np
import yaml
from astropy.table import Table
from astropy.coordinates import SkyCoord
import astropy.units as u


def gaia_err(G, rng):
    s = 10 ** (-1.7 + 0.28 * (G - 15))          # ~0.02 mas at G=15, ~0.07 at 17, ~0.5 at 20
    return s * rng.uniform(0.8, 1.25, len(G))


def corr3(n, rng):
    while True:
        r = rng.uniform(-0.3, 0.3, (n, 3))
        det = 1 - (r ** 2).sum(1) + 2 * r[:, 0] * r[:, 1] * r[:, 2]
        if (det > 0.05).all():
            return r


def astrometry(true, G, rng):
    n = len(G)
    e = np.column_stack([gaia_err(G, rng), 1.1 * gaia_err(G, rng), 1.1 * gaia_err(G, rng)])
    r = corr3(n, rng)
    C = np.zeros((n, 3, 3))
    for i in range(3):
        C[:, i, i] = e[:, i] ** 2
    for (i, j), k in zip([(0, 1), (0, 2), (1, 2)], range(3)):
        C[:, i, j] = C[:, j, i] = r[:, k] * e[:, i] * e[:, j]
    L = np.linalg.cholesky(C)
    obs = true + np.einsum("nij,nj->ni", L, rng.normal(size=(n, 3)))
    return obs, e, r


def plummer(n, a, e, pa, rng):
    uu = rng.random(n)
    R = a * np.sqrt(uu / (1 - uu))
    ph = 2 * np.pi * rng.random(n)
    xmaj, xmin = R * np.cos(ph), (1 - e) * R * np.sin(ph)
    p = np.radians(pa)
    return xmaj * np.sin(p) + xmin * np.cos(p), xmaj * np.cos(p) - xmin * np.sin(p)


def foreground(n, rng):
    G = 14.5 + 6.5 * rng.random(n) ** (1 / 2.2)                  # counts rise toward faint mags
    pop = rng.random(n) < 0.6                                   # 60% halo turnoff, 40% disk red dwarfs
    BR = np.where(pop, rng.normal(0.8, 0.12, n), rng.uniform(1.1, 2.5, n))
    plx = np.where(pop, rng.lognormal(np.log(0.12), 0.5, n), rng.lognormal(np.log(1.0), 0.6, n))
    pm = np.where(pop[:, None], rng.normal([3.0, -4.0], 4.5, (n, 2)), rng.normal([8.0, -10.0], 12.0, (n, 2)))
    logg = np.where(pop, rng.normal(4.2, 0.35, n), rng.normal(4.7, 0.25, n))
    return G, BR, np.column_stack([plx, pm]), logg


def dwarf_stars(n, d, rng):
    G_trgb = 16.3
    hb = rng.random(n) < 0.12
    G = np.where(hb, rng.normal(20.1, 0.1, n), G_trgb + (20.5 - G_trgb) * rng.random(n) ** (1 / 2.5))
    BR = np.where(hb, rng.uniform(0.2, 0.9, n), 1.0 + 0.14 * (20.5 - G) + rng.normal(0, 0.04, n))
    s_int = d["sigma_v_kms"] / (4.74 * d["distance_kpc"])
    true = np.column_stack([np.full(n, 1 / d["distance_kpc"]),
                            d["pmra"] + rng.normal(0, s_int, n), d["pmdec"] + rng.normal(0, s_int, n)])
    logg = np.where(hb, rng.normal(2.4, 0.3, n), 0.5 + 0.55 * (G - G_trgb) + rng.normal(0, 0.3, n))
    return G, BR, true, logg


def to_table(xi, eta, center, G, BR, true, logg, rng, label):
    c0 = SkyCoord(center[0] * u.deg, center[1] * u.deg)
    c = c0.spherical_offsets_by(xi * u.deg, eta * u.deg)
    obs, e, r = astrometry(true, G, rng)
    lg = logg + rng.normal(0, 0.2, len(G))
    bad = rng.random(len(G)) < 0.12                              # catastrophic XP log g values
    lg[bad] = rng.uniform(0, 5.5, bad.sum())
    lg[(G > 19) | (rng.random(len(G)) < 0.15)] = np.nan          # XP parameters only for G < 19, not all
    return Table(dict(source_id=np.arange(len(G)), ra=c.ra.deg, dec=c.dec.deg, parallax=obs[:, 0],
                      parallax_error=e[:, 0], pmra=obs[:, 1], pmra_error=e[:, 1], pmdec=obs[:, 2], pmdec_error=e[:, 2],
                      parallax_pmra_corr=r[:, 0], parallax_pmdec_corr=r[:, 1], pmra_pmdec_corr=r[:, 2],
                      ruwe=rng.lognormal(0, 0.08, len(G)), phot_g_mean_mag=G, bp_rp=BR + rng.normal(0, 0.02, len(G)),
                      logg_gspphot=lg, mock_label=label))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--n_dwarf", type=int, default=8000)
    ap.add_argument("--halo_frac", type=float, default=0.08)
    ap.add_argument("--fg_density", type=float, default=800.0, help="foreground stars per deg^2")
    ap.add_argument("--seed", type=int, default=3)
    ap.add_argument("--halo_dcol", type=float, default=0.0, help="BP-RP shift of the halo stars (negative = bluer)")
    a = ap.parse_args()
    cfg = yaml.safe_load(open(a.config))
    d, c = cfg["dwarf"], cfg["control"]
    rng = np.random.default_rng(a.seed)
    R = d["radius_deg"]
    # target: foreground + main + halo
    nf = int(a.fg_density * np.pi * R ** 2)
    r = R * np.sqrt(rng.random(nf)); ph = 2 * np.pi * rng.random(nf)
    Gf, BRf, Af, lgf = foreground(nf, rng)
    nh = int(a.halo_frac * a.n_dwarf); nm = a.n_dwarf - nh
    xm, ym = plummer(nm, d["a_plummer_deg"], d["ellipticity"], d["pa_deg"], rng)
    xh, yh = plummer(nh, 3.5 * d["a_plummer_deg"], 0.15, d["pa_deg"], rng)
    Gd, BRd, Ad, lgd = dwarf_stars(a.n_dwarf, d, rng)
    BRd[nm:] += a.halo_dcol                                     # the last nh dwarf stars are the halo
    xi = np.concatenate([r * np.cos(ph), xm, xh]); eta = np.concatenate([r * np.sin(ph), ym, yh])
    lab = np.array(["fg"] * nf + ["main"] * nm + ["halo"] * nh)
    T = to_table(xi, eta, (d["ra"], d["dec"]), np.concatenate([Gf, Gd]), np.concatenate([BRf, BRd]),
                 np.concatenate([Af, Ad]), np.concatenate([lgf, lgd]), rng, lab)
    T = T[np.hypot(xi, eta) < R]
    base = os.path.join(os.path.dirname(os.path.abspath(a.config)), cfg["name"])
    T.write(f"{base}_mock_target.fits", overwrite=True)
    # control: foreground only
    Rc = c["radius_deg"]
    nc = int(a.fg_density * np.pi * Rc ** 2)
    r = Rc * np.sqrt(rng.random(nc)); ph = 2 * np.pi * rng.random(nc)
    Gc, BRc, Ac, lgc = foreground(nc, rng)
    Tc = to_table(r * np.cos(ph), r * np.sin(ph), (c["ra"], c["dec"]), Gc, BRc, Ac, lgc, rng, np.array(["fg"] * nc))
    Tc.write(f"{base}_mock_control.fits", overwrite=True)
    print(f"mock target: {len(T):,} stars ({(T['mock_label'] == 'main').sum():,} main, "
          f"{(T['mock_label'] == 'halo').sum():,} halo); control: {len(Tc):,}")


if __name__ == "__main__":
    main()
