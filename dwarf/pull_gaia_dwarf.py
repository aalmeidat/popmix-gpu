"""Download a Gaia DR3 cone around a dwarf galaxy plus a control field, for dwarf/fit_dwarf.py.

Run this on your own computer (it needs network access to the Gaia archive):
    pip install astroquery astropy pyyaml
    python pull_gaia_dwarf.py --config sculptor.yaml
Writes <name>_target.fits and <name>_control.fits next to the config.
With --xp, also fetches the Zhang, Green & Rix (2023) XP parameters (GAVO TAP, G < 17.65) for both cones
and merges them by source_id as logg_zgr, feh_zgr, teff_zgr (+ errors) and qf_zgr (quality flags).

The archive caps the rows returned per query (50,000 for these anonymous jobs as of Oct 2026), so each
cone is fetched in declination strips; any strip that comes back at the cap is split again, and the
pieces are stacked and de-duplicated by source_id. If the archive rejects the RR Lyrae join, rerun
with --no-rrlyrae.
"""
import argparse, os
import numpy as np
import yaml

COLUMNS = """g.source_id, g.ra, g.dec, g.l, g.b,
  g.parallax, g.parallax_error, g.pmra, g.pmra_error, g.pmdec, g.pmdec_error,
  g.parallax_pmra_corr, g.parallax_pmdec_corr, g.pmra_pmdec_corr,
  g.ruwe, g.astrometric_params_solved, g.astrometric_excess_noise, g.visibility_periods_used,
  g.nu_eff_used_in_astrometry, g.pseudocolour, g.ecl_lat,
  g.phot_g_mean_mag, g.phot_bp_mean_mag, g.phot_rp_mean_mag, g.bp_rp,
  g.phot_bp_rp_excess_factor, g.phot_g_mean_flux_over_error,
  g.radial_velocity, g.radial_velocity_error, g.has_xp_continuous,
  g.teff_gspphot, g.logg_gspphot, g.mh_gspphot, g.distance_gspphot, g.ag_gspphot, g.ebpminrp_gspphot"""

RR_COLUMNS = ", rr.pf, rr.p1_o, rr.best_classification, rr.int_average_g"


def adql(ra, dec, radius, gmax, rrlyrae=True, dec_lo=None, dec_hi=None):
    cols = COLUMNS + (RR_COLUMNS if rrlyrae else "")
    join = "LEFT OUTER JOIN gaiadr3.vari_rrlyrae AS rr ON g.source_id = rr.source_id" if rrlyrae else ""
    strip = f"\n  AND g.dec >= {dec_lo:.6f} AND g.dec < {dec_hi:.6f}" if dec_lo is not None else ""
    return f"""SELECT {cols}
FROM gaiadr3.gaia_source AS g
{join}
WHERE 1 = CONTAINS(POINT('ICRS', g.ra, g.dec), CIRCLE('ICRS', {ra}, {dec}, {radius}))
  AND g.parallax IS NOT NULL
  AND g.phot_g_mean_mag < {gmax}{strip}"""


def clean(t):
    for name in t.colnames:  # FITS cannot store object/mixed-type columns (e.g. masked strings)
        if t[name].dtype.kind == "O":
            t[name] = [("" if (v is None or v is np.ma.masked) else str(v)) for v in t[name]]
    t.rename_columns(t.colnames, [c.lower() for c in t.colnames])
    return t


def fetch_cone(Gaia, ra, dec, radius, gmax, rrlyrae, cap, nstrips, depth=0):
    from astropy.table import vstack, unique
    edges = np.linspace(dec - radius, dec + radius + 1e-6, nstrips + 1)
    parts = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        t = clean(Gaia.launch_job_async(adql(ra, dec, radius, gmax, rrlyrae, lo, hi), verbose=False).get_results())
        print(f"{'  ' * (depth + 1)}dec [{lo:+.3f}, {hi:+.3f}): {len(t):,} rows", flush=True)
        if len(t) >= cap:  # hit the cap: split this strip again
            t = fetch_strip(Gaia, ra, dec, radius, gmax, rrlyrae, cap, lo, hi, depth + 1)
        parts.append(t)
    out = vstack(parts, metadata_conflicts="silent")
    return unique(out, keys="source_id")


def fetch_strip(Gaia, ra, dec, radius, gmax, rrlyrae, cap, lo, hi, depth):
    from astropy.table import vstack
    mid = 0.5 * (lo + hi)
    parts = []
    for a, b in ((lo, mid), (mid, hi)):
        t = clean(Gaia.launch_job_async(adql(ra, dec, radius, gmax, rrlyrae, a, b), verbose=False).get_results())
        print(f"{'  ' * (depth + 1)}dec [{a:+.3f}, {b:+.3f}): {len(t):,} rows", flush=True)
        if len(t) >= cap:
            t = fetch_strip(Gaia, ra, dec, radius, gmax, rrlyrae, cap, a, b, depth + 1)
        parts.append(t)
    return vstack(parts, metadata_conflicts="silent")


# ----------------------------------------------------------------------------- XP parameters (Zhang, Green & Rix 2023)
XP_TAP = "https://dc.zah.uni-heidelberg.de/tap"   # GAVO DaCHS; table xpparams.main


def xp_where(ra, dec, radius, lo, hi):
    return (f"1 = CONTAINS(POINT('ICRS', ra, dec), CIRCLE('ICRS', {ra}, {dec}, {radius}))"
            f" AND dec >= {lo:.6f} AND dec < {hi:.6f}")


def fetch_xp(ra, dec, radius, nstrips, maxrec=200000, timeout=600):
    """Cone of Zhang et al. XP parameters in small declination strips, as synchronous queries with an
    explicit MAXREC and no COUNT(*) pre-check (the count scanned the full table and hung). A strip whose
    result reaches maxrec is split again. Progress prints after every strip."""
    import requests, io
    from astropy.table import Table, vstack, unique
    url = XP_TAP.rstrip("/") + "/sync"

    def run(lo, hi):
        q = ("SELECT source_id, teff, err_teff, fe_h, err_fe_h, logg, err_logg, quality_flags "
             f"FROM xpparams.main WHERE {xp_where(ra, dec, radius, lo, hi)}")
        r = requests.post(url, data=dict(REQUEST="doQuery", LANG="ADQL", FORMAT="votable", MAXREC=maxrec, QUERY=q),
                          timeout=timeout)
        r.raise_for_status()
        return Table.read(io.BytesIO(r.content), format="votable")

    def strip(lo, hi, depth):
        t = run(lo, hi)
        print(f"{'  ' * (depth + 1)}XP dec [{lo:+.3f}, {hi:+.3f}): {len(t):,} rows", flush=True)
        if len(t) >= maxrec:
            mid = 0.5 * (lo + hi)
            return strip(lo, mid, depth + 1) + strip(mid, hi, depth + 1)
        return [t] if len(t) else []

    edges = np.linspace(dec - radius, dec + radius + 1e-6, nstrips + 1)
    parts = sum((strip(a, b, 0) for a, b in zip(edges[:-1], edges[1:])), [])
    t = unique(vstack(parts, metadata_conflicts="silent"), keys="source_id")
    t.rename_columns(t.colnames, [c.lower() for c in t.colnames])
    ren = dict(teff="teff_zgr", err_teff="err_teff_zgr", fe_h="feh_zgr", err_fe_h="err_feh_zgr",
               logg="logg_zgr", err_logg="err_logg_zgr", quality_flags="qf_zgr")
    t.rename_columns(list(ren), list(ren.values()))
    return t


def merge_xp(path, xp):
    from astropy.table import Table, join
    t = Table.read(path)
    for c in [c for c in t.colnames if c.endswith("_zgr")]:
        t.remove_column(c)
    m = join(t, xp, keys="source_id", join_type="left")
    for c in [c for c in m.colnames if c.endswith("_zgr")]:
        if hasattr(m[c], "filled"):
            m[c] = m[c].filled(np.nan if m[c].dtype.kind == "f" else -1)
    m.write(path, overwrite=True)
    n = np.isfinite(np.asarray(m["logg_zgr"], float)).sum()
    print(f"  merged XP parameters into {os.path.basename(path)}: {n:,} of {len(m):,} stars have logg_zgr")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="sculptor.yaml")
    ap.add_argument("--no-rrlyrae", action="store_true")
    ap.add_argument("--cap", type=int, default=50000, help="archive row cap per query")
    ap.add_argument("--strips", type=int, default=None, help="initial declination strips per cone (default ~4 per deg)")
    ap.add_argument("--xp", action="store_true", help="also fetch Zhang et al. 2023 XP parameters and merge them")
    ap.add_argument("--xp-only", action="store_true", help="only fetch/merge XP parameters into existing files")
    ap.add_argument("--print-only", action="store_true", help="print the full-cone ADQL (for the web form)")
    a = ap.parse_args()
    cfg = yaml.safe_load(open(a.config))
    d, c = cfg["dwarf"], cfg["control"]
    rr = not a.no_rrlyrae
    if a.print_only:
        print("-- TARGET\n" + adql(d["ra"], d["dec"], d["radius_deg"], cfg["gmax_query"], rr))
        print("\n-- CONTROL\n" + adql(c["ra"], c["dec"], c["radius_deg"], cfg["gmax_query"], rr))
        print("\n-- the web form also caps rows; add e.g. AND g.dec >= -35.7 AND g.dec < -34.7 to fetch strips")
        return
    Gaia = None
    if not a.xp_only:
        from astroquery.gaia import Gaia
        Gaia.ROW_LIMIT = -1
    base = os.path.join(os.path.dirname(os.path.abspath(a.config)), cfg["name"])
    for tag, f in (("target", d), ("control", c)):
        out = f"{base}_{tag}.fits"
        ns = a.strips or max(8, int(round(4 * f["radius_deg"])))
        if not a.xp_only:
            print(f"querying {tag} ({f['radius_deg']} deg cone, {ns} strips) ...", flush=True)
            t = fetch_cone(Gaia, f["ra"], f["dec"], f["radius_deg"], cfg["gmax_query"], rr, a.cap, ns)
            t.write(out, overwrite=True)
            print(f"  total {len(t):,} unique rows -> {out}", flush=True)
        if a.xp or a.xp_only:
            if not os.path.exists(out):
                print(f"skipping XP for {tag}: {os.path.basename(out)} not found", flush=True)
                continue
            print(f"querying XP parameters for {tag} ...", flush=True)
            merge_xp(out, fetch_xp(f["ra"], f["dec"], f["radius_deg"], ns))


if __name__ == "__main__":
    main()
