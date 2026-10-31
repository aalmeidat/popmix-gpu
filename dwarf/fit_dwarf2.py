"""Dwarf halo search, version 2: the four fixes after the first Sculptor run.

1. Local foreground: the foreground template (astrometry XD, CMD, log g) comes from an outer annulus of
   the same wide cone (e.g. 5.5-8 deg) instead of a distant control field, and the foreground surface
   density gets a planar gradient exp(gx*xi + gy*eta) fitted with everything else. (A radial term was
   tried and removed: a foreground rising toward the center is degenerate with an extended halo, and
   on a mock with an injected halo it absorbed most of it.)
2. The halo has its own CMD: the main body's CMD shifted in BP-RP by a fitted dcol (|dcol| < 0.15 mag;
   negative = bluer, i.e. a more metal-poor RGB). A free membership-weighted halo histogram (confined
   to the main locus) was tried first and failed the mock test: built from halo weights over the whole
   field it was dominated by foreground stars inside the locus, and suppressed the injected halo.
3. CMD exclusion boxes (cuts.cmd_exclude) remove the faint-blue region where the dwarf's horizontal
   branch overlaps distant Milky Way turnoff stars and Gaia PMs are too noisy to separate them.
4. log g from a configurable column: Zhang, Green & Rix (2023) XP parameters ('logg_zgr', quality
   flags < 8) instead of GSP-Phot, which fails for faint metal-poor giants.
The halo scale stays bounded (a < a_h < 0.5 R_fit) and the main-body CMD comes from the main body
inside 3 Plummer radii, as in version 1.

Usage: python dwarf/fit_dwarf2.py --config dwarf/sculptor8.yaml [--target X.fits] [--iters 5]
"""
import argparse, json, os, sys, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import yaml
from astropy.table import Table
from astropy.coordinates import SkyCoord
import astropy.units as u
from scipy.optimize import minimize
from scipy.ndimage import gaussian_filter, binary_dilation
from scipy.special import logsumexp, expit
from proto import xd
from fit_dwarf import col, Hist2D, LoggTerm, gauss3_logpdf, ell_radius, plummer_log


def logit(p):
    return np.log(p / (1 - p))


# ----------------------------------------------------------------------------- data
def load_field(path, center, rmin, rmax, cuts, logg_col):
    t = Table.read(path)
    t.rename_columns(t.colnames, [c.lower() for c in t.colnames])
    c0 = SkyCoord(center[0] * u.deg, center[1] * u.deg)
    c = SkyCoord(col(t, "ra") * u.deg, col(t, "dec") * u.deg)
    dlon, dlat = c0.spherical_offsets_to(c)
    xi, eta = dlon.deg, dlat.deg
    r = np.hypot(xi, eta)
    G, BR = col(t, "phot_g_mean_mag"), col(t, "bp_rp")
    plx, pmra, pmdec = col(t, "parallax"), col(t, "pmra"), col(t, "pmdec")
    e = np.column_stack([col(t, "parallax_error"), col(t, "pmra_error"), col(t, "pmdec_error")])
    r12, r13, r23 = (np.nan_to_num(col(t, k)) for k in ("parallax_pmra_corr", "parallax_pmdec_corr", "pmra_pmdec_corr"))
    ruwe = col(t, "ruwe") if "ruwe" in t.colnames else np.ones(len(t))
    if logg_col in t.colnames:
        logg = col(t, logg_col)
        if logg_col == "logg_zgr" and "qf_zgr" in t.colnames:   # Zhang et al. recommended cut
            q = col(t, "qf_zgr")
            logg[~(np.isfinite(q) & (q < 8))] = np.nan
    else:
        print(f"  note: {logg_col} not in {os.path.basename(path)}; log g treated as missing for all stars")
        logg = np.full(len(t), np.nan)
    ok = (r >= rmin) & (r < rmax) & np.isfinite(G) & np.isfinite(BR) & np.isfinite(plx) & np.isfinite(pmra) \
        & np.isfinite(pmdec) & np.isfinite(e).all(1) & (G > cuts["gmin"]) & (G < cuts["gmax"]) \
        & (BR > cuts["color"][0]) & (BR < cuts["color"][1]) & (np.nan_to_num(ruwe, nan=99) < cuts["ruwe_max"])
    for (c_lo, c_hi, g_lo, g_hi) in cuts.get("cmd_exclude", []) or []:
        ok &= ~((BR > c_lo) & (BR < c_hi) & (G > g_lo) & (G < g_hi))
    for (m_ra, m_dec, m_r) in cuts.get("masks", []) or []:          # e.g. foreground globular clusters
        ok &= SkyCoord(m_ra * u.deg, m_dec * u.deg).separation(c).deg > m_r
    C = np.zeros((len(t), 3, 3))
    for i in range(3):
        C[:, i, i] = e[:, i] ** 2
    C[:, 0, 1] = C[:, 1, 0] = r12 * e[:, 0] * e[:, 1]
    C[:, 0, 2] = C[:, 2, 0] = r13 * e[:, 0] * e[:, 2]
    C[:, 1, 2] = C[:, 2, 1] = r23 * e[:, 1] * e[:, 2]
    sid = np.asarray(t["source_id"]) if "source_id" in t.colnames else np.arange(len(t))
    return dict(xi=xi[ok], eta=eta[ok], A=np.column_stack([plx, pmra, pmdec])[ok], C=C[ok], G=G[ok], BR=BR[ok],
                logg=logg[ok], id=sid[ok], n_raw=len(t), rmin=rmin, radius=rmax)


class LocusHist:
    """Membership-weighted CMD density confined to a (dilated) mask of allowed cells."""

    def __init__(self, x, y, w, ref, mask, smooth=1.5):
        H, _, _ = np.histogram2d(x, y, bins=[ref.xe, ref.ye], weights=w)
        H = gaussian_filter(H, smooth) * mask + 1e-6 * max(H.sum(), 1e-12) / H.size
        area = np.diff(ref.xe)[0] * np.diff(ref.ye)[0]
        self.xe, self.ye = ref.xe, ref.ye
        self.logH = np.log(H / (H.sum() * area))

    __call__ = Hist2D.__call__


# ----------------------------------------------------------------------------- model
class Model:
    NG = 2  # foreground spatial terms: planar gradient gx, gy (a radial term is degenerate with a halo)
    DMAX = 0.15  # max |BP-RP shift| of the halo CMD relative to the main body (mag)

    def __init__(self, tgt, tmpl, cfg, ncomp):
        self.t, self.cfg, self.ncomp = tgt, cfg, ncomp
        d, m = cfg["dwarf"], cfg["model"]
        self.R = tgt["radius"]
        self.area = np.pi * self.R ** 2
        rng = np.random.default_rng(1)
        rr = self.R * np.sqrt(rng.random(200000)); ph = 2 * np.pi * rng.random(200000)
        mx, my = rr * np.cos(ph), rr * np.sin(ph)
        keep = np.ones(len(mx), bool)
        c0 = SkyCoord(d["ra"] * u.deg, d["dec"] * u.deg)
        for (m_ra, m_dec, m_r) in cfg["cuts"].get("masks", []) or []:   # masked holes inside the fit region
            dl, db = c0.spherical_offsets_to(SkyCoord(m_ra * u.deg, m_dec * u.deg))
            keep &= np.hypot(mx - dl.deg, my - db.deg) > m_r
        self.area *= keep.mean()
        self.mc = (mx[keep], my[keep])
        self.s_int = d["sigma_v_kms"] / (4.74 * d["distance_kpc"])
        # foreground template: astrometry XD, CMD, log g
        Ac, Cc = tmpl["A"], tmpl["C"]
        self.fmu, self.fsd = Ac.mean(0), Ac.std(0)
        Zc, Sc = (Ac - self.fmu) / self.fsd, Cc / (self.fsd[:, None] * self.fsd[None, :])
        best = None
        for s in range(3):
            f = xd.fit(Zc, Sc, m["K_xd_foreground"], max_iter=400, tol=1e-7, reg=1e-6, seed=s)
            if best is None or f["logL"] > best["logL"]:
                best = f
        self.fxd = best
        cr, gr = cfg["cuts"]["color"], [cfg["cuts"]["gmin"], cfg["cuts"]["gmax"]]
        self.cmd_box = (cr, gr, m["cmd_bins"])
        fg_cmd = Hist2D(tmpl["BR"], tmpl["G"], np.ones(len(tmpl["G"])), cr, gr, m["cmd_bins"])
        fg_logg = LoggTerm(tmpl["logg"], np.ones(len(tmpl["G"])), m["logg_bins"])
        self.lp_fg_nonspatial = self._fg_ast(tgt["A"], tgt["C"]) + fg_cmd(tgt["BR"], tgt["G"]) + fg_logg(tgt["logg"])
        th = [0.0] * ncomp + [1 / d["distance_kpc"], d["pmra"], d["pmdec"],
                              np.log(d["a_plummer_deg"]), logit(d["ellipticity"]), d["pa_deg"]]
        if ncomp == 2:
            th[1] = -1.5
            th += [logit(2 * d["a_plummer_deg"] / (0.5 * self.R - d["a_plummer_deg"])), logit(0.2), 0.0]
        th += [0.0] * self.NG
        self.theta = np.array(th, float)
        self.h_cmd = None

    def _fg_ast(self, A, C):
        import torch
        Z, S = (A - self.fmu) / self.fsd, C / (self.fsd[:, None] * self.fsd[None, :])
        prm = [torch.as_tensor(v, dtype=torch.float64) for v in (self.fxd["alpha"], self.fxd["mu"], self.fxd["V"])]
        out = [xd._estep(torch.as_tensor(Z[i:i + 100000], dtype=torch.float64),
                         torch.as_tensor(S[i:i + 100000], dtype=torch.float64), *prm, need_moments=False)[0].numpy()
               for i in range(0, len(Z), 100000)]
        return np.concatenate(out) - np.log(self.fsd).sum()

    def unpack(self, th):
        nc = self.ncomp
        z = np.concatenate([[0.0], th[:nc]])
        p = dict(w=np.exp(z - logsumexp(z)), plx=th[nc], pmra=th[nc + 1], pmdec=th[nc + 2], a=np.exp(th[nc + 3]),
                 e=expit(th[nc + 4]), pa=th[nc + 5], g=th[-self.NG:])
        if nc == 2:
            p.update(a_h=p["a"] + (0.5 * self.R - p["a"]) * expit(th[nc + 6]), e_h=expit(th[nc + 7]),
                     dcol=self.DMAX * np.tanh(th[nc + 8]))
        return p

    def fg_spatial(self, p, xi, eta):
        gx, gy = p["g"]
        f = lambda x, y: gx * x + gy * y
        norm = np.log(np.exp(f(*self.mc)).mean() * self.area)
        return f(xi, eta) - norm

    def dwarf_spatial(self, p, xi, eta):
        prof = [(p["a"], p["e"])] + ([(p["a_h"], p["e_h"])] if self.ncomp == 2 else [])
        out = []
        for a, e in prof:
            norm = np.log(np.exp(plummer_log(self.mc[0], self.mc[1], a, e, p["pa"])).mean() * self.area)
            out.append(plummer_log(xi, eta, a, e, p["pa"]) - norm)
        return out

    def comp_logl(self, th):
        p, t = self.unpack(th), self.t
        mean = np.array([p["plx"], p["pmra"], p["pmdec"]])
        lp_ast = gauss3_logpdf(t["A"], t["C"], mean, np.diag([1e-8, self.s_int ** 2, self.s_int ** 2]))
        lg = self.d_logg(t["logg"])
        sp = self.dwarf_spatial(p, t["xi"], t["eta"])
        L = [np.log(p["w"][0]) + self.fg_spatial(p, t["xi"], t["eta"]) + self.lp_fg_nonspatial,
             np.log(p["w"][1]) + sp[0] + lp_ast + self.d_cmd(t["BR"], t["G"]) + lg]
        if self.ncomp == 2:   # halo CMD = main-body CMD shifted in colour (metal-poor RGB -> bluer, dcol < 0)
            L.append(np.log(p["w"][2]) + sp[1] + lp_ast + self.d_cmd(t["BR"] - p["dcol"], t["G"]) + lg)
        return np.column_stack(L)

    def set_hists(self, w_main, w_halo=None):
        cr, gr, b = self.cmd_box
        t = self.t
        self.d_cmd = Hist2D(t["BR"], t["G"], np.maximum(w_main, 1e-6), cr, gr, b)
        self.d_logg = LoggTerm(t["logg"], np.maximum(w_main, 1e-6), self.cfg["model"]["logg_bins"])
        if w_halo is not None and w_halo.sum() > 5:
            H = np.exp(self.d_cmd.logH)
            mask = binary_dilation(H > 0.005 * H.max(), iterations=self.cfg["model"].get("halo_locus_dilate", 2))
            self.h_cmd = LocusHist(t["BR"], t["G"], w_halo, self.d_cmd, mask)
            self.locus_mask = mask

    def fit(self, iters, verbose=True):
        t, d = self.t, self.cfg["dwarf"]
        r = ell_radius(t["xi"], t["eta"], d["ellipticity"], d["pa_deg"])
        dpm = np.hypot(t["A"][:, 1] - d["pmra"], t["A"][:, 2] - d["pmdec"])
        spm = np.sqrt(t["C"][:, 1, 1] + t["C"][:, 2, 2])
        w_main = ((r < 3 * d["a_plummer_deg"]) & (dpm < 3 * spm + 0.2)
                  & (np.abs(t["A"][:, 0]) < 3 * np.sqrt(t["C"][:, 0, 0]) + 0.05)).astype(float)
        w_halo, hist = None, []
        for it in range(iters):
            self.set_hists(w_main, w_halo)
            res = minimize(lambda th: -logsumexp(self.comp_logl(th), axis=1).sum(), self.theta, method="L-BFGS-B")
            self.theta = res.x
            L = self.comp_logl(self.theta)
            lnl = logsumexp(L, axis=1)
            R = np.exp(L - lnl[:, None])
            p = self.unpack(self.theta)
            w_main = R[:, 1] * (ell_radius(t["xi"], t["eta"], p["e"], p["pa"]) < 3 * p["a"])
            w_halo = R[:, 2] if self.ncomp == 2 else None
            hist.append(float(lnl.sum()))
            if verbose:
                print(f"  K={self.ncomp} it={it} lnL={lnl.sum():.1f}  N_dwarf={R[:, 1:].sum():.0f}  a={p['a'] * 60:.2f}'"
                      f"  fg grad=({p['g'][0]:+.3f},{p['g'][1]:+.3f})/deg"
                      + (f"  a_h={p['a_h'] * 60:.1f}'  f_halo={p['w'][2] / p['w'][1:].sum():.3f}  dcol={p['dcol']:+.3f}" if self.ncomp == 2 else ""),
                      flush=True)
        self.R_resp, self.lnL, self.history = R, float(lnl.sum()), hist
        return self


# ----------------------------------------------------------------------------- plots
def plots(models, tgt, tmpl, out, name, cfg):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    m1, m2 = models[1], models[2]
    p1 = m1.unpack(m1.theta)
    r = ell_radius(tgt["xi"], tgt["eta"], p1["e"], p1["pa"])
    rmc = ell_radius(*m1.mc, p1["e"], p1["pa"])
    edges = np.logspace(np.log10(0.03), np.log10(r.max()), 24)
    nmc = len(rmc)
    inb = [(rmc >= a) & (rmc < b) for a, b in zip(edges[:-1], edges[1:])]
    area = np.array([s.sum() / nmc * m1.area for s in inb])
    rc = np.sqrt(edges[:-1] * edges[1:])
    N = len(r)
    cnt = np.histogram(r, edges)[0]
    ok = area > 0
    fig, (ax, bx, cx) = plt.subplots(1, 3, figsize=(16, 4.6))
    ax.errorbar(rc[ok] * 60, cnt[ok] / area[ok], yerr=np.sqrt(np.maximum(cnt[ok], 1)) / area[ok], fmt="o", ms=4,
                color="#0b0b0b", label="all stars")
    colors = {1: "#2a78d6", 2: "#eb6834"}
    for K, m in models.items():
        p = m.unpack(m.theta)
        fg = np.exp(m.fg_spatial(p, *m.mc)); sps = [np.exp(s) for s in m.dwarf_spatial(p, *m.mc)]
        tot = N * p["w"][0] * np.array([fg[s].sum() for s in inb]) / nmc * m.area / np.where(ok, area, np.nan)
        if K == 1:
            ax.plot(rc * 60, tot, color="#9a9994", lw=1, ls="-.", label="foreground (1-comp. fit)")
        for k, sp in enumerate(sps):
            comp = N * p["w"][k + 1] * np.array([sp[s].sum() for s in inb]) / nmc * m.area / np.where(ok, area, np.nan)
            tot = tot + comp
            if K == 2:
                ax.plot(rc * 60, comp, color=colors[2], lw=1, ls=":" if k else "--")
        ax.plot(rc * 60, tot, color=colors[K], lw=2, label=f"{K}-component model")
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("elliptical radius (arcmin)"); ax.set_ylabel("stars per deg$^2$")
    ax.set_title(f"{name}: surface density (2-comp. dashed/dotted: main/halo)", fontsize=9)
    ax.legend(fontsize=8, frameon=False)
    # residual of all-star counts against the 1-component model
    p = m1.unpack(m1.theta)
    fg = np.exp(m1.fg_spatial(p, *m1.mc)); sp = np.exp(m1.dwarf_spatial(p, *m1.mc)[0])
    model1 = N * (p["w"][0] * np.array([fg[s].sum() for s in inb]) + p["w"][1] * np.array([sp[s].sum() for s in inb])) / nmc * m1.area
    gd = ok & (model1 > 0)
    bx.errorbar(rc[gd] * 60, (cnt[gd] - model1[gd]) / np.sqrt(model1[gd]), fmt="o", ms=4, color="#2a78d6")
    bx.axhline(0, color="#9a9994", lw=1); bx.set_xscale("log")
    bx.set_xlabel("elliptical radius (arcmin)"); bx.set_ylabel("(observed − 1-comp. model) / √model")
    bx.set_title("all-star residuals against the 1-component model", fontsize=9)
    # CMDs: main members, halo members, foreground template, and the exclusion boxes
    pm, phl = m2.R_resp[:, 1], m2.R_resp[:, 2]
    cx.scatter(tmpl["BR"], tmpl["G"], s=0.3, c="#d9d8d4", lw=0, rasterized=True, label="foreground template")
    cx.scatter(tgt["BR"][pm > 0.9], tgt["G"][pm > 0.9], s=1, c="#2a78d6", lw=0, rasterized=True, label="main, P > 0.9")
    cx.scatter(tgt["BR"][phl > 0.5], tgt["G"][phl > 0.5], s=4, c="#eb6834", lw=0, rasterized=True, label="halo, P > 0.5")
    for (c0, c1, g0, g1) in cfg["cuts"].get("cmd_exclude", []) or []:
        cx.add_patch(plt.Rectangle((c0, g0), c1 - c0, g1 - g0, fill=False, ec="#0b0b0b", ls="--", lw=1))
    cx.invert_yaxis(); cx.set_xlabel("BP − RP"); cx.set_ylabel("G")
    cx.set_title("CMD (dashed box: excluded region)", fontsize=9); cx.legend(fontsize=7, frameon=False, markerscale=4)
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(out, f"profile_v2.{ext}"), dpi=160, bbox_inches="tight")
    plt.close(fig)


# ----------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--target")
    ap.add_argument("--control")
    ap.add_argument("--iters", type=int, default=None)
    a = ap.parse_args()
    cfg = yaml.safe_load(open(a.config))
    base = os.path.dirname(os.path.abspath(a.config))
    tp = a.target or os.path.join(base, f"{cfg['name']}_target.fits")
    out = os.path.join(base, f"results_{cfg['name']}")
    os.makedirs(out, exist_ok=True)
    d, fgc = cfg["dwarf"], cfg["foreground"]
    lc = cfg["model"].get("logg_column", "logg_gspphot")
    t0 = time.time()
    tgt = load_field(tp, (d["ra"], d["dec"]), 0.0, fgc["fit_radius_deg"], cfg["cuts"], lc)
    if fgc["mode"] == "annulus":
        tmpl = load_field(tp, (d["ra"], d["dec"]), fgc["fit_radius_deg"], d["radius_deg"], cfg["cuts"], lc)
        tdesc = f"annulus {fgc['fit_radius_deg']}-{d['radius_deg']} deg"
    else:
        c = cfg["control"]
        cp = a.control or os.path.join(base, f"{cfg['name']}_control.fits")
        tmpl = load_field(cp, (c["ra"], c["dec"]), 0.0, c["radius_deg"], cfg["cuts"], lc)
        tdesc = "control field"
    print(f"fit region r < {fgc['fit_radius_deg']} deg: {len(tgt['G']):,} stars; foreground template ({tdesc}): {len(tmpl['G']):,}")
    print(f"log g column '{lc}': available for {np.isfinite(tgt['logg']).mean():.1%} of fit-region stars")
    iters = a.iters or cfg["model"]["em_iterations"]
    models = {}
    for K in (1, 2):
        print(f"fitting {K}-component dwarf model")
        m = Model(tgt, tmpl, cfg, K)
        if K == 2:
            t1 = models[1].theta  # [w_main, plx, pmra, pmdec, log a, logit e, pa, gx, gy]
            m.theta[0] = t1[0]; m.theta[1] = t1[0] - 2.5
            m.theta[2:8] = t1[1:7]; m.theta[-Model.NG:] = t1[-Model.NG:]
        models[K] = m.fit(iters)
    n = len(tgt["G"])
    res = {}
    for K, m in models.items():
        p = m.unpack(m.theta)
        res[K] = dict(lnL=m.lnL, n_params=len(m.theta), bic=-2 * m.lnL + len(m.theta) * np.log(n),
                      N_dwarf=float(m.R_resp[:, 1:].sum()), weights=p["w"].tolist(), plx_mas=p["plx"],
                      pmra=p["pmra"], pmdec=p["pmdec"], a_arcmin=p["a"] * 60, ellipticity=p["e"], pa_deg=p["pa"] % 180,
                      fg_gradient=p["g"].tolist())
        if K == 2:
            res[K].update(a_halo_arcmin=p["a_h"] * 60, e_halo=p["e_h"], at_bound=bool(p["a_h"] > 0.98 * 0.5 * m.R),
                          halo_color_shift=float(p["dcol"]),
                          halo_fraction_of_dwarf=float(p["w"][2] / p["w"][1:].sum()), N_halo=float(m.R_resp[:, 2].sum()))
    summary = dict(name=cfg["name"], version=2, n_fit=n, n_template=len(tmpl["G"]), template=tdesc, logg_column=lc,
                   cmd_exclude=cfg["cuts"].get("cmd_exclude"), delta_lnL_2_minus_1=res[2]["lnL"] - res[1]["lnL"],
                   delta_BIC_1_minus_2=res[1]["bic"] - res[2]["bic"], models=res, seconds=time.time() - t0)
    json.dump(summary, open(os.path.join(out, "fit_v2.json"), "w"), indent=2, default=float)
    m2 = models[2]
    Table(dict(source_id=tgt["id"], xi=tgt["xi"], eta=tgt["eta"], G=tgt["G"], bp_rp=tgt["BR"],
               p_dwarf_1comp=models[1].R_resp[:, 1:].sum(1), p_main_2comp=m2.R_resp[:, 1],
               p_halo_2comp=m2.R_resp[:, 2])).write(os.path.join(out, "members_v2.fits"), overwrite=True)
    plots(models, tgt, tmpl, out, cfg["name"], cfg)
    print(json.dumps({k: v for k, v in summary.items() if k != "models"}, indent=1, default=float))
    for K in (1, 2):
        print(K, {k: (round(float(v), 4) if isinstance(v, (float, np.floating)) else v) for k, v in res[K].items()})


if __name__ == "__main__":
    main()
