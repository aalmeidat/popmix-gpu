"""Halo search in a dwarf galaxy: spatial + CMD + astrometry (+ XP log g) mixture with an empirical foreground.

Components
  foreground : uniform on the sky; astrometry from an error-deconvolved mixture (XD) fit to a control
               field; CMD and log g from the control field
  main       : elliptical Plummer; astrometry = one systemic (parallax, pmra, pmdec) + intrinsic PM width
  halo       : (2-component model only) a second, more extended elliptical Plummer, same center and PA,
               sharing the main body's astrometry, CMD and log g distributions
Every term is a density in the quantities a star has; log g is a categorical term that includes
"missing", so stars without XP parameters still enter correctly.
Fitting: EM-style -- dwarf CMD and log g histograms are rebuilt from the current memberships, then the
continuous parameters are optimized; repeated em_iterations times.
Outputs (dwarf/results_<name>/): fit.json (1- vs 2-component lnL, BIC, parameters), members.fits,
profile.png/pdf (radial profile with both models), map_cmd.png/pdf.

Usage: python dwarf/fit_dwarf.py --config dwarf/sculptor.yaml [--target X.fits --control Y.fits]
"""
import argparse, json, os, sys, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import yaml
from astropy.table import Table
from astropy.coordinates import SkyCoord
import astropy.units as u
from scipy.optimize import minimize
from scipy.ndimage import gaussian_filter
from scipy.special import logsumexp, expit
from proto import xd

LOG2PI = np.log(2 * np.pi)


# ----------------------------------------------------------------------------- data
def col(t, name):
    c = t[name]
    if hasattr(c, "filled"):
        c = c.filled(np.nan)
    return np.asarray(c, dtype=float)


def load_field(path, center, radius, cuts):
    t = Table.read(path)
    t.rename_columns(t.colnames, [c.lower() for c in t.colnames])
    c0 = SkyCoord(center[0] * u.deg, center[1] * u.deg)
    c = SkyCoord(col(t, "ra") * u.deg, col(t, "dec") * u.deg)
    dlon, dlat = c0.spherical_offsets_to(c)
    xi, eta = dlon.deg, dlat.deg
    G, BR = col(t, "phot_g_mean_mag"), col(t, "bp_rp")
    plx, pmra, pmdec = col(t, "parallax"), col(t, "pmra"), col(t, "pmdec")
    e = np.column_stack([col(t, "parallax_error"), col(t, "pmra_error"), col(t, "pmdec_error")])
    r12, r13, r23 = (np.nan_to_num(col(t, k)) for k in ("parallax_pmra_corr", "parallax_pmdec_corr", "pmra_pmdec_corr"))
    ruwe = col(t, "ruwe") if "ruwe" in t.colnames else np.ones(len(t))
    logg = col(t, "logg_gspphot") if "logg_gspphot" in t.colnames else np.full(len(t), np.nan)
    ok = (np.hypot(xi, eta) < radius) & np.isfinite(G) & np.isfinite(BR) & np.isfinite(plx) & np.isfinite(pmra) \
        & np.isfinite(pmdec) & np.isfinite(e).all(1) & (G > cuts["gmin"]) & (G < cuts["gmax"]) \
        & (BR > cuts["color"][0]) & (BR < cuts["color"][1]) & (np.nan_to_num(ruwe, nan=99) < cuts["ruwe_max"])
    C = np.zeros((len(t), 3, 3))
    for i in range(3):
        C[:, i, i] = e[:, i] ** 2
    C[:, 0, 1] = C[:, 1, 0] = r12 * e[:, 0] * e[:, 1]
    C[:, 0, 2] = C[:, 2, 0] = r13 * e[:, 0] * e[:, 2]
    C[:, 1, 2] = C[:, 2, 1] = r23 * e[:, 1] * e[:, 2]
    sid = np.asarray(t["source_id"]) if "source_id" in t.colnames else np.arange(len(t))
    return dict(xi=xi[ok], eta=eta[ok], A=np.column_stack([plx, pmra, pmdec])[ok], C=C[ok], G=G[ok], BR=BR[ok],
                logg=logg[ok], id=sid[ok], n_raw=len(t), radius=radius)


# ----------------------------------------------------------------------------- densities
class Hist2D:
    def __init__(self, x, y, w, xr, yr, bins, smooth=1.0):
        H, self.xe, self.ye = np.histogram2d(x, y, bins=bins, range=[xr, yr], weights=w)
        H = gaussian_filter(H, smooth) + 1e-3 * max(H.mean(), 1e-12)
        area = np.diff(self.xe)[0] * np.diff(self.ye)[0]
        self.logH = np.log(H / (H.sum() * area))

    def __call__(self, x, y):
        i = np.clip(np.searchsorted(self.xe, x) - 1, 0, len(self.xe) - 2)
        j = np.clip(np.searchsorted(self.ye, y) - 1, 0, len(self.ye) - 2)
        return self.logH[i, j]


class LoggTerm:
    """Categorical over log g bins plus 'missing'; returns a log probability mass."""

    def __init__(self, logg, w, nb):
        self.edges = np.linspace(0, 5.5, nb + 1)
        miss = ~np.isfinite(logg)
        p_miss = (w * miss).sum() / w.sum()
        h, _ = np.histogram(np.clip(logg[~miss], 0, 5.499), bins=self.edges, weights=w[~miss])
        h = gaussian_filter(h.astype(float), 0.7) + 1e-3 * max(h.mean(), 1e-12)
        self.lp_miss = np.log(np.clip(p_miss, 1e-6, 1 - 1e-6))
        self.lp_bin = np.log(1 - np.exp(self.lp_miss)) + np.log(h / h.sum())

    def __call__(self, logg):
        out = np.full(len(logg), self.lp_miss)
        m = np.isfinite(logg)
        k = np.clip(np.searchsorted(self.edges, np.clip(logg[m], 0, 5.499)) - 1, 0, len(self.edges) - 2)
        out[m] = self.lp_bin[k]
        return out


def gauss3_logpdf(A, C, mean, extra):
    T = C + extra[None]
    L = np.linalg.cholesky(T)
    d = A - mean[None]
    z = np.linalg.solve(L, d[..., None])[..., 0]
    return -0.5 * (z ** 2).sum(1) - np.log(np.diagonal(L, axis1=1, axis2=2)).sum(1) - 1.5 * LOG2PI


def ell_radius(xi, eta, e, pa):
    p = np.radians(pa)
    xmaj = xi * np.sin(p) + eta * np.cos(p)
    xmin = xi * np.cos(p) - eta * np.sin(p)
    return np.sqrt(xmaj ** 2 + (xmin / (1 - e)) ** 2)


def plummer_log(xi, eta, a, e, pa):
    R = ell_radius(xi, eta, e, pa)
    return -np.log(np.pi * a ** 2 * (1 - e)) - 2 * np.log1p((R / a) ** 2)


# ----------------------------------------------------------------------------- model
class Model:
    def __init__(self, tgt, ctl, cfg, ncomp):
        self.t, self.cfg, self.ncomp = tgt, cfg, ncomp
        d = cfg["dwarf"]
        self.R = tgt["radius"]
        self.area = np.pi * self.R ** 2
        rng = np.random.default_rng(1)
        r = self.R * np.sqrt(rng.random(200000))
        ph = 2 * np.pi * rng.random(200000)
        self.mc = (r * np.cos(ph), r * np.sin(ph))          # uniform points in the footprint
        self.s_int = d["sigma_v_kms"] / (4.74 * d["distance_kpc"])
        m = cfg["model"]
        # foreground astrometry: XD on the control field (standardized)
        Ac, Cc = ctl["A"], ctl["C"]
        self.fmu, self.fsd = Ac.mean(0), Ac.std(0)
        Zc = (Ac - self.fmu) / self.fsd
        Sc = Cc / (self.fsd[:, None] * self.fsd[None, :])
        best = None
        for s in range(3):
            f = xd.fit(Zc, Sc, m["K_xd_foreground"], max_iter=400, tol=1e-7, reg=1e-6, seed=s)
            if best is None or f["logL"] > best["logL"]:
                best = f
        self.fxd = best
        self.lp_fg_ast = self._fg_ast(tgt["A"], tgt["C"])
        cr, gr = cfg["cuts"]["color"], [cfg["cuts"]["gmin"], cfg["cuts"]["gmax"]]
        self.cmd_box = (cr, gr, m["cmd_bins"])
        self.fg_cmd = Hist2D(ctl["BR"], ctl["G"], np.ones(len(ctl["G"])), cr, gr, m["cmd_bins"])
        self.fg_logg = LoggTerm(ctl["logg"], np.ones(len(ctl["G"])), m["logg_bins"])
        self.lp_fg = np.log(1 / self.area) + self.lp_fg_ast + self.fg_cmd(tgt["BR"], tgt["G"]) + self.fg_logg(tgt["logg"])
        # parameters: [w logits (ncomp), plx, pmra, pmdec, log a, logit e, pa, (log a_h, logit e_h)]
        logit = lambda p: np.log(p / (1 - p))
        th = [0.0] * ncomp + [1 / d["distance_kpc"], d["pmra"], d["pmdec"],
                              np.log(d["a_plummer_deg"]), logit(d["ellipticity"]), d["pa_deg"]]
        if ncomp == 2:
            th[1] = -1.5  # halo weight logit (relative to foreground) starts small
            # halo scale is bounded: a < a_h < 0.5 R (a flat component is degenerate with the foreground)
            th += [logit(2 * d["a_plummer_deg"] / (0.5 * self.R - d["a_plummer_deg"])), logit(0.2)]
        self.theta = np.array(th, float)

    def _fg_ast(self, A, C):
        Z = (A - self.fmu) / self.fsd
        S = C / (self.fsd[:, None] * self.fsd[None, :])
        import torch
        f = self.fxd
        prm = [torch.as_tensor(v, dtype=torch.float64) for v in (f["alpha"], f["mu"], f["V"])]
        lse = []
        for i in range(0, len(Z), 100000):
            zi = torch.as_tensor(Z[i:i + 100000], dtype=torch.float64)
            si = torch.as_tensor(S[i:i + 100000], dtype=torch.float64)
            lse.append(xd._estep(zi, si, *prm, need_moments=False)[0].numpy())
        return np.concatenate(lse) - np.log(self.fsd).sum()

    def unpack(self, th):
        nc = self.ncomp
        w = np.exp(np.concatenate([[0.0], th[:nc]]) - logsumexp(np.concatenate([[0.0], th[:nc]])))  # [fg, main, (halo)]
        p = dict(w=w, plx=th[nc], pmra=th[nc + 1], pmdec=th[nc + 2], a=np.exp(th[nc + 3]),
                 e=expit(th[nc + 4]), pa=th[nc + 5])
        if nc == 2:
            p.update(a_h=p["a"] + (0.5 * self.R - p["a"]) * expit(th[nc + 6]), e_h=expit(th[nc + 7]))
        return p

    def _spatial(self, p, xi, eta):
        out = [plummer_log(xi, eta, p["a"], p["e"], p["pa"])]
        if self.ncomp == 2:
            out.append(plummer_log(xi, eta, p["a_h"], p["e_h"], p["pa"]))
        # renormalize each profile to the circular footprint
        norms = []
        for k, lp in enumerate(out):
            a, e = (p["a"], p["e"]) if k == 0 else (p["a_h"], p["e_h"])
            frac = np.exp(plummer_log(self.mc[0], self.mc[1], a, e, p["pa"])).mean() * self.area
            norms.append(np.log(frac))
        return [lp - n for lp, n in zip(out, norms)]

    def comp_logl(self, th):
        p = self.unpack(th)
        t = self.t
        mean = np.array([p["plx"], p["pmra"], p["pmdec"]])
        extra = np.diag([1e-8, self.s_int ** 2, self.s_int ** 2])
        lp_ast = gauss3_logpdf(t["A"], t["C"], mean, extra)
        base = lp_ast + self.d_cmd(t["BR"], t["G"]) + self.d_logg(t["logg"])
        sp = self._spatial(p, t["xi"], t["eta"])
        L = [np.log(p["w"][0]) + self.lp_fg] + [np.log(p["w"][k + 1]) + sp[k] + base for k in range(self.ncomp)]
        return np.column_stack(L)

    def set_dwarf_hist(self, wd):
        cr, gr, b = self.cmd_box
        self.d_cmd = Hist2D(self.t["BR"], self.t["G"], wd, cr, gr, b)
        self.d_logg = LoggTerm(self.t["logg"], wd, self.cfg["model"]["logg_bins"])

    def fit(self, iters, verbose=True):
        t, d = self.t, self.cfg["dwarf"]
        # initial memberships: near the center, PM close to systemic, parallax consistent with zero
        r = ell_radius(t["xi"], t["eta"], d["ellipticity"], d["pa_deg"])
        dpm = np.hypot(t["A"][:, 1] - d["pmra"], t["A"][:, 2] - d["pmdec"])
        spm = np.sqrt(t["C"][:, 1, 1] + t["C"][:, 2, 2])
        wd = ((r < 3 * d["a_plummer_deg"]) & (dpm < 3 * spm + 0.2) & (np.abs(t["A"][:, 0]) < 3 * np.sqrt(t["C"][:, 0, 0]) + 0.05)).astype(float)
        hist = []
        for it in range(iters):
            self.set_dwarf_hist(np.maximum(wd, 1e-6))
            res = minimize(lambda th: -logsumexp(self.comp_logl(th), axis=1).sum(), self.theta, method="L-BFGS-B")
            self.theta = res.x
            L = self.comp_logl(self.theta)
            lnl = logsumexp(L, axis=1)
            R = np.exp(L - lnl[:, None])
            # dwarf CMD / log g histograms come from the main body only, inside 3 Plummer radii, so an
            # extended component cannot widen them by absorbing foreground-like stars (EM runaway)
            p = self.unpack(self.theta)
            r = ell_radius(t["xi"], t["eta"], p["e"], p["pa"])
            wd = R[:, 1] * (r < 3 * p["a"])
            hist.append(float(lnl.sum()))
            if verbose:
                p = self.unpack(self.theta)
                print(f"  K={self.ncomp} it={it} lnL={lnl.sum():.1f}  N_dwarf={R[:, 1:].sum():.0f}  a={p['a'] * 60:.2f}'"
                      + (f"  a_h={p['a_h'] * 60:.2f}'  f_halo={p['w'][2] / p['w'][1:].sum():.3f}" if self.ncomp == 2 else ""),
                      flush=True)
        self.R_resp, self.lnL, self.history = R, float(lnl.sum()), hist
        return self


# ----------------------------------------------------------------------------- plots
def profile_plot(models, tgt, out, name):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    m1 = models[1]
    p1 = m1.unpack(m1.theta)
    r = ell_radius(tgt["xi"], tgt["eta"], p1["e"], p1["pa"])
    rmc = ell_radius(*m1.mc, p1["e"], p1["pa"])
    edges = np.logspace(np.log10(0.02), np.log10(r.max()), 26)
    n_mc = len(rmc)
    area = np.array([((rmc >= a) & (rmc < b)).sum() / n_mc * m1.area for a, b in zip(edges[:-1], edges[1:])])
    cnt = np.histogram(r, edges)[0]
    rc = np.sqrt(edges[:-1] * edges[1:])
    ok = area > 0
    N = len(r)
    fig, (ax, bx) = plt.subplots(1, 2, figsize=(11, 4.4))
    ax.errorbar(rc[ok] * 60, cnt[ok] / area[ok], yerr=np.sqrt(np.maximum(cnt[ok], 1)) / area[ok], fmt="o", ms=4,
                color="#0b0b0b", label="all stars (observed)")
    colors = {1: "#2a78d6", 2: "#eb6834"}
    for K, m in models.items():
        p = m.unpack(m.theta)
        sp = m._spatial(p, *m.mc)
        tot = np.full(len(edges) - 1, N * p["w"][0] / m.area)
        for k in range(m.ncomp):
            dens = np.exp(sp[k])
            mod = np.array([dens[(rmc >= a) & (rmc < b)].sum() / n_mc * m.area for a, b in zip(edges[:-1], edges[1:])])
            comp = N * p["w"][k + 1] * mod / np.where(area > 0, area, np.nan)
            tot = tot + comp
            if K == 2:
                ax.plot(rc * 60, comp, color=colors[2], lw=1, ls=":" if k == 1 else "--")
        ax.plot(rc * 60, tot, color=colors[K], lw=2, label=f"{K}-component model")
        ax.axhline(N * p["w"][0] / m.area, color="#9a9994", lw=1, ls="-.") if K == 1 else None
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("elliptical radius (arcmin)")
    ax.set_ylabel("stars per deg$^2$")
    ax.legend(fontsize=8, frameon=False)
    ax.set_title(f"{name}: surface density (dashed/dotted: main/halo; dash-dot: foreground)", fontsize=9)
    # members-only residual: weighted dwarf counts relative to the 1-component Plummer
    wd = m1.R_resp[:, 1:].sum(1)
    cw = np.histogram(r, edges, weights=wd)[0]
    sp1 = np.exp(m1._spatial(p1, *m1.mc)[0])
    mod1 = np.array([sp1[(rmc >= a) & (rmc < b)].sum() / n_mc * m1.area for a, b in zip(edges[:-1], edges[1:])])
    exp1 = N * p1["w"][1] * mod1
    good = ok & (exp1 > 0.5)
    bx.errorbar(rc[good] * 60, cw[good] / exp1[good], yerr=np.sqrt(np.maximum(cw[good], 1)) / exp1[good], fmt="o", ms=4,
                color="#2a78d6")
    bx.axhline(1, color="#9a9994", lw=1)
    bx.set_xscale("log")
    bx.set_yscale("log")
    bx.set_xlabel("elliptical radius (arcmin)")
    bx.set_ylabel("member-weighted counts / 1-Plummer model")
    bx.set_title("excess over a single Plummer (members, 1-component fit)", fontsize=9)
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(out, f"profile.{ext}"), dpi=160, bbox_inches="tight")
    plt.close(fig)


def map_cmd_plot(m, tgt, ctl, out, name):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    pd = m.R_resp[:, 1:].sum(1)
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.3))
    ax = axes[0]
    ax.scatter(tgt["xi"], tgt["eta"], s=0.5, c="#c9c8c3", lw=0, rasterized=True)
    s = pd > 0.5
    ax.scatter(tgt["xi"][s], tgt["eta"][s], s=1.5, c="#2a78d6", lw=0, rasterized=True)
    if m.ncomp == 2:
        h = m.R_resp[:, 2] > 0.5
        ax.scatter(tgt["xi"][h], tgt["eta"][h], s=4, c="#eb6834", lw=0, rasterized=True)
    ax.set_xlim(tgt["radius"], -tgt["radius"])
    ax.set_ylim(-tgt["radius"], tgt["radius"])
    ax.set_aspect("equal")
    ax.set_xlabel("ξ (deg)  ← east")
    ax.set_ylabel("η (deg)")
    ax.set_title(f"P(dwarf) > 0.5: {s.sum():,}" + (f";  P(halo) > 0.5: {h.sum():,}" if m.ncomp == 2 else ""), fontsize=9)
    for ax, (BR, G, ttl) in zip(axes[1:], [(ctl["BR"], ctl["G"], "control field (foreground)"),
                                          (tgt["BR"][s], tgt["G"][s], "members, P(dwarf) > 0.5")]):
        ax.scatter(BR, G, s=1, c="#52514e", lw=0, alpha=0.5, rasterized=True)
        ax.invert_yaxis()
        ax.set_xlabel("BP − RP")
        ax.set_ylabel("G")
        ax.set_title(ttl, fontsize=9)
    fig.suptitle(f"{name}: {m.ncomp}-component fit", fontsize=10)
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(out, f"map_cmd.{ext}"), dpi=160, bbox_inches="tight")
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
    cp = a.control or os.path.join(base, f"{cfg['name']}_control.fits")
    out = os.path.join(base, f"results_{cfg['name']}")
    os.makedirs(out, exist_ok=True)
    d, c = cfg["dwarf"], cfg["control"]
    t0 = time.time()
    tgt = load_field(tp, (d["ra"], d["dec"]), d["radius_deg"], cfg["cuts"])
    ctl = load_field(cp, (c["ra"], c["dec"]), c["radius_deg"], cfg["cuts"])
    print(f"target: {len(tgt['G']):,} of {tgt['n_raw']:,} stars pass cuts; control: {len(ctl['G']):,} of {ctl['n_raw']:,}")
    iters = a.iters or cfg["model"]["em_iterations"]
    models = {}
    for K in (1, 2):
        print(f"fitting {K}-component dwarf model")
        m = Model(tgt, ctl, cfg, K)
        if K == 2 and 1 in models:  # start the 2-component fit from the 1-component solution
            t1 = models[1].theta        # [w_main, plx, pmra, pmdec, log a, logit e, pa]
            m.theta[0] = t1[0]
            m.theta[1] = t1[0] - 2.5    # halo weight starts ~8% of the main body
            m.theta[2:8] = t1[1:7]
        models[K] = m.fit(iters)
    n = len(tgt["G"])
    res = {}
    for K, m in models.items():
        p = m.unpack(m.theta)
        k = len(m.theta)
        res[K] = dict(lnL=m.lnL, n_params=k, bic=-2 * m.lnL + k * np.log(n),
                      N_dwarf=float(m.R_resp[:, 1:].sum()), weights=p["w"].tolist(),
                      plx_mas=p["plx"], pmra=p["pmra"], pmdec=p["pmdec"],
                      a_arcmin=p["a"] * 60, ellipticity=p["e"], pa_deg=p["pa"] % 180)
        if K == 2:
            res[K].update(a_halo_arcmin=p["a_h"] * 60, e_halo=p["e_h"],
                          halo_fraction_of_dwarf=float(p["w"][2] / p["w"][1:].sum()),
                          N_halo=float(m.R_resp[:, 2].sum()))
    summary = dict(name=cfg["name"], n_target=n, n_control=len(ctl["G"]),
                   delta_lnL_2_minus_1=res[2]["lnL"] - res[1]["lnL"],
                   delta_BIC_1_minus_2=res[1]["bic"] - res[2]["bic"],
                   models=res, seconds=time.time() - t0,
                   caveat="dwarf CMD/log g histograms are rebuilt from memberships (EM); lnL values are "
                          "conditional on those histograms, so treat the comparison as indicative")
    json.dump(summary, open(os.path.join(out, "fit.json"), "w"), indent=2, default=float)
    m2 = models[2]
    T = Table(dict(source_id=tgt["id"], xi=tgt["xi"], eta=tgt["eta"], G=tgt["G"], bp_rp=tgt["BR"],
                   p_dwarf_1comp=models[1].R_resp[:, 1:].sum(1), p_main_2comp=m2.R_resp[:, 1],
                   p_halo_2comp=m2.R_resp[:, 2]))
    T.write(os.path.join(out, "members.fits"), overwrite=True)
    profile_plot(models, tgt, out, cfg["name"])
    map_cmd_plot(m2, tgt, ctl, out, cfg["name"])
    print(json.dumps({k: v for k, v in summary.items() if k != "models"}, indent=1, default=float))
    for K in (1, 2):
        print(K, {k: (round(v, 4) if isinstance(v, float) else v) for k, v in res[K].items()})


if __name__ == "__main__":
    main()
