"""Dwarf halo search, version 2, on the GPU (PyTorch): the same model as fit_dwarf2.py, with exact gradients.

What changes relative to fit_dwarf2.py (CPU, scipy L-BFGS-B with finite-difference gradients):
  * The likelihood is evaluated in PyTorch on the GPU and minimized with torch L-BFGS (strong Wolfe line
    search) using autograd gradients, instead of ~(n_params + 1) likelihood evaluations per gradient.
  * Bug fix: the halo CMD is the main-body CMD shifted in BP-RP by dcol. In fit_dwarf2.py the CMD is a
    nearest-cell histogram lookup, which is piecewise constant in dcol, so the finite-difference gradient
    was zero and dcol never left its starting value (0.000 in every CPU run). Here the CMD and log g
    histograms are read with bilinear (linear) interpolation, so dcol is actually fitted.
  * --compat reproduces the CPU model (nearest-cell lookup, dcol frozen at 0), to check the port against
    the CPU fit_v2.json results.
Everything else (data cuts, local foreground template, planar foreground gradient, bounded halo scale,
main-body CMD from the core, 5 EM-style histogram updates) is inherited from fit_dwarf2.py unchanged.

Usage: python dwarf/fit_dwarf_gpu.py --config dwarf/sculptor8.yaml [--target X.fits] [--compat] [--tag gpu]
Writes results_<name>/fit_<tag>.json (and members_<tag>.fits).
"""
import argparse, json, os, sys, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import torch
import yaml
from astropy.table import Table
from scipy.special import logsumexp
from fit_dwarf import Hist2D, LoggTerm, ell_radius
from fit_dwarf2 import Model as CPUModel, load_field

LOG2PI = np.log(2 * np.pi)

# The foreground astrometry template (an XD fit to the outer annulus) is identical for the 1- and
# 2-component models; memoize it so it is fitted once per field instead of twice.
from proto import xd as _xd
_xd_fit, _xd_memo = _xd.fit, {}


def _xd_fit_memo(X, S, K, *args, **kw):
    key = (X.shape, float(np.asarray(X).sum()), float(np.asarray(S).sum()), K, kw.get("seed"))
    if key not in _xd_memo:
        _xd_memo[key] = _xd_fit(X, S, K, *args, **kw)
    return _xd_memo[key]


_xd.fit = _xd_fit_memo


def sync(dev):
    if dev.type == "cuda":
        torch.cuda.synchronize()


class GridLookup:
    """A 2-D log-density grid (Hist2D) read at continuous positions: nearest cell or bilinear on cell centers."""

    def __init__(self, h, dev, mode):
        self.logH = torch.as_tensor(h.logH, dtype=torch.float64, device=dev)
        self.x0, self.dx = float(h.xe[0]), float(h.xe[1] - h.xe[0])
        self.y0, self.dy = float(h.ye[0]), float(h.ye[1] - h.ye[0])
        self.nx, self.ny = h.logH.shape
        self.mode = mode

    def __call__(self, x, y):
        if self.mode == "nearest":
            i = torch.clamp(torch.floor((x - self.x0) / self.dx), 0, self.nx - 1).long()
            j = torch.clamp(torch.floor((y - self.y0) / self.dy), 0, self.ny - 1).long()
            return self.logH[i, j]
        fx = torch.clamp((x - self.x0) / self.dx - 0.5, 0, self.nx - 1)
        fy = torch.clamp((y - self.y0) / self.dy - 0.5, 0, self.ny - 1)
        i0 = torch.clamp(torch.floor(fx), 0, self.nx - 2).long(); j0 = torch.clamp(torch.floor(fy), 0, self.ny - 2).long()
        wx, wy = fx - i0, fy - j0
        H = self.logH
        return ((1 - wx) * (1 - wy) * H[i0, j0] + wx * (1 - wy) * H[i0 + 1, j0]
                + (1 - wx) * wy * H[i0, j0 + 1] + wx * wy * H[i0 + 1, j0 + 1])


def logg_values(term, logg, mode):
    """Per-star log probability from LoggTerm, with nearest or linear interpolation over the bins."""
    if mode == "nearest":
        return term(logg)
    out = np.full(len(logg), term.lp_miss)
    m = np.isfinite(logg)
    c = 0.5 * (term.edges[1:] + term.edges[:-1])
    out[m] = np.interp(np.clip(logg[m], 0, 5.499), c, term.lp_bin)
    return out


class GPUModel(CPUModel):
    """fit_dwarf2.Model with the likelihood in PyTorch. __init__ (foreground XD template, theta init) is inherited."""

    def __init__(self, tgt, tmpl, cfg, ncomp, dev, compat=False):
        super().__init__(tgt, tmpl, cfg, ncomp)
        self.dev, self.compat = dev, compat
        self.mode = "nearest" if compat else "linear"
        T = lambda v: torch.as_tensor(np.asarray(v), dtype=torch.float64, device=dev)
        t = tgt
        self.xi, self.eta, self.BR, self.G = T(t["xi"]), T(t["eta"]), T(t["BR"]), T(t["G"])
        self.A = T(t["A"])
        Ctot = t["C"] + np.diag([1e-8, self.s_int ** 2, self.s_int ** 2])[None]
        L = torch.linalg.cholesky(T(Ctot))
        self.Linv = torch.linalg.inv(L)                                   # fixed: only the mean depends on theta
        self.LinvA = torch.einsum("nij,nj->ni", self.Linv, self.A)
        self.lp_ast_const = -torch.log(torch.diagonal(L, dim1=1, dim2=2)).sum(1) - 1.5 * LOG2PI
        self.fg_ns = T(self.lp_fg_nonspatial)
        self.mcx, self.mcy = T(self.mc[0]), T(self.mc[1])
        self.n_evals = 0

    # ------------------------------------------------------------------ torch versions of the model pieces
    def tunpack(self, th):
        nc = self.ncomp
        z = torch.cat([th.new_zeros(1), th[:nc]])
        p = dict(lw=z - torch.logsumexp(z, 0), plx=th[nc], pmra=th[nc + 1], pmdec=th[nc + 2], a=torch.exp(th[nc + 3]),
                 e=torch.sigmoid(th[nc + 4]), pa=th[nc + 5], g=th[-self.NG:])
        if nc == 2:
            p.update(a_h=p["a"] + (0.5 * self.R - p["a"]) * torch.sigmoid(th[nc + 6]), e_h=torch.sigmoid(th[nc + 7]),
                     dcol=th.new_zeros(()) if self.compat else self.DMAX * torch.tanh(th[nc + 8]))
        return p

    @staticmethod
    def tplummer(xi, eta, a, e, pa):
        p = torch.deg2rad(pa)
        xmaj = xi * torch.sin(p) + eta * torch.cos(p)
        xmin = xi * torch.cos(p) - eta * torch.sin(p)
        R2 = xmaj ** 2 + (xmin / (1 - e)) ** 2
        return -torch.log(np.pi * a ** 2 * (1 - e)) - 2 * torch.log1p(R2 / a ** 2)

    def tcomp_logl(self, th):
        p = self.tunpack(th)
        mean = torch.stack([p["plx"], p["pmra"], p["pmdec"]])
        z = self.LinvA - self.Linv @ mean
        lp_ast = -0.5 * (z ** 2).sum(1) + self.lp_ast_const
        gx, gy = p["g"][0], p["g"][1]
        fg_norm = torch.log(torch.exp(gx * self.mcx + gy * self.mcy).mean() * self.area)
        fg_sp = gx * self.xi + gy * self.eta - fg_norm

        def sp(a, e):
            norm = torch.log(torch.exp(self.tplummer(self.mcx, self.mcy, a, e, p["pa"])).mean() * self.area)
            return self.tplummer(self.xi, self.eta, a, e, p["pa"]) - norm

        L = [p["lw"][0] + fg_sp + self.fg_ns,
             p["lw"][1] + sp(p["a"], p["e"]) + lp_ast + self.cmd_main + self.lg]
        if self.ncomp == 2:
            L.append(p["lw"][2] + sp(p["a_h"], p["e_h"]) + lp_ast + self.cmd_lut(self.BR - p["dcol"], self.G) + self.lg)
        return torch.stack(L, 1)

    def set_hists(self, w_main, w_halo=None):
        super().set_hists(w_main, w_halo)           # builds self.d_cmd (Hist2D) and self.d_logg (LoggTerm) in numpy
        self.cmd_lut = GridLookup(self.d_cmd, self.dev, self.mode)
        self.cmd_main = self.cmd_lut(self.BR, self.G).detach()
        self.lg = torch.as_tensor(logg_values(self.d_logg, self.t["logg"], self.mode), dtype=torch.float64, device=self.dev)

    def comp_logl(self, th):                         # numpy interface, used by the inherited plots
        with torch.no_grad():
            return self.tcomp_logl(torch.as_tensor(th, dtype=torch.float64, device=self.dev)).cpu().numpy()

    def loss(self, th):
        self.n_evals += 1
        return -torch.logsumexp(self.tcomp_logl(th), 1).mean()

    def fit(self, iters, verbose=True):
        t, d = self.t, self.cfg["dwarf"]
        r = ell_radius(t["xi"], t["eta"], d["ellipticity"], d["pa_deg"])
        dpm = np.hypot(t["A"][:, 1] - d["pmra"], t["A"][:, 2] - d["pmdec"])
        spm = np.sqrt(t["C"][:, 1, 1] + t["C"][:, 2, 2])
        w_main = ((r < 3 * d["a_plummer_deg"]) & (dpm < 3 * spm + 0.2)
                  & (np.abs(t["A"][:, 0]) < 3 * np.sqrt(t["C"][:, 0, 0]) + 0.05)).astype(float)
        w_halo, hist = None, []
        th = torch.tensor(self.theta, dtype=torch.float64, device=self.dev, requires_grad=True)
        self.t_opt = 0.0
        for it in range(iters):
            self.set_hists(w_main, w_halo)
            opt = torch.optim.LBFGS([th], lr=1.0, max_iter=500, max_eval=1250, tolerance_grad=1e-7,
                                    tolerance_change=1e-10, history_size=30, line_search_fn="strong_wolfe")

            def closure():
                opt.zero_grad()
                f = self.loss(th)
                f.backward()
                return f

            sync(self.dev); t0 = time.time()
            opt.step(closure)
            sync(self.dev); self.t_opt += time.time() - t0
            self.theta = th.detach().cpu().numpy().copy()
            L = self.comp_logl(self.theta)
            lnl = logsumexp(L, axis=1)
            R = np.exp(L - lnl[:, None])
            p = self.unpack(self.theta)
            if self.ncomp == 2 and self.compat:
                p["dcol"] = 0.0
            w_main = R[:, 1] * (ell_radius(t["xi"], t["eta"], p["e"], p["pa"]) < 3 * p["a"])
            w_halo = R[:, 2] if self.ncomp == 2 else None
            hist.append(float(lnl.sum()))
            if verbose:
                print(f"  K={self.ncomp} it={it} lnL={lnl.sum():.1f}  N_dwarf={R[:, 1:].sum():.0f}  a={p['a'] * 60:.2f}'"
                      f"  fg grad=({p['g'][0]:+.3f},{p['g'][1]:+.3f})/deg"
                      + (f"  a_h={p['a_h'] * 60:.1f}'  f_halo={p['w'][2] / p['w'][1:].sum():.3f}  dcol={p['dcol']:+.3f}"
                         if self.ncomp == 2 else "") + f"  [{self.n_evals} evals so far]", flush=True)
        self.R_resp, self.lnL, self.history = R, float(lnl.sum()), hist
        return self


def summarize(models, n, R_fit):
    res = {}
    for K, m in models.items():
        p = m.unpack(m.theta)
        if K == 2 and m.compat:
            p["dcol"] = 0.0
        res[K] = dict(lnL=m.lnL, n_params=len(m.theta) - (1 if (K == 2 and m.compat) else 0),
                      N_dwarf=float(m.R_resp[:, 1:].sum()), weights=p["w"].tolist(), plx_mas=p["plx"], pmra=p["pmra"],
                      pmdec=p["pmdec"], a_arcmin=p["a"] * 60, ellipticity=p["e"], pa_deg=p["pa"] % 180,
                      fg_gradient=p["g"].tolist(), n_evals=m.n_evals, seconds_optimizer=m.t_opt)
        res[K]["bic"] = -2 * m.lnL + res[K]["n_params"] * np.log(n)
        if K == 2:
            res[K].update(a_halo_arcmin=p["a_h"] * 60, e_halo=p["e_h"], at_bound=bool(p["a_h"] > 0.98 * 0.5 * R_fit),
                          halo_color_shift=float(p["dcol"]), halo_fraction_of_dwarf=float(p["w"][2] / p["w"][1:].sum()),
                          N_halo=float(m.R_resp[:, 2].sum()))
    return res


def run(cfg, target, compat, iters=None, dev=None, verbose=True):
    """Fit the 1- and 2-component models. Returns (models, summary dict)."""
    dev = dev or (torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu"))
    d, fgc = cfg["dwarf"], cfg["foreground"]
    lc = cfg["model"].get("logg_column", "logg_gspphot")
    t0 = time.time()
    tgt = load_field(target, (d["ra"], d["dec"]), 0.0, fgc["fit_radius_deg"], cfg["cuts"], lc)
    tmpl = load_field(target, (d["ra"], d["dec"]), fgc["fit_radius_deg"], d["radius_deg"], cfg["cuts"], lc)
    t_load = time.time() - t0
    if verbose:
        print(f"fit region: {len(tgt['G']):,} stars; foreground template: {len(tmpl['G']):,}; device {dev}"
              f"; mode {'compat (nearest, dcol=0)' if compat else 'linear, dcol free'}", flush=True)
    iters = iters or cfg["model"]["em_iterations"]
    models = {}
    for K in (1, 2):
        sync(dev); tk = time.time()
        m = GPUModel(tgt, tmpl, cfg, K, dev, compat)
        sync(dev); t_init = time.time() - tk
        if K == 2:
            t1 = models[1].theta
            m.theta[0] = t1[0]; m.theta[1] = t1[0] - 2.5
            m.theta[2:8] = t1[1:7]; m.theta[-CPUModel.NG:] = t1[-CPUModel.NG:]
        if verbose:
            print(f"fitting {K}-component model (setup incl. foreground XD: {t_init:.1f}s)", flush=True)
        models[K] = m.fit(iters, verbose)
        models[K].t_init = t_init
    n = len(tgt["G"])
    res = summarize(models, n, tgt["radius"])
    summary = dict(name=cfg["name"], version="2-gpu", mode="compat" if compat else "linear", device=str(dev),
                   gpu=torch.cuda.get_device_name(0) if dev.type == "cuda" else "cpu", n_fit=n, n_template=len(tmpl["G"]),
                   delta_lnL_2_minus_1=res[2]["lnL"] - res[1]["lnL"], delta_BIC_1_minus_2=res[1]["bic"] - res[2]["bic"],
                   seconds_load=t_load, seconds_setup=sum(m.t_init for m in models.values()),
                   seconds_optimizer=sum(m.t_opt for m in models.values()), seconds_total=time.time() - t0, models=res)
    return models, summary, tgt, tmpl


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--target")
    ap.add_argument("--compat", action="store_true", help="reproduce the CPU model: nearest-cell lookup, dcol = 0")
    ap.add_argument("--iters", type=int, default=None)
    ap.add_argument("--tag", default=None)
    ap.add_argument("--reference", default=None, help="CPU fit_v2.json to compare against")
    ap.add_argument("--plots", action="store_true")
    a = ap.parse_args()
    cfg = yaml.safe_load(open(a.config))
    base = os.path.dirname(os.path.abspath(a.config))
    tp = a.target or os.path.join(base, f"{cfg['name']}_target.fits")
    tag = a.tag or ("gpu_compat" if a.compat else "gpu")
    out = os.path.join(base, f"results_{cfg['name']}")
    os.makedirs(out, exist_ok=True)
    models, summary, tgt, tmpl = run(cfg, tp, a.compat, a.iters)
    if a.reference and os.path.exists(a.reference):
        ref = json.load(open(a.reference))
        cmp = {}
        for K in ("1", "2"):
            r, g = ref["models"][K], summary["models"][int(K)]
            cmp[K] = dict(lnL_cpu=r["lnL"], lnL_gpu=g["lnL"], dlnL_gpu_minus_cpu=g["lnL"] - r["lnL"],
                          a_arcmin=(r["a_arcmin"], g["a_arcmin"]), N_dwarf=(r["N_dwarf"], g["N_dwarf"]))
            if K == "2":
                cmp[K].update(a_halo_arcmin=(r["a_halo_arcmin"], g["a_halo_arcmin"]),
                              halo_fraction=(r["halo_fraction_of_dwarf"], g["halo_fraction_of_dwarf"]),
                              dcol=(r["halo_color_shift"], g["halo_color_shift"]))
        summary["compare_to_cpu"] = dict(reference=os.path.relpath(a.reference, base), cpu_seconds=ref.get("seconds"), models=cmp)
        print("comparison with the CPU run (cpu, gpu):")
        print(json.dumps(cmp, indent=1, default=float))
    json.dump(summary, open(os.path.join(out, f"fit_{tag}.json"), "w"), indent=2, default=float)
    m2 = models[2]
    Table(dict(source_id=tgt["id"], xi=tgt["xi"], eta=tgt["eta"], G=tgt["G"], bp_rp=tgt["BR"],
               p_dwarf_1comp=models[1].R_resp[:, 1:].sum(1), p_main_2comp=m2.R_resp[:, 1],
               p_halo_2comp=m2.R_resp[:, 2])).write(os.path.join(out, f"members_{tag}.fits"), overwrite=True)
    if a.plots:
        from fit_dwarf2 import plots
        plots(models, tgt, tmpl, out, cfg["name"], cfg)
    print(json.dumps({k: v for k, v in summary.items() if k not in ("models", "compare_to_cpu")}, indent=1, default=float))
    for K in (1, 2):
        print(K, {k: (round(float(v), 4) if isinstance(v, (float, np.floating)) else v)
                  for k, v in summary["models"][K].items()})


if __name__ == "__main__":
    main()
