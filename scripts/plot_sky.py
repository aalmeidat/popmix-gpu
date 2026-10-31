"""Where the mixture components sit in space: sky maps and depth, from the XD membership catalog.

Figures (in output_dir/):
  sky_components.pdf/.png    one panel per component: hard members over the full sample,
                             SMC-centered orthographic X/Y (east to the left), LMC direction marked
  depth_components.pdf/.png  X (east-west) vs per-star distance, and distance distributions
Positions are not used in the fit, so spatial coherence here is a result.
Usage: python scripts/plot_sky.py --config config_smc16k_raw_K4.yaml --dist Dist
"""
import argparse, json, os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from astropy.table import Table
from proto.data import load_config

# validated categorical order (dataviz reference palette, light mode)
SLOTS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6"]
INK, INK2, GRID = "#0b0b0b", "#52514e", "#d9d8d4"
RA0, DEC0 = 13.19, -72.83          # adopted SMC center (paper, Sec. 2.2)
LMC_PA = 115.0                     # direction to the LMC, deg east of north (paper)


def ortho(ra, dec, ra0=RA0, dec0=DEC0):
    a, d = np.radians(ra), np.radians(dec)
    a0, d0 = np.radians(ra0), np.radians(dec0)
    X = np.cos(d) * np.sin(a - a0)
    Y = np.sin(d) * np.cos(d0) - np.cos(d) * np.sin(d0) * np.cos(a - a0)
    return np.degrees(X), np.degrees(Y)


def name_components(comps):
    """Descriptive labels from fitted properties (heuristic, for display only)."""
    K = len(comps)
    names = [None] * K
    sf = [c["sigma_feh"] for c in comps]
    broad = int(np.argmax(sf)) if max(sf) > 2.5 * np.median(sf) else None
    rest = [k for k in range(K) if k != broad]
    fg = max(rest, key=lambda k: comps[k]["mean_pmra"])
    rest = [k for k in rest if k != fg]
    halo = min(rest, key=lambda k: comps[k]["mean_feh"])
    rest = [k for k in rest if k != halo]
    names[fg], names[halo] = "Foreground-like", "Halo-like"
    if broad is not None:
        names[broad] = "Broad [Fe/H]"
    # naming heuristic for SMC-like samples: components more metal-poor than this are labeled metal-poor
    mp = [k for k in rest if comps[k]["mean_feh"] < -1.28]
    for i, k in enumerate(sorted(mp, key=lambda k: comps[k]["mean_feh"])):
        names[k] = "Metal-poor, extended" if len(mp) == 1 else f"Metal-poor {i + 1}"
    rest = [k for k in rest if k not in mp]
    rest.sort(key=lambda k: comps[k]["mean_vlos"])
    for i, k in enumerate(rest):
        names[k] = "Main body" if len(rest) == 1 else f"Main body {chr(97 + i)}"
    rest = mp + rest
    order = [fg, halo] + rest + ([broad] if broad is not None else [])
    return names, order


def style(ax):
    for s in ax.spines.values():
        s.set_color(GRID)
    ax.tick_params(colors=INK2, labelsize=8)
    ax.set_facecolor("#fcfcfb")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--ra", default="RACAT")
    ap.add_argument("--dec", default="DECCAT")
    ap.add_argument("--dist", default="Dist", help="per-star distance column [kpc]; '' to skip depth figure")
    ap.add_argument("--pmin", type=float, default=0.0, help="only plot members with max probability above this")
    a = ap.parse_args()
    cfg = load_config(a.config)
    od = cfg["output_dir"]
    cc = cfg["catalog"]
    t = Table.read(cc["path"], format=cc.get("format")) if cc.get("format") else Table.read(cc["path"])
    m = Table.read(os.path.join(od, "xd", "membership.fits"))
    summ = json.load(open(os.path.join(od, "xd", "xd_summary.json")))
    comps = summ["components"]
    K = len(comps)
    if len(m) != len(t):
        if cc.get("id_column"):
            pos = {v: i for i, v in enumerate(np.asarray(t[cc["id_column"]]))}
            t = t[[pos[v] for v in np.asarray(m["id"])]]
        else:                      # ids are row indices into the input file
            t = t[np.asarray(m["id"], int)]
    X, Y = ortho(np.asarray(t[a.ra], float), np.asarray(t[a.dec], float))
    P = np.column_stack([m[f"p_comp{k}"] for k in range(K)])
    best, pmax = P.argmax(1), P.max(1)
    names, order = name_components(comps)
    color = {k: SLOTS[i % len(SLOTS)] for i, k in enumerate(order)}

    # ---- sky maps
    ncol = 4 if K > 4 else K
    nrow = int(np.ceil(K / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(3.2 * ncol, 3.3 * nrow), squeeze=False)
    lim = 9.0
    for i, k in enumerate(order):
        ax = axes.flat[i]
        style(ax)
        ax.scatter(X, Y, s=1, c="#c9c8c3", lw=0, rasterized=True)
        sel = (best == k) & (pmax > a.pmin)
        ax.scatter(X[sel], Y[sel], s=2.5, c=color[k], lw=0, alpha=0.8, rasterized=True)
        ax.plot(0, 0, "+", color=INK, ms=8, mew=1.2)
        u = np.radians(LMC_PA)
        ax.annotate("", xy=(6.5 * np.sin(u), 6.5 * np.cos(u)), xytext=(4.0 * np.sin(u), 4.0 * np.cos(u)),
                    arrowprops=dict(arrowstyle="->", color=INK2, lw=1.0, ls="--"))
        ax.text(6.8 * np.sin(u), 6.8 * np.cos(u) - 0.6, "LMC", color=INK2, fontsize=7, ha="center")
        ax.set_xlim(lim, -lim)
        ax.set_ylim(-lim, lim)
        ax.set_aspect("equal")
        ax.set_title(f"{names[k]}  (N = {sel.sum():,}, w = {comps[k]['weight']:.2f})\n"
                     f"[Fe/H] = {comps[k]['mean_feh']:.2f},  \u03c3v = {comps[k]['sigma_vlos']:.0f} km/s,  "
                     f"\u03bc\u03b1* = {comps[k]['mean_pmra']:.2f}", fontsize=8.5, color=INK)
        if i % ncol == 0:
            ax.set_ylabel("Y (deg)", color=INK2, fontsize=9)
        if i >= K - ncol:
            ax.set_xlabel("X (deg)  ← east", color=INK2, fontsize=9)
    for j in range(K, nrow * ncol):
        axes.flat[j].axis("off")
    cut = f", max P > {a.pmin:g}" if a.pmin > 0 else ""
    fig.suptitle(f"Sky distribution of the K = {K} components (hard members{cut}); positions not used in the fit",
                 fontsize=10, color=INK)
    fig.tight_layout()
    tag = f"_p{a.pmin:g}" if a.pmin > 0 else ""
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(od, f"sky_components{tag}.{ext}"), dpi=170, bbox_inches="tight")
    plt.close(fig)

    # ---- depth
    if a.dist:
        D = np.asarray(t[a.dist], float)
        fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 4.2), gridspec_kw=dict(width_ratios=[1.4, 1]))
        style(a1)
        style(a2)
        bins = np.linspace(np.nanpercentile(D, 0.5), np.nanpercentile(D, 99.5), 50)
        xb = np.linspace(-7, 7, 15)
        for k in order:
            sel = (best == k) & (pmax > a.pmin)
            # median distance in X bins, with interquartile band
            xc, med, lo, hi = [], [], [], []
            for x0, x1 in zip(xb[:-1], xb[1:]):
                s = sel & (X >= x0) & (X < x1)
                if s.sum() >= 15:
                    q = np.nanpercentile(D[s], [25, 50, 75])
                    xc.append(0.5 * (x0 + x1)); lo.append(q[0]); med.append(q[1]); hi.append(q[2])
            a1.fill_between(xc, lo, hi, color=color[k], alpha=0.15, lw=0)
            a1.plot(xc, med, color=color[k], lw=2, marker="o", ms=4, label=names[k])
            a2.hist(D[sel], bins, histtype="step", density=True, lw=2, color=color[k], label=names[k])
        a1.set_xlim(7, -7)
        a1.set_xlabel("X (deg)  ← east (toward the LMC)", color=INK2, fontsize=9)
        a1.set_ylabel(f"median {a.dist} (kpc), 25–75% band", color=INK2, fontsize=9)
        a1.grid(color=GRID, lw=0.6)
        a1.legend(fontsize=8, frameon=False, labelcolor=INK)
        a2.set_xlabel(f"{a.dist} (kpc)", color=INK2, fontsize=9)
        a2.set_yticks([])
        a2.set_title("distance distributions (hard members)", fontsize=9, color=INK)
        fig.suptitle(f"Depth: per-star spectrophotometric distances by component (K = {K})", fontsize=10, color=INK)
        fig.tight_layout()
        for ext in ("pdf", "png"):
            fig.savefig(os.path.join(od, f"depth_components{tag}.{ext}"), dpi=170, bbox_inches="tight")
        plt.close(fig)
    print("labels:", {names[k]: int(((best == k) & (pmax > a.pmin)).sum()) for k in order})


if __name__ == "__main__":
    main()
