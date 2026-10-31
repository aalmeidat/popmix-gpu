"""Corner-style comparison: observed stars, XD components (deconvolved), and the deconvolving flow.

Lower triangle: observed stars (gray) with XD component 2-sigma ellipses (intrinsic widths).
Upper triangle: density contours of the deconvolved flow (color) vs the XD mixture (black).
Diagonal: 1D distributions -- observed, XD intrinsic mixture, deconvolved flow (and plain flow if present).
Usage: python scripts/plot_compare.py --config config.yaml
"""
import argparse, os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
from proto.data import load_config, load_catalog
from proto import xd

COLORS = ["#2a6fdb", "#3c9d5d", "#e8a33d", "#7b5ea7", "#4f9da6", "#8c6d31", "#555555"]  # red reserved for the flow


def ellipse(ax, mean, cov, color, nsig=2.0):
    vals, vecs = np.linalg.eigh(cov)
    ang = np.degrees(np.arctan2(vecs[1, 1], vecs[0, 1]))
    w, h = 2 * nsig * np.sqrt(vals[::-1])
    ax.add_patch(Ellipse(mean, w, h, angle=ang, fill=False, lw=1.6, color=color))


def contour(ax, x, y, rng, color, levels=(0.39, 0.86), lw=1.2, ls="-"):
    H, xe, ye = np.histogram2d(x, y, bins=60, range=rng)
    from scipy.ndimage import gaussian_filter
    H = gaussian_filter(H.T, 1.0)
    Hs = np.sort(H.ravel())[::-1]
    cs = np.cumsum(Hs) / Hs.sum()
    lv = sorted({Hs[np.searchsorted(cs, l)] for l in levels})
    if len(lv) < 1 or lv[-1] <= 0:
        return
    xc, yc = 0.5 * (xe[1:] + xe[:-1]), 0.5 * (ye[1:] + ye[:-1])
    ax.contour(xc, yc, H, levels=lv, colors=color, linewidths=lw, linestyles=ls)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    cfg = load_config(a.config)
    od = cfg["output_dir"]
    cat = load_catalog(dict(cfg, catalog=dict(cfg["catalog"], missing="drop")))
    X, names, labels = cat["X"], cat["names"], cat["labels"]
    D = X.shape[1]
    m = np.load(os.path.join(od, "xd", "xd_best.npz"))
    alpha, mu_p, V_p = m["alpha"], m["mu_phys"], m["V_phys"]
    rng = np.random.default_rng(0)
    xs, _ = xd.sample(alpha, mu_p, V_p, 200000, rng)
    fpath = os.path.join(od, "flow", "flow_deconv_samples.npy")
    fs = np.load(fpath) if os.path.exists(fpath) else None
    ppath = os.path.join(od, "flow", "flow_plain_samples.npy")
    ps = np.load(ppath) if os.path.exists(ppath) else None

    lo = np.percentile(X, 0.5, 0) - 0.05 * np.ptp(X, 0)
    hi = np.percentile(X, 99.5, 0) + 0.05 * np.ptp(X, 0)
    fig, axes = plt.subplots(D, D, figsize=(2.6 * D, 2.6 * D))
    for i in range(D):
        for j in range(D):
            ax = axes[i, j]
            if i == j:
                bins = np.linspace(lo[i], hi[i], 60)
                ax.hist(X[:, i], bins, density=True, color="0.75", label="observed")
                ax.hist(xs[:, i], bins, density=True, histtype="step", color="k", lw=1.3, label="XD intrinsic")
                if fs is not None:
                    ax.hist(fs[:, i], bins, density=True, histtype="step", color="#d1495b", lw=1.5,
                            label="flow (deconvolved)")
                if ps is not None:
                    ax.hist(ps[:, i], bins, density=True, histtype="step", color="#2a6fdb", lw=1.0,
                            ls="--", label="flow (plain)")
                ax.set_yticks([])
            elif i > j:
                ax.scatter(X[:, j], X[:, i], s=1, c="0.7", alpha=0.3, rasterized=True)
                for k in range(len(alpha)):
                    sub = V_p[k][np.ix_([j, i], [j, i])]
                    ellipse(ax, mu_p[k, [j, i]], sub, COLORS[k % len(COLORS)])
            else:
                r = [[lo[j], hi[j]], [lo[i], hi[i]]]
                contour(ax, xs[:, j], xs[:, i], r, "k")
                if fs is not None:
                    contour(ax, fs[:, j], fs[:, i], r, "#d1495b", lw=1.5)
            if i != j:
                ax.set_xlim(lo[j], hi[j])
                ax.set_ylim(lo[i], hi[i])
            else:
                ax.set_xlim(lo[i], hi[i])
            if i == D - 1:
                ax.set_xlabel(labels[j])
            else:
                ax.set_xticklabels([])
            if j == 0 and i != 0:
                ax.set_ylabel(labels[i])
            elif i != j:
                ax.set_yticklabels([])
    axes[0, 0].legend(fontsize=7, loc="upper left", bbox_to_anchor=(0, 1.45), ncol=2, frameon=False)
    fig.suptitle(f"{len(X):,} stars: XD components (lower) and deconvolved flow vs XD (upper)", y=0.995, fontsize=11)
    fig.tight_layout()
    out = a.out or os.path.join(od, "compare_xd_flow.pdf")
    fig.savefig(out, dpi=200)
    fig.savefig(out.replace(".pdf", ".png"), dpi=150)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
