"""Bootstrap assignment stability for the XD mixture, as in the paper's Figure 1(e, f).

For each of B bootstrap resamples, fit K = K_min..K_max (same resamples for every K, so
comparisons are paired). Each fit then labels ALL original stars (most probable component,
with per-star errors). Stability(K) = adjusted Rand index (ARI) between every pair of resamples.
Paired test: for each resample pair, ARI(K_a) - ARI(K_b).
Usage: python scripts/stability.py --config config_smc16k_raw.yaml --B 20 --K 3 4 5 6 7
"""
import argparse, itertools, os, sys, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from proto.data import load_config, load_catalog, Scaler, save_json
from proto import xd


def ari(a, b):
    """Adjusted Rand index (Hubert & Arabie 1985)."""
    _, ai = np.unique(a, return_inverse=True)
    _, bi = np.unique(b, return_inverse=True)
    C = np.zeros((ai.max() + 1, bi.max() + 1))
    np.add.at(C, (ai, bi), 1)
    c2 = lambda x: x * (x - 1) / 2
    s_ij = c2(C).sum()
    s_a, s_b = c2(C.sum(1)).sum(), c2(C.sum(0)).sum()
    n2 = c2(len(a))
    exp = s_a * s_b / n2
    mx = 0.5 * (s_a + s_b)
    return float((s_ij - exp) / (mx - exp)) if mx != exp else 1.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--K", type=int, nargs="+", default=[3, 4, 5, 6, 7])
    ap.add_argument("--B", type=int, default=20)
    ap.add_argument("--max_iter", type=int, default=300)
    ap.add_argument("--compare", type=int, nargs="*", default=[4, 5, 6, 7],
                    help="paired tests: K=4 against each of these")
    a = ap.parse_args()
    cfg = load_config(a.config)
    c = cfg["xd"]
    out = os.path.join(cfg["output_dir"], "stability")
    os.makedirs(out, exist_ok=True)
    cat = load_catalog(cfg)
    sc = Scaler().fit(cat["X"])
    Z, SZ = sc.x(cat["X"]), sc.cov(cat["S"])
    N = len(Z)
    rng = np.random.default_rng(c["seed"])
    idx = [rng.integers(0, N, N) for _ in range(a.B)]
    labels = {K: [] for K in a.K}
    t0 = time.time()
    for b, ib in enumerate(idx):
        for K in a.K:
            f = xd.fit(Z[ib], SZ[ib], K, a.max_iter, c["tol"], c["reg"], seed=c["seed"] + 7919 * b, chunk=c["chunk"])
            R = xd.responsibilities(Z, SZ, f["alpha"], f["mu"], f["V"], chunk=c["chunk"])
            labels[K].append(R.argmax(1))
        print(f"resample {b + 1}/{a.B} done ({time.time() - t0:.0f}s)", flush=True)
        np.savez(os.path.join(out, "labels.npz"), **{f"K{K}": np.array(labels[K]) for K in a.K})

    pairs = list(itertools.combinations(range(a.B), 2))
    A = {K: np.array([ari(labels[K][i], labels[K][j]) for i, j in pairs]) for K in a.K}
    rows = [dict(K=K, mean_ari=float(A[K].mean()), sd_ari=float(A[K].std(ddof=1)),
                 median_ari=float(np.median(A[K]))) for K in a.K]
    paired = []
    for Kb in a.compare:
        if Kb == 4 or 4 not in A or Kb not in A:
            continue
        d = A[4] - A[Kb]
        # pairs share resamples, so the naive SE is optimistic; report it alongside a jackknife over resamples
        jk = []
        for r in range(a.B):
            keep = [p for p, (i, j) in enumerate(pairs) if i != r and j != r]
            jk.append(d[keep].mean())
        jk = np.array(jk)
        se_jk = np.sqrt((a.B - 1) / a.B * ((jk - jk.mean()) ** 2).sum())
        paired.append(dict(vs=Kb, mean_diff=float(d.mean()), se_jackknife=float(se_jk),
                           frac_pairs_K4_better=float((d > 0).mean())))
    save_json(dict(B=a.B, n_pairs=len(pairs), N=N, rows=rows, paired_K4=paired,
                   seconds=time.time() - t0), os.path.join(out, "stability.json"))

    fig, ax = plt.subplots(figsize=(4.8, 3.4))
    for K in a.K:
        ax.scatter(np.full(len(A[K]), K) + np.random.default_rng(K).uniform(-0.12, 0.12, len(A[K])), A[K],
                   s=4, color="#9ec5f4", lw=0)
    ax.plot([r["K"] for r in rows], [r["mean_ari"] for r in rows], color="#2a78d6", marker="o", ms=6, lw=2,
            label="mean over pairs")
    ax.set_xlabel("K")
    ax.set_ylabel("pairwise ARI between resamples")
    ax.set_ylim(0, 1)
    ax.set_title(f"Assignment stability, N = {N:,}, B = {a.B} ({len(pairs)} pairs)", fontsize=9)
    ax.grid(color="#d9d8d4", lw=0.6)
    ax.legend(fontsize=8, frameon=False)
    fig.tight_layout()
    fig.savefig(os.path.join(out, "stability.png"), dpi=170)
    fig.savefig(os.path.join(out, "stability.pdf"))
    for r in rows:
        print(f"K={r['K']}: mean ARI {r['mean_ari']:.3f} (sd {r['sd_ari']:.3f})")
    for p in paired:
        print(f"K=4 vs K={p['vs']}: dARI = {p['mean_diff']:+.3f} +/- {p['se_jackknife']:.3f} (jackknife); "
              f"K=4 better in {100 * p['frac_pairs_K4_better']:.0f}% of pairs")


if __name__ == "__main__":
    main()
