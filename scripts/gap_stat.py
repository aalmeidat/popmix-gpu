"""Likelihood gap statistic for the XD mixture (Tibshirani et al. 2001, with log-likelihood in place of log W).

Gap(K) = (1/N) [ log L_data(K) - mean_b log L_ref,b(K) ]
Reference sets: N points drawn uniformly in the PCA-aligned bounding box of the standardized data,
each blurred by a randomly assigned real per-star error covariance, then fit with the same XD
settings as the data. Rule: smallest K with Gap(K) >= Gap(K+1) - s(K+1), s = sd * sqrt(1 + 1/B).

This is my implementation; it may differ in detail from the gap statistic used in the paper.
Usage: python scripts/gap_stat.py --config config_smc16k_raw.yaml --B 10 --K 2 3 4 5 6 7
"""
import argparse, os, sys, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from proto.data import load_config, load_catalog, Scaler, save_json
from proto import xd


def best_fit(Z, S, K, inits, max_iter, tol, reg, seed, chunk):
    fits = [xd.fit(Z, S, K, max_iter, tol, reg, seed=seed + i, chunk=chunk) for i in range(inits)]
    return max(f["logL"] for f in fits)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--K", type=int, nargs="+", default=[2, 3, 4, 5, 6, 7])
    ap.add_argument("--B", type=int, default=10)
    ap.add_argument("--inits", type=int, default=2)
    ap.add_argument("--max_iter", type=int, default=300)
    a = ap.parse_args()
    cfg = load_config(a.config)
    c = cfg["xd"]
    out = os.path.join(cfg["output_dir"], "gap")
    os.makedirs(out, exist_ok=True)
    cat = load_catalog(cfg)
    sc = Scaler().fit(cat["X"])
    Z, SZ = sc.x(cat["X"]), sc.cov(cat["S"])
    N, D = Z.shape
    rng = np.random.default_rng(c["seed"])
    # PCA-aligned box
    U, s, Vt = np.linalg.svd(Z - Z.mean(0), full_matrices=False)
    R = (Z - Z.mean(0)) @ Vt.T
    lo, hi = R.min(0), R.max(0)
    t0 = time.time()
    data_ll = {}
    for K in a.K:
        data_ll[K] = best_fit(Z, SZ, K, a.inits, a.max_iter, c["tol"], c["reg"], c["seed"], c["chunk"])
        print(f"data K={K}: logL={data_ll[K]:.1f}  ({time.time() - t0:.0f}s)", flush=True)
    ref_ll = {K: [] for K in a.K}
    for b in range(a.B):
        Rb = rng.uniform(lo, hi, size=(N, D))
        Sb = SZ[rng.integers(0, N, N)]
        Lb = np.linalg.cholesky(Sb + 1e-12 * np.eye(D))
        Zb = Rb @ Vt + Z.mean(0) + np.einsum("nij,nj->ni", Lb, rng.normal(size=(N, D)))
        for K in a.K:
            ref_ll[K].append(best_fit(Zb, Sb, K, a.inits, a.max_iter, c["tol"], c["reg"], c["seed"] + 1000 * b, c["chunk"]))
        print(f"reference {b + 1}/{a.B} done ({time.time() - t0:.0f}s)", flush=True)
    rows = []
    for K in a.K:
        r = np.array(ref_ll[K]) / N
        gap = data_ll[K] / N - r.mean()
        sk = r.std(ddof=1) * np.sqrt(1 + 1 / a.B) if a.B > 1 else np.nan
        rows.append(dict(K=K, gap=float(gap), s=float(sk), data_mean_logL=data_ll[K] / N, ref_mean_logL=float(r.mean())))
    pick = None
    for i in range(len(rows) - 1):
        if rows[i]["gap"] >= rows[i + 1]["gap"] - rows[i + 1]["s"]:
            pick = rows[i]["K"]
            break
    save_json(dict(B=a.B, inits=a.inits, max_iter=a.max_iter, N=N, rows=rows, tibshirani_K=pick,
                   seconds=time.time() - t0), os.path.join(out, "gap.json"))
    fig, ax = plt.subplots(figsize=(4.6, 3.4))
    Ks = [r["K"] for r in rows]
    ax.errorbar(Ks, [r["gap"] for r in rows], yerr=[r["s"] for r in rows], color="#2a78d6", marker="o",
                ms=6, lw=2, capsize=3)
    ax.set_xlabel("K")
    ax.set_ylabel("Gap(K)  [nats per star]")
    ax.set_title(f"Likelihood gap statistic, N = {N:,}, B = {a.B}", fontsize=9)
    ax.grid(color="#d9d8d4", lw=0.6)
    fig.tight_layout()
    fig.savefig(os.path.join(out, "gap.png"), dpi=170)
    fig.savefig(os.path.join(out, "gap.pdf"))
    for r in rows:
        print(f"K={r['K']}: Gap={r['gap']:.5f} +/- {r['s']:.5f}")
    print("Tibshirani rule picks K =", pick)


if __name__ == "__main__":
    main()
