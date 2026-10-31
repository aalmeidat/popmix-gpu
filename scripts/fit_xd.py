"""Fit the error-aware mixture (extreme deconvolution) on GPU: BIC over K, restarts, bootstrap.

Outputs (in output_dir/xd/):
  xd_best.npz           best model (physical units) + scaler
  xd_summary.json       BIC table, timings, GPU name, bootstrap spreads
  membership.fits       per-star component probabilities
Usage: python scripts/fit_xd.py --config config.yaml
"""
import argparse, os, sys, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import torch
from astropy.table import Table
from proto.data import load_config, load_catalog, Scaler, save_json
from proto import xd


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--quick", action="store_true", help="few restarts/bootstraps, for testing")
    a = ap.parse_args()
    cfg = load_config(a.config)
    c = cfg["xd"]
    if a.quick:
        c = dict(c, n_inits=2, n_bootstrap=3, max_iter=150)
    out = os.path.join(cfg["output_dir"], "xd")
    os.makedirs(out, exist_ok=True)

    dev = xd.device_auto()
    gpu = torch.cuda.get_device_name(0) if dev.type == "cuda" else "cpu"
    cat = load_catalog(cfg)
    sc = Scaler().fit(cat["X"])
    Z, SZ = sc.x(cat["X"]), sc.cov(cat["S"])
    print(f"device={gpu}  stars kept {cat['n_kept']}/{cat['n_input']}  dims={cat['names']}")

    t_all = time.time()
    table, best = [], None
    for K in c["K_values"]:
        fits = [xd.fit(Z, SZ, K, c["max_iter"], c["tol"], c["reg"], seed=c["seed"] + i,
                       device=dev, chunk=c["chunk"]) for i in range(c["n_inits"])]
        f = min(fits, key=lambda r: r["bic"])
        table.append(dict(K=K, bic=f["bic"], logL=f["logL"], n_iter=f["n_iter"],
                          seconds_per_init=float(np.mean([r["seconds"] for r in fits])),
                          bic_spread_over_inits=float(np.ptp([r["bic"] for r in fits]))))
        print(f"K={K}: BIC={f['bic']:.1f}  iters={f['n_iter']}  {table[-1]['seconds_per_init']:.2f}s/init")
        if best is None or f["bic"] < best["bic"]:
            best, bestK = f, K

    # bootstrap the best model
    rng = np.random.default_rng(c["seed"])
    n = len(Z)
    boots = []
    t_b = time.time()
    for b in range(c["n_bootstrap"]):
        i = rng.integers(0, n, n)
        f = xd.fit(Z[i], SZ[i], bestK, c["max_iter"], c["tol"], c["reg"], device=dev,
                   chunk=c["chunk"], init=(best["alpha"], best["mu"], best["V"]))
        perm = xd.match_components(best["mu"], f["mu"])
        boots.append((f["alpha"][perm], f["mu"][perm], f["V"][perm]))
    t_boot = time.time() - t_b

    # physical units
    mu_p = sc.inv_x(best["mu"])
    V_p = np.array([sc.inv_cov(v) for v in best["V"]])
    comp_rows = []
    for k in range(bestK):
        row = dict(weight=float(best["alpha"][k]))
        if boots:
            aw = np.array([bb[0][k] for bb in boots])
            am = np.array([sc.inv_x(bb[1][k]) for bb in boots])
            ad = np.array([np.sqrt(np.diag(sc.inv_cov(bb[2][k]))) for bb in boots])
            row["weight_err"] = float(aw.std())
        for d, name in enumerate(cat["names"]):
            row[f"mean_{name}"] = float(mu_p[k, d])
            row[f"sigma_{name}"] = float(np.sqrt(V_p[k, d, d]))
            if boots:
                row[f"mean_{name}_err"] = float(am[:, d].std())
                row[f"sigma_{name}_err"] = float(ad[:, d].std())
        comp_rows.append(row)

    np.savez(os.path.join(out, "xd_best.npz"), alpha=best["alpha"], mu=best["mu"], V=best["V"],
             mu_phys=mu_p, V_phys=V_p, scaler_mean=sc.mean, scaler_std=sc.std,
             names=np.array(cat["names"]))
    R = xd.responsibilities(Z, SZ, best["alpha"], best["mu"], best["V"], device=dev, chunk=c["chunk"])
    mt = Table()
    mt["id"] = cat["ids"]
    for k in range(bestK):
        mt[f"p_comp{k}"] = R[:, k]
    mt["best_comp"] = R.argmax(1)
    mt.write(os.path.join(out, "membership.fits"), overwrite=True)

    save_json(dict(device=gpu, n_stars=cat["n_kept"], n_input=cat["n_input"], dims=cat["names"],
                   missing_policy=cfg["catalog"].get("missing", "drop"),
                   bic_table=table, best_K=bestK, components=comp_rows,
                   n_bootstrap=len(boots), bootstrap_seconds=t_boot,
                   total_seconds=time.time() - t_all),
              os.path.join(out, "xd_summary.json"))
    print(f"best K={bestK}; wrote {out}/xd_best.npz, xd_summary.json, membership.fits "
          f"(total {time.time() - t_all:.1f}s)")


if __name__ == "__main__":
    main()
