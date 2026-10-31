"""Write an illustrative mock catalog with the config's column names, for testing the pipeline.

The component parameters are made up to look SMC-like; they are NOT measurements.
Usage: python scripts/make_mock.py --config config.yaml --n 16000
"""
import argparse, os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
from astropy.table import Table
from proto.data import load_config

# intrinsic components in (vlos km/s, pmra, pmdec mas/yr, [Fe/H])
COMPONENTS = [
    dict(w=0.55, mu=[150, 0.70, -1.25, -0.95], sd=[22, 0.04, 0.04, 0.20]),   # main body
    dict(w=0.30, mu=[170, 0.80, -1.20, -1.05], sd=[28, 0.06, 0.05, 0.25]),   # offset / tidal
    dict(w=0.15, mu=[155, 0.72, -1.22, -1.55], sd=[42, 0.10, 0.09, 0.30]),   # metal-poor, hot
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--n", type=int, default=16000)
    ap.add_argument("--out", default=None)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--missing_frac", type=float, default=0.0,
                    help="fraction of stars with [Fe/H] set to NaN (tests the inflate policy)")
    a = ap.parse_args()
    cfg = load_config(a.config)
    rng = np.random.default_rng(a.seed)
    w = np.array([c["w"] for c in COMPONENTS])
    comp = rng.choice(len(w), a.n, p=w / w.sum())
    truth = np.empty((a.n, 4))
    for k, c in enumerate(COMPONENTS):
        m = comp == k
        truth[m] = rng.normal(c["mu"], c["sd"], size=(m.sum(), 4))
    err = np.column_stack([
        rng.uniform(0.3, 2.0, a.n),        # vlos
        rng.uniform(0.02, 0.12, a.n),      # pmra
        rng.uniform(0.02, 0.10, a.n),      # pmdec
        rng.uniform(0.05, 0.20, a.n),      # [Fe/H]
    ])
    rho = rng.uniform(-0.4, 0.4, a.n)
    obs = truth.copy()
    obs[:, [0, 3]] += rng.normal(0, 1, (a.n, 2)) * err[:, [0, 3]]
    z1, z2 = rng.normal(size=(2, a.n))
    obs[:, 1] += err[:, 1] * z1
    obs[:, 2] += err[:, 2] * (rho * z1 + np.sqrt(1 - rho**2) * z2)
    if a.missing_frac > 0:
        m = rng.random(a.n) < a.missing_frac
        obs[m, 3] = np.nan

    dims = cfg["dimensions"]
    if [d["name"] for d in dims] != ["vlos", "pmra", "pmdec", "feh"]:
        sys.exit("make_mock.py expects the four default dimensions (vlos, pmra, pmdec, feh).")
    t = Table()
    for k, d in enumerate(dims):
        t[d["column"]] = obs[:, k]
        t[d["error"]] = err[:, k]
    for c in cfg.get("correlations", []) or []:
        t[c["column"]] = rho
    t["true_component"] = comp
    if cfg["catalog"].get("id_column"):
        t[cfg["catalog"]["id_column"]] = np.arange(a.n)
    out = a.out or cfg["catalog"]["path"]
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    t.write(out, overwrite=True)
    print(f"wrote {a.n} mock stars to {out}")


if __name__ == "__main__":
    main()
