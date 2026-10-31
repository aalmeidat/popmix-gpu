"""Catalog loading, per-star covariance construction, and standardization."""
import json
import numpy as np
import yaml
from astropy.table import Table


def load_config(path):
    with open(path) as f:
        return yaml.safe_load(f)


def _raw(t, name):
    c = t[name]
    if hasattr(c, "filled"):
        c = c.filled(np.nan)
    return np.asarray(c, dtype=float)


def _col(t, name):
    """A column name, or an arithmetic expression of columns, e.g. "GAIAEDR3_PMRA - PM_ra_model"."""
    if name in t.colnames:
        return _raw(t, name)
    ns = {c: _raw(t, c) for c in t.colnames if c in name}
    ns.update(np=np, sqrt=np.sqrt, abs=np.abs)
    return np.asarray(eval(name, {"__builtins__": {}}, ns), dtype=float)


def load_catalog(cfg):
    """Return dict with X (N,D), S (N,D,D), names, labels, ids, n_input, n_kept, missing_mask."""
    cc = cfg["catalog"]
    t = Table.read(cc["path"], format=cc.get("format")) if cc.get("format") else Table.read(cc["path"])
    dims = cfg["dimensions"]
    names = [d["name"] for d in dims]
    labels = [d.get("label", d["name"]) for d in dims]
    N, D = len(t), len(dims)

    X = np.column_stack([_col(t, d["column"]) for d in dims])
    E = np.column_stack([
        _col(t, d["error"]) if d.get("error") else np.zeros(N) for d in dims
    ])
    floors = np.array([d.get("error_floor", 0.0) or 0.0 for d in dims])
    E = np.sqrt(E**2 + floors**2)

    S = np.zeros((N, D, D))
    idx = np.arange(D)
    S[:, idx, idx] = E**2
    for c in cfg.get("correlations", []) or []:
        i, j = names.index(c["a"]), names.index(c["b"])
        rho = np.nan_to_num(_col(t, c["column"]), nan=0.0)
        S[:, i, j] = S[:, j, i] = rho * E[:, i] * E[:, j]

    missing = ~np.isfinite(X) | ~np.isfinite(E)
    policy = cc.get("missing", "drop")
    if policy == "drop":
        keep = ~missing.any(1)
    elif policy == "inflate":
        keep = ~missing.all(1)
        med = np.nanmedian(X, 0)
        sd = np.nanstd(X, 0)
        big = (cc.get("inflate_factor", 100.0) * sd) ** 2
        for k in range(D):
            m = missing[:, k]
            X[m, k] = med[k]
            S[m, k, :] = 0.0
            S[m, :, k] = 0.0
            S[m, k, k] = big[k]
    else:
        raise ValueError(f"unknown missing policy {policy!r}")

    # quality cuts: catalog.cuts = [{column: FeH, min: -3.0, max: -0.5}, ...]; NaN fails a cut
    for cut in cc.get("cuts", []) or []:
        v = _col(t, cut["column"])
        ok = np.isfinite(v)
        if cut.get("min") is not None:
            ok &= v > cut["min"]
        if cut.get("max") is not None:
            ok &= v <= cut["max"]
        keep &= ok

    ids =np.asarray(t[cc["id_column"]]) if cc.get("id_column") else np.arange(N)
    return dict(X=X[keep], S=S[keep], names=names, labels=labels, ids=ids[keep],
                n_input=N, n_kept=int(keep.sum()), missing=missing[keep], keep=keep)


class Scaler:
    """Standardize each dimension; transforms covariances consistently."""

    def __init__(self, mean=None, std=None):
        self.mean, self.std = mean, std

    def fit(self, X):
        self.mean, self.std = X.mean(0), X.std(0)
        return self

    def x(self, X):
        return (X - self.mean) / self.std

    def cov(self, S):
        s = self.std
        return S / (s[:, None] * s[None, :])

    def inv_x(self, Z):
        return Z * self.std + self.mean

    def inv_cov(self, C):
        s = self.std
        return C * (s[:, None] * s[None, :])

    def to_dict(self):
        return dict(mean=self.mean.tolist(), std=self.std.tolist())

    @classmethod
    def from_dict(cls, d):
        return cls(np.array(d["mean"]), np.array(d["std"]))


def save_json(obj, path):
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)


def load_json(path):
    with open(path) as f:
        return json.load(f)
