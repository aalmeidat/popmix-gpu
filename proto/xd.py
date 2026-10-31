"""Extreme deconvolution (Bovy, Hogg & Roweis 2011) in PyTorch, chunked for GPUs.

Fits the intrinsic (error-free) distribution as a Gaussian mixture, with every star
carrying its own measurement covariance S_i. This is the same model class as the
error-aware GMM in the current SMC analysis, minus pyGMMis's selection-function term.
"""
import math
import time
import numpy as np
import torch

LOG2PI = math.log(2 * math.pi)


def device_auto():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _estep(x, S, alpha, mu, V, need_moments=True):
    """Per-chunk E-step. Returns log-likelihood per star, responsibilities and moments."""
    D = x.shape[1]
    T = V[None] + S[:, None]                              # n,K,D,D
    L = torch.linalg.cholesky(T)
    diff = x[:, None, :] - mu[None]                       # n,K,D
    sol = torch.cholesky_solve(diff.unsqueeze(-1), L).squeeze(-1)
    maha = (diff * sol).sum(-1)
    logdet = 2 * torch.log(torch.diagonal(L, dim1=-2, dim2=-1)).sum(-1)
    logp = torch.log(alpha)[None] - 0.5 * (maha + logdet + D * LOG2PI)
    lse = torch.logsumexp(logp, 1)
    r = torch.exp(logp - lse[:, None])
    if not need_moments:
        return lse, r, None, None
    b = mu[None] + (V[None] @ sol.unsqueeze(-1)).squeeze(-1)          # n,K,D
    TinvV = torch.cholesky_solve(V[None].expand_as(T).contiguous(), L)
    B = V[None] - V[None] @ TinvV                                      # n,K,D,D
    return lse, r, b, B


def _kmeans_init(x, K, gen, iters=20):
    n = x.shape[0]
    centers = x[torch.randint(n, (1,), generator=gen, device="cpu").to(x.device)]
    for _ in range(1, K):  # k-means++ seeding
        d2 = torch.cdist(x, centers).min(1).values ** 2
        p = (d2 / d2.sum()).cpu()
        idx = torch.multinomial(p, 1, generator=gen).to(x.device)
        centers = torch.cat([centers, x[idx]])
    for _ in range(iters):
        lab = torch.cdist(x, centers).argmin(1)
        for k in range(K):
            m = lab == k
            if m.any():
                centers[k] = x[m].mean(0)
    lab = torch.cdist(x, centers).argmin(1)
    D = x.shape[1]
    V = torch.stack([
        torch.cov(x[lab == k].T) if (lab == k).sum() > D else torch.eye(D, dtype=x.dtype, device=x.device)
        for k in range(K)
    ]) + 1e-3 * torch.eye(D, dtype=x.dtype, device=x.device)
    alpha = torch.stack([(lab == k).float().mean() for k in range(K)]).to(x.dtype).clamp_min(1e-3)
    return alpha / alpha.sum(), centers.clone(), V


def fit(X, S, K, max_iter=500, tol=1e-7, reg=1e-6, seed=0, device=None,
        chunk=200000, init=None, dtype=torch.float64, verbose=False):
    """Fit an XD mixture. X, S in standardized units (numpy). Returns dict of numpy arrays."""
    device = device or device_auto()
    x = torch.as_tensor(X, dtype=dtype, device=device)
    s = torch.as_tensor(S, dtype=dtype, device=device)
    n, D = x.shape
    eye = torch.eye(D, dtype=dtype, device=device)
    gen = torch.Generator().manual_seed(seed)
    if init is None:
        alpha, mu, V = _kmeans_init(x, K, gen)
    else:
        alpha, mu, V = (torch.as_tensor(a, dtype=dtype, device=device) for a in init)
    hist, t0, prev = [], time.time(), -np.inf
    for it in range(max_iter):
        Nk = torch.zeros(K, dtype=dtype, device=device)
        Sb = torch.zeros(K, D, dtype=dtype, device=device)
        Sbb = torch.zeros(K, D, D, dtype=dtype, device=device)
        ll = 0.0
        for a in range(0, n, chunk):
            lse, r, b, B = _estep(x[a:a + chunk], s[a:a + chunk], alpha, mu, V)
            ll += lse.sum().item()
            Nk += r.sum(0)
            Sb += torch.einsum("nk,nkd->kd", r, b)
            Sbb += torch.einsum("nk,nkd,nke->kde", r, b, b) + torch.einsum("nk,nkde->kde", r, B)
        Nk = Nk.clamp_min(1e-10)
        alpha = Nk / n
        mu = Sb / Nk[:, None]
        V = Sbb / Nk[:, None, None] - mu[:, :, None] * mu[:, None, :]
        V = 0.5 * (V + V.transpose(-1, -2)) + reg * eye
        mll = ll / n
        hist.append(mll)
        if verbose and it % 25 == 0:
            print(f"  K={K} it={it} mean logL={mll:.6f}")
        if abs(mll - prev) < tol:
            break
        prev = mll
    if device.type == "cuda":
        torch.cuda.synchronize()
    elapsed = time.time() - t0
    # final log-likelihood at the returned parameters
    ll = sum(_estep(x[a:a + chunk], s[a:a + chunk], alpha, mu, V, need_moments=False)[0].sum().item()
             for a in range(0, n, chunk))
    p = (K - 1) + K * D + K * D * (D + 1) // 2
    return dict(alpha=alpha.cpu().numpy(), mu=mu.cpu().numpy(), V=V.cpu().numpy(),
                logL=ll, bic=-2 * ll + p * math.log(n), n_params=p, n_iter=it + 1,
                seconds=elapsed, history=hist)


def responsibilities(X, S, alpha, mu, V, device=None, chunk=200000, dtype=torch.float64):
    device = device or device_auto()
    x = torch.as_tensor(X, dtype=dtype, device=device)
    s = torch.as_tensor(S, dtype=dtype, device=device)
    a = torch.as_tensor(alpha, dtype=dtype, device=device)
    m = torch.as_tensor(mu, dtype=dtype, device=device)
    v = torch.as_tensor(V, dtype=dtype, device=device)
    out = [_estep(x[i:i + chunk], s[i:i + chunk], a, m, v, need_moments=False)[1].cpu().numpy()
           for i in range(0, x.shape[0], chunk)]
    return np.concatenate(out)


def sample(alpha, mu, V, n, rng):
    comp = rng.choice(len(alpha), size=n, p=alpha / alpha.sum())
    out = np.empty((n, mu.shape[1]))
    for k in range(len(alpha)):
        m = comp == k
        out[m] = rng.multivariate_normal(mu[k], V[k], size=m.sum())
    return out, comp


def match_components(ref_mu, mu):
    """Permutation aligning a bootstrap fit's components to the reference fit."""
    from scipy.optimize import linear_sum_assignment
    cost = ((ref_mu[:, None, :] - mu[None, :, :]) ** 2).sum(-1)
    _, col = linear_sum_assignment(cost)
    return col
