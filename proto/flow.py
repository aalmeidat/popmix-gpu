"""Deconvolving normalizing flow.

The observed density is the intrinsic density q convolved with each star's noise:
    p(x_i) = E_{e ~ N(0, S_i)} [ q(x_i - e) ].
We train q (a neural spline flow) by maximizing a Monte Carlo estimate of log p(x_i)
with M noise draws per star (an importance-weighted lower bound that tightens as M grows).
With M = 1 and no noise it reduces to an ordinary flow fit to the noisy data.
"""
import math
import torch
import zuko


def make_flow(D, transforms=5, hidden=(128, 128)):
    return zuko.flows.NSF(features=D, transforms=transforms, hidden_features=list(hidden))


def deconv_nll(flow, x, Lc, M, deconvolve=True):
    """Mean negative log-likelihood of a batch. x: (n,D), Lc: Cholesky factors of S, (n,D,D)."""
    n, D = x.shape
    if not deconvolve:
        return -flow().log_prob(x).mean()
    eps = torch.randn(n, M, D, device=x.device, dtype=x.dtype)
    e = (Lc[:, None] @ eps.unsqueeze(-1)).squeeze(-1)
    v = (x[:, None, :] - e).reshape(-1, D)
    lq = flow().log_prob(v).reshape(n, M)
    return -(torch.logsumexp(lq, 1) - math.log(M)).mean()


def cholesky_safe(S, jitter=1e-9):
    D = S.shape[-1]
    eye = torch.eye(D, dtype=S.dtype, device=S.device)
    return torch.linalg.cholesky(S + jitter * eye)
