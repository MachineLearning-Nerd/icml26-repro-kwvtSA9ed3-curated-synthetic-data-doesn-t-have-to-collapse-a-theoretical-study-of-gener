"""Optimal-transport conditional flow matching (OT-CFM) on CIFAR-10, CPU-sized.

The paper (Appendix C.5) trains "a normalizing flow using optimal transport conditional
flow matching (OT-CFM)" following Lipman et al. 2022 / Tong et al. 2023. That algorithm
is implemented here rather than substituted for something nearby:

  * draw a data batch x1 and a noise batch x0 ~ N(0, I);
  * couple them by EXACT minibatch optimal transport (Hungarian assignment on the
    squared-Euclidean cost), which is the "OT" in OT-CFM -- plain CFM uses the
    independent coupling and is a different algorithm;
  * sample t ~ U(0,1), form x_t = (1-t) x0 + t x1, and regress the network onto the
    conditional velocity u = x1 - x0;
  * sample by integrating dx/dt = v(x, t) from t=0 to t=1.

The network is a small time-conditioned UNet so the whole loop fits CPU budget.
"""

from __future__ import annotations

import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment


# --------------------------------------------------------------------------- #
def timestep_embedding(t: torch.Tensor, dim: int) -> torch.Tensor:
    half = dim // 2
    freqs = torch.exp(-math.log(10000) * torch.arange(half, dtype=torch.float32) / half)
    args = t[:, None] * freqs[None]
    return torch.cat([torch.cos(args), torch.sin(args)], dim=-1)


class Block(nn.Module):
    def __init__(self, cin: int, cout: int, tdim: int) -> None:
        super().__init__()
        self.c1 = nn.Conv2d(cin, cout, 3, padding=1)
        self.c2 = nn.Conv2d(cout, cout, 3, padding=1)
        self.emb = nn.Linear(tdim, cout)
        self.n1 = nn.GroupNorm(8, cout)
        self.n2 = nn.GroupNorm(8, cout)
        self.skip = nn.Conv2d(cin, cout, 1) if cin != cout else nn.Identity()

    def forward(self, x: torch.Tensor, te: torch.Tensor) -> torch.Tensor:
        h = F.silu(self.n1(self.c1(x)))
        h = h + self.emb(te)[:, :, None, None]
        h = F.silu(self.n2(self.c2(h)))
        return h + self.skip(x)


class UNet(nn.Module):
    """Small time-conditioned UNet velocity field v(x, t)."""

    def __init__(self, ch: int = 48, tdim: int = 96, in_ch: int = 3) -> None:
        super().__init__()
        self.tdim = tdim
        self.tmlp = nn.Sequential(nn.Linear(tdim, tdim), nn.SiLU(), nn.Linear(tdim, tdim))
        self.d1 = Block(in_ch, ch, tdim)
        self.d2 = Block(ch, ch * 2, tdim)
        self.d3 = Block(ch * 2, ch * 2, tdim)
        self.mid = Block(ch * 2, ch * 2, tdim)
        self.u3 = Block(ch * 4, ch * 2, tdim)
        self.u2 = Block(ch * 4, ch, tdim)
        self.u1 = Block(ch * 2, ch, tdim)
        self.out = nn.Conv2d(ch, in_ch, 3, padding=1)

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        te = self.tmlp(timestep_embedding(t, self.tdim))
        h1 = self.d1(x, te)                              # 32
        h2 = self.d2(F.avg_pool2d(h1, 2), te)            # 16
        h3 = self.d3(F.avg_pool2d(h2, 2), te)            # 8
        m = self.mid(h3, te)
        u3 = self.u3(torch.cat([m, h3], 1), te)
        u3 = F.interpolate(u3, scale_factor=2, mode="nearest")
        u2 = self.u2(torch.cat([u3, h2], 1), te)
        u2 = F.interpolate(u2, scale_factor=2, mode="nearest")
        u1 = self.u1(torch.cat([u2, h1], 1), te)
        return self.out(u1)


# --------------------------------------------------------------------------- #
def ot_pair(x0: torch.Tensor, x1: torch.Tensor) -> torch.Tensor:
    """Exact minibatch OT coupling: permute x0 to minimise sum ||x0_i - x1_i||^2."""
    a = x0.reshape(len(x0), -1)
    b = x1.reshape(len(x1), -1)
    cost = torch.cdist(a, b).pow(2).numpy()
    r, c = linear_sum_assignment(cost)
    perm = np.empty(len(x0), dtype=np.int64)
    perm[c] = r
    return x0[torch.from_numpy(perm)]


def cfm_loss(model: UNet, x1: torch.Tensor, generator: torch.Generator) -> torch.Tensor:
    x0 = torch.randn(x1.shape, generator=generator)
    x0 = ot_pair(x0, x1)                     # <-- the OT in OT-CFM
    t = torch.rand(len(x1), generator=generator)
    xt = (1 - t[:, None, None, None]) * x0 + t[:, None, None, None] * x1
    return F.mse_loss(model(xt, t), x1 - x0)


@torch.no_grad()
def sample(model: UNet, n: int, steps: int, batch: int,
           generator: torch.Generator, shape=(3, 32, 32)) -> torch.Tensor:
    """Euler integration of dx/dt = v(x,t) from t=0 (noise) to t=1."""
    model.eval()
    outs = []
    dt = 1.0 / steps
    for s in range(0, n, batch):
        b = min(batch, n - s)
        x = torch.randn((b, *shape), generator=generator)
        for k in range(steps):
            t = torch.full((b,), k * dt)
            x = x + dt * model(x, t)
        outs.append(x)
    model.train()
    return torch.cat(outs)[:n]
