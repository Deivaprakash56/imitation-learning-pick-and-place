"""Behaviour-cloning policy: MLP with observation normalisation and optional action chunking.

With chunk > 1 the network predicts the next `chunk` actions (ACT-style); at run time the
overlapping predictions are blended with temporal ensembling, which smooths jitter.
"""
import numpy as np
import torch
import torch.nn as nn


class BCPolicy(nn.Module):
    def __init__(self, obs_dim, act_dim, chunk=1, hidden=(512, 512, 512), dropout=0.1):
        super().__init__()
        self.obs_dim, self.act_dim, self.chunk = obs_dim, act_dim, chunk
        layers, d = [], obs_dim
        for h in hidden:
            layers += [nn.Linear(d, h), nn.LayerNorm(h), nn.GELU(), nn.Dropout(dropout)]
            d = h
        layers.append(nn.Linear(d, act_dim * chunk))
        self.net = nn.Sequential(*layers)
        self.register_buffer('obs_mean', torch.zeros(obs_dim))
        self.register_buffer('obs_std', torch.ones(obs_dim))

    def set_norm(self, mean, std):
        self.obs_mean.copy_(torch.as_tensor(mean))
        self.obs_std.copy_(torch.as_tensor(np.maximum(std, 1e-3)))

    def forward(self, obs):
        x = (obs - self.obs_mean) / self.obs_std
        return torch.tanh(self.net(x)).view(-1, self.chunk, self.act_dim)

    def save(self, path, **meta):
        torch.save({'state_dict': self.state_dict(), 'obs_dim': self.obs_dim, 'act_dim': self.act_dim,
                    'chunk': self.chunk, 'meta': meta}, path)

    @classmethod
    def load(cls, path, device='cpu'):
        ck = torch.load(path, map_location=device, weights_only=False)
        hidden = tuple(ck['meta'].get('hidden', (512, 512, 512)))
        m = cls(ck['obs_dim'], ck['act_dim'], ck['chunk'], hidden=hidden)
        m.load_state_dict(ck['state_dict'])
        return m.eval().to(device)


class ChunkExecutor:
    """Temporal ensembling over overlapping action chunks (exponential weights, newest first)."""

    def __init__(self, policy: BCPolicy, k=0.1, device='cpu'):
        self.p, self.k, self.device = policy, k, device
        self.reset()

    def reset(self):
        self.buf = []   # list of (t_start, chunk array)
        self.t = 0

    @torch.no_grad()
    def act(self, obs: np.ndarray) -> np.ndarray:
        chunk = self.p(torch.as_tensor(obs, dtype=torch.float32, device=self.device)[None])[0].cpu().numpy()
        if self.p.chunk == 1:
            self.t += 1
            return chunk[0]
        self.buf.append((self.t, chunk))
        self.buf = [(s, c) for s, c in self.buf if self.t - s < self.p.chunk]
        preds = np.stack([c[self.t - s] for s, c in self.buf])          # oldest -> newest
        w = np.exp(-self.k * np.arange(len(preds)))                       # ACT: w_0 (oldest) is largest
        self.t += 1
        return (w[:, None] * preds).sum(0) / w.sum()
