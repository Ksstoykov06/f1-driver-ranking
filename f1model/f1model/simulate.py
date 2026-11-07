"""Puts every driver through the same simulated weekends in identical cars.

Scores are mean finishing positions on a 1-20 scale (lower is better), which makes the two
pipelines comparable even though their raw coefficients have different units.
"""
import numpy as np
from .config import *

def _rank_to_pos(x):
    """Rank entrants on x (higher better); rescale to a 1-20 field."""
    n = x.shape[-1]
    ranks = (-x).argsort(-1).argsort(-1)
    return 1 + ranks * (FIELD_SIZE - 1) / max(n - 1, 1)

def _grid(pos):
    return (pos - 10.5) / 9.5

def _fit_draw(ent, i, rng, N):
    """A fresh car for every weekend: each driver's fit comes from their own spread."""
    return ent["fit_sd"][i] * rng.normal(size=N)

def sim_results(ent, wet_pairs, rng, weekends):
    """ent: dict of (draws, entrants) arrays. Returns mean finish, shape (draws, entrants)."""
    D, N = ent["q"].shape
    out = np.zeros((D, N))
    for i in range(D):
        w = wet_pairs[rng.integers(len(wet_pairs), size=weekends)]
        for k in range(weekends):
            f = _fit_draw(ent, i, rng, N)
            pq = ent["q"][i] + f + ent["wq"][i] * w[k, 0] + rng.gumbel(size=N)
            grid = _rank_to_pos(pq)
            pr = ent["r"][i] + f * ent["fit_r"][i] + ent["wr"][i] * w[k, 1] + ent["cr"][i] * w[k, 2] + ent["g"][i] * _grid(grid) + rng.gumbel(size=N)
            pr = np.where(rng.random(N) < (ent["perr_w"][i] if (w[k, 1] or w[k, 2]) else ent["perr"][i]), -1e6, pr)
            out[i] += _rank_to_pos(pr)
    return out / weekends

def sim_laps(ent, wet_pairs, rng, weekends, push_laps=3):
    """Qualifying is the best of `push_laps` draws around each driver's own lap-to-lap spread."""
    D, N = ent["q"].shape
    out = np.zeros((D, N))
    for i in range(D):
        w = wet_pairs[rng.integers(len(wet_pairs), size=weekends)]
        for k in range(weekends):
            f = _fit_draw(ent, i, rng, N)
            base = ent["q"][i] + f + ent["wq"][i] * w[k, 0] + ent["sig_q"][i] * rng.standard_t(4, N)
            laps = base[:, None] + ent["lap_sd"][:, None] * rng.normal(size=(N, push_laps))
            grid = _rank_to_pos(laps.max(1))
            pace = ent["r"][i] + f * ent["fit_r"][i] + ent["wr"][i] * w[k, 1] + ent["sig_r"][i] * rng.standard_t(4, N)
            u = (ent["craft"][i] + ent["cw"][i] * w[k, 1] + ent["cr"][i] * w[k, 2] + ent["b"][i] * pace + ent["g"][i] * _grid(grid) + rng.gumbel(size=N))
            u = np.where(rng.random(N) < (ent["perr_w"][i] if (w[k, 1] or w[k, 2]) else ent["perr"][i]), -1e6, u)
            out[i] += _rank_to_pos(u)
    return out / weekends
