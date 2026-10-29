"""Dataset sources that all return the interaction frame defined in :mod:`recsys.data`.

``load_movielens`` reads a local copy of a MovieLens release (``ratings.csv`` from the
``ml-latest-small`` or ``ml-25m`` archives). ``synthetic_interactions`` generates data with
the same schema for environments that cannot download MovieLens, so the pipeline and its
tests run anywhere; results produced on synthetic data are labelled as such.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from recsys.data import validate_interactions


def load_movielens(directory: Path) -> pd.DataFrame:
    """Read ``ratings.csv`` (userId, movieId, rating, timestamp) from ``directory``."""
    path = directory / "ratings.csv"
    if not path.is_file():
        raise FileNotFoundError(f"{path} not found; download a MovieLens release first")
    raw = pd.read_csv(
        path,
        dtype={"userId": "int64", "movieId": "int64", "rating": "float64", "timestamp": "int64"},
    )
    frame = raw.rename(columns={"userId": "user_id", "movieId": "item_id"})
    return validate_interactions(frame)


def synthetic_interactions(
    n_users: int = 500,
    n_items: int = 1000,
    interactions_per_user: tuple[int, int] = (20, 120),
    dim: int = 8,
    seed: int = 0,
) -> pd.DataFrame:
    """Generate interactions from a latent-factor model with popularity skew.

    Each user and item gets a ``dim``-dimensional preference vector. A user's items are
    sampled with probability proportional to ``exp(u . v)`` times a Zipf-like popularity
    prior, so the data has both personal structure for the two-tower model to learn and
    the long-tail popularity that makes naive baselines look stronger than they are.
    Ratings are the affinity mapped onto the MovieLens 0.5 to 5.0 half-star scale.
    """
    if n_users < 1 or n_items < 2:
        raise ValueError("need at least one user and two items")
    low, high = interactions_per_user
    if not 1 <= low <= high <= n_items:
        raise ValueError("interactions_per_user must satisfy 1 <= low <= high <= n_items")

    rng = np.random.default_rng(seed)
    users = rng.normal(size=(n_users, dim)) / np.sqrt(dim)
    items = rng.normal(size=(n_items, dim)) / np.sqrt(dim)
    popularity = 1.0 / np.arange(1, n_items + 1) ** 0.8
    rng.shuffle(popularity)

    start = 1_500_000_000
    rows: list[tuple[int, int, float, int]] = []
    for user in range(n_users):
        affinity = items @ users[user]
        weights = np.exp(2.0 * affinity) * popularity
        count = int(rng.integers(low, high + 1))
        chosen = rng.choice(n_items, size=count, replace=False, p=weights / weights.sum())
        times = np.sort(rng.integers(start, start + 365 * 86_400, size=count))
        for item, ts in zip(chosen, times, strict=True):
            score = 3.0 + 4.0 * affinity[item] + rng.normal(scale=0.5)
            rating = float(np.clip(np.round(score * 2) / 2, 0.5, 5.0))
            rows.append((user + 1, int(item) + 1, rating, int(ts)))

    frame = pd.DataFrame(rows, columns=["user_id", "item_id", "rating", "timestamp"])
    return validate_interactions(frame.drop_duplicates(["user_id", "item_id", "timestamp"]))
