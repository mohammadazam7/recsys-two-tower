"""Train the two-tower model and measure retrieval recall against a popularity baseline.

Evaluation follows the temporal split: for each evaluated user, every item is scored,
items the user already interacted with in training are removed, and a hit is counted if
the held-out item is among the top ``k``. Recall@k is the share of users with a hit. The
popularity baseline ranks items by training frequency under the same exclusion, so the
model is only credited for what it learns beyond "recommend what is popular".
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
import torch

from recsys.model import TowerConfig, TwoTower, item_log_probability, sampled_softmax_loss


@dataclass(frozen=True)
class TrainConfig:
    epochs: int = 10
    batch_size: int = 512
    learning_rate: float = 3e-3
    log_q_correction: bool = True
    seed: int = 0


@dataclass
class History:
    losses: list[float] = field(default_factory=list)


def _tensors(frame: pd.DataFrame) -> tuple[torch.Tensor, torch.Tensor]:
    if (frame["user_idx"] < 0).any() or (frame["item_idx"] < 0).any():
        raise ValueError("frame contains IDs outside the ID map; encode with the training map")
    users = torch.as_tensor(frame["user_idx"].to_numpy(copy=True), dtype=torch.long)
    items = torch.as_tensor(frame["item_idx"].to_numpy(copy=True), dtype=torch.long)
    return users, items


def fit(train: pd.DataFrame, tower: TowerConfig, config: TrainConfig) -> tuple[TwoTower, History]:
    """Train on ``train`` (with ``user_idx`` and ``item_idx`` columns), deterministically."""
    torch.manual_seed(config.seed)
    users, items = _tensors(train)
    model = TwoTower(tower)
    optimiser = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    counts = torch.bincount(items, minlength=tower.n_items)
    log_q = item_log_probability(counts) if config.log_q_correction else None
    generator = torch.Generator().manual_seed(config.seed)
    history = History()
    model.train()
    for _ in range(config.epochs):
        order = torch.randperm(len(users), generator=generator)
        total, batches = 0.0, 0
        for start in range(0, len(order), config.batch_size):
            batch = order[start : start + config.batch_size]
            if len(batch) < 2:
                continue
            logits = model.in_batch_logits(users[batch], items[batch])
            loss = sampled_softmax_loss(logits, items[batch], log_q)
            optimiser.zero_grad()
            loss.backward()  # type: ignore[no-untyped-call]
            optimiser.step()
            total += loss.item()
            batches += 1
        history.losses.append(total / max(batches, 1))
    model.eval()
    return model, history


def _seen(frames: Sequence[pd.DataFrame]) -> dict[int, set[int]]:
    seen: dict[int, set[int]] = {}
    for frame in frames:
        for user, item in zip(frame["user_idx"], frame["item_idx"], strict=True):
            seen.setdefault(int(user), set()).add(int(item))
    return seen


def _recall(
    scores: np.ndarray, users: np.ndarray, targets: np.ndarray, seen: dict[int, set[int]], k: int
) -> float:
    hits = 0
    for row, (user, target) in enumerate(zip(users, targets, strict=True)):
        ranked = scores[row].copy()
        for item in seen.get(int(user), ()):
            ranked[item] = -np.inf
        top = np.argpartition(-ranked, k)[:k]
        hits += int(target in top)
    return hits / len(users)


@dataclass(frozen=True)
class Evaluation:
    users: int
    k: int
    model_recall: float
    popularity_recall: float


def evaluate(
    model: TwoTower,
    held_out: pd.DataFrame,
    history: Sequence[pd.DataFrame],
    k: int = 10,
) -> Evaluation:
    """Recall@k of the model and of the popularity baseline on ``held_out``.

    ``history`` holds every interaction the user is known to have had before the held-out
    one (training, plus validation when scoring the test set); those items are excluded.
    """
    if held_out.empty:
        raise ValueError("nothing to evaluate")
    users, targets = (t.numpy() for t in _tensors(held_out))
    n_items = model.config.n_items
    if not 0 < k < n_items:
        raise ValueError("k must be between 1 and the number of items")
    seen = _seen(history)
    with torch.no_grad():
        scores = (model.user_embeddings(torch.as_tensor(users)) @ model.item_embeddings().T).numpy()
    counts = np.bincount(
        np.concatenate([f["item_idx"].to_numpy() for f in history]), minlength=n_items
    ).astype(np.float64)
    popularity = np.broadcast_to(counts, (len(users), n_items))
    return Evaluation(
        users=len(users),
        k=k,
        model_recall=_recall(scores, users, targets, seen, k),
        popularity_recall=_recall(popularity, users, targets, seen, k),
    )
