"""The two-tower retrieval model.

A user tower and an item tower each map an ID to an L2-normalised embedding, and the
score of a (user, item) pair is their dot product divided by a temperature. Because the
score is an inner product, every item embedding can be computed once and searched with
an approximate nearest neighbour index at serving time.

Training uses in-batch sampled softmax: for each (user, item) pair in a batch, the other
items in the batch are its negatives. Popular items appear in more batches, so they are
used as negatives more often than a full softmax would use them, which pushes their
scores down unfairly. The log-Q correction subtracts the log of each item's sampling
probability from its logit to undo that bias (Yi et al., "Sampling-Bias-Corrected Neural
Modeling for Large Corpus Item Recommendations", RecSys 2019).
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import torch
from torch import Tensor, nn
from torch.nn import functional as F


@dataclass(frozen=True)
class TowerConfig:
    n_users: int
    n_items: int
    embedding_dim: int = 64
    hidden_dim: int = 128
    output_dim: int = 32
    temperature: float = 0.1

    def __post_init__(self) -> None:
        if self.n_users < 1 or self.n_items < 2:
            raise ValueError("need at least one user and two items")
        if self.temperature <= 0:
            raise ValueError("temperature must be positive")


class Tower(nn.Module):
    """ID embedding followed by a small MLP, output normalised to unit length."""

    def __init__(self, count: int, config: TowerConfig) -> None:
        super().__init__()
        self.embedding = nn.Embedding(count, config.embedding_dim)
        self.mlp = nn.Sequential(
            nn.Linear(config.embedding_dim, config.hidden_dim),
            nn.ReLU(),
            nn.Linear(config.hidden_dim, config.output_dim),
        )
        nn.init.normal_(self.embedding.weight, std=0.05)

    def forward(self, ids: Tensor) -> Tensor:
        out: Tensor = F.normalize(self.mlp(self.embedding(ids)), dim=-1)
        return out


class TwoTower(nn.Module):
    def __init__(self, config: TowerConfig) -> None:
        super().__init__()
        self.config = config
        self.user_tower = Tower(config.n_users, config)
        self.item_tower = Tower(config.n_items, config)

    def user_embeddings(self, users: Tensor) -> Tensor:
        out: Tensor = self.user_tower(users)
        return out

    def item_embeddings(self, items: Tensor | None = None) -> Tensor:
        """Embeddings for ``items``, or for the whole catalog when ``items`` is None."""
        if items is None:
            items = torch.arange(self.config.n_items)
        out: Tensor = self.item_tower(items)
        return out

    def in_batch_logits(self, users: Tensor, items: Tensor) -> Tensor:
        """Scores of every user in the batch against every item in the batch."""
        return self.user_embeddings(users) @ self.item_embeddings(items).T / self.config.temperature


def sampled_softmax_loss(logits: Tensor, items: Tensor, log_q: Tensor | None = None) -> Tensor:
    """In-batch softmax cross-entropy, the diagonal being each row's positive.

    ``log_q`` is the log sampling probability of every catalog item. When given, each
    column's logit is reduced by its item's log probability before the softmax.

    If an item appears twice in a batch, the second copy would be counted as a negative
    for the first user; those collisions are masked out so a positive is never also a
    negative.
    """
    if logits.shape[0] != logits.shape[1] or logits.shape[0] != items.shape[0]:
        raise ValueError("logits must be square with one row per (user, item) pair")
    if log_q is not None:
        logits = logits - log_q[items].unsqueeze(0)
    same_item = items.unsqueeze(0) == items.unsqueeze(1)
    off_diagonal = ~torch.eye(len(items), dtype=torch.bool, device=items.device)
    logits = logits.masked_fill(same_item & off_diagonal, float("-inf"))
    targets = torch.arange(len(items), device=items.device)
    return F.cross_entropy(logits, targets)


def item_log_probability(item_counts: Tensor) -> Tensor:
    """Log of each item's share of training interactions, the in-batch sampling rate.

    Items never seen in training get the smallest observed probability rather than
    log(0), so a cold item is not given an infinite boost."""
    counts = item_counts.to(torch.float64)
    floor = counts[counts > 0].min() if (counts > 0).any() else torch.tensor(1.0)
    probability = torch.clamp(counts, min=float(floor)) / counts.clamp(min=0).sum().clamp(min=1)
    return torch.log(probability).to(torch.float32)


def save(model: TwoTower, path: str, **extra: object) -> None:
    torch.save({"config": asdict(model.config), "state": model.state_dict(), **extra}, path)


def load(path: str) -> tuple[TwoTower, dict[str, object]]:
    payload = torch.load(path, map_location="cpu", weights_only=True)
    model = TwoTower(TowerConfig(**payload.pop("config")))
    model.load_state_dict(payload.pop("state"))
    model.eval()
    return model, payload
