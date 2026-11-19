"""Train the two-tower model on the synthetic dataset and compare it with popularity.

Usage: python examples/train_synthetic.py [--users 2000] [--items 400] [--epochs 20]

Prints recall@20 on each user's held-out most recent interactions for the model with and
without the log-Q correction, and for ranking every user's unseen items by popularity.
"""

from __future__ import annotations

import argparse
import time

from recsys.data import IdMap, temporal_split
from recsys.datasets import synthetic_interactions
from recsys.model import TowerConfig
from recsys.train import TrainConfig, evaluate, fit


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--users", type=int, default=2000)
    parser.add_argument("--items", type=int, default=400)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--k", type=int, default=20)
    args = parser.parse_args()

    frame = synthetic_interactions(n_users=args.users, n_items=args.items, seed=1)
    ids = IdMap.fit(frame)
    split = temporal_split(ids.encode(frame))
    tower = TowerConfig(n_users=len(ids.users), n_items=len(ids.items), output_dim=16)
    print(f"{len(split.train)} training interactions, {len(split.validation)} held out")

    print(f"| Configuration | recall@{args.k} | train seconds |")
    print("|---|---|---|")
    popularity = None
    for name, log_q in [("Two-tower, log-Q correction", True), ("Two-tower, no correction", False)]:
        started = time.perf_counter()
        model, _ = fit(
            split.train,
            tower,
            TrainConfig(epochs=args.epochs, batch_size=512, log_q_correction=log_q, seed=0),
        )
        seconds = time.perf_counter() - started
        result = evaluate(model, split.validation, [split.train], k=args.k)
        popularity = result.popularity_recall
        print(f"| {name} | {result.model_recall:.3f} | {seconds:.0f} |")
    print(f"| Popularity | {popularity:.3f} | - |")


if __name__ == "__main__":
    main()
