"""Measure approximate search against exact search, in two settings.

Usage: python examples/ann_benchmark.py [--items 20000] [--users 4000] [--scale 1000000]

1. Trained embeddings. A two-tower model is trained on the synthetic dataset, its item
   embeddings are indexed, and each user's held-out item is looked for through the exact
   index and through IVF. This answers: does approximate search cost recommendation
   quality?
2. Scale. Random clustered unit vectors, many more than the trained catalog, to show
   where exact search becomes slow and what IVF trades for speed.

Latency is the median over single-query searches on one thread.
"""

from __future__ import annotations

import argparse

import faiss
import numpy as np
import torch

from recsys.data import IdMap, temporal_split
from recsys.datasets import synthetic_interactions
from recsys.index import IndexConfig, ItemIndex, compare, hit_rate, overlap_recall, retrieve
from recsys.model import TowerConfig
from recsys.train import TrainConfig, fit


def clustered(rows: int, dim: int, clusters: int, rng: np.random.Generator) -> np.ndarray:
    centres = rng.normal(size=(clusters, dim)).astype(np.float32)
    values = centres[rng.integers(clusters, size=rows)]
    values = values + 0.35 * rng.normal(size=(rows, dim)).astype(np.float32)
    return values / np.linalg.norm(values, axis=1, keepdims=True)


def trained(args: argparse.Namespace) -> None:
    frame = synthetic_interactions(n_users=args.users, n_items=args.items, seed=1)
    ids = IdMap.fit(frame)
    split = temporal_split(ids.encode(frame))
    tower = TowerConfig(n_users=len(ids.users), n_items=len(ids.items), output_dim=32)
    model, _ = fit(split.train, tower, TrainConfig(epochs=args.epochs, seed=0))
    with torch.no_grad():
        items = model.item_embeddings().numpy()
        users = split.validation["user_idx"].to_numpy(copy=True)
        queries = model.user_embeddings(torch.as_tensor(users)).numpy()
    targets = split.validation["item_idx"].to_numpy()
    seen: dict[int, set[int]] = {}
    for user, item in zip(split.train["user_idx"], split.train["item_idx"], strict=True):
        seen.setdefault(int(user), set()).add(int(item))
    exclude = [seen.get(int(user), set()) for user in users]

    print(f"\nTrained embeddings: {len(items)} items, {len(queries)} users, 32 dimensions")
    print(f"| Index | recall@{args.k} of the held-out item | overlap with exact top {args.k} |")
    print("|---|---|---|")
    flat = ItemIndex(items)
    exact = retrieve(flat, queries, args.k, exclude)
    print(f"| flat (exact) | {hit_rate(exact, targets):.3f} | 1.000 |")
    ivf = ItemIndex(items, IndexConfig("ivf", nlist=args.nlist, nprobe=1))
    for nprobe in (1, 4, 16):
        ivf.nprobe = nprobe
        found = retrieve(ivf, queries, args.k, exclude)
        name = f"ivf, nlist {args.nlist}, nprobe {nprobe}"
        print(f"| {name} | {hit_rate(found, targets):.3f} | {overlap_recall(found, exact):.3f} |")


def scale(args: argparse.Namespace) -> None:
    rng = np.random.default_rng(0)
    items = clustered(args.scale, 32, clusters=2000, rng=rng)
    queries = clustered(2000, 32, clusters=2000, rng=rng)
    nlist = int(4 * np.sqrt(args.scale))
    print(f"\nScale: {args.scale:,} random clustered vectors, 32 dimensions, top {args.k}")
    print("| Index | overlap with exact | ms per query | build seconds |")
    print("|---|---|---|---|")
    for row in compare(items, queries, k=args.k, nlist=nlist, nprobes=(1, 4, 16, 64)):
        print(
            f"| {row.name} | {row.recall:.3f} | {row.ms_per_query:.3f} | {row.build_seconds:.1f} |"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--items", type=int, default=20_000)
    parser.add_argument("--users", type=int, default=4_000)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--nlist", type=int, default=128)
    parser.add_argument("--scale", type=int, default=1_000_000)
    parser.add_argument("--k", type=int, default=20)
    args = parser.parse_args()
    faiss.omp_set_num_threads(1)
    trained(args)
    scale(args)


if __name__ == "__main__":
    main()
