"""Nearest-neighbour search over item embeddings, exact and approximate.

Retrieval asks: of all items, which have the largest inner product with this user's
embedding? Scoring every item is exact and its cost grows with the catalog. An inverted
file (IVF) index clusters the items once, and at query time looks only inside the
``nprobe`` clusters closest to the query. That is faster and can miss a neighbour that
sits in a cluster it did not open, so an approximate index is always reported with its
recall against the exact answer, not only with its speed.

Both towers produce unit-length vectors, so the inner product used here is also the
cosine similarity.
"""

from __future__ import annotations

import time
from collections.abc import Collection, Sequence
from dataclasses import dataclass
from typing import Any

import faiss
import numpy as np

KINDS = ("flat", "ivf")


@dataclass(frozen=True)
class IndexConfig:
    kind: str = "flat"
    nlist: int = 100  # clusters the catalog is divided into (ivf only)
    nprobe: int = 8  # clusters opened per query (ivf only)
    seed: int = 0

    def __post_init__(self) -> None:
        if self.kind not in KINDS:
            raise ValueError(f"kind must be one of {', '.join(KINDS)}")
        if self.nlist < 1:
            raise ValueError("nlist must be at least 1")
        if not 1 <= self.nprobe <= self.nlist:
            raise ValueError("nprobe must be between 1 and nlist")


def _matrix(values: np.ndarray, name: str) -> np.ndarray:
    matrix = np.ascontiguousarray(values, dtype=np.float32)
    if matrix.ndim != 2 or matrix.shape[0] == 0:
        raise ValueError(f"{name} must be a non-empty matrix of shape (rows, dimensions)")
    return matrix


class ItemIndex:
    """Item embeddings behind a search interface. Row ``i`` of the embeddings is item
    ``i``, so the ids a search returns are item indexes."""

    def __init__(self, embeddings: np.ndarray, config: IndexConfig | None = None) -> None:
        self.config = config or IndexConfig()
        items = _matrix(embeddings, "embeddings")
        self.size, self.dim = items.shape
        started = time.perf_counter()
        # The library's flat and IVF classes share search and add but not a typed base.
        self._index: Any
        if self.config.kind == "flat":
            self._index = faiss.IndexFlatIP(self.dim)
        else:
            if self.config.nlist > self.size:
                raise ValueError(f"nlist {self.config.nlist} exceeds the {self.size} items indexed")
            self._index = faiss.index_factory(
                self.dim, f"IVF{self.config.nlist},Flat", faiss.METRIC_INNER_PRODUCT
            )
            self._index.cp.seed = self.config.seed  # the clustering is seeded
            self._index.train(items)
            self._index.nprobe = self.config.nprobe
        self._index.add(items)
        self.build_seconds = time.perf_counter() - started

    @property
    def nprobe(self) -> int:
        return int(self._index.nprobe) if self.config.kind == "ivf" else 1

    @nprobe.setter
    def nprobe(self, value: int) -> None:
        if self.config.kind != "ivf":
            raise ValueError("only an ivf index has nprobe")
        if not 1 <= value <= self.config.nlist:
            raise ValueError("nprobe must be between 1 and nlist")
        self._index.nprobe = value

    def search(self, queries: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
        """Scores and item ids of the ``k`` best items per query, best first.

        An approximate index can find fewer than ``k`` items when the clusters it opened
        are small; the missing places hold id -1.
        """
        rows = _matrix(queries, "queries")
        if rows.shape[1] != self.dim:
            raise ValueError(f"queries have {rows.shape[1]} dimensions, the index has {self.dim}")
        if not 1 <= k <= self.size:
            raise ValueError(f"k must be between 1 and the {self.size} items indexed")
        scores, ids = self._index.search(rows, k)
        return scores, ids.astype(np.int64)


def retrieve(
    index: ItemIndex,
    queries: np.ndarray,
    k: int,
    exclude: Sequence[Collection[int]] | None = None,
) -> np.ndarray:
    """The ``k`` best item ids per query after removing each query's excluded items.

    A recommender must not return what the user already has. Filtering after the search
    could leave fewer than ``k`` items, so the search asks for ``k`` plus the largest
    exclusion list and the filter keeps the first ``k`` that survive. Places that cannot
    be filled hold -1.
    """
    rows = _matrix(queries, "queries")
    if exclude is None:
        return index.search(rows, k)[1]
    if len(exclude) != len(rows):
        raise ValueError("exclude needs one collection per query")
    if not 1 <= k <= index.size:
        raise ValueError(f"k must be between 1 and the {index.size} items indexed")
    fetch = min(index.size, k + max(len(e) for e in exclude))
    _, found = index.search(rows, fetch)
    out = np.full((len(rows), k), -1, dtype=np.int64)
    for row, (ids, skip) in enumerate(zip(found, exclude, strict=True)):
        kept = [int(i) for i in ids if i >= 0 and int(i) not in skip][:k]
        out[row, : len(kept)] = kept
    return out


def overlap_recall(found: np.ndarray, exact: np.ndarray) -> float:
    """The share of the exact top-k that the approximate search also returned, averaged
    over queries. 1.0 means the approximate index lost nothing."""
    if found.shape != exact.shape or found.ndim != 2 or found.size == 0:
        raise ValueError("found and exact must be non-empty matrices of the same shape")
    hits = sum(
        len(set(a.tolist()) & set(b.tolist()) - {-1}) for a, b in zip(found, exact, strict=True)
    )
    return hits / exact.size


def hit_rate(found: np.ndarray, targets: np.ndarray) -> float:
    """Recall@k for one held-out item per query: the share of queries whose target is
    among the ids found."""
    if len(found) != len(targets) or len(targets) == 0:
        raise ValueError("need one target per query")
    return float(np.mean([target in row for row, target in zip(found, targets, strict=True)]))


@dataclass(frozen=True)
class Measurement:
    name: str
    recall: float  # overlap with the exact top-k
    ms_per_query: float  # median over single-query searches
    build_seconds: float


def _single_query_ms(index: ItemIndex, queries: np.ndarray, k: int) -> float:
    """Median time of one search with one query, the way a request arrives when serving."""
    times = []
    for row in queries:
        started = time.perf_counter()
        index.search(row[None, :], k)
        times.append(time.perf_counter() - started)
    return float(np.median(times)) * 1000


def compare(
    embeddings: np.ndarray,
    queries: np.ndarray,
    *,
    k: int = 20,
    nlist: int = 100,
    nprobes: Sequence[int] = (1, 4, 16),
    timed_queries: int = 500,
    seed: int = 0,
) -> list[Measurement]:
    """Exact search and IVF at several ``nprobe`` values on the same queries: recall
    against the exact answer and single-query latency for each."""
    flat = ItemIndex(embeddings, IndexConfig("flat"))
    exact = flat.search(queries, k)[1]
    timed = _matrix(queries, "queries")[:timed_queries]
    rows = [Measurement("flat (exact)", 1.0, _single_query_ms(flat, timed, k), flat.build_seconds)]
    ivf = ItemIndex(embeddings, IndexConfig("ivf", nlist=nlist, nprobe=min(nprobes), seed=seed))
    for nprobe in sorted(nprobes):
        ivf.nprobe = nprobe
        found = ivf.search(queries, k)[1]
        rows.append(
            Measurement(
                f"ivf, nlist {nlist}, nprobe {nprobe}",
                overlap_recall(found, exact),
                _single_query_ms(ivf, timed, k),
                ivf.build_seconds,
            )
        )
    return rows
