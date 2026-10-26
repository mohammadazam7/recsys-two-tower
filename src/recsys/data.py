"""Interaction data: schema validation, ID mapping and a temporal per-user split.

Every stage downstream consumes the same columns, ``user_id``, ``item_id``, ``rating`` and
``timestamp``, so the loaders for MovieLens and for the synthetic fallback both produce
this frame and nothing else.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

COLUMNS = ("user_id", "item_id", "rating", "timestamp")


def validate_interactions(frame: pd.DataFrame) -> pd.DataFrame:
    """Return ``frame`` restricted to the schema columns, or raise ``ValueError``."""
    missing = [c for c in COLUMNS if c not in frame.columns]
    if missing:
        raise ValueError(f"missing columns: {missing}")
    out = frame.loc[:, list(COLUMNS)]
    if out.isna().to_numpy().any():
        raise ValueError("interactions contain null values")
    if not pd.api.types.is_integer_dtype(out["timestamp"]):
        raise ValueError("timestamp must be integer seconds since the epoch")
    if out.duplicated(["user_id", "item_id", "timestamp"]).any():
        raise ValueError("duplicate (user_id, item_id, timestamp) rows")
    return out.reset_index(drop=True)


@dataclass(frozen=True)
class IdMap:
    """Dense, contiguous indices for raw IDs; embedding tables are indexed by these."""

    users: pd.Index
    items: pd.Index

    @classmethod
    def fit(cls, frame: pd.DataFrame) -> IdMap:
        return cls(
            users=pd.Index(np.sort(frame["user_id"].unique())),
            items=pd.Index(np.sort(frame["item_id"].unique())),
        )

    def encode(self, frame: pd.DataFrame) -> pd.DataFrame:
        """Add ``user_idx`` and ``item_idx`` columns; unknown IDs map to -1."""
        out = frame.copy()
        out["user_idx"] = self.users.get_indexer(pd.Index(out["user_id"]))
        out["item_idx"] = self.items.get_indexer(pd.Index(out["item_id"]))
        return out


@dataclass(frozen=True)
class Split:
    train: pd.DataFrame
    validation: pd.DataFrame
    test: pd.DataFrame


def temporal_split(frame: pd.DataFrame, holdout: int = 1, min_train: int = 3) -> Split:
    """Hold out each user's ``holdout`` latest interactions for test and the
    ``holdout`` before those for validation.

    Users with fewer than ``min_train + 2 * holdout`` interactions stay entirely in the
    training set, so every evaluated user has history to learn from. Ties on timestamp
    are broken by item ID so the split is deterministic.
    """
    if holdout < 1 or min_train < 1:
        raise ValueError("holdout and min_train must be positive")
    ordered = frame.sort_values(["user_id", "timestamp", "item_id"], kind="mergesort")
    rank_from_end = ordered.groupby("user_id").cumcount(ascending=False)
    counts = ordered.groupby("user_id")["item_id"].transform("size")
    eligible = counts >= min_train + 2 * holdout

    is_test = eligible & (rank_from_end < holdout)
    is_val = eligible & (rank_from_end >= holdout) & (rank_from_end < 2 * holdout)
    is_train = ~(is_test | is_val)
    return Split(
        train=ordered[is_train].reset_index(drop=True),
        validation=ordered[is_val].reset_index(drop=True),
        test=ordered[is_test].reset_index(drop=True),
    )
