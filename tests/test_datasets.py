from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from recsys.data import temporal_split
from recsys.datasets import load_movielens, synthetic_interactions


def test_load_movielens_renames_and_validates(tmp_path: Path) -> None:
    (tmp_path / "ratings.csv").write_text(
        "userId,movieId,rating,timestamp\n1,31,2.5,1260759144\n1,1029,3.0,1260759179\n"
    )
    frame = load_movielens(tmp_path)
    assert list(frame.columns) == ["user_id", "item_id", "rating", "timestamp"]
    assert frame["item_id"].tolist() == [31, 1029]


def test_load_movielens_reports_missing_file(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        load_movielens(tmp_path)


def test_synthetic_is_deterministic_for_a_seed() -> None:
    a = synthetic_interactions(n_users=20, n_items=50, interactions_per_user=(5, 10), seed=3)
    b = synthetic_interactions(n_users=20, n_items=50, interactions_per_user=(5, 10), seed=3)
    pd.testing.assert_frame_equal(a, b)


def test_synthetic_respects_bounds_and_rating_scale() -> None:
    frame = synthetic_interactions(n_users=30, n_items=80, interactions_per_user=(5, 15))
    per_user = frame.groupby("user_id").size()
    assert per_user.between(5, 15).all()
    assert frame["rating"].between(0.5, 5.0).all()
    assert ((frame["rating"] * 2) % 1 == 0).all()


def test_synthetic_has_popularity_skew() -> None:
    frame = synthetic_interactions(n_users=200, n_items=400, interactions_per_user=(20, 40))
    counts = frame["item_id"].value_counts()
    top_decile = counts.head(len(counts) // 10).sum()
    assert top_decile / counts.sum() > 0.2


def test_synthetic_works_with_temporal_split() -> None:
    split = temporal_split(
        synthetic_interactions(n_users=25, n_items=60, interactions_per_user=(10, 30))
    )
    assert split.test["user_id"].nunique() == 25
