from __future__ import annotations

import pandas as pd
import pytest

from recsys.data import IdMap, temporal_split, validate_interactions


def _frame(rows: list[tuple[int, int, float, int]]) -> pd.DataFrame:
    return pd.DataFrame(rows, columns=["user_id", "item_id", "rating", "timestamp"])


@pytest.fixture
def interactions() -> pd.DataFrame:
    rows = [(1, item, 4.0, t) for item, t in zip(range(10, 16), range(100, 106), strict=True)]
    rows += [(2, 20, 3.0, 50), (2, 21, 5.0, 60)]
    return _frame(rows)


def test_validate_rejects_missing_columns() -> None:
    with pytest.raises(ValueError, match="missing columns"):
        validate_interactions(pd.DataFrame({"user_id": [1]}))


def test_validate_rejects_duplicates_and_nulls() -> None:
    with pytest.raises(ValueError, match="duplicate"):
        validate_interactions(_frame([(1, 2, 4.0, 10), (1, 2, 5.0, 10)]))
    with pytest.raises(ValueError, match="null"):
        validate_interactions(_frame([(1, 2, float("nan"), 10)]))


def test_validate_drops_extra_columns(interactions: pd.DataFrame) -> None:
    extra = interactions.assign(genre="drama")
    assert list(validate_interactions(extra).columns) == [
        "user_id",
        "item_id",
        "rating",
        "timestamp",
    ]


def test_split_holds_out_latest_interactions(interactions: pd.DataFrame) -> None:
    split = temporal_split(interactions, holdout=1, min_train=3)
    assert split.test["item_id"].tolist() == [15]
    assert split.validation["item_id"].tolist() == [14]
    assert sorted(split.train.loc[split.train["user_id"] == 1, "item_id"]) == [10, 11, 12, 13]


def test_split_keeps_short_histories_in_train(interactions: pd.DataFrame) -> None:
    split = temporal_split(interactions, holdout=1, min_train=3)
    assert set(split.train.loc[split.train["user_id"] == 2, "item_id"]) == {20, 21}
    assert 2 not in set(split.test["user_id"])


def test_split_never_trains_on_the_future(interactions: pd.DataFrame) -> None:
    split = temporal_split(interactions)
    for user, test_rows in split.test.groupby("user_id"):
        train_rows = split.train[split.train["user_id"] == user]
        assert train_rows["timestamp"].max() < test_rows["timestamp"].min()


def test_split_partitions_every_row(interactions: pd.DataFrame) -> None:
    split = temporal_split(interactions)
    assert len(split.train) + len(split.validation) + len(split.test) == len(interactions)


def test_id_map_is_dense_and_marks_unknown_ids(interactions: pd.DataFrame) -> None:
    ids = IdMap.fit(interactions)
    encoded = ids.encode(interactions)
    assert encoded["user_idx"].tolist()[:1] == [0]
    assert sorted(encoded["item_idx"].unique()) == list(range(len(ids.items)))
    unknown = ids.encode(_frame([(99, 999, 1.0, 1)]))
    assert unknown[["user_idx", "item_idx"]].iloc[0].tolist() == [-1, -1]
