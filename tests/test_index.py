import numpy as np
import pytest

from recsys.index import (
    IndexConfig,
    ItemIndex,
    compare,
    hit_rate,
    overlap_recall,
    retrieve,
)


def unit(rows: int, dim: int, seed: int) -> np.ndarray:
    values = np.random.default_rng(seed).normal(size=(rows, dim)).astype(np.float32)
    return values / np.linalg.norm(values, axis=1, keepdims=True)


def clustered(rows: int, dim: int, clusters: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    centres = rng.normal(size=(clusters, dim))
    values = centres[rng.integers(clusters, size=rows)] + 0.3 * rng.normal(size=(rows, dim))
    return (values / np.linalg.norm(values, axis=1, keepdims=True)).astype(np.float32)


ITEMS = clustered(4000, 16, clusters=40, seed=0)
QUERIES = clustered(200, 16, clusters=40, seed=1)


def brute_force(items: np.ndarray, queries: np.ndarray, k: int) -> np.ndarray:
    return np.argsort(-(queries @ items.T), axis=1, kind="stable")[:, :k]


def test_the_flat_index_agrees_with_scoring_every_item() -> None:
    scores, ids = ItemIndex(ITEMS).search(QUERIES, 10)
    assert np.array_equal(ids, brute_force(ITEMS, QUERIES, 10))
    assert np.all(np.diff(scores, axis=1) <= 0)  # best first
    assert scores[0, 0] == pytest.approx(float(QUERIES[0] @ ITEMS[ids[0, 0]]), abs=1e-5)


def test_an_ivf_index_that_opens_every_cluster_is_exact() -> None:
    ivf = ItemIndex(ITEMS, IndexConfig("ivf", nlist=32, nprobe=32))
    exact = ItemIndex(ITEMS).search(QUERIES, 10)[1]
    assert overlap_recall(ivf.search(QUERIES, 10)[1], exact) == 1.0


def test_opening_more_clusters_never_loses_recall_and_one_cluster_is_not_enough() -> None:
    exact = ItemIndex(ITEMS).search(QUERIES, 10)[1]
    ivf = ItemIndex(ITEMS, IndexConfig("ivf", nlist=64, nprobe=1))
    recalls = []
    for nprobe in (1, 2, 4, 8, 16, 64):
        ivf.nprobe = nprobe
        recalls.append(overlap_recall(ivf.search(QUERIES, 10)[1], exact))
    assert recalls == sorted(recalls)
    assert recalls[0] < 1.0
    assert recalls[-1] == 1.0


def test_the_same_seed_builds_the_same_index() -> None:
    config = IndexConfig("ivf", nlist=32, nprobe=2, seed=3)
    a = ItemIndex(ITEMS, config).search(QUERIES, 10)[1]
    b = ItemIndex(ITEMS, config).search(QUERIES, 10)[1]
    assert np.array_equal(a, b)


def test_retrieve_leaves_out_excluded_items_and_still_returns_k() -> None:
    index = ItemIndex(ITEMS)
    plain = retrieve(index, QUERIES[:3], 5)
    exclude = [set(plain[0, :2].tolist()), set(), {int(plain[2, 4])}]
    filtered = retrieve(index, QUERIES[:3], 5, exclude)
    assert filtered.shape == (3, 5)
    assert not set(filtered[0].tolist()) & exclude[0]
    # The two removed items are replaced by the next two best, in order.
    assert filtered[0].tolist() == brute_force(ITEMS, QUERIES[:1], 7)[0, 2:].tolist()
    assert np.array_equal(filtered[1], plain[1])
    assert int(plain[2, 4]) not in filtered[2]


def test_retrieve_pads_with_minus_one_when_too_few_items_remain() -> None:
    items = unit(6, 4, seed=2)
    found = retrieve(ItemIndex(items), unit(1, 4, seed=3), 5, [{0, 1, 2}])
    assert sorted(found[0, :3].tolist()) == [3, 4, 5]
    assert found[0, 3:].tolist() == [-1, -1]


def test_overlap_recall_and_hit_rate_on_a_case_worked_by_hand() -> None:
    exact = np.array([[1, 2, 3, 4], [5, 6, 7, 8]])
    found = np.array([[1, 2, 9, 4], [5, -1, -1, -1]])
    assert overlap_recall(found, exact) == pytest.approx(4 / 8)
    assert hit_rate(found, np.array([9, 6])) == 0.5
    with pytest.raises(ValueError, match="same shape"):
        overlap_recall(found, exact[:, :2])
    with pytest.raises(ValueError, match="one target per query"):
        hit_rate(found, np.array([1]))


def test_compare_reports_exact_search_first_and_ivf_by_nprobe() -> None:
    rows = compare(ITEMS, QUERIES, k=10, nlist=32, nprobes=(8, 1), timed_queries=20)
    assert [r.name for r in rows] == [
        "flat (exact)",
        "ivf, nlist 32, nprobe 1",
        "ivf, nlist 32, nprobe 8",
    ]
    assert rows[0].recall == 1.0
    assert rows[1].recall <= rows[2].recall <= 1.0
    assert all(r.ms_per_query > 0 and r.build_seconds >= 0 for r in rows)


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"kind": "tree"}, "kind must be one of"),
        ({"kind": "ivf", "nlist": 0}, "nlist must be at least 1"),
        ({"kind": "ivf", "nlist": 4, "nprobe": 5}, "nprobe must be between 1 and nlist"),
    ],
)
def test_impossible_index_settings_are_refused(kwargs: dict[str, object], message: str) -> None:
    with pytest.raises(ValueError, match=message):
        IndexConfig(**kwargs)  # type: ignore[arg-type]


def test_bad_inputs_are_refused_with_the_reason() -> None:
    index = ItemIndex(ITEMS)
    with pytest.raises(ValueError, match="non-empty matrix"):
        ItemIndex(np.zeros((0, 4)))
    with pytest.raises(ValueError, match="exceeds the 10 items"):
        ItemIndex(unit(10, 4, seed=0), IndexConfig("ivf", nlist=50))
    with pytest.raises(ValueError, match="8 dimensions, the index has 16"):
        index.search(unit(2, 8, seed=0), 5)
    with pytest.raises(ValueError, match="k must be between 1 and the 4000"):
        index.search(QUERIES, 4001)
    with pytest.raises(ValueError, match="only an ivf index"):
        index.nprobe = 2
    with pytest.raises(ValueError, match="one collection per query"):
        retrieve(index, QUERIES[:2], 5, [set()])
    with pytest.raises(ValueError, match="k must be between"):
        retrieve(index, QUERIES[:1], 0, [set()])
    ivf = ItemIndex(ITEMS, IndexConfig("ivf", nlist=8, nprobe=2))
    assert ivf.nprobe == 2 and index.nprobe == 1
    with pytest.raises(ValueError, match="between 1 and nlist"):
        ivf.nprobe = 9
