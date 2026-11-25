import math
from pathlib import Path

import pytest
import torch

from recsys.data import IdMap, temporal_split
from recsys.datasets import synthetic_interactions
from recsys.model import (
    TowerConfig,
    TwoTower,
    item_log_probability,
    load,
    sampled_softmax_loss,
    save,
)
from recsys.train import TrainConfig, evaluate, fit


def test_towers_produce_unit_vectors_and_scaled_scores() -> None:
    model = TwoTower(TowerConfig(n_users=5, n_items=7, output_dim=4, temperature=0.5))
    users = model.user_embeddings(torch.tensor([0, 1]))
    items = model.item_embeddings()
    assert users.shape == (2, 4) and items.shape == (7, 4)
    assert torch.allclose(users.norm(dim=-1), torch.ones(2), atol=1e-5)
    logits = model.in_batch_logits(torch.tensor([0, 1]), torch.tensor([3, 4]))
    assert logits.shape == (2, 2)
    assert logits.abs().max() <= 1 / 0.5 + 1e-4


def test_loss_is_cross_entropy_with_the_diagonal_as_target() -> None:
    logits = torch.tensor([[2.0, 0.0], [0.0, 2.0]])
    expected = -math.log(math.exp(2) / (math.exp(2) + 1))
    assert float(sampled_softmax_loss(logits, torch.tensor([0, 1]))) == pytest.approx(expected)


def test_log_q_correction_subtracts_each_columns_log_probability() -> None:
    logits = torch.zeros(2, 2)
    log_q = torch.log(torch.tensor([0.9, 0.1]))
    corrected = float(sampled_softmax_loss(logits, torch.tensor([0, 1]), log_q))
    # Every logit becomes -log q of its column: about 0.105 for item 0, 2.303 for item 1.
    a, b = -math.log(0.9), -math.log(0.1)
    normaliser = math.log(math.exp(a) + math.exp(b))
    assert corrected == pytest.approx(((normaliser - a) + (normaliser - b)) / 2, rel=1e-5)


def test_a_repeated_item_in_the_batch_is_not_its_own_negative() -> None:
    logits = torch.tensor([[1.0, 1.0], [1.0, 1.0]])
    # Both rows are the same item: masking leaves only the positive, so the loss is zero.
    assert float(sampled_softmax_loss(logits, torch.tensor([5, 5]))) == pytest.approx(0.0)


def test_item_log_probability_floors_unseen_items() -> None:
    log_q = item_log_probability(torch.tensor([3, 1, 0]))
    assert torch.allclose(log_q[:2], torch.log(torch.tensor([0.75, 0.25])))
    assert log_q[2] == log_q[1] and torch.isfinite(log_q).all()


def test_shape_mismatch_is_refused() -> None:
    with pytest.raises(ValueError, match="square"):
        sampled_softmax_loss(torch.zeros(2, 3), torch.tensor([0, 1]))


@pytest.fixture(scope="module")
def trained():  # type: ignore[no-untyped-def]
    frame = synthetic_interactions(n_users=800, n_items=400, seed=1)
    ids = IdMap.fit(frame)
    split = temporal_split(ids.encode(frame))
    tower = TowerConfig(n_users=len(ids.users), n_items=len(ids.items), output_dim=16)
    model, history = fit(split.train, tower, TrainConfig(epochs=10, batch_size=512, seed=0))
    return split, model, history


def test_training_reduces_the_loss(trained) -> None:  # type: ignore[no-untyped-def]
    _, _, history = trained
    # The in-batch loss starts near log(512) = 6.24 and most of it cannot be removed: on
    # this data a user's next item is weakly predictable, so the drop is small but steady.
    assert history.losses[-1] < history.losses[0] - 0.1
    assert all(b <= a + 1e-3 for a, b in zip(history.losses, history.losses[1:], strict=False))


def test_the_model_beats_popularity_on_held_out_items(trained) -> None:  # type: ignore[no-untyped-def]
    split, model, _ = trained
    result = evaluate(model, split.validation, [split.train], k=20)
    assert result.users == split.validation["user_id"].nunique()
    assert result.model_recall > result.popularity_recall


def test_training_is_deterministic(trained) -> None:  # type: ignore[no-untyped-def]
    split, model, history = trained
    tower = model.config
    _, again_history = fit(split.train, tower, TrainConfig(epochs=10, batch_size=512, seed=0))
    assert again_history.losses == history.losses


def test_a_checkpoint_round_trips(trained, tmp_path: Path) -> None:  # type: ignore[no-untyped-def]
    _, model, _ = trained
    path = tmp_path / "model.pt"
    save(model, str(path), epoch=8)
    loaded, extra = load(str(path))
    assert extra == {"epoch": 8}
    ids = torch.arange(5)
    assert torch.equal(loaded.item_embeddings(ids), model.item_embeddings(ids))


def test_evaluation_refuses_impossible_settings(trained) -> None:  # type: ignore[no-untyped-def]
    split, model, _ = trained
    with pytest.raises(ValueError, match="between 1"):
        evaluate(model, split.validation, [split.train], k=10_000)
    with pytest.raises(ValueError, match="nothing"):
        evaluate(model, split.validation.iloc[:0], [split.train])
