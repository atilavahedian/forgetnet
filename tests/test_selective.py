from __future__ import annotations

from dataclasses import replace

import pytest
import torch

from forgetnet.data import IGNORE_INDEX, QUERY_STABLE, TaskBatch, make_task_batch
from forgetnet.experiment import supervised_answer_accuracy, supervised_answer_loss
from forgetnet.models import MemoryStats, ModelOutput
from forgetnet.selective import counterfactual_locality_loss


def _output(answer_logits: torch.Tensor) -> ModelOutput:
    return ModelOutput(
        logits=answer_logits[:, -1, :],
        answer_logits=answer_logits,
        memory_stats=MemoryStats(
            final_memory_shape=(answer_logits.shape[0], 0, answer_logits.shape[-1]),
            write_frequency=0.0,
            mean_write_strength=0.0,
            mean_surprise=0.0,
        ),
    )


def _permute_batch(batch: TaskBatch, order: torch.Tensor) -> TaskBatch:
    fields = (
        "input_ids",
        "labels",
        "answer_targets",
        "query_kinds",
        "query_keys",
        "stale_targets",
        "event_targets",
        "lag_tokens",
        "pair_ids",
        "pair_variants",
        "changed_keys",
        "update_positions",
    )
    return replace(
        batch,
        **{field: getattr(batch, field)[order] for field in fields},
    )


def test_supervised_answer_objective_uses_every_query_and_ignores_other_tokens() -> None:
    batch = make_task_batch(
        "conflict_stream",
        batch_size=4,
        seq_len=24,
        seed=901,
        window_size=2,
        active_keys=2,
        updates_per_key=1,
        paired=True,
    )
    logits = torch.zeros(4, 24, batch.vocab_size)
    query_mask = batch.answer_targets != IGNORE_INDEX
    logits[query_mask, batch.answer_targets[query_mask]] = 20.0
    logits[~query_mask, 0] = 1_000.0
    output = _output(logits)

    assert query_mask.sum().item() == 8
    assert supervised_answer_loss(output, batch).item() < 1e-6
    assert supervised_answer_accuracy(output, batch) == 1.0


def test_counterfactual_locality_is_zero_for_identical_pairs_and_order_invariant() -> None:
    batch = make_task_batch(
        "conflict_stream",
        batch_size=4,
        seq_len=24,
        seed=902,
        window_size=2,
        active_keys=2,
        updates_per_key=1,
        paired=True,
    )
    logits = torch.zeros(4, 24, batch.vocab_size)
    assert counterfactual_locality_loss(_output(logits), batch).item() == pytest.approx(0.0)

    shifted = logits.clone()
    variant_one = batch.pair_variants == 1
    stable_positions = batch.query_kinds == QUERY_STABLE
    shifted[variant_one.unsqueeze(-1) & stable_positions, 0] = 5.0
    original_loss = counterfactual_locality_loss(_output(shifted), batch)

    order = torch.tensor([2, 0, 3, 1])
    permuted_batch = _permute_batch(batch, order)
    permuted_loss = counterfactual_locality_loss(
        _output(shifted[order]),
        permuted_batch,
    )

    assert original_loss.item() > 0.0
    assert permuted_loss.item() == pytest.approx(original_loss.item(), rel=1e-6)


def test_counterfactual_locality_rejects_unpaired_batches() -> None:
    batch = make_task_batch(
        "conflict_stream",
        batch_size=2,
        seq_len=24,
        seed=903,
        window_size=2,
        active_keys=2,
        updates_per_key=1,
        paired=False,
    )
    logits = torch.zeros(2, 24, batch.vocab_size)

    with pytest.raises(ValueError, match="paired examples"):
        counterfactual_locality_loss(_output(logits), batch)
