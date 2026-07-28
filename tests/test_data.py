from __future__ import annotations

import pytest
import torch

from forgetnet.data import (
    IGNORE_INDEX,
    KEY_START,
    QUERY_STABLE,
    QUERY_TOKEN,
    QUERY_UPDATED,
    SEP_TOKEN,
    SET_TOKEN,
    VALUE_START,
    conflict_oracle_targets,
    make_task_batch,
)


def test_batches_are_deterministic_for_same_seed() -> None:
    first = make_task_batch("associative_lookup", batch_size=4, seq_len=32, seed=17)
    second = make_task_batch("associative_lookup", batch_size=4, seq_len=32, seed=17)

    assert torch.equal(first.input_ids, second.input_ids)
    assert torch.equal(first.labels, second.labels)
    assert first.metadata == second.metadata


def test_changing_facts_label_uses_latest_value_before_query() -> None:
    batch = make_task_batch("changing_facts", batch_size=8, seq_len=40, seed=9)

    for row, label in zip(batch.input_ids.tolist(), batch.labels.tolist(), strict=True):
        query_pos = row.index(QUERY_TOKEN)
        query_key = row[query_pos + 1]
        latest_value = None
        for idx in range(0, query_pos - 2, 3):
            key, value, sep = row[idx], row[idx + 1], row[idx + 2]
            if sep == SEP_TOKEN and key == query_key:
                latest_value = value
        assert latest_value == label


def test_changing_facts_has_no_perfect_fixed_tail_shortcut() -> None:
    batch = make_task_batch("changing_facts", batch_size=512, seq_len=40, seed=91)

    best_tail_accuracy = max(
        (batch.input_ids[:, -offset] == batch.labels).float().mean().item()
        for offset in range(1, 16)
    )

    assert best_tail_accuracy < 0.30


def test_multi_hop_label_follows_two_edges() -> None:
    batch = make_task_batch("multi_hop", batch_size=8, seq_len=36, seed=22)

    for row, label in zip(batch.input_ids.tolist(), batch.labels.tolist(), strict=True):
        query_pos = row.index(QUERY_TOKEN)
        start_key = row[query_pos + 1]
        edges = {}
        for idx in range(0, query_pos - 2, 3):
            src, dst, sep = row[idx], row[idx + 1], row[idx + 2]
            if sep == SEP_TOKEN:
                edges[src] = dst
        assert edges[edges[start_key]] == label


def test_multi_hop_remains_valid_across_many_seeds() -> None:
    for seed in range(250):
        batch = make_task_batch("multi_hop", batch_size=16, seq_len=36, seed=seed)
        for row, label in zip(batch.input_ids.tolist(), batch.labels.tolist(), strict=True):
            query_pos = row.index(QUERY_TOKEN)
            start_key = row[query_pos + 1]
            edges = {
                row[index]: row[index + 1]
                for index in range(0, query_pos - 2, 3)
                if row[index + 2] == SEP_TOKEN
            }
            assert edges[edges[start_key]] == label


def test_generated_tokens_stay_inside_declared_vocab() -> None:
    batch = make_task_batch("needle_recall", batch_size=16, seq_len=48, seed=3)

    assert int(batch.input_ids.min()) >= 0
    assert int(batch.input_ids.max()) < batch.vocab_size
    assert int(batch.labels.min()) >= VALUE_START
    assert int(batch.labels.max()) < batch.vocab_size
    assert KEY_START < VALUE_START


def test_conflict_stream_oracle_and_query_lag_are_exact() -> None:
    window_size = 6
    batch = make_task_batch(
        "conflict_stream",
        batch_size=8,
        seq_len=60,
        seed=101,
        window_size=window_size,
        active_keys=4,
        updates_per_key=2,
    )

    assert torch.equal(conflict_oracle_targets(batch.input_ids), batch.answer_targets)
    assert (batch.answer_targets != IGNORE_INDEX).sum(dim=1).tolist() == [4] * 8
    assert set(batch.query_kinds[batch.query_kinds > 0].tolist()) == {
        QUERY_STABLE,
        QUERY_UPDATED,
    }

    for row_index, row in enumerate(batch.input_ids.tolist()):
        last_set: dict[int, int] = {}
        for position in range(len(row) - 2):
            if row[position] == SET_TOKEN:
                last_set[row[position + 1]] = position + 2
            if batch.answer_targets[row_index, position] != IGNORE_INDEX:
                key = int(batch.query_keys[row_index, position])
                assert position - last_set[key] > 2 * window_size
                assert batch.lag_tokens[row_index, position] == position - last_set[key]
                stale_or_current = {
                    int(value)
                    for value in batch.stale_targets[row_index, position]
                    if value != IGNORE_INDEX
                }
                stale_or_current.add(int(batch.answer_targets[row_index, position]))
                local_tail = row[max(0, position - 2 * window_size) : position]
                assert stale_or_current.isdisjoint(local_tail)


def test_conflict_stream_pairs_change_one_event_and_preserve_controls() -> None:
    batch = make_task_batch(
        "conflict_stream",
        batch_size=6,
        seq_len=60,
        seed=303,
        window_size=6,
        active_keys=4,
        updates_per_key=2,
        paired=True,
    )

    for pair_id in batch.pair_ids.unique():
        members = (batch.pair_ids == pair_id).nonzero(as_tuple=False).flatten()
        assert members.numel() == 2
        first = int(members[batch.pair_variants[members].argmin()])
        second = int(members[batch.pair_variants[members].argmax()])
        assert batch.pair_ids[first] == batch.pair_ids[second]
        assert batch.pair_variants[first].item() == 0
        assert batch.pair_variants[second].item() == 1
        differences = batch.input_ids[first] != batch.input_ids[second]
        assert differences.sum().item() == 1
        changed_key = batch.changed_keys[first]
        stable_mask = (batch.query_kinds[first] == QUERY_STABLE)
        changed_mask = batch.query_keys[first] == changed_key
        assert torch.equal(
            batch.answer_targets[first][stable_mask],
            batch.answer_targets[second][stable_mask],
        )
        assert changed_mask.any()
        assert not torch.equal(
            batch.answer_targets[first][changed_mask],
            batch.answer_targets[second][changed_mask],
        )


def test_conflict_stream_is_deterministic_and_rejects_short_sequences() -> None:
    kwargs = {
        "task": "conflict_stream",
        "batch_size": 4,
        "seq_len": 60,
        "seed": 707,
        "window_size": 6,
        "active_keys": 4,
        "updates_per_key": 2,
        "paired": True,
    }
    first = make_task_batch(**kwargs)
    second = make_task_batch(**kwargs)

    for field in (
        "input_ids",
        "answer_targets",
        "query_kinds",
        "query_keys",
        "stale_targets",
        "event_targets",
        "pair_ids",
    ):
        assert torch.equal(getattr(first, field), getattr(second, field))

    with pytest.raises(ValueError, match="too short"):
        make_task_batch(
            "conflict_stream",
            batch_size=2,
            seq_len=30,
            seed=1,
            window_size=8,
            active_keys=4,
            updates_per_key=2,
        )


def test_conflict_stream_supports_explicit_long_lag_and_capacity_stress() -> None:
    minimum_lag = 32
    batch = make_task_batch(
        "conflict_stream",
        batch_size=2,
        seq_len=156,
        seed=909,
        num_keys=24,
        num_values=48,
        window_size=4,
        active_keys=16,
        updates_per_key=1,
        paired=True,
        minimum_query_lag=minimum_lag,
    )
    query_mask = batch.answer_targets != IGNORE_INDEX

    assert torch.equal(conflict_oracle_targets(batch.input_ids), batch.answer_targets)
    assert batch.metadata["active_keys"] == 16
    assert batch.metadata["minimum_query_lag_exclusive"] == minimum_lag
    assert (batch.lag_tokens[query_mask] > minimum_lag).all()

    with pytest.raises(ValueError, match="full local receptive field"):
        make_task_batch(
            "conflict_stream",
            batch_size=2,
            seq_len=48,
            seed=910,
            window_size=4,
            active_keys=4,
            updates_per_key=1,
            paired=True,
            minimum_query_lag=7,
        )
