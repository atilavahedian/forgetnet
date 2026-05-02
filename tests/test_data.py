from __future__ import annotations

import torch

from forgetnet.data import (
    KEY_START,
    QUERY_TOKEN,
    SEP_TOKEN,
    VALUE_START,
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


def test_generated_tokens_stay_inside_declared_vocab() -> None:
    batch = make_task_batch("needle_recall", batch_size=16, seq_len=48, seed=3)

    assert int(batch.input_ids.min()) >= 0
    assert int(batch.input_ids.max()) < batch.vocab_size
    assert int(batch.labels.min()) >= VALUE_START
    assert int(batch.labels.max()) < batch.vocab_size
    assert KEY_START < VALUE_START
