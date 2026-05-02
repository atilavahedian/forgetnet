from __future__ import annotations

from dataclasses import dataclass
import random
from typing import Any

import torch

PAD_TOKEN = 0
SEP_TOKEN = 1
QUERY_TOKEN = 2
NOISE_TOKEN = 3
KEY_START = 10
VALUE_START = 64
DEFAULT_VOCAB_SIZE = 128
DEFAULT_NUM_KEYS = 24
DEFAULT_NUM_VALUES = 48

TASKS = (
    "associative_lookup",
    "changing_facts",
    "needle_recall",
    "multi_hop",
    "length_extrapolation",
)


@dataclass(frozen=True)
class TaskBatch:
    input_ids: torch.Tensor
    labels: torch.Tensor
    task: str
    vocab_size: int
    metadata: dict[str, Any]

    def to(self, device: torch.device | str) -> "TaskBatch":
        return TaskBatch(
            input_ids=self.input_ids.to(device),
            labels=self.labels.to(device),
            task=self.task,
            vocab_size=self.vocab_size,
            metadata=self.metadata,
        )


def make_task_batch(
    task: str,
    batch_size: int,
    seq_len: int,
    seed: int,
    vocab_size: int = DEFAULT_VOCAB_SIZE,
    num_keys: int = DEFAULT_NUM_KEYS,
    num_values: int = DEFAULT_NUM_VALUES,
) -> TaskBatch:
    if task == "all":
        raise ValueError("task='all' is only valid for evaluation orchestration")
    if task not in TASKS:
        raise ValueError(f"unknown task: {task}")
    if seq_len < 14:
        raise ValueError("seq_len must be at least 14")
    if VALUE_START + num_values > vocab_size:
        raise ValueError("vocab_size is too small for requested values")

    rng = random.Random(seed)
    rows: list[list[int]] = []
    labels: list[int] = []
    for _ in range(batch_size):
        row, label = _make_one(task, seq_len, rng, num_keys, num_values)
        rows.append(row)
        labels.append(label)

    return TaskBatch(
        input_ids=torch.tensor(rows, dtype=torch.long),
        labels=torch.tensor(labels, dtype=torch.long),
        task=task,
        vocab_size=vocab_size,
        metadata={
            "task": task,
            "batch_size": batch_size,
            "seq_len": seq_len,
            "seed": seed,
            "vocab_size": vocab_size,
            "num_keys": num_keys,
            "num_values": num_values,
        },
    )


def _make_one(
    task: str,
    seq_len: int,
    rng: random.Random,
    num_keys: int,
    num_values: int,
) -> tuple[list[int], int]:
    if task == "associative_lookup":
        return _associative_lookup(seq_len, rng, num_keys, num_values)
    if task == "changing_facts":
        return _changing_facts(seq_len, rng, num_keys, num_values)
    if task == "needle_recall":
        return _needle_recall(seq_len, rng, num_keys, num_values)
    if task == "multi_hop":
        return _multi_hop(seq_len, rng, num_keys, num_values)
    if task == "length_extrapolation":
        return _associative_lookup(seq_len, rng, num_keys, num_values)
    raise ValueError(f"unknown task: {task}")


def _key(rng: random.Random, num_keys: int) -> int:
    return KEY_START + rng.randrange(num_keys)


def _value(rng: random.Random, num_values: int) -> int:
    return VALUE_START + rng.randrange(num_values)


def _records_for(seq_len: int) -> int:
    return max(3, (seq_len - 2) // 3)


def _finish(records: list[tuple[int, int]], query_key: int, label: int, seq_len: int) -> tuple[list[int], int]:
    row: list[int] = []
    for key, value in records:
        row.extend([key, value, SEP_TOKEN])
    row.extend([QUERY_TOKEN, query_key])
    if len(row) < seq_len:
        row.extend([NOISE_TOKEN] * (seq_len - len(row)))
    return row[:seq_len], label


def _associative_lookup(
    seq_len: int,
    rng: random.Random,
    num_keys: int,
    num_values: int,
) -> tuple[list[int], int]:
    n_records = _records_for(seq_len)
    records: list[tuple[int, int]] = []
    state: dict[int, int] = {}
    for _ in range(n_records):
        key = _key(rng, num_keys)
        value = _value(rng, num_values)
        records.append((key, value))
        state[key] = value
    query_key = rng.choice(list(state.keys()))
    return _finish(records, query_key, state[query_key], seq_len)


def _changing_facts(
    seq_len: int,
    rng: random.Random,
    num_keys: int,
    num_values: int,
) -> tuple[list[int], int]:
    n_records = _records_for(seq_len)
    query_key = _key(rng, num_keys)
    stale_value = _value(rng, num_values)
    latest_value = _value(rng, num_values)
    while latest_value == stale_value:
        latest_value = _value(rng, num_values)

    records: list[tuple[int, int]] = [(query_key, stale_value)]
    for _ in range(max(0, n_records - 2)):
        key = _key(rng, num_keys)
        if key == query_key:
            key = KEY_START + ((key - KEY_START + 1) % num_keys)
        records.append((key, _value(rng, num_values)))
    records.append((query_key, latest_value))
    return _finish(records, query_key, latest_value, seq_len)


def _needle_recall(
    seq_len: int,
    rng: random.Random,
    num_keys: int,
    num_values: int,
) -> tuple[list[int], int]:
    n_records = _records_for(seq_len)
    query_key = _key(rng, num_keys)
    label = _value(rng, num_values)
    needle_index = rng.randrange(n_records)
    records: list[tuple[int, int]] = []
    for idx in range(n_records):
        if idx == needle_index:
            records.append((query_key, label))
        else:
            key = _key(rng, num_keys)
            if key == query_key:
                key = KEY_START + ((key - KEY_START + 1) % num_keys)
            records.append((key, _value(rng, num_values)))
    return _finish(records, query_key, label, seq_len)


def _multi_hop(
    seq_len: int,
    rng: random.Random,
    num_keys: int,
    num_values: int,
) -> tuple[list[int], int]:
    n_records = _records_for(seq_len)
    start_key = _key(rng, num_keys)
    mid_key = _key(rng, num_keys)
    while mid_key == start_key:
        mid_key = _key(rng, num_keys)
    label = _value(rng, num_values)

    records: list[tuple[int, int]] = [(start_key, mid_key), (mid_key, label)]
    blocked = {start_key, mid_key}
    for _ in range(max(0, n_records - 2)):
        key = _key(rng, num_keys)
        if key in blocked:
            key = KEY_START + ((key - KEY_START + 2) % num_keys)
        records.append((key, _value(rng, num_values)))
    rng.shuffle(records)
    return _finish(records, start_key, label, seq_len)
