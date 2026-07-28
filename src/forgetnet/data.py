from __future__ import annotations

from dataclasses import dataclass
import random
from typing import Any

import torch

PAD_TOKEN = 0
SEP_TOKEN = 1
QUERY_TOKEN = 2
NOISE_TOKEN = 3
SET_TOKEN = 4
READ_TOKEN = 5
KEY_START = 10
VALUE_START = 64
DEFAULT_VOCAB_SIZE = 128
DEFAULT_NUM_KEYS = 24
DEFAULT_NUM_VALUES = 48
IGNORE_INDEX = -100

QUERY_NONE = 0
QUERY_STABLE = 1
QUERY_UPDATED = 2

TASKS = (
    "associative_lookup",
    "changing_facts",
    "needle_recall",
    "multi_hop",
    "length_extrapolation",
    "conflict_stream",
)


@dataclass(frozen=True)
class TaskBatch:
    input_ids: torch.Tensor
    labels: torch.Tensor
    answer_targets: torch.Tensor
    query_kinds: torch.Tensor
    query_keys: torch.Tensor
    stale_targets: torch.Tensor
    event_targets: torch.Tensor
    lag_tokens: torch.Tensor
    pair_ids: torch.Tensor
    pair_variants: torch.Tensor
    changed_keys: torch.Tensor
    update_positions: torch.Tensor
    task: str
    vocab_size: int
    metadata: dict[str, Any]

    def to(self, device: torch.device | str) -> "TaskBatch":
        return TaskBatch(
            input_ids=self.input_ids.to(device),
            labels=self.labels.to(device),
            answer_targets=self.answer_targets.to(device),
            query_kinds=self.query_kinds.to(device),
            query_keys=self.query_keys.to(device),
            stale_targets=self.stale_targets.to(device),
            event_targets=self.event_targets.to(device),
            lag_tokens=self.lag_tokens.to(device),
            pair_ids=self.pair_ids.to(device),
            pair_variants=self.pair_variants.to(device),
            changed_keys=self.changed_keys.to(device),
            update_positions=self.update_positions.to(device),
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
    window_size: int = 8,
    active_keys: int = 4,
    updates_per_key: int = 2,
    paired: bool = False,
) -> TaskBatch:
    if task == "all":
        raise ValueError("task='all' is only valid for evaluation orchestration")
    if task not in TASKS:
        raise ValueError(f"unknown task: {task}")
    if seq_len < 14:
        raise ValueError("seq_len must be at least 14")
    if VALUE_START + num_values > vocab_size:
        raise ValueError("vocab_size is too small for requested values")

    if task == "conflict_stream":
        return _make_conflict_stream_batch(
            batch_size=batch_size,
            seq_len=seq_len,
            seed=seed,
            vocab_size=vocab_size,
            num_keys=num_keys,
            num_values=num_values,
            window_size=window_size,
            active_keys=active_keys,
            updates_per_key=updates_per_key,
            paired=paired,
        )

    rng = random.Random(seed)
    rows: list[list[int]] = []
    labels: list[int] = []
    for _ in range(batch_size):
        row, label = _make_one(task, seq_len, rng, num_keys, num_values)
        rows.append(row)
        labels.append(label)

    input_ids = torch.tensor(rows, dtype=torch.long)
    label_tensor = torch.tensor(labels, dtype=torch.long)
    supervision = _legacy_supervision(input_ids, label_tensor, task)
    return TaskBatch(
        input_ids=input_ids,
        labels=label_tensor,
        answer_targets=supervision["answer_targets"],
        query_kinds=supervision["query_kinds"],
        query_keys=supervision["query_keys"],
        stale_targets=supervision["stale_targets"],
        event_targets=supervision["event_targets"],
        lag_tokens=torch.full((batch_size, input_ids.shape[1]), -1, dtype=torch.long),
        pair_ids=torch.full((batch_size,), -1, dtype=torch.long),
        pair_variants=torch.full((batch_size,), -1, dtype=torch.long),
        changed_keys=torch.full((batch_size,), -1, dtype=torch.long),
        update_positions=torch.full((batch_size,), -1, dtype=torch.long),
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

    distractors: list[tuple[int, int]] = []
    for _ in range(max(0, n_records - 2)):
        key = _key(rng, num_keys)
        if key == query_key:
            key = KEY_START + ((key - KEY_START + 1) % num_keys)
        distractors.append((key, _value(rng, num_values)))

    # Put the correction away from the query instead of leaking it in the final
    # record. Its position varies, so no fixed tail offset can solve the task.
    latest_index = rng.randint(1, max(1, n_records - 3))
    records = [(query_key, stale_value), *distractors]
    records.insert(latest_index, (query_key, latest_value))
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
    allowed_distractor_keys = [
        KEY_START + offset
        for offset in range(num_keys)
        if KEY_START + offset not in blocked
    ]
    if not allowed_distractor_keys:
        raise ValueError("multi_hop requires at least three distinct keys")
    for _ in range(max(0, n_records - 2)):
        key = rng.choice(allowed_distractor_keys)
        records.append((key, _value(rng, num_values)))
    rng.shuffle(records)
    return _finish(records, start_key, label, seq_len)


@dataclass(frozen=True)
class _ConflictExample:
    tokens: list[int]
    answer_targets: list[int]
    query_kinds: list[int]
    query_keys: list[int]
    stale_targets: list[list[int]]
    event_targets: list[int]
    lag_tokens: list[int]
    pair_id: int
    pair_variant: int
    changed_key: int
    update_position: int


def _make_conflict_stream_batch(
    *,
    batch_size: int,
    seq_len: int,
    seed: int,
    vocab_size: int,
    num_keys: int,
    num_values: int,
    window_size: int,
    active_keys: int,
    updates_per_key: int,
    paired: bool,
) -> TaskBatch:
    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    if paired and batch_size % 2:
        raise ValueError("paired conflict batches require an even batch_size")
    if window_size < 1:
        raise ValueError("window_size must be positive")
    if active_keys < 2 or active_keys >= num_keys:
        raise ValueError("active_keys must be at least two and smaller than num_keys")
    if updates_per_key < 1:
        raise ValueError("updates_per_key must be positive")

    rng = random.Random(seed)
    examples: list[_ConflictExample] = []
    example_groups = batch_size // 2 if paired else batch_size
    for pair_index in range(example_groups):
        pair_id = seed * 1_000_000 + pair_index
        first, second = _make_conflict_pair(
            seq_len=seq_len,
            rng=rng,
            num_keys=num_keys,
            num_values=num_values,
            window_size=window_size,
            active_keys=active_keys,
            updates_per_key=updates_per_key,
            pair_id=pair_id,
        )
        examples.append(first)
        if paired:
            examples.append(second)

    # Pair membership must not be inferable from a fixed row position.
    rng.shuffle(examples)

    max_stale = updates_per_key + 1
    input_ids = torch.tensor([example.tokens for example in examples], dtype=torch.long)
    answer_targets = torch.tensor(
        [example.answer_targets for example in examples], dtype=torch.long
    )
    query_kinds = torch.tensor([example.query_kinds for example in examples], dtype=torch.long)
    query_keys = torch.tensor([example.query_keys for example in examples], dtype=torch.long)
    stale_targets = torch.tensor(
        [example.stale_targets for example in examples], dtype=torch.long
    )
    event_targets = torch.tensor(
        [example.event_targets for example in examples], dtype=torch.float32
    )
    lag_tokens = torch.tensor([example.lag_tokens for example in examples], dtype=torch.long)
    labels = torch.tensor(
        [next(target for target in reversed(example.answer_targets) if target != IGNORE_INDEX)
         for example in examples],
        dtype=torch.long,
    )
    return TaskBatch(
        input_ids=input_ids,
        labels=labels,
        answer_targets=answer_targets,
        query_kinds=query_kinds,
        query_keys=query_keys,
        stale_targets=stale_targets,
        event_targets=event_targets,
        lag_tokens=lag_tokens,
        pair_ids=torch.tensor([example.pair_id for example in examples], dtype=torch.long),
        pair_variants=torch.tensor(
            [example.pair_variant for example in examples], dtype=torch.long
        ),
        changed_keys=torch.tensor(
            [example.changed_key for example in examples], dtype=torch.long
        ),
        update_positions=torch.tensor(
            [example.update_position for example in examples], dtype=torch.long
        ),
        task="conflict_stream",
        vocab_size=vocab_size,
        metadata={
            "task": "conflict_stream",
            "batch_size": batch_size,
            "seq_len": seq_len,
            "seed": seed,
            "vocab_size": vocab_size,
            "num_keys": num_keys,
            "num_values": num_values,
            "window_size": window_size,
            "active_keys": active_keys,
            "updates_per_key": updates_per_key,
            "paired": paired,
            "max_stale_values": max_stale,
            "minimum_query_lag_exclusive": 2 * window_size,
        },
    )


def _make_conflict_pair(
    *,
    seq_len: int,
    rng: random.Random,
    num_keys: int,
    num_values: int,
    window_size: int,
    active_keys: int,
    updates_per_key: int,
    pair_id: int,
) -> tuple[_ConflictExample, _ConflictExample]:
    operation_count = seq_len // 3
    prefix_padding = seq_len - operation_count * 3
    conflict_count = max(1, active_keys // 2)
    query_count = active_keys
    minimum_gap_operations = (2 * window_size + 2) // 3
    required_operations = active_keys + conflict_count * updates_per_key + query_count
    filler_count = operation_count - required_operations
    if filler_count < minimum_gap_operations:
        minimum_length = 3 * (required_operations + minimum_gap_operations)
        raise ValueError(
            "seq_len is too short for leakage-resistant conflict_stream; "
            f"need at least {minimum_length} tokens"
        )

    active = rng.sample(range(KEY_START, KEY_START + num_keys), active_keys)
    rng.shuffle(active)
    conflict_keys = active[:conflict_count]
    stable_keys = active[conflict_count:]
    changed_key = conflict_keys[0]
    distractor_keys = [
        key for key in range(KEY_START, KEY_START + num_keys) if key not in active
    ]

    values_needed = active_keys + conflict_count * updates_per_key + 1
    value_pool = list(range(VALUE_START, VALUE_START + num_values))
    if values_needed > len(value_pool):
        raise ValueError("num_values is too small for distinct conflict values")
    reserved_values = rng.sample(value_pool, values_needed)
    chosen_values = iter(reserved_values[:-1])
    alternate_value = reserved_values[-1]
    background_values = [value for value in value_pool if value not in reserved_values]
    if not background_values:
        raise ValueError("num_values must leave at least one background value")

    initial_events = [(SET_TOKEN, key, next(chosen_values)) for key in active]
    rng.shuffle(initial_events)
    update_events: list[tuple[int, int, int]] = []
    for _ in range(updates_per_key):
        round_keys = list(conflict_keys)
        rng.shuffle(round_keys)
        update_events.extend((SET_TOKEN, key, next(chosen_values)) for key in round_keys)

    protected_gap = minimum_gap_operations
    movable_fillers = filler_count - protected_gap
    prefix_fillers = [
        (SET_TOKEN, rng.choice(distractor_keys), rng.choice(background_values))
        for _ in range(movable_fillers)
    ]
    middle_events = [*update_events, *prefix_fillers]
    rng.shuffle(middle_events)
    gap_events = [
        (SET_TOKEN, rng.choice(distractor_keys), rng.choice(background_values))
        for _ in range(protected_gap)
    ]
    query_order = [*conflict_keys, *stable_keys]
    rng.shuffle(query_order)
    query_events = [(QUERY_TOKEN, key, READ_TOKEN) for key in query_order]
    events = [*initial_events, *middle_events, *gap_events, *query_events]

    changed_event_indices = [
        index
        for index, (operation, key, _) in enumerate(events)
        if operation == SET_TOKEN and key == changed_key
    ]
    changed_event_index = changed_event_indices[-1]
    alternate_events = list(events)
    operation, key, _ = alternate_events[changed_event_index]
    alternate_events[changed_event_index] = (operation, key, alternate_value)

    first = _encode_conflict_events(
        events,
        prefix_padding=prefix_padding,
        seq_len=seq_len,
        conflict_keys=set(conflict_keys),
        changed_key=changed_key,
        pair_id=pair_id,
        pair_variant=0,
        max_stale=updates_per_key + 1,
        update_event_index=changed_event_index,
    )
    second = _encode_conflict_events(
        alternate_events,
        prefix_padding=prefix_padding,
        seq_len=seq_len,
        conflict_keys=set(conflict_keys),
        changed_key=changed_key,
        pair_id=pair_id,
        pair_variant=1,
        max_stale=updates_per_key + 1,
        update_event_index=changed_event_index,
    )
    return first, second


def _encode_conflict_events(
    events: list[tuple[int, int, int]],
    *,
    prefix_padding: int,
    seq_len: int,
    conflict_keys: set[int],
    changed_key: int,
    pair_id: int,
    pair_variant: int,
    max_stale: int,
    update_event_index: int,
) -> _ConflictExample:
    tokens = [PAD_TOKEN] * prefix_padding
    answer_targets = [IGNORE_INDEX] * prefix_padding
    query_kinds = [QUERY_NONE] * prefix_padding
    query_keys = [IGNORE_INDEX] * prefix_padding
    stale_targets = [[IGNORE_INDEX] * max_stale for _ in range(prefix_padding)]
    event_targets = [0] * prefix_padding
    lag_tokens = [-1] * prefix_padding
    state: dict[int, int] = {}
    history: dict[int, list[int]] = {}
    last_set_position: dict[int, int] = {}

    for operation, key, value_or_sep in events:
        base = len(tokens)
        tokens.extend([operation, key, value_or_sep])
        answer_targets.extend([IGNORE_INDEX, IGNORE_INDEX, IGNORE_INDEX])
        query_kinds.extend([QUERY_NONE, QUERY_NONE, QUERY_NONE])
        query_keys.extend([IGNORE_INDEX, IGNORE_INDEX, IGNORE_INDEX])
        stale_targets.extend([[IGNORE_INDEX] * max_stale for _ in range(3)])
        event_targets.extend([0, 0, 0])
        lag_tokens.extend([-1, -1, -1])
        target_position = base + 2
        if operation == SET_TOKEN:
            previous = state.get(key)
            if previous is not None:
                history.setdefault(key, []).append(previous)
            state[key] = value_or_sep
            event_targets[target_position] = 1
            last_set_position[key] = target_position
        else:
            answer_targets[target_position] = state[key]
            query_keys[target_position] = key
            query_kinds[target_position] = (
                QUERY_UPDATED if key in conflict_keys else QUERY_STABLE
            )
            stale = history.get(key, [])[-max_stale:]
            padded = [*stale, *([IGNORE_INDEX] * (max_stale - len(stale)))]
            stale_targets[target_position] = padded
            lag_tokens[target_position] = target_position - last_set_position[key]

    if len(tokens) != seq_len:
        raise AssertionError("conflict stream encoding did not fill seq_len")
    return _ConflictExample(
        tokens=tokens,
        answer_targets=answer_targets,
        query_kinds=query_kinds,
        query_keys=query_keys,
        stale_targets=stale_targets,
        event_targets=event_targets,
        pair_id=pair_id,
        pair_variant=pair_variant,
        changed_key=changed_key,
        update_position=prefix_padding + update_event_index * 3 + 2,
        lag_tokens=lag_tokens,
    )


def conflict_oracle_targets(input_ids: torch.Tensor) -> torch.Tensor:
    """Return exact last-write-wins targets at ConflictStream query events."""
    if input_ids.ndim != 2:
        raise ValueError("input_ids must have shape [batch, sequence]")
    targets = torch.full_like(input_ids, IGNORE_INDEX)
    for row_index, row in enumerate(input_ids.tolist()):
        state: dict[int, int] = {}
        cursor = 0
        while cursor < len(row):
            if row[cursor] == PAD_TOKEN:
                cursor += 1
                continue
            if cursor + 2 >= len(row):
                raise ValueError("truncated ConflictStream event")
            operation, key, value_or_sep = row[cursor : cursor + 3]
            if operation == SET_TOKEN:
                state[key] = value_or_sep
            elif operation == QUERY_TOKEN:
                if key not in state:
                    raise ValueError(f"query references unset key {key}")
                targets[row_index, cursor + 2] = state[key]
            else:
                raise ValueError(f"unknown ConflictStream operation token: {operation}")
            cursor += 3
    return targets


def _legacy_supervision(
    input_ids: torch.Tensor,
    labels: torch.Tensor,
    task: str,
) -> dict[str, torch.Tensor]:
    batch_size, seq_len = input_ids.shape
    answer_targets = torch.full((batch_size, seq_len), IGNORE_INDEX, dtype=torch.long)
    answer_targets[:, -1] = labels
    query_kinds = torch.full((batch_size, seq_len), QUERY_NONE, dtype=torch.long)
    query_kinds[:, -1] = QUERY_UPDATED if task == "changing_facts" else QUERY_STABLE
    query_keys = torch.full((batch_size, seq_len), IGNORE_INDEX, dtype=torch.long)
    event_targets = torch.zeros((batch_size, seq_len), dtype=torch.float32)
    stale_targets = torch.full((batch_size, seq_len, 1), IGNORE_INDEX, dtype=torch.long)
    for row_index, row in enumerate(input_ids.tolist()):
        query_position = row.index(QUERY_TOKEN)
        query_keys[row_index, -1] = row[query_position + 1]
        for position in range(1, query_position, 3):
            event_targets[row_index, position] = 1
    return {
        "answer_targets": answer_targets,
        "query_kinds": query_kinds,
        "query_keys": query_keys,
        "stale_targets": stale_targets,
        "event_targets": event_targets,
    }
