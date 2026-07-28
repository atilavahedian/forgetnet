from __future__ import annotations

import torch
import torch.nn.functional as F

from forgetnet.data import IGNORE_INDEX, make_task_batch
from forgetnet.models import (
    ConflictLocalizedMemory,
    ConflictLocalizedMemoryCell,
    ForgetNet,
    TinyTransformer,
    build_model,
    count_parameters,
)


def test_forgetnet_forward_returns_logits_and_bounded_memory_stats() -> None:
    batch = make_task_batch("changing_facts", batch_size=3, seq_len=24, seed=5)
    model = ForgetNet(vocab_size=batch.vocab_size, d_model=32, memory_slots=6, window_size=4)

    output = model(batch.input_ids)

    assert output.logits.shape == (3, batch.vocab_size)
    assert output.aux_logits is not None
    assert output.aux_logits.shape == (3, 24, batch.vocab_size)
    assert output.answer_logits is not None
    assert output.answer_logits.shape == (3, 24, batch.vocab_size)
    assert output.memory_stats.final_memory_shape == (3, 6, 32)
    assert 0.0 <= output.memory_stats.write_frequency <= 1.0
    assert 0.0 <= output.memory_stats.mean_write_strength <= 1.0
    assert 0.0 <= output.memory_stats.mean_surprise <= 1.0


def test_tiny_transformer_baseline_matches_forgetnet_output_contract() -> None:
    batch = make_task_batch("associative_lookup", batch_size=2, seq_len=20, seed=1)
    model = TinyTransformer(vocab_size=batch.vocab_size, d_model=32, n_layers=1, n_heads=4)

    output = model(batch.input_ids)

    assert output.logits.shape == (2, batch.vocab_size)
    assert output.aux_logits is not None
    assert output.aux_logits.shape == (2, 20, batch.vocab_size)
    assert output.answer_logits is not None
    assert output.answer_logits.shape == (2, 20, batch.vocab_size)
    assert output.memory_stats.final_memory_shape == (2, 0, 32)


def test_model_factory_builds_memory_ablations() -> None:
    batch = make_task_batch("needle_recall", batch_size=2, seq_len=20, seed=4)

    for name in ["forgetnet", "no_forget", "no_surprise", "random_write", "fifo_memory"]:
        model = build_model(name, vocab_size=batch.vocab_size, d_model=24, memory_slots=4)
        output = model(batch.input_ids)
        assert output.logits.shape == (2, batch.vocab_size)


def test_parameter_count_is_positive() -> None:
    model = ForgetNet(vocab_size=128, d_model=32, memory_slots=4)

    assert count_parameters(model) > 0


def test_cldm_cell_allocates_matches_and_localizes_replacement() -> None:
    cell = ConflictLocalizedMemoryCell(d_model=4, memory_slots=2, variant="cldm")
    state = cell.empty_state(1, device=torch.device("cpu"), dtype=torch.float32)
    empty_read, empty_weights = cell.read(state, torch.tensor([[1.0, 0.0, 0.0, 0.0]]))

    assert torch.equal(empty_read, torch.zeros_like(empty_read))
    assert torch.equal(empty_weights, torch.zeros_like(empty_weights))
    assert torch.isfinite(empty_read).all()

    key = torch.tensor([[1.0, 0.0, 0.0, 0.0]])
    first_value = torch.tensor([[0.0, 1.0, 0.0, 0.0]])
    state, first_trace = cell.write(
        state,
        key,
        first_value,
        torch.tensor([True]),
    )
    first_slot = int(first_trace.slot_index[0])
    first_memory_value = state.values[0, first_slot].clone()

    assert state.occupied.sum().item() == 1
    assert first_trace.localization.item() == 1.0

    state, repeat_trace = cell.write(
        state,
        key,
        first_value,
        torch.tensor([True]),
    )
    assert state.occupied.sum().item() == 1
    assert int(repeat_trace.slot_index[0]) == first_slot
    assert repeat_trace.match_probability.item() > 0.98

    changed_value = torch.tensor([[0.0, 0.0, 1.0, 0.0]])
    before_distance = torch.linalg.vector_norm(first_memory_value - torch.tanh(changed_value))
    state, changed_trace = cell.write(
        state,
        key,
        changed_value,
        torch.tensor([True]),
    )
    after_distance = torch.linalg.vector_norm(
        state.values[0, first_slot] - torch.tanh(changed_value)
    )

    assert int(changed_trace.slot_index[0]) == first_slot
    assert changed_trace.conflict.item() > 0.25
    assert after_distance < before_distance


def test_cldm_no_replace_preserves_matched_value() -> None:
    cell = ConflictLocalizedMemoryCell(
        d_model=4,
        memory_slots=2,
        variant="cldm_no_replace",
    )
    state = cell.empty_state(1, device=torch.device("cpu"), dtype=torch.float32)
    key = torch.tensor([[1.0, 0.0, 0.0, 0.0]])
    state, trace = cell.write(
        state,
        key,
        torch.tensor([[0.0, 1.0, 0.0, 0.0]]),
        torch.tensor([True]),
    )
    slot = int(trace.slot_index[0])
    preserved = state.values[0, slot].clone()

    state, trace = cell.write(
        state,
        key,
        torch.tensor([[0.0, 0.0, 1.0, 0.0]]),
        torch.tensor([True]),
    )

    assert trace.match_probability.item() > 0.98
    assert trace.write_gate.item() < 1e-6
    assert torch.allclose(state.values[0, slot], preserved, atol=1e-7, rtol=0.0)


def test_cldm_full_capacity_insertion_evicts_one_slot() -> None:
    cell = ConflictLocalizedMemoryCell(d_model=3, memory_slots=2, variant="cldm")
    state = cell.empty_state(1, device=torch.device("cpu"), dtype=torch.float32)
    keys = [
        torch.tensor([[1.0, 0.0, 0.0]]),
        torch.tensor([[0.0, 1.0, 0.0]]),
        torch.tensor([[0.0, 0.0, 1.0]]),
    ]
    for key in keys:
        state, _ = cell.write(state, key, key, torch.tensor([True]))

    assert state.occupied.sum().item() == 2
    similarities = torch.einsum("sd,d->s", state.keys[0], keys[-1][0])
    assert similarities.max().item() > 0.99


def test_cldm_variants_are_parameter_matched() -> None:
    counts = {
        name: count_parameters(
            build_model(name, vocab_size=128, d_model=16, memory_slots=4)
        )
        for name in (
            "cldm",
            "cldm_entangled",
            "cldm_no_conflict",
            "cldm_no_replace",
            "cldm_shuffled_conflict",
            "cldm_soft_route",
        )
    }

    assert len(set(counts.values())) == 1


def test_cldm_entangled_ablation_overwrites_its_address_with_content() -> None:
    cell = ConflictLocalizedMemoryCell(
        d_model=4,
        memory_slots=2,
        variant="cldm_entangled",
    )
    state = cell.empty_state(1, device=torch.device("cpu"), dtype=torch.float32)
    key = torch.tensor([[1.0, 0.0, 0.0, 0.0]])
    value = torch.tensor([[0.0, 1.0, 0.0, 0.0]])

    state, trace = cell.write(state, key, value, torch.tensor([True]))
    slot = int(trace.slot_index[0])

    assert F.cosine_similarity(state.keys[:, slot], value).item() > 0.99
    assert F.cosine_similarity(state.keys[:, slot], key).item() < 0.01


def test_shuffled_conflict_control_uses_another_streams_gate() -> None:
    cell = ConflictLocalizedMemoryCell(
        d_model=2,
        memory_slots=1,
        variant="cldm_shuffled_conflict",
    )
    state = cell.empty_state(2, device=torch.device("cpu"), dtype=torch.float32)
    key = torch.tensor([[1.0, 0.0], [1.0, 0.0]])
    initial = torch.tensor([[1.0, 0.0], [1.0, 0.0]])
    state, _ = cell.write(state, key, initial, torch.tensor([True, True]))
    changed = torch.tensor([[1.0, 0.0], [-1.0, 0.0]])
    _, trace = cell.write(state, key, changed, torch.tensor([True, True]))

    assert trace.conflict[0].item() < trace.conflict[1].item()
    assert trace.write_gate[0].item() > 0.5
    assert trace.write_gate[1].item() < 0.02


def test_cldm_sequence_trace_and_gradients_follow_delayed_queries() -> None:
    batch = make_task_batch(
        "conflict_stream",
        batch_size=4,
        seq_len=48,
        seed=808,
        window_size=4,
        active_keys=4,
        updates_per_key=1,
        paired=True,
    )
    model = ConflictLocalizedMemory(
        vocab_size=batch.vocab_size,
        d_model=16,
        memory_slots=4,
        window_size=4,
        max_seq_len=64,
    )
    output = model(batch.input_ids, trace_mode="full", return_state=True)

    assert output.answer_logits is not None
    assert output.answer_logits.shape == (4, 48, batch.vocab_size)
    assert output.memory_trace is not None
    assert output.memory_trace.write_mask.shape == (4, 48)
    assert output.memory_trace.route_weights is not None
    assert output.memory_trace.route_weights.shape == (4, 48, 4)
    assert output.memory_state is not None
    assert output.memory_state.occupied.all()
    assert torch.isfinite(output.answer_logits).all()
    assert torch.equal(
        output.memory_trace.slot_index[~output.memory_trace.write_mask],
        torch.full_like(
            output.memory_trace.slot_index[~output.memory_trace.write_mask], -1
        ),
    )

    query_mask = batch.answer_targets != IGNORE_INDEX
    loss = F.cross_entropy(
        output.answer_logits[query_mask],
        batch.answer_targets[query_mask],
    )
    loss.backward()

    for parameter in (
        model.token_embedding.weight,
        model.key_proj.weight,
        model.value_proj.weight,
        model.answer_head.weight,
    ):
        assert parameter.grad is not None
        assert torch.isfinite(parameter.grad).all()
        assert parameter.grad.abs().sum().item() > 0.0


def test_gru_factory_matches_sequence_answer_contract() -> None:
    batch = make_task_batch("associative_lookup", batch_size=2, seq_len=20, seed=19)
    model = build_model("gru", vocab_size=batch.vocab_size, d_model=16)

    output = model(batch.input_ids)

    assert output.logits.shape == (2, batch.vocab_size)
    assert output.answer_logits is not None
    assert output.answer_logits.shape == (2, 20, batch.vocab_size)
