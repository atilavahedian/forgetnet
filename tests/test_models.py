from __future__ import annotations

import torch

from forgetnet.data import make_task_batch
from forgetnet.models import ForgetNet, TinyTransformer, build_model, count_parameters


def test_forgetnet_forward_returns_logits_and_bounded_memory_stats() -> None:
    batch = make_task_batch("changing_facts", batch_size=3, seq_len=24, seed=5)
    model = ForgetNet(vocab_size=batch.vocab_size, d_model=32, memory_slots=6, window_size=4)

    output = model(batch.input_ids)

    assert output.logits.shape == (3, batch.vocab_size)
    assert output.memory_stats.final_memory_shape == (3, 6, 32)
    assert 0.0 <= output.memory_stats.write_frequency <= 1.0
    assert 0.0 <= output.memory_stats.mean_write_strength <= 1.0


def test_tiny_transformer_baseline_matches_forgetnet_output_contract() -> None:
    batch = make_task_batch("associative_lookup", batch_size=2, seq_len=20, seed=1)
    model = TinyTransformer(vocab_size=batch.vocab_size, d_model=32, n_layers=1, n_heads=4)

    output = model(batch.input_ids)

    assert output.logits.shape == (2, batch.vocab_size)
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
