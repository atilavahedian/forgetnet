from __future__ import annotations

from dataclasses import replace
import math

import pytest
import torch

from forgetnet.data import IGNORE_INDEX, QUERY_STABLE, QUERY_UPDATED, make_task_batch
from forgetnet.models import MemoryStats, ModelOutput
from forgetnet.selective_metrics import (
    area_under_risk_coverage,
    conflict_acceptance,
    expected_calibration_error,
    extract_pair_locality_rows,
    extract_query_rows,
    paired_seed_summary,
    read_jsonl_gz,
    relative_stale_reduction,
    summarize_pair_locality_rows,
    summarize_query_rows,
    tensor_sha256,
    write_jsonl_gz,
)


def _output(answer_logits: torch.Tensor | None, *, batch_size: int = 1) -> ModelOutput:
    vocab_size = answer_logits.shape[-1] if answer_logits is not None else 8
    logits = (
        answer_logits[:, -1, :]
        if answer_logits is not None
        else torch.zeros(batch_size, vocab_size)
    )
    return ModelOutput(
        logits=logits,
        answer_logits=answer_logits,
        memory_stats=MemoryStats(
            final_memory_shape=(batch_size, 0, vocab_size),
            write_frequency=0.0,
            mean_write_strength=0.0,
            mean_surprise=0.0,
        ),
    )


def _paired_batch(*, batch_size: int = 2):
    return make_task_batch(
        "conflict_stream",
        batch_size=batch_size,
        seq_len=24,
        seed=1_901,
        window_size=2,
        active_keys=2,
        updates_per_key=1,
        paired=True,
    )


def _calibrated_logits(batch) -> torch.Tensor:
    batch_size, sequence_length = batch.answer_targets.shape
    vocab_size = batch.vocab_size
    logits = torch.zeros(batch_size, sequence_length, vocab_size)
    for batch_index, position in (batch.answer_targets != IGNORE_INDEX).nonzero().tolist():
        target = int(batch.answer_targets[batch_index, position])
        kind = int(batch.query_kinds[batch_index, position])
        probabilities = torch.empty(vocab_size)
        if kind == QUERY_UPDATED:
            stale = next(
                int(value)
                for value in batch.stale_targets[batch_index, position]
                if int(value) != IGNORE_INDEX
            )
            probabilities.fill_(0.1 / (vocab_size - 2))
            probabilities[target] = 0.6
            probabilities[stale] = 0.3
        else:
            probabilities.fill_(0.2 / (vocab_size - 1))
            probabilities[target] = 0.8
        logits[batch_index, position] = probabilities.log()
    return logits


def test_query_extraction_deduplicates_stale_mass_and_computes_metrics() -> None:
    batch = _paired_batch()
    stale_targets = batch.stale_targets.clone()
    for batch_index, position in (batch.query_kinds == QUERY_UPDATED).nonzero().tolist():
        stale_targets[batch_index, position, 1] = stale_targets[
            batch_index,
            position,
            0,
        ]
    batch = replace(batch, stale_targets=stale_targets)
    rows = extract_query_rows(
        _output(_calibrated_logits(batch), batch_size=2),
        batch,
        stream_ids=[101, 102],
        condition="iid",
    )

    assert len(rows) == 4
    updated = [row for row in rows if row.query_kind == QUERY_UPDATED]
    assert all(len(row.stale_values) == 1 for row in updated)
    assert all(row.target_probability == pytest.approx(0.6, abs=1e-6) for row in updated)
    assert all(row.stale_probability == pytest.approx(0.3, abs=1e-6) for row in updated)
    assert all(row.nll == pytest.approx(-math.log(0.6), abs=1e-6) for row in updated)
    expected_brier = 0.6**2 + 0.3**2 + 126 * (0.1 / 126) ** 2 - 1.2 + 1.0
    assert all(row.brier == pytest.approx(expected_brier, abs=1e-6) for row in updated)

    summary = summarize_query_rows(rows, ece_bins=4)
    assert summary["stable_accuracy"] == 1.0
    assert summary["update_accuracy"] == 1.0
    assert summary["stale_intrusion_rate"] == 0.0
    assert summary["stale_probability"] == pytest.approx(0.3, abs=1e-6)
    assert 0.0 <= summary["ece"] <= 1.0
    assert summary["aurc"] == 0.0


def test_query_extraction_fails_closed_on_invalid_supervision_or_logits() -> None:
    batch = _paired_batch()
    logits = _calibrated_logits(batch)
    updated_index = (batch.query_kinds == QUERY_UPDATED).nonzero()[0]
    batch_index, position = (int(updated_index[0]), int(updated_index[1]))

    stale_targets = batch.stale_targets.clone()
    stale_targets[batch_index, position, 0] = batch.answer_targets[
        batch_index,
        position,
    ]
    invalid_batch = replace(batch, stale_targets=stale_targets)
    with pytest.raises(ValueError, match="also marked stale"):
        extract_query_rows(_output(logits, batch_size=2), invalid_batch)

    invalid_logits = logits.clone()
    invalid_logits[batch_index, position, 0] = float("nan")
    with pytest.raises(ValueError, match="non-finite"):
        extract_query_rows(_output(invalid_logits, batch_size=2), batch)

    with pytest.raises(ValueError, match="per-token answer logits"):
        extract_query_rows(_output(None, batch_size=2), batch)


def test_calibration_metrics_have_hand_computed_values() -> None:
    confidences = [0.9, 0.8]
    correctness = [1, 0]

    assert expected_calibration_error(confidences, correctness, bins=2) == pytest.approx(0.45)
    assert area_under_risk_coverage(confidences, correctness) == pytest.approx(0.25)

    with pytest.raises(ValueError, match="aligned"):
        expected_calibration_error([0.5], [], bins=2)
    with pytest.raises(ValueError, match=r"\[0, 1\]"):
        area_under_risk_coverage([1.1], [1])


def test_pair_locality_is_symmetric_and_macro_averaged() -> None:
    batch = make_task_batch(
        "conflict_stream",
        batch_size=4,
        seq_len=60,
        seed=1_902,
        window_size=6,
        active_keys=4,
        updates_per_key=2,
        paired=True,
    )
    logits = torch.zeros(4, 60, batch.vocab_size)
    for batch_index, position in (batch.answer_targets != IGNORE_INDEX).nonzero().tolist():
        logits[batch_index, position, batch.answer_targets[batch_index, position]] = 8.0

    identical = extract_pair_locality_rows(_output(logits, batch_size=4), batch)
    assert all(row.correctness_disagreement == 0.0 for row in identical)
    assert all(row.target_probability_shift == 0.0 for row in identical)
    assert all(row.symmetric_kl == pytest.approx(0.0, abs=1e-12) for row in identical)

    shifted = logits.clone()
    variant_one = batch.pair_variants == 1
    stable = batch.query_kinds == QUERY_STABLE
    shifted_rows = shifted[variant_one.unsqueeze(1) & stable]
    shifted_rows.zero_()
    shifted_rows[:, 0] = 8.0
    shifted[variant_one.unsqueeze(1) & stable] = shifted_rows
    locality = extract_pair_locality_rows(_output(shifted, batch_size=4), batch)
    summary = summarize_pair_locality_rows(locality)

    assert summary["pairs"] == 2
    assert summary["correctness_disagreement"] == 1.0
    assert summary["target_probability_shift"] > 0.0
    assert summary["symmetric_kl"] > 0.0
    untouched = extract_pair_locality_rows(
        _output(shifted, batch_size=4),
        batch,
        scope="untouched",
    )
    assert len(untouched) > len(locality)

    broken = replace(batch, pair_variants=torch.zeros_like(batch.pair_variants))
    with pytest.raises(ValueError, match="duplicate variant"):
        extract_pair_locality_rows(_output(shifted, batch_size=4), broken)


def test_gzip_jsonl_and_tensor_hashing_are_deterministic(tmp_path) -> None:
    batch = _paired_batch()
    rows = extract_query_rows(_output(_calibrated_logits(batch), batch_size=2), batch)
    first = write_jsonl_gz(tmp_path / "first.jsonl.gz", rows)
    second = write_jsonl_gz(tmp_path / "second.jsonl.gz", rows)

    assert first.read_bytes() == second.read_bytes()
    loaded = read_jsonl_gz(first)
    assert len(loaded) == len(rows)
    assert loaded[0]["stale_values"] == list(rows[0].stale_values)

    base = torch.arange(12, dtype=torch.int64).reshape(3, 4)
    same_values_noncontiguous = base.t().contiguous().t()
    assert not same_values_noncontiguous.is_contiguous()
    assert tensor_sha256({"x": base}) == tensor_sha256(
        {"x": same_values_noncontiguous}
    )
    assert tensor_sha256({"x": base}) != tensor_sha256({"x": base.float()})
    assert tensor_sha256({"x": base}) != tensor_sha256({"y": base})


def test_paired_seed_inference_is_deterministic_and_exact_for_small_n() -> None:
    candidate = {3: 0.8, 1: 0.6, 2: 0.7}
    baseline = {1: 0.5, 2: 0.6, 3: 0.7}
    summary = paired_seed_summary(
        candidate,
        baseline,
        direction="higher",
        bootstrap_draws=512,
        statistics_seed=77,
    )
    repeated = paired_seed_summary(
        candidate,
        baseline,
        direction="higher",
        bootstrap_draws=512,
        statistics_seed=77,
    )

    assert summary == repeated
    assert summary["benefit_delta"]["mean"] == pytest.approx(0.1)
    assert summary["benefit_delta"]["ci95"] == pytest.approx([0.1, 0.1])
    assert summary["sign_flip"]["method"] == "exact"
    assert summary["sign_flip"]["pvalue_two_sided"] == pytest.approx(0.25)

    with pytest.raises(ValueError, match="paired units differ"):
        paired_seed_summary({1: 1.0, 2: 1.0}, {1: 0.0, 3: 0.0}, direction="higher")


def test_relative_stale_reduction_and_acceptance_gates_fail_closed() -> None:
    stale_candidate = {1: 0.07, 2: 0.14, 3: 0.21}
    stale_baseline = {1: 0.10, 2: 0.20, 3: 0.30}
    reduction = relative_stale_reduction(
        stale_candidate,
        stale_baseline,
        bootstrap_draws=512,
        statistics_seed=91,
    )
    assert reduction["estimate"] == pytest.approx(0.30)
    assert reduction["ci95"] == pytest.approx([0.30, 0.30])

    baseline = {
        "stable_accuracy": {1: 0.80, 2: 0.82, 3: 0.84},
        "update_accuracy": {1: 0.60, 2: 0.62, 3: 0.64},
        "stale_probability": stale_baseline,
    }
    candidate = {
        "stable_accuracy": {1: 0.79, 2: 0.81, 3: 0.83},
        "update_accuracy": {1: 0.66, 2: 0.68, 3: 0.70},
        "stale_probability": stale_candidate,
    }
    accepted = conflict_acceptance(
        candidate,
        baseline,
        candidate_parameters=104,
        baseline_parameters=100,
        candidate_flops=104.0,
        baseline_flops=100.0,
        bootstrap_draws=512,
        statistics_seed=101,
    )
    assert accepted["primary_pass"] is True
    assert all(accepted["gates"].values())

    compute_failure = conflict_acceptance(
        candidate,
        baseline,
        candidate_parameters=104,
        baseline_parameters=100,
        candidate_flops=110.0,
        baseline_flops=100.0,
        bootstrap_draws=128,
    )
    assert compute_failure["primary_pass"] is False
    assert compute_failure["gates"]["flop_match"] is False

    malformed = conflict_acceptance(
        {"stable_accuracy": candidate["stable_accuracy"]},
        baseline,
        candidate_parameters=104,
        baseline_parameters=100,
        candidate_flops=104.0,
        baseline_flops=100.0,
        bootstrap_draws=128,
    )
    assert malformed["primary_pass"] is False
    assert not any(malformed["gates"].values())
    assert "missing required metric" in malformed["error"]
