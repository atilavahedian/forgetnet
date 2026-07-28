from __future__ import annotations

import json

import pytest
import torch
from torch import nn

from forgetnet.profiling import (
    ComputeProfile,
    allocate_equal_flop_steps,
    profile_training_step,
    realized_training_flops,
    validate_equal_flop_allocation,
)


def test_profile_exact_production_step_is_serializable_and_timed() -> None:
    torch.manual_seed(7)
    model = nn.Linear(4, 3)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    inputs = torch.randn(2, 4)

    def production_step() -> None:
        optimizer.zero_grad(set_to_none=True)
        loss = model(inputs).square().mean()
        loss.backward()
        optimizer.step()

    profile = profile_training_step(
        production_step,
        device="cpu",
        examples_per_step=2,
        warmup_steps=0,
        timed_steps=3,
    )

    assert profile.device == "cpu"
    assert profile.peak_cuda_memory_bytes is None
    assert len(profile.wall_time_samples_seconds) == 3
    assert profile.wall_time_seconds_median > 0.0
    assert profile.wall_time_seconds_iqr >= 0.0
    assert profile.throughput_examples_per_second == pytest.approx(
        2 / profile.wall_time_seconds_median
    )

    if profile.flop_counter is not None:
        assert profile.flop_counter == "torch.utils.flop_counter.FlopCounterMode"
        assert profile.estimated_matmul_flops == 96
        assert profile.op_breakdown == {"aten.addmm": 48, "aten.mm": 48}
        assert sum(profile.op_breakdown.values()) == profile.estimated_matmul_flops

    payload = json.loads(profile.to_json())
    restored = ComputeProfile.from_dict(payload)
    assert restored == profile


def test_profile_allocation_uses_counter_derived_matmul_estimate() -> None:
    reference = _profile(estimated=100)
    comparison = _profile(estimated=125)

    steps = allocate_equal_flop_steps(
        {"reference": reference, "comparison": comparison},
        reference_model="reference",
        reference_steps=10,
    )

    assert steps == {"reference": 10, "comparison": 8}
    assert realized_training_flops(
        {"reference": reference, "comparison": comparison}, steps
    ) == {"reference": 1000, "comparison": 1000}
    assert validate_equal_flop_allocation(
        {"reference": reference, "comparison": comparison}, steps
    ) == 1.0


def test_equal_flop_allocation_rejects_more_than_five_percent_rounding_error() -> None:
    with pytest.raises(ValueError, match="exceeds allowed"):
        allocate_equal_flop_steps(
            {"reference": 100, "expensive": 260},
            reference_model="reference",
            reference_steps=3,
        )

    with pytest.raises(ValueError, match="protocol limit"):
        validate_equal_flop_allocation(
            {"a": 100, "b": 100},
            {"a": 1, "b": 1},
            max_ratio=1.06,
        )


def test_allocation_requires_positive_available_flops_and_matching_models() -> None:
    unavailable = _profile(estimated=None)
    with pytest.raises(ValueError, match="no positive"):
        allocate_equal_flop_steps(
            {"missing": unavailable},
            reference_model="missing",
            reference_steps=10,
        )

    with pytest.raises(ValueError, match="missing=.*b"):
        realized_training_flops({"a": 10, "b": 10}, {"a": 2})


def test_profile_argument_validation_is_fast_and_explicit() -> None:
    with pytest.raises(ValueError, match="examples_per_step"):
        profile_training_step(lambda: None, device="cpu", examples_per_step=0)
    with pytest.raises(ValueError, match="timed_steps"):
        profile_training_step(
            lambda: None,
            device="cpu",
            examples_per_step=1,
            timed_steps=0,
        )
    with pytest.raises(ValueError, match="warmup_steps"):
        profile_training_step(
            lambda: None,
            device="cpu",
            examples_per_step=1,
            warmup_steps=0.5,  # type: ignore[arg-type]
        )


def _profile(*, estimated: int | None) -> ComputeProfile:
    return ComputeProfile(
        device="cpu",
        flop_counter=(
            "torch.utils.flop_counter.FlopCounterMode" if estimated is not None else None
        ),
        estimated_matmul_flops=estimated,
        op_breakdown={},
        warmup_steps=0,
        timed_steps=1,
        examples_per_step=1,
        wall_time_samples_seconds=(1.0,),
        wall_time_seconds_median=1.0,
        wall_time_seconds_iqr=0.0,
        throughput_examples_per_second=1.0,
        peak_cuda_memory_bytes=None,
    )
