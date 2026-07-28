from __future__ import annotations

from dataclasses import dataclass
import json
import math
from statistics import median
import time
from typing import Any, Callable, Mapping

import torch

try:
    from torch.utils.flop_counter import FlopCounterMode as _FlopCounterMode
except (AttributeError, ImportError):  # pragma: no cover - depends on the torch build
    _FlopCounterMode = None


@dataclass(frozen=True)
class ComputeProfile:
    """Serializable measurements for one production optimizer step.

    ``estimated_matmul_flops`` is produced by PyTorch's dispatch-based FLOP
    counter over the exact callable supplied to :func:`profile_training_step`.
    PyTorch covers selected operations rather than every hardware instruction,
    so the result is deliberately labelled an estimate. Wall time is measured.
    """

    device: str
    flop_counter: str | None
    estimated_matmul_flops: int | None
    op_breakdown: dict[str, int]
    warmup_steps: int
    timed_steps: int
    examples_per_step: int
    wall_time_samples_seconds: tuple[float, ...]
    wall_time_seconds_median: float
    wall_time_seconds_iqr: float
    throughput_examples_per_second: float
    peak_cuda_memory_bytes: int | None
    schema_version: int = 1

    @property
    def flops_per_step_for_allocation(self) -> int:
        """Return the counter-derived estimate used for budget allocation."""

        if self.estimated_matmul_flops is not None and self.estimated_matmul_flops > 0:
            return self.estimated_matmul_flops
        raise ValueError(
            "profile has no positive counter-derived estimated_matmul_flops"
        )

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-compatible representation."""

        return {
            "schema_version": self.schema_version,
            "device": self.device,
            "flop_counter": self.flop_counter,
            "estimated_matmul_flops": self.estimated_matmul_flops,
            "op_breakdown": dict(self.op_breakdown),
            "warmup_steps": self.warmup_steps,
            "timed_steps": self.timed_steps,
            "examples_per_step": self.examples_per_step,
            "wall_time_samples_seconds": list(self.wall_time_samples_seconds),
            "wall_time_seconds_median": self.wall_time_seconds_median,
            "wall_time_seconds_iqr": self.wall_time_seconds_iqr,
            "throughput_examples_per_second": self.throughput_examples_per_second,
            "peak_cuda_memory_bytes": self.peak_cuda_memory_bytes,
        }

    def to_json(self, *, indent: int | None = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent, sort_keys=True)

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "ComputeProfile":
        """Reconstruct a profile from :meth:`to_dict` output."""

        return cls(
            schema_version=int(payload.get("schema_version", 1)),
            device=str(payload["device"]),
            flop_counter=(
                None if payload.get("flop_counter") is None else str(payload["flop_counter"])
            ),
            estimated_matmul_flops=_optional_int(payload.get("estimated_matmul_flops")),
            op_breakdown={
                str(name): int(count)
                for name, count in dict(payload.get("op_breakdown", {})).items()
            },
            warmup_steps=int(payload["warmup_steps"]),
            timed_steps=int(payload["timed_steps"]),
            examples_per_step=int(payload["examples_per_step"]),
            wall_time_samples_seconds=tuple(
                float(value) for value in payload["wall_time_samples_seconds"]
            ),
            wall_time_seconds_median=float(payload["wall_time_seconds_median"]),
            wall_time_seconds_iqr=float(payload["wall_time_seconds_iqr"]),
            throughput_examples_per_second=float(
                payload["throughput_examples_per_second"]
            ),
            peak_cuda_memory_bytes=_optional_int(payload.get("peak_cuda_memory_bytes")),
        )


def profile_training_step(
    production_training_step: Callable[[], Any],
    *,
    device: torch.device | str,
    examples_per_step: int,
    warmup_steps: int = 2,
    timed_steps: int = 7,
) -> ComputeProfile:
    """Profile the exact training step used by an experiment.

    ``production_training_step`` must perform the complete step whose compute is
    being compared: zeroing gradients, forward pass, all production losses,
    backward pass, and optimizer update. The callable is executed once inside
    ``FlopCounterMode`` when that API is available, then ``warmup_steps`` times,
    then ``timed_steps`` times. Accelerator timings are synchronized around
    every sample.
    """

    resolved_device = torch.device(device)
    _validate_profile_arguments(
        examples_per_step=examples_per_step,
        warmup_steps=warmup_steps,
        timed_steps=timed_steps,
    )

    estimated_matmul_flops: int | None = None
    op_breakdown: dict[str, int] = {}
    flop_counter_name: str | None = None
    if _FlopCounterMode is not None:
        counter = _FlopCounterMode(display=False)
        _synchronize(resolved_device)
        with counter:
            production_training_step()
        _synchronize(resolved_device)
        estimated_matmul_flops = int(counter.get_total_flops())
        op_breakdown = _global_op_breakdown(counter.get_flop_counts())
        flop_counter_name = "torch.utils.flop_counter.FlopCounterMode"

    for _ in range(warmup_steps):
        production_training_step()
    _synchronize(resolved_device)

    if resolved_device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(resolved_device)

    timings: list[float] = []
    for _ in range(timed_steps):
        _synchronize(resolved_device)
        started = time.perf_counter()
        production_training_step()
        _synchronize(resolved_device)
        timings.append(time.perf_counter() - started)

    median_seconds = float(median(timings))
    iqr_seconds = _percentile(timings, 0.75) - _percentile(timings, 0.25)
    peak_memory = (
        int(torch.cuda.max_memory_allocated(resolved_device))
        if resolved_device.type == "cuda"
        else None
    )
    return ComputeProfile(
        device=str(resolved_device),
        flop_counter=flop_counter_name,
        estimated_matmul_flops=estimated_matmul_flops,
        op_breakdown=op_breakdown,
        warmup_steps=warmup_steps,
        timed_steps=timed_steps,
        examples_per_step=examples_per_step,
        wall_time_samples_seconds=tuple(timings),
        wall_time_seconds_median=median_seconds,
        wall_time_seconds_iqr=float(iqr_seconds),
        throughput_examples_per_second=examples_per_step / median_seconds,
        peak_cuda_memory_bytes=peak_memory,
    )


def allocate_equal_flop_steps(
    profiles_or_flops: Mapping[str, ComputeProfile | int],
    *,
    reference_model: str,
    reference_steps: int,
    max_ratio: float = 1.05,
) -> dict[str, int]:
    """Allocate integer steps around one reference model's FLOP budget.

    The nearest positive integer step count is selected for every non-reference
    model. The allocation is rejected when integer rounding cannot satisfy the
    requested max/min total-FLOP ratio.
    """

    flops_per_step = _resolve_flops_per_step(profiles_or_flops)
    if reference_model not in flops_per_step:
        raise ValueError(f"unknown reference_model: {reference_model}")
    if not _is_int(reference_steps) or reference_steps < 1:
        raise ValueError("reference_steps must be a positive integer")
    _validate_max_ratio(max_ratio)

    target_flops = flops_per_step[reference_model] * reference_steps
    allocation: dict[str, int] = {}
    for model, per_step in flops_per_step.items():
        if model == reference_model:
            allocation[model] = reference_steps
            continue
        ideal_steps = target_flops / per_step
        lower = max(1, math.floor(ideal_steps))
        upper = max(1, math.ceil(ideal_steps))
        allocation[model] = min(
            {lower, upper},
            key=lambda steps: (abs(steps * per_step - target_flops), steps),
        )

    validate_equal_flop_allocation(
        flops_per_step,
        allocation,
        max_ratio=max_ratio,
    )
    return allocation


def realized_training_flops(
    profiles_or_flops: Mapping[str, ComputeProfile | int],
    steps: Mapping[str, int],
) -> dict[str, int]:
    """Return total training FLOPs implied by an allocation."""

    flops_per_step = _resolve_flops_per_step(profiles_or_flops)
    if set(steps) != set(flops_per_step):
        missing = sorted(set(flops_per_step) - set(steps))
        extra = sorted(set(steps) - set(flops_per_step))
        raise ValueError(f"step models do not match profiles; missing={missing}, extra={extra}")
    totals: dict[str, int] = {}
    for model, step_count in steps.items():
        if not _is_int(step_count) or step_count < 1:
            raise ValueError(f"steps for {model!r} must be a positive integer")
        totals[model] = flops_per_step[model] * step_count
    return totals


def validate_equal_flop_allocation(
    profiles_or_flops: Mapping[str, ComputeProfile | int],
    steps: Mapping[str, int],
    *,
    max_ratio: float = 1.05,
) -> float:
    """Validate and return the realized largest/smallest FLOP ratio."""

    _validate_max_ratio(max_ratio)
    totals = realized_training_flops(profiles_or_flops, steps)
    ratio = max(totals.values()) / min(totals.values())
    if ratio > max_ratio + 1e-12:
        raise ValueError(
            f"equal-FLOP allocation ratio {ratio:.6f} exceeds allowed {max_ratio:.6f}; "
            "increase the reference budget to reduce integer-rounding error"
        )
    return ratio


def _resolve_flops_per_step(
    profiles_or_flops: Mapping[str, ComputeProfile | int],
) -> dict[str, int]:
    if not profiles_or_flops:
        raise ValueError("at least one model profile is required")
    resolved: dict[str, int] = {}
    for model, value in profiles_or_flops.items():
        if not model:
            raise ValueError("model names must be nonempty")
        if isinstance(value, ComputeProfile):
            flops = value.flops_per_step_for_allocation
        elif _is_int(value):
            flops = value
        else:
            raise TypeError(
                f"FLOPs for {model!r} must be an int or ComputeProfile, got {type(value)!r}"
            )
        if flops <= 0:
            raise ValueError(f"FLOPs per step for {model!r} must be positive")
        resolved[model] = int(flops)
    return resolved


def _global_op_breakdown(
    counts: Mapping[str, Mapping[Any, int]],
) -> dict[str, int]:
    global_counts = counts.get("Global", {})
    breakdown: dict[str, int] = {}
    for operation, flop_count in global_counts.items():
        name = str(operation)
        breakdown[name] = breakdown.get(name, 0) + int(flop_count)
    return dict(sorted(breakdown.items()))


def _synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elif device.type == "mps" and hasattr(torch, "mps"):
        torch.mps.synchronize()
    elif device.type == "xpu" and hasattr(torch, "xpu"):
        torch.xpu.synchronize(device)


def _percentile(values: list[float], quantile: float) -> float:
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    position = (len(ordered) - 1) * quantile
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def _validate_profile_arguments(
    *,
    examples_per_step: int,
    warmup_steps: int,
    timed_steps: int,
) -> None:
    if not _is_int(examples_per_step) or examples_per_step < 1:
        raise ValueError("examples_per_step must be a positive integer")
    if not _is_int(warmup_steps) or warmup_steps < 0:
        raise ValueError("warmup_steps must be a nonnegative integer")
    if not _is_int(timed_steps) or timed_steps < 1:
        raise ValueError("timed_steps must be a positive integer")


def _validate_max_ratio(max_ratio: float) -> None:
    if not 1.0 <= max_ratio <= 1.05:
        raise ValueError("max_ratio must be between 1.0 and the protocol limit 1.05")


def _optional_int(value: Any) -> int | None:
    return None if value is None else int(value)


def _is_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)
