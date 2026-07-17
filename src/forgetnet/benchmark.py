from __future__ import annotations

import csv
from dataclasses import asdict, dataclass, replace
import json
import math
from pathlib import Path
from statistics import mean, pstdev
import time
from typing import Any

from forgetnet.continual import ContinualConfig, DEFAULT_CONTINUAL_TASKS, run_continual
from forgetnet.data import TASKS
from forgetnet.experiment import ModelConfig
from forgetnet.runtime import to_jsonable

DEFAULT_BENCHMARK_MODELS = ("forgetnet", "no_forget", "local_transformer")


@dataclass(frozen=True)
class BenchmarkConfig:
    models: tuple[str, ...] = DEFAULT_BENCHMARK_MODELS
    seeds: tuple[int, ...] = (42, 43, 44)
    task_sequence: tuple[str, ...] = DEFAULT_CONTINUAL_TASKS
    eval_tasks: tuple[str, ...] = TASKS
    steps_per_task: int = 100
    eval_steps: int = 5
    batch_size: int = 32
    seq_len: int = 64
    extrapolate_len: int = 192
    lr: float = 3e-4
    aux_loss_weight: float = 0.1
    eval_seed: int = 12_345
    device: str = "auto"
    output_dir: str = "runs"
    quiet: bool = True
    model_config: ModelConfig = ModelConfig(model="forgetnet")
    model_widths: tuple[tuple[str, int], ...] = ()


def run_benchmark(config: BenchmarkConfig) -> Path:
    _validate_config(config)
    benchmark_dir = _new_benchmark_dir(config.output_dir)
    rows: list[dict[str, Any]] = []
    model_widths = dict(config.model_widths)
    for model_name in config.models:
        for seed in config.seeds:
            continual_config = ContinualConfig(
                task_sequence=config.task_sequence,
                eval_tasks=config.eval_tasks,
                steps_per_task=config.steps_per_task,
                eval_steps=config.eval_steps,
                batch_size=config.batch_size,
                seq_len=config.seq_len,
                extrapolate_len=config.extrapolate_len,
                lr=config.lr,
                aux_loss_weight=config.aux_loss_weight,
                seed=seed,
                eval_seed=config.eval_seed,
                device=config.device,
                output_dir=str(benchmark_dir / "runs"),
                quiet=config.quiet,
                model_config=replace(
                    config.model_config,
                    model=model_name,
                    d_model=model_widths.get(model_name, config.model_config.d_model),
                ),
            )
            run_dir = run_continual(continual_config)
            metrics = json.loads((run_dir / "metrics.json").read_text())
            continual = metrics["continual"]
            rows.append(
                {
                    "model": model_name,
                    "seed": seed,
                    "parameter_count": metrics["parameter_count"],
                    "wall_time_seconds": metrics["wall_time_seconds"],
                    "final_average_accuracy": continual["final_average_accuracy"],
                    "final_learned_task_accuracy": continual["final_learned_task_accuracy"],
                    "mean_immediate_accuracy": continual["mean_immediate_accuracy"],
                    "mean_forgetting": continual["mean_forgetting"],
                    "mean_backward_transfer": continual["mean_backward_transfer"],
                    "run_dir": str(run_dir.relative_to(benchmark_dir)),
                }
            )

    aggregates = {
        model_name: _aggregate_model([row for row in rows if row["model"] == model_name])
        for model_name in config.models
    }
    ranking = sorted(
        config.models,
        key=lambda model: (
            -aggregates[model]["final_learned_task_accuracy"]["mean"],
            aggregates[model]["mean_forgetting"]["mean"],
        ),
    )
    summary = {
        "kind": "continual_benchmark",
        "protocol": "equal-update-paired-evaluation-v2",
        "config": to_jsonable(asdict(config)),
        "run_count": len(rows),
        "paired_eval_seed": config.eval_seed,
        "rows": rows,
        "aggregates": aggregates,
        "parameter_count_ratio": _parameter_count_ratio(aggregates),
        "ranking": ranking,
        "deltas_from_forgetnet": _forgetnet_deltas(aggregates),
    }
    (benchmark_dir / "benchmark_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n"
    )
    _write_csv(benchmark_dir / "benchmark_runs.csv", rows)
    return benchmark_dir


def _aggregate_model(rows: list[dict[str, Any]]) -> dict[str, Any]:
    metrics = (
        "parameter_count",
        "wall_time_seconds",
        "final_average_accuracy",
        "final_learned_task_accuracy",
        "mean_immediate_accuracy",
        "mean_forgetting",
        "mean_backward_transfer",
    )
    aggregate: dict[str, Any] = {"seeds": [row["seed"] for row in rows]}
    for metric in metrics:
        values = [float(row[metric]) for row in rows]
        standard_deviation = pstdev(values) if len(values) > 1 else 0.0
        aggregate[metric] = {
            "mean": mean(values),
            "std": standard_deviation,
            "ci95_half_width": 1.96 * standard_deviation / math.sqrt(len(values)),
            "values": values,
        }
    return aggregate


def _forgetnet_deltas(aggregates: dict[str, Any]) -> dict[str, Any]:
    reference = aggregates.get("forgetnet")
    if reference is None:
        return {}
    deltas = {}
    for model, aggregate in aggregates.items():
        if model == "forgetnet":
            continue
        deltas[model] = {
            "final_learned_task_accuracy": (
                aggregate["final_learned_task_accuracy"]["mean"]
                - reference["final_learned_task_accuracy"]["mean"]
            ),
            "mean_forgetting": (
                aggregate["mean_forgetting"]["mean"]
                - reference["mean_forgetting"]["mean"]
            ),
        }
    return deltas


def _parameter_count_ratio(aggregates: dict[str, Any]) -> float:
    counts = [aggregate["parameter_count"]["mean"] for aggregate in aggregates.values()]
    return max(counts) / min(counts)


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fieldnames = [
        "model",
        "seed",
        "parameter_count",
        "wall_time_seconds",
        "final_average_accuracy",
        "final_learned_task_accuracy",
        "mean_immediate_accuracy",
        "mean_forgetting",
        "mean_backward_transfer",
        "run_dir",
    ]
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _validate_config(config: BenchmarkConfig) -> None:
    if not config.models or len(set(config.models)) != len(config.models):
        raise ValueError("models must be a nonempty unique sequence")
    if not config.seeds or len(set(config.seeds)) != len(config.seeds):
        raise ValueError("seeds must be a nonempty unique sequence")
    widths = dict(config.model_widths)
    if len(widths) != len(config.model_widths):
        raise ValueError("model_widths must contain unique model names")
    unknown = set(widths) - set(config.models)
    if unknown:
        raise ValueError(f"model_widths contains models outside the benchmark: {sorted(unknown)}")
    if any(width < 1 for width in widths.values()):
        raise ValueError("model widths must be positive")


def _new_benchmark_dir(root: str | Path) -> Path:
    root_dir = Path(root)
    root_dir.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime("%Y%m%d-%H%M%S")
    candidate = root_dir / f"continual-benchmark-{stamp}"
    suffix = 1
    while candidate.exists():
        suffix += 1
        candidate = root_dir / f"continual-benchmark-{stamp}-{suffix}"
    candidate.mkdir(parents=True)
    return candidate
