from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path
import time
from typing import Any

import torch
from tqdm import trange

from forgetnet.data import IGNORE_INDEX, TASKS, make_task_batch
from forgetnet.experiment import (
    ModelConfig,
    next_token_auxiliary_loss,
    supervised_answer_accuracy,
    supervised_answer_loss,
)
from forgetnet.models import build_model, count_parameters
from forgetnet.runtime import seed_everything, select_device, to_jsonable

DEFAULT_CONTINUAL_TASKS = (
    "associative_lookup",
    "changing_facts",
    "needle_recall",
    "multi_hop",
)


@dataclass(frozen=True)
class ContinualConfig:
    task_sequence: tuple[str, ...] = DEFAULT_CONTINUAL_TASKS
    eval_tasks: tuple[str, ...] = TASKS
    steps_per_task: int = 100
    eval_steps: int = 5
    batch_size: int = 32
    seq_len: int = 64
    extrapolate_len: int = 192
    lr: float = 3e-4
    aux_loss_weight: float = 0.1
    seed: int = 42
    eval_seed: int = 12_345
    device: str = "auto"
    output_dir: str = "runs"
    quiet: bool = False
    model_config: ModelConfig = ModelConfig(model="forgetnet")


def run_continual(config: ContinualConfig) -> Path:
    _validate_config(config)
    seed_everything(config.seed)
    device = select_device(config.device)
    model = build_model(**asdict(config.model_config)).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.lr)
    run_dir = _new_run_dir(config.output_dir, f"continual-{config.model_config.model}")
    started = time.perf_counter()

    stages = [
        {
            "stage": 0,
            "trained_task": None,
            "training": None,
            "tasks": _evaluate_tasks(model, config, device),
        }
    ]
    for stage_index, task in enumerate(config.task_sequence, start=1):
        training = _train_stage(model, optimizer, task, stage_index, config, device)
        stages.append(
            {
                "stage": stage_index,
                "trained_task": task,
                "training": training,
                "tasks": _evaluate_tasks(model, config, device),
            }
        )

    continual = summarize_continual_metrics(stages, config.task_sequence, config.eval_tasks)
    metrics = {
        "kind": "continual",
        "model": config.model_config.model,
        "parameter_count": count_parameters(model),
        "device": str(device),
        "wall_time_seconds": time.perf_counter() - started,
        "config": to_jsonable(asdict(config)),
        "stages": stages,
        "continual": continual,
    }
    _write_json(run_dir / "metrics.json", metrics)
    torch.save(
        {
            "model_state": model.state_dict(),
            "model_config": asdict(config.model_config),
            "continual_config": to_jsonable(asdict(config)),
            "metrics": metrics,
        },
        run_dir / "checkpoint.pt",
    )
    return run_dir


def summarize_continual_metrics(
    stages: list[dict[str, Any]],
    task_sequence: tuple[str, ...],
    eval_tasks: tuple[str, ...],
) -> dict[str, Any]:
    if len(stages) != len(task_sequence) + 1:
        raise ValueError("stages must contain an initial evaluation and one stage per task")
    final_tasks = stages[-1]["tasks"]
    per_task: dict[str, Any] = {}
    for learned_index, task in enumerate(task_sequence, start=1):
        initial = float(stages[0]["tasks"][task]["accuracy"])
        immediate = float(stages[learned_index]["tasks"][task]["accuracy"])
        history = [float(stage["tasks"][task]["accuracy"]) for stage in stages[learned_index:]]
        best = max(history)
        final = float(final_tasks[task]["accuracy"])
        per_task[task] = {
            "initial_accuracy": initial,
            "immediate_accuracy": immediate,
            "best_accuracy_after_learning": best,
            "final_accuracy": final,
            "forgetting": max(0.0, best - final),
            "backward_transfer": final - immediate,
            "retention_ratio": final / best if best > 0.0 else None,
        }

    learned = list(per_task.values())
    return {
        "task_sequence": list(task_sequence),
        "eval_tasks": list(eval_tasks),
        "accuracy_matrix": [
            {
                "stage": stage["stage"],
                "trained_task": stage["trained_task"],
                "accuracies": {
                    task: float(stage["tasks"][task]["accuracy"]) for task in eval_tasks
                },
            }
            for stage in stages
        ],
        "per_task": per_task,
        "initial_average_accuracy": _mean(
            [float(stages[0]["tasks"][task]["accuracy"]) for task in eval_tasks]
        ),
        "final_average_accuracy": _mean(
            [float(final_tasks[task]["accuracy"]) for task in eval_tasks]
        ),
        "final_learned_task_accuracy": _mean([item["final_accuracy"] for item in learned]),
        "mean_immediate_accuracy": _mean([item["immediate_accuracy"] for item in learned]),
        "mean_forgetting": _mean([item["forgetting"] for item in learned]),
        "mean_backward_transfer": _mean([item["backward_transfer"] for item in learned]),
    }


def _train_stage(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    task: str,
    stage_index: int,
    config: ContinualConfig,
    device: torch.device,
) -> dict[str, Any]:
    model.train()
    history: list[dict[str, float | int]] = []
    iterator = trange(
        config.steps_per_task,
        disable=config.quiet,
        desc=f"stage {stage_index}: {task}",
    )
    for step in iterator:
        batch = make_task_batch(
            task,
            batch_size=config.batch_size,
            seq_len=config.seq_len,
            seed=config.seed + stage_index * 100_000 + step,
            vocab_size=config.model_config.vocab_size,
        ).to(device)
        output = model(batch.input_ids)
        answer_loss = supervised_answer_loss(output, batch)
        auxiliary_loss = next_token_auxiliary_loss(output.aux_logits, batch.input_ids)
        loss = answer_loss + config.aux_loss_weight * auxiliary_loss
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        with torch.no_grad():
            accuracy = supervised_answer_accuracy(output, batch)
        record = {
            "step": step + 1,
            "loss": float(loss.detach().cpu()),
            "answer_loss": float(answer_loss.detach().cpu()),
            "auxiliary_loss": float(auxiliary_loss.detach().cpu()),
            "accuracy": accuracy,
            "mean_surprise": output.memory_stats.mean_surprise,
        }
        history.append(record)
        if not config.quiet:
            iterator.set_postfix(loss=f"{record['loss']:.3f}", acc=f"{accuracy:.3f}")
    return {
        "task": task,
        "steps": config.steps_per_task,
        "final": history[-1] if history else {},
        "history": history,
    }


def _evaluate_tasks(
    model: torch.nn.Module,
    config: ContinualConfig,
    device: torch.device,
) -> dict[str, Any]:
    model.eval()
    task_metrics: dict[str, Any] = {}
    with torch.no_grad():
        for task_index, task in enumerate(config.eval_tasks):
            seq_len = config.extrapolate_len if task == "length_extrapolation" else config.seq_len
            correct = 0
            total = 0
            write_strengths: list[float] = []
            write_frequencies: list[float] = []
            surprises: list[float] = []
            for step in range(config.eval_steps):
                batch = make_task_batch(
                    task,
                    batch_size=config.batch_size,
                    seq_len=seq_len,
                    seed=config.eval_seed + task_index * 10_000 + step,
                    vocab_size=config.model_config.vocab_size,
                ).to(device)
                output = model(batch.input_ids)
                if output.answer_logits is None:
                    correct += int((output.logits.argmax(dim=-1) == batch.labels).sum().cpu())
                    total += int(batch.labels.numel())
                else:
                    mask = batch.answer_targets != IGNORE_INDEX
                    predictions = output.answer_logits.argmax(dim=-1)
                    correct += int(
                        (predictions[mask] == batch.answer_targets[mask]).sum().cpu()
                    )
                    total += int(mask.sum().cpu())
                write_strengths.append(output.memory_stats.mean_write_strength)
                write_frequencies.append(output.memory_stats.write_frequency)
                surprises.append(output.memory_stats.mean_surprise)
            task_metrics[task] = {
                "accuracy": correct / max(1, total),
                "examples": total,
                "seq_len": seq_len,
                "mean_write_strength": _mean(write_strengths),
                "write_frequency": _mean(write_frequencies),
                "mean_surprise": _mean(surprises),
            }
    return task_metrics


def _validate_config(config: ContinualConfig) -> None:
    if not config.task_sequence:
        raise ValueError("task_sequence must not be empty")
    if len(set(config.task_sequence)) != len(config.task_sequence):
        raise ValueError("task_sequence must contain unique tasks")
    for task in (*config.task_sequence, *config.eval_tasks):
        if task not in TASKS:
            raise ValueError(f"unknown task: {task}")
    missing = set(config.task_sequence) - set(config.eval_tasks)
    if missing:
        raise ValueError(f"eval_tasks must contain every trained task: {sorted(missing)}")
    if config.steps_per_task < 1 or config.eval_steps < 1:
        raise ValueError("training and evaluation steps must be positive")


def _new_run_dir(root: str | Path, prefix: str) -> Path:
    root_dir = Path(root)
    root_dir.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime("%Y%m%d-%H%M%S")
    candidate = root_dir / f"{prefix}-{stamp}"
    suffix = 1
    while candidate.exists():
        suffix += 1
        candidate = root_dir / f"{prefix}-{stamp}-{suffix}"
    candidate.mkdir(parents=True)
    return candidate


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(to_jsonable(payload), indent=2) + "\n")


def _mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0
