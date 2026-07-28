from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path
import time
from typing import Any

import torch
import torch.nn.functional as F
from tqdm import trange

from forgetnet.data import (
    DEFAULT_VOCAB_SIZE,
    IGNORE_INDEX,
    TASKS,
    TaskBatch,
    make_task_batch,
)
from forgetnet.models import ModelOutput, build_model, count_parameters
from forgetnet.runtime import ensure_dir, seed_everything, select_device, to_jsonable


@dataclass(frozen=True)
class ModelConfig:
    model: str
    vocab_size: int = DEFAULT_VOCAB_SIZE
    d_model: int = 64
    memory_slots: int = 16
    window_size: int = 8
    max_seq_len: int = 512
    n_heads: int = 4


@dataclass(frozen=True)
class TrainConfig:
    task: str
    steps: int = 300
    batch_size: int = 32
    seq_len: int = 64
    lr: float = 3e-4
    aux_loss_weight: float = 0.1
    seed: int = 42
    device: str = "auto"
    output_dir: str = "runs"
    quiet: bool = False
    model_config: ModelConfig = ModelConfig(model="forgetnet")


@dataclass(frozen=True)
class EvalConfig:
    task: str = "all"
    eval_steps: int = 5
    batch_size: int = 32
    seq_len: int = 64
    extrapolate_len: int = 192
    seed: int = 123
    device: str = "auto"
    output_dir: str = "runs"
    checkpoint: str | None = None
    model_config: ModelConfig = ModelConfig(model="forgetnet")


def train(config: TrainConfig) -> Path:
    seed_everything(config.seed)
    device = select_device(config.device)
    model = build_model(**asdict(config.model_config)).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.lr)
    run_dir = _new_run_dir(config.output_dir, f"{config.model_config.model}-{config.task}")
    history: list[dict[str, Any]] = []
    started = time.perf_counter()

    iterator = trange(config.steps, disable=config.quiet, desc="train")
    for step in iterator:
        batch = make_task_batch(
            config.task,
            batch_size=config.batch_size,
            seq_len=config.seq_len,
            seed=config.seed + step,
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
            "write_frequency": output.memory_stats.write_frequency,
            "mean_write_strength": output.memory_stats.mean_write_strength,
            "mean_surprise": output.memory_stats.mean_surprise,
        }
        history.append(record)
        if not config.quiet:
            iterator.set_postfix(loss=f"{record['loss']:.3f}", acc=f"{accuracy:.3f}")

    metrics = {
        "kind": "train",
        "task": config.task,
        "model": config.model_config.model,
        "parameter_count": count_parameters(model),
        "device": str(device),
        "wall_time_seconds": time.perf_counter() - started,
        "config": to_jsonable(asdict(config)),
        "history": history,
        "final": history[-1] if history else {},
    }
    _write_json(run_dir / "metrics.json", metrics)
    torch.save(
        {
            "model_state": model.state_dict(),
            "model_config": asdict(config.model_config),
            "train_config": to_jsonable(asdict(config)),
            "metrics": metrics,
        },
        run_dir / "checkpoint.pt",
    )
    return run_dir


def next_token_auxiliary_loss(aux_logits: torch.Tensor | None, input_ids: torch.Tensor) -> torch.Tensor:
    if aux_logits is None:
        raise ValueError("model must return auxiliary token logits")
    if aux_logits.shape[:2] != input_ids.shape:
        raise ValueError("auxiliary logits must align with input_ids")
    if input_ids.shape[1] < 2:
        return aux_logits.sum() * 0.0
    predictions = aux_logits[:, :-1, :].reshape(-1, aux_logits.shape[-1])
    targets = input_ids[:, 1:].reshape(-1)
    return F.cross_entropy(predictions, targets)


def supervised_answer_loss(output: ModelOutput, batch: TaskBatch) -> torch.Tensor:
    if output.answer_logits is None:
        return F.cross_entropy(output.logits, batch.labels)
    mask = batch.answer_targets != IGNORE_INDEX
    if not bool(mask.any()):
        raise ValueError("batch has no supervised answer positions")
    return F.cross_entropy(output.answer_logits[mask], batch.answer_targets[mask])


def supervised_answer_accuracy(output: ModelOutput, batch: TaskBatch) -> float:
    if output.answer_logits is None:
        predictions = output.logits.argmax(dim=-1)
        return float((predictions == batch.labels).float().mean().detach().cpu())
    mask = batch.answer_targets != IGNORE_INDEX
    predictions = output.answer_logits.argmax(dim=-1)
    return float(
        (predictions[mask] == batch.answer_targets[mask]).float().mean().detach().cpu()
    )


def evaluate(config: EvalConfig) -> Path:
    seed_everything(config.seed)
    device = select_device(config.device)
    model_config = config.model_config
    checkpoint_payload: dict[str, Any] | None = None
    if config.checkpoint:
        checkpoint_payload = torch.load(config.checkpoint, map_location=device)
        model_config = ModelConfig(**checkpoint_payload["model_config"])

    model = build_model(**asdict(model_config)).to(device)
    if checkpoint_payload is not None:
        model.load_state_dict(checkpoint_payload["model_state"])
    model.eval()

    tasks = list(TASKS) if config.task == "all" else [config.task]
    run_dir = _new_run_dir(config.output_dir, "eval")
    started = time.perf_counter()
    task_metrics: dict[str, Any] = {}
    with torch.no_grad():
        for task in tasks:
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
                    seed=config.seed + step,
                    vocab_size=model_config.vocab_size,
                ).to(device)
                output = model(batch.input_ids)
                if output.answer_logits is None:
                    predictions = output.logits.argmax(dim=-1)
                    correct += int((predictions == batch.labels).sum().cpu())
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
            accuracy = correct / max(1, total)
            task_metrics[task] = {
                "accuracy": accuracy,
                "examples": total,
                "seq_len": seq_len,
                "mean_write_strength": sum(write_strengths) / len(write_strengths),
                "write_frequency": sum(write_frequencies) / len(write_frequencies),
                "mean_surprise": sum(surprises) / len(surprises),
            }
            if task == "changing_facts":
                task_metrics[task]["overwrite_accuracy"] = accuracy

    metrics = {
        "kind": "eval",
        "model": model_config.model,
        "parameter_count": count_parameters(model),
        "device": str(device),
        "wall_time_seconds": time.perf_counter() - started,
        "checkpoint": config.checkpoint,
        "config": to_jsonable(asdict(config)),
        "tasks": task_metrics,
    }
    _write_json(run_dir / "metrics.json", metrics)
    return run_dir


def _new_run_dir(root: str | Path, prefix: str) -> Path:
    root_dir = ensure_dir(root)
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
