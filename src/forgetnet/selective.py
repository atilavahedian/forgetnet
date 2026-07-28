from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path
import time
from typing import Any

import torch
import torch.nn.functional as F
from tqdm import trange

from forgetnet.data import IGNORE_INDEX, QUERY_STABLE, QUERY_UPDATED, TaskBatch, make_task_batch
from forgetnet.experiment import (
    ModelConfig,
    next_token_auxiliary_loss,
    supervised_answer_loss,
)
from forgetnet.models import ModelOutput, build_model, count_parameters
from forgetnet.runtime import seed_everything, select_device, to_jsonable


@dataclass(frozen=True)
class SelectiveConfig:
    steps: int = 200
    eval_steps: int = 10
    batch_size: int = 32
    seq_len: int = 60
    active_keys: int = 4
    updates_per_key: int = 2
    num_keys: int = 24
    num_values: int = 48
    minimum_query_lag: int | None = None
    lr: float = 3e-4
    aux_loss_weight: float = 0.05
    clr_loss_weight: float = 0.1
    seed: int = 42
    eval_seed: int = 12_345
    device: str = "auto"
    output_dir: str = "runs"
    quiet: bool = False
    model_config: ModelConfig = ModelConfig(model="cldm", max_seq_len=128)


@dataclass(frozen=True)
class SelectiveStepResult:
    loss: torch.Tensor
    answer_loss: torch.Tensor
    auxiliary_loss: torch.Tensor
    locality_loss: torch.Tensor
    query_accuracy: torch.Tensor


def run_selective(config: SelectiveConfig) -> Path:
    _validate_config(config)
    seed_everything(config.seed)
    device = select_device(config.device)
    model = build_model(**asdict(config.model_config)).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.lr)
    run_dir = _new_run_dir(config.output_dir, f"selective-{config.model_config.model}")
    history: list[dict[str, float | int]] = []
    started = time.perf_counter()

    iterator = trange(config.steps, disable=config.quiet, desc="selective")
    for step in iterator:
        batch = _make_batch(config, config.seed + step, device)
        step_result = selective_training_step(
            model,
            optimizer,
            batch,
            aux_loss_weight=config.aux_loss_weight,
            clr_loss_weight=config.clr_loss_weight,
        )

        record = {
            "step": step + 1,
            "loss": float(step_result.loss.cpu()),
            "answer_loss": float(step_result.answer_loss.cpu()),
            "auxiliary_loss": float(step_result.auxiliary_loss.cpu()),
            "locality_loss": float(step_result.locality_loss.cpu()),
            "query_accuracy": float(step_result.query_accuracy.cpu()),
        }
        history.append(record)
        if not config.quiet:
            iterator.set_postfix(
                loss=f"{record['loss']:.3f}",
                acc=f"{record['query_accuracy']:.3f}",
            )

    evaluation = evaluate_selective_model(model, config, device)
    metrics = {
        "kind": "selective_conflict",
        "protocol": "conflict-stream-equal-data-v1",
        "model": config.model_config.model,
        "parameter_count": count_parameters(model),
        "device": str(device),
        "wall_time_seconds": time.perf_counter() - started,
        "config": to_jsonable(asdict(config)),
        "history": history,
        "evaluation": evaluation,
    }
    _write_json(run_dir / "metrics.json", metrics)
    torch.save(
        {
            "model_state": model.state_dict(),
            "model_config": asdict(config.model_config),
            "selective_config": to_jsonable(asdict(config)),
            "metrics": metrics,
        },
        run_dir / "checkpoint.pt",
    )
    return run_dir


def selective_training_step(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    batch: TaskBatch,
    *,
    aux_loss_weight: float,
    clr_loss_weight: float,
) -> SelectiveStepResult:
    """Run the exact optimizer step shared by training and compute profiling."""

    if min(aux_loss_weight, clr_loss_weight) < 0.0:
        raise ValueError("loss weights must be nonnegative")
    model.train()
    output = model(batch.input_ids)
    answer_loss = supervised_answer_loss(output, batch)
    auxiliary_loss = next_token_auxiliary_loss(output.aux_logits, batch.input_ids)
    locality_loss = counterfactual_locality_loss(output, batch)
    loss = (
        answer_loss
        + aux_loss_weight * auxiliary_loss
        + clr_loss_weight * locality_loss
    )
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optimizer.step()

    if output.answer_logits is None:
        raise ValueError("selective training requires per-token answer logits")
    query_mask = batch.answer_targets != IGNORE_INDEX
    query_accuracy = (
        output.answer_logits.argmax(dim=-1)[query_mask]
        == batch.answer_targets[query_mask]
    ).float().mean()
    return SelectiveStepResult(
        loss=loss.detach(),
        answer_loss=answer_loss.detach(),
        auxiliary_loss=auxiliary_loss.detach(),
        locality_loss=locality_loss.detach(),
        query_accuracy=query_accuracy.detach(),
    )


def counterfactual_locality_loss(output: ModelOutput, batch: TaskBatch) -> torch.Tensor:
    if output.answer_logits is None:
        raise ValueError("counterfactual locality requires per-token answer logits")
    left, right = _paired_rows(batch)
    stable_mask = (batch.query_kinds[left] == QUERY_STABLE) & (
        batch.query_kinds[right] == QUERY_STABLE
    )
    if not bool(stable_mask.any()):
        return output.answer_logits.sum() * 0.0
    left_logp = F.log_softmax(output.answer_logits[left].float(), dim=-1)
    right_logp = F.log_softmax(output.answer_logits[right].float(), dim=-1)
    left_p = left_logp.exp()
    right_p = right_logp.exp()
    left_to_right = (left_p * (left_logp - right_logp)).sum(dim=-1)
    right_to_left = (right_p * (right_logp - left_logp)).sum(dim=-1)
    return (0.5 * (left_to_right + right_to_left))[stable_mask].mean()


@torch.no_grad()
def evaluate_selective_model(
    model: torch.nn.Module,
    config: SelectiveConfig,
    device: torch.device,
) -> dict[str, Any]:
    model.eval()
    totals = {
        "queries": 0,
        "correct": 0,
        "stable_queries": 0,
        "stable_correct": 0,
        "updated_queries": 0,
        "updated_correct": 0,
        "stale_intrusions": 0,
        "stale_probability_sum": 0.0,
        "pair_disagreement_sum": 0.0,
        "pair_probability_shift_sum": 0.0,
        "pair_symmetric_kl_sum": 0.0,
        "pair_stable_queries": 0,
    }
    trace_totals = {
        "event_write_gate_sum": 0.0,
        "event_conflict_sum": 0.0,
        "event_localization_sum": 0.0,
        "events": 0,
    }
    for eval_step in range(config.eval_steps):
        batch = _make_batch(config, config.eval_seed + eval_step, device)
        output = model(batch.input_ids)
        _accumulate_query_totals(totals, output, batch)
        if output.memory_trace is not None:
            event_mask = batch.event_targets.bool()
            trace_totals["events"] += int(event_mask.sum().cpu())
            trace_totals["event_write_gate_sum"] += float(
                output.memory_trace.write_gate[event_mask].sum().cpu()
            )
            trace_totals["event_conflict_sum"] += float(
                output.memory_trace.conflict[event_mask].sum().cpu()
            )
            trace_totals["event_localization_sum"] += float(
                output.memory_trace.localization[event_mask].sum().cpu()
            )
    model.train()

    return {
        "query_accuracy": _ratio(totals["correct"], totals["queries"]),
        "stable_accuracy": _ratio(
            totals["stable_correct"], totals["stable_queries"]
        ),
        "update_accuracy": _ratio(
            totals["updated_correct"], totals["updated_queries"]
        ),
        "stale_intrusion_rate": _ratio(
            totals["stale_intrusions"], totals["updated_queries"]
        ),
        "stale_probability": _ratio(
            totals["stale_probability_sum"], totals["updated_queries"]
        ),
        "pair_stable_disagreement": _ratio(
            totals["pair_disagreement_sum"], totals["pair_stable_queries"]
        ),
        "pair_stable_probability_shift": _ratio(
            totals["pair_probability_shift_sum"], totals["pair_stable_queries"]
        ),
        "pair_stable_symmetric_kl": _ratio(
            totals["pair_symmetric_kl_sum"], totals["pair_stable_queries"]
        ),
        "mean_event_write_gate": _ratio(
            trace_totals["event_write_gate_sum"], trace_totals["events"]
        ),
        "mean_event_conflict": _ratio(
            trace_totals["event_conflict_sum"], trace_totals["events"]
        ),
        "mean_event_localization": _ratio(
            trace_totals["event_localization_sum"], trace_totals["events"]
        ),
        "queries": totals["queries"],
        "events": trace_totals["events"],
    }


def _accumulate_query_totals(
    totals: dict[str, int | float],
    output: ModelOutput,
    batch: TaskBatch,
) -> None:
    if output.answer_logits is None:
        raise ValueError("selective evaluation requires per-token answer logits")
    probabilities = F.softmax(output.answer_logits.float(), dim=-1)
    predictions = probabilities.argmax(dim=-1)
    query_mask = batch.answer_targets != IGNORE_INDEX
    correct = predictions == batch.answer_targets
    stable_mask = query_mask & (batch.query_kinds == QUERY_STABLE)
    updated_mask = query_mask & (batch.query_kinds == QUERY_UPDATED)

    totals["queries"] += int(query_mask.sum().cpu())
    totals["correct"] += int(correct[query_mask].sum().cpu())
    totals["stable_queries"] += int(stable_mask.sum().cpu())
    totals["stable_correct"] += int(correct[stable_mask].sum().cpu())
    totals["updated_queries"] += int(updated_mask.sum().cpu())
    totals["updated_correct"] += int(correct[updated_mask].sum().cpu())

    stale = batch.stale_targets.clamp_min(0)
    stale_valid = batch.stale_targets != IGNORE_INDEX
    stale_probabilities = probabilities.gather(-1, stale)
    stale_probability = (stale_probabilities * stale_valid).sum(dim=-1)
    stale_hit = (
        (predictions.unsqueeze(-1) == batch.stale_targets) & stale_valid
    ).any(dim=-1)
    totals["stale_probability_sum"] += float(stale_probability[updated_mask].sum().cpu())
    totals["stale_intrusions"] += int(stale_hit[updated_mask].sum().cpu())

    left, right = _paired_rows(batch)
    stable_pair_mask = (batch.query_kinds[left] == QUERY_STABLE) & (
        batch.query_kinds[right] == QUERY_STABLE
    )
    left_correct = correct[left]
    right_correct = correct[right]
    target = batch.answer_targets[left].clamp_min(0)
    left_target_probability = probabilities[left].gather(-1, target.unsqueeze(-1)).squeeze(-1)
    right_target_probability = probabilities[right].gather(-1, target.unsqueeze(-1)).squeeze(-1)
    left_logp = probabilities[left].clamp_min(1e-12).log()
    right_logp = probabilities[right].clamp_min(1e-12).log()
    symmetric_kl = 0.5 * (
        (probabilities[left] * (left_logp - right_logp)).sum(dim=-1)
        + (probabilities[right] * (right_logp - left_logp)).sum(dim=-1)
    )
    totals["pair_stable_queries"] += int(stable_pair_mask.sum().cpu())
    totals["pair_disagreement_sum"] += float(
        (left_correct.float() - right_correct.float()).abs()[stable_pair_mask].sum().cpu()
    )
    totals["pair_probability_shift_sum"] += float(
        (left_target_probability - right_target_probability)
        .abs()[stable_pair_mask]
        .sum()
        .cpu()
    )
    totals["pair_symmetric_kl_sum"] += float(
        symmetric_kl[stable_pair_mask].sum().cpu()
    )


def _paired_rows(batch: TaskBatch) -> tuple[torch.Tensor, torch.Tensor]:
    if not batch.metadata.get("paired") or bool((batch.pair_ids < 0).any()):
        raise ValueError("counterfactual locality requires paired examples")
    order = torch.argsort(batch.pair_ids * 2 + batch.pair_variants)
    if order.numel() % 2:
        raise ValueError("paired batch has an odd row count")
    left = order[0::2]
    right = order[1::2]
    valid = (batch.pair_ids[left] == batch.pair_ids[right]) & (
        batch.pair_variants[left] == 0
    ) & (batch.pair_variants[right] == 1)
    if not bool(valid.all()):
        raise ValueError("paired batch is missing a 0/1 member")
    return left, right


def _make_batch(
    config: SelectiveConfig,
    seed: int,
    device: torch.device,
) -> TaskBatch:
    return make_task_batch(
        "conflict_stream",
        batch_size=config.batch_size,
        seq_len=config.seq_len,
        seed=seed,
        vocab_size=config.model_config.vocab_size,
        window_size=config.model_config.window_size,
        active_keys=config.active_keys,
        updates_per_key=config.updates_per_key,
        paired=True,
        num_keys=config.num_keys,
        num_values=config.num_values,
        minimum_query_lag=config.minimum_query_lag,
    ).to(device)


def _validate_config(config: SelectiveConfig) -> None:
    if config.steps < 1 or config.eval_steps < 1:
        raise ValueError("steps and eval_steps must be positive")
    if config.batch_size < 2 or config.batch_size % 2:
        raise ValueError("batch_size must be a positive even number")
    if config.seq_len > config.model_config.max_seq_len:
        raise ValueError("seq_len exceeds model max_seq_len")
    if (
        config.minimum_query_lag is not None
        and config.minimum_query_lag < 2 * config.model_config.window_size
    ):
        raise ValueError("minimum_query_lag must cover the full local receptive field")
    if min(config.aux_loss_weight, config.clr_loss_weight) < 0.0:
        raise ValueError("loss weights must be nonnegative")


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


def _ratio(numerator: int | float, denominator: int | float) -> float | None:
    return float(numerator) / float(denominator) if denominator else None
