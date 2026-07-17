from __future__ import annotations

import json
from pathlib import Path

import pytest

from forgetnet.continual import ContinualConfig, run_continual, summarize_continual_metrics
from forgetnet.experiment import ModelConfig


def test_continual_metrics_measure_forgetting_and_backward_transfer() -> None:
    stages = [
        _stage(0, None, associative_lookup=0.1, changing_facts=0.1),
        _stage(1, "associative_lookup", associative_lookup=0.8, changing_facts=0.1),
        _stage(2, "changing_facts", associative_lookup=0.5, changing_facts=0.7),
    ]

    metrics = summarize_continual_metrics(
        stages,
        ("associative_lookup", "changing_facts"),
        ("associative_lookup", "changing_facts"),
    )

    assert metrics["final_learned_task_accuracy"] == pytest.approx(0.6)
    assert metrics["mean_forgetting"] == pytest.approx(0.15)
    assert metrics["mean_backward_transfer"] == pytest.approx(-0.15)
    assert metrics["per_task"]["associative_lookup"]["retention_ratio"] == pytest.approx(0.625)


def test_continual_run_writes_accuracy_matrix_and_checkpoint(tmp_path: Path) -> None:
    config = ContinualConfig(
        task_sequence=("associative_lookup", "changing_facts"),
        eval_tasks=("associative_lookup", "changing_facts"),
        steps_per_task=1,
        eval_steps=1,
        batch_size=2,
        seq_len=14,
        seed=61,
        eval_seed=71,
        device="cpu",
        output_dir=str(tmp_path),
        quiet=True,
        model_config=ModelConfig(
            model="forgetnet",
            d_model=8,
            memory_slots=2,
            window_size=4,
            max_seq_len=32,
        ),
    )

    run_dir = run_continual(config)
    metrics = json.loads((run_dir / "metrics.json").read_text())

    assert metrics["kind"] == "continual"
    assert len(metrics["stages"]) == 3
    assert len(metrics["continual"]["accuracy_matrix"]) == 3
    assert set(metrics["continual"]["per_task"]) == {
        "associative_lookup",
        "changing_facts",
    }
    assert (run_dir / "checkpoint.pt").exists()


def _stage(stage: int, trained_task: str | None, **accuracies: float) -> dict:
    return {
        "stage": stage,
        "trained_task": trained_task,
        "tasks": {task: {"accuracy": accuracy} for task, accuracy in accuracies.items()},
    }
