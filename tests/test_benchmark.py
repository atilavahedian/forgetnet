from __future__ import annotations

import csv
import json
from pathlib import Path

from forgetnet.benchmark import BenchmarkConfig, run_benchmark
from forgetnet.experiment import ModelConfig


def test_benchmark_compares_models_under_shared_budget(tmp_path: Path) -> None:
    config = BenchmarkConfig(
        models=("forgetnet", "local_transformer"),
        seeds=(81,),
        task_sequence=("changing_facts",),
        eval_tasks=("changing_facts",),
        steps_per_task=1,
        eval_steps=1,
        batch_size=2,
        seq_len=14,
        eval_seed=91,
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

    benchmark_dir = run_benchmark(config)
    summary = json.loads((benchmark_dir / "benchmark_summary.json").read_text())

    assert summary["protocol"] == "equal-update-paired-evaluation-v1"
    assert summary["run_count"] == 2
    assert set(summary["aggregates"]) == {"forgetnet", "local_transformer"}
    assert len(summary["ranking"]) == 2
    assert "local_transformer" in summary["deltas_from_forgetnet"]
    with (benchmark_dir / "benchmark_runs.csv").open(newline="") as handle:
        assert len(list(csv.DictReader(handle))) == 2
