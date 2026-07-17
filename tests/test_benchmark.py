from __future__ import annotations

import csv
import json
from pathlib import Path

from forgetnet.benchmark import BenchmarkConfig, run_benchmark
from forgetnet.experiment import ModelConfig
from forgetnet.plotting import plot_benchmark


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
        model_widths=(("local_transformer", 12),),
    )

    benchmark_dir = run_benchmark(config)
    summary = json.loads((benchmark_dir / "benchmark_summary.json").read_text())

    assert summary["protocol"] == "equal-update-paired-evaluation-v2"
    assert summary["run_count"] == 2
    assert set(summary["aggregates"]) == {"forgetnet", "local_transformer"}
    assert len(summary["ranking"]) == 2
    paired = summary["paired_deltas_from_forgetnet"]["local_transformer"]
    assert paired["seeds"] == [81]
    assert len(paired["final_learned_task_accuracy"]["values"]) == 1
    assert summary["config"]["model_widths"] == [["local_transformer", 12]]
    assert summary["parameter_count_ratio"] >= 1.0
    with (benchmark_dir / "benchmark_runs.csv").open(newline="") as handle:
        assert len(list(csv.DictReader(handle))) == 2

    plot_path = plot_benchmark(
        benchmark_dir / "benchmark_summary.json",
        tmp_path / "plots",
    )
    assert plot_path.exists()
