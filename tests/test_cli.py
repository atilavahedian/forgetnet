from __future__ import annotations

import json
from pathlib import Path

from forgetnet.cli import main


def test_train_eval_plot_smoke_flow(tmp_path: Path) -> None:
    runs_dir = tmp_path / "runs"

    main(
        [
            "train",
            "--task",
            "changing_facts",
            "--model",
            "forgetnet",
            "--steps",
            "2",
            "--batch-size",
            "4",
            "--seq-len",
            "24",
            "--d-model",
            "24",
            "--memory-slots",
            "4",
            "--output-dir",
            str(runs_dir),
            "--seed",
            "11",
        ]
    )
    checkpoint = next(runs_dir.glob("*/checkpoint.pt"))

    main(
        [
            "eval",
            "--checkpoint",
            str(checkpoint),
            "--task",
            "all",
            "--eval-steps",
            "2",
            "--batch-size",
            "4",
            "--output-dir",
            str(runs_dir),
        ]
    )
    metrics_path = sorted(runs_dir.glob("eval-*/metrics.json"))[-1]
    metrics = json.loads(metrics_path.read_text())

    assert "changing_facts" in metrics["tasks"]
    assert metrics["parameter_count"] > 0

    main(["plot", "--runs", str(runs_dir), "--output-dir", str(tmp_path / "plots")])
    assert (tmp_path / "plots" / "accuracy_by_task.png").exists()


def test_demo_command_runs_without_checkpoint(capsys) -> None:
    main(
        [
            "demo",
            "--task",
            "associative_lookup",
            "--batch-size",
            "1",
            "--seq-len",
            "20",
            "--d-model",
            "16",
        ]
    )

    captured = capsys.readouterr()
    assert "Prediction" in captured.out


def test_continual_command_writes_retention_metrics(tmp_path: Path) -> None:
    main(
        [
            "continual",
            "--tasks",
            "associative_lookup,changing_facts",
            "--eval-tasks",
            "associative_lookup,changing_facts",
            "--steps-per-task",
            "1",
            "--eval-steps",
            "1",
            "--batch-size",
            "2",
            "--seq-len",
            "14",
            "--d-model",
            "8",
            "--memory-slots",
            "2",
            "--max-seq-len",
            "32",
            "--device",
            "cpu",
            "--output-dir",
            str(tmp_path),
            "--quiet",
        ]
    )

    metrics_path = next(tmp_path.glob("continual-*/metrics.json"))
    metrics = json.loads(metrics_path.read_text())
    assert metrics["continual"]["task_sequence"] == [
        "associative_lookup",
        "changing_facts",
    ]


def test_benchmark_command_writes_model_comparison(tmp_path: Path) -> None:
    main(
        [
            "benchmark",
            "--models",
            "forgetnet",
            "--seeds",
            "101",
            "--tasks",
            "changing_facts",
            "--eval-tasks",
            "changing_facts",
            "--steps-per-task",
            "1",
            "--eval-steps",
            "1",
            "--batch-size",
            "2",
            "--seq-len",
            "14",
            "--d-model",
            "8",
            "--memory-slots",
            "2",
            "--max-seq-len",
            "32",
            "--device",
            "cpu",
            "--output-dir",
            str(tmp_path),
        ]
    )

    summary_path = next(tmp_path.glob("continual-benchmark-*/benchmark_summary.json"))
    summary = json.loads(summary_path.read_text())
    assert summary["run_count"] == 1


def test_selective_command_writes_multi_query_metrics_and_checkpoint(
    tmp_path: Path,
) -> None:
    main(
        [
            "selective",
            "--model",
            "cldm",
            "--steps",
            "1",
            "--eval-steps",
            "1",
            "--batch-size",
            "2",
            "--seq-len",
            "24",
            "--active-keys",
            "2",
            "--updates-per-key",
            "1",
            "--d-model",
            "8",
            "--memory-slots",
            "2",
            "--window-size",
            "2",
            "--max-seq-len",
            "32",
            "--n-heads",
            "2",
            "--device",
            "cpu",
            "--output-dir",
            str(tmp_path),
            "--quiet",
        ]
    )

    run_dir = next(tmp_path.glob("selective-cldm-*"))
    metrics = json.loads((run_dir / "metrics.json").read_text())
    assert (run_dir / "checkpoint.pt").exists()
    assert metrics["protocol"] == "conflict-stream-equal-data-v1"
    assert metrics["evaluation"]["queries"] == 4
    assert metrics["evaluation"]["events"] > 0
