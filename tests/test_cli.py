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
