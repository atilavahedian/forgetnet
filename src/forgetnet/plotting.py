from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from forgetnet.runtime import ensure_dir


def plot_runs(runs: str | Path, output_dir: str | Path) -> Path:
    runs_dir = Path(runs)
    out_dir = ensure_dir(output_dir)
    records = _collect_eval_records(runs_dir)
    if not records:
        raise ValueError(f"no eval metrics found under {runs_dir}")

    (out_dir / "plot_data.json").write_text(json.dumps(records, indent=2) + "\n")

    labels = [f"{record['model']} {record['source']}\n{record['task']}" for record in records]
    values = [record["accuracy"] for record in records]
    plt.figure(figsize=(max(7, len(labels) * 1.1), 4.5))
    bars = plt.bar(range(len(values)), values, color="#334155")
    plt.ylim(0.0, 1.0)
    plt.ylabel("Accuracy")
    plt.title("ForgetNet synthetic memory accuracy")
    plt.xticks(range(len(labels)), labels, rotation=30, ha="right")
    for bar, value in zip(bars, values, strict=True):
        plt.text(bar.get_x() + bar.get_width() / 2, value + 0.02, f"{value:.2f}", ha="center", fontsize=8)
    plt.tight_layout()
    path = out_dir / "accuracy_by_task.png"
    plt.savefig(path, dpi=180)
    plt.close()
    return path


def plot_benchmark(summary_path: str | Path, output_dir: str | Path) -> Path:
    payload = json.loads(Path(summary_path).read_text())
    if payload.get("kind") != "continual_benchmark":
        raise ValueError("summary is not a continual benchmark")
    models = list(payload["config"]["models"])
    aggregates = payload["aggregates"]
    accuracy = [aggregates[model]["final_learned_task_accuracy"]["mean"] for model in models]
    accuracy_error = [
        aggregates[model]["final_learned_task_accuracy"]["ci95_half_width"] for model in models
    ]
    forgetting = [aggregates[model]["mean_forgetting"]["mean"] for model in models]
    forgetting_error = [
        aggregates[model]["mean_forgetting"]["ci95_half_width"] for model in models
    ]

    out_dir = ensure_dir(output_dir)
    figure, axes = plt.subplots(1, 2, figsize=(10, 4.5))
    palette = ["#2563eb", "#7c3aed", "#475569"]
    colors = [palette[index % len(palette)] for index in range(len(models))]
    axes[0].bar(models, accuracy, yerr=accuracy_error, color=colors, capsize=4)
    axes[0].set_title("Final learned-task accuracy")
    axes[0].set_ylabel("Accuracy")
    axes[0].set_ylim(bottom=0.0)
    axes[1].bar(models, forgetting, yerr=forgetting_error, color=colors, capsize=4)
    axes[1].set_title("Mean forgetting (lower is better)")
    axes[1].set_ylabel("Accuracy drop")
    axes[1].set_ylim(bottom=0.0)
    for axis in axes:
        axis.tick_params(axis="x", rotation=20)
        axis.grid(axis="y", alpha=0.2)
    figure.suptitle("ForgetNet parameter-matched continual benchmark")
    figure.tight_layout()
    path = out_dir / "continual_benchmark.png"
    figure.savefig(path, dpi=180)
    plt.close(figure)
    return path


def _collect_eval_records(runs_dir: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for path in sorted(runs_dir.rglob("metrics.json")):
        payload = json.loads(path.read_text())
        if payload.get("kind") != "eval":
            continue
        model = payload.get("model", "unknown")
        checkpoint = payload.get("checkpoint")
        source = Path(checkpoint).parent.name if checkpoint else "fresh"
        for task, metrics in payload.get("tasks", {}).items():
            records.append(
                {
                    "model": model,
                    "source": source,
                    "task": task,
                    "accuracy": float(metrics["accuracy"]),
                }
            )
    return records
