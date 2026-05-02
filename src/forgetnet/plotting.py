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
