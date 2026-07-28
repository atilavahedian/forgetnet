from __future__ import annotations

import argparse
from dataclasses import asdict
from pathlib import Path

import torch

from forgetnet.benchmark import BenchmarkConfig, run_benchmark
from forgetnet.continual import DEFAULT_CONTINUAL_TASKS, ContinualConfig, run_continual
from forgetnet.data import TASKS, make_task_batch
from forgetnet.experiment import EvalConfig, ModelConfig, TrainConfig, evaluate, train
from forgetnet.models import build_model
from forgetnet.runtime import seed_everything, select_device


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        prog="forgetnet",
        description="Train and evaluate plastic-memory sequence models.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    _add_train_parser(subparsers)
    _add_eval_parser(subparsers)
    _add_continual_parser(subparsers)
    _add_benchmark_parser(subparsers)
    _add_plot_parser(subparsers)
    _add_plot_benchmark_parser(subparsers)
    _add_demo_parser(subparsers)
    args = parser.parse_args(argv)

    if args.command == "train":
        run_dir = train(_train_config(args))
        print(f"Wrote run to {run_dir}")
    elif args.command == "eval":
        run_dir = evaluate(_eval_config(args))
        print(f"Wrote eval metrics to {run_dir / 'metrics.json'}")
    elif args.command == "continual":
        run_dir = run_continual(_continual_config(args))
        print(f"Wrote continual metrics to {run_dir / 'metrics.json'}")
    elif args.command == "benchmark":
        run_dir = run_benchmark(_benchmark_config(args))
        print(f"Wrote benchmark summary to {run_dir / 'benchmark_summary.json'}")
    elif args.command == "plot":
        from forgetnet.plotting import plot_runs

        path = plot_runs(args.runs, args.output_dir)
        print(f"Wrote plot to {path}")
    elif args.command == "plot-benchmark":
        from forgetnet.plotting import plot_benchmark

        path = plot_benchmark(args.summary, args.output_dir)
        print(f"Wrote benchmark plot to {path}")
    elif args.command == "demo":
        _demo(args)
    else:
        parser.error(f"unknown command: {args.command}")


def _add_common_model_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--model", default="forgetnet")
    parser.add_argument("--d-model", type=int, default=64)
    parser.add_argument("--memory-slots", type=int, default=16)
    parser.add_argument("--window-size", type=int, default=8)
    parser.add_argument("--max-seq-len", type=int, default=512)


def _add_train_parser(subparsers: argparse._SubParsersAction[argparse.ArgumentParser]) -> None:
    parser = subparsers.add_parser("train", help="train a model on one synthetic memory task")
    parser.add_argument("--task", choices=TASKS, default="changing_facts")
    parser.add_argument("--steps", type=int, default=300)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--seq-len", type=int, default=64)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--aux-loss-weight", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--output-dir", default="runs")
    parser.add_argument("--quiet", action="store_true")
    _add_common_model_args(parser)


def _add_eval_parser(subparsers: argparse._SubParsersAction[argparse.ArgumentParser]) -> None:
    parser = subparsers.add_parser("eval", help="evaluate a checkpoint or fresh model")
    parser.add_argument("--checkpoint")
    parser.add_argument("--task", choices=("all", *TASKS), default="all")
    parser.add_argument("--eval-steps", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--seq-len", type=int, default=64)
    parser.add_argument("--extrapolate-len", type=int, default=192)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--output-dir", default="runs")
    _add_common_model_args(parser)


def _add_continual_parser(subparsers: argparse._SubParsersAction[argparse.ArgumentParser]) -> None:
    parser = subparsers.add_parser(
        "continual",
        help="train tasks sequentially and measure retention and forgetting",
    )
    parser.add_argument(
        "--tasks",
        default=",".join(DEFAULT_CONTINUAL_TASKS),
        help="comma-separated training sequence",
    )
    parser.add_argument("--eval-tasks", default=",".join(TASKS))
    parser.add_argument("--steps-per-task", type=int, default=100)
    parser.add_argument("--eval-steps", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--seq-len", type=int, default=64)
    parser.add_argument("--extrapolate-len", type=int, default=192)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--aux-loss-weight", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--eval-seed", type=int, default=12_345)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--output-dir", default="runs")
    parser.add_argument("--quiet", action="store_true")
    _add_common_model_args(parser)


def _add_benchmark_parser(subparsers: argparse._SubParsersAction[argparse.ArgumentParser]) -> None:
    parser = subparsers.add_parser(
        "benchmark",
        help="compare models under equal sequential-training and paired-evaluation budgets",
    )
    parser.add_argument("--models", default="forgetnet,no_forget,local_transformer")
    parser.add_argument(
        "--model-widths",
        default="",
        help="optional model=width pairs, for example forgetnet=64,local_transformer=48",
    )
    parser.add_argument("--seeds", default="42,43,44")
    parser.add_argument("--tasks", default=",".join(DEFAULT_CONTINUAL_TASKS))
    parser.add_argument("--eval-tasks", default=",".join(TASKS))
    parser.add_argument("--steps-per-task", type=int, default=100)
    parser.add_argument("--eval-steps", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--seq-len", type=int, default=64)
    parser.add_argument("--extrapolate-len", type=int, default=192)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--aux-loss-weight", type=float, default=0.1)
    parser.add_argument("--eval-seed", type=int, default=12_345)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--output-dir", default="runs")
    parser.add_argument("--show-progress", action="store_true")
    _add_common_model_args(parser)


def _add_plot_parser(subparsers: argparse._SubParsersAction[argparse.ArgumentParser]) -> None:
    parser = subparsers.add_parser("plot", help="plot accuracy from eval metrics")
    parser.add_argument("--runs", default="runs")
    parser.add_argument("--output-dir", default="results")


def _add_plot_benchmark_parser(
    subparsers: argparse._SubParsersAction[argparse.ArgumentParser],
) -> None:
    parser = subparsers.add_parser(
        "plot-benchmark",
        help="plot accuracy and forgetting from a continual benchmark summary",
    )
    parser.add_argument("--summary", required=True)
    parser.add_argument("--output-dir", default="results/continual-benchmark")


def _add_demo_parser(subparsers: argparse._SubParsersAction[argparse.ArgumentParser]) -> None:
    parser = subparsers.add_parser("demo", help="print one model prediction on a synthetic example")
    parser.add_argument("--task", choices=TASKS, default="changing_facts")
    parser.add_argument("--checkpoint")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--seq-len", type=int, default=40)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--device", default="auto")
    _add_common_model_args(parser)


def _model_config(args: argparse.Namespace) -> ModelConfig:
    return ModelConfig(
        model=args.model,
        d_model=args.d_model,
        memory_slots=args.memory_slots,
        window_size=args.window_size,
        max_seq_len=args.max_seq_len,
    )


def _train_config(args: argparse.Namespace) -> TrainConfig:
    return TrainConfig(
        task=args.task,
        steps=args.steps,
        batch_size=args.batch_size,
        seq_len=args.seq_len,
        lr=args.lr,
        aux_loss_weight=args.aux_loss_weight,
        seed=args.seed,
        device=args.device,
        output_dir=args.output_dir,
        quiet=args.quiet,
        model_config=_model_config(args),
    )


def _eval_config(args: argparse.Namespace) -> EvalConfig:
    return EvalConfig(
        task=args.task,
        eval_steps=args.eval_steps,
        batch_size=args.batch_size,
        seq_len=args.seq_len,
        extrapolate_len=args.extrapolate_len,
        seed=args.seed,
        device=args.device,
        output_dir=args.output_dir,
        checkpoint=args.checkpoint,
        model_config=_model_config(args),
    )


def _continual_config(args: argparse.Namespace) -> ContinualConfig:
    return ContinualConfig(
        task_sequence=_csv_tasks(args.tasks),
        eval_tasks=_csv_tasks(args.eval_tasks),
        steps_per_task=args.steps_per_task,
        eval_steps=args.eval_steps,
        batch_size=args.batch_size,
        seq_len=args.seq_len,
        extrapolate_len=args.extrapolate_len,
        lr=args.lr,
        aux_loss_weight=args.aux_loss_weight,
        seed=args.seed,
        eval_seed=args.eval_seed,
        device=args.device,
        output_dir=args.output_dir,
        quiet=args.quiet,
        model_config=_model_config(args),
    )


def _benchmark_config(args: argparse.Namespace) -> BenchmarkConfig:
    return BenchmarkConfig(
        models=_csv_tasks(args.models),
        seeds=tuple(int(seed.strip()) for seed in args.seeds.split(",") if seed.strip()),
        task_sequence=_csv_tasks(args.tasks),
        eval_tasks=_csv_tasks(args.eval_tasks),
        steps_per_task=args.steps_per_task,
        eval_steps=args.eval_steps,
        batch_size=args.batch_size,
        seq_len=args.seq_len,
        extrapolate_len=args.extrapolate_len,
        lr=args.lr,
        aux_loss_weight=args.aux_loss_weight,
        eval_seed=args.eval_seed,
        device=args.device,
        output_dir=args.output_dir,
        quiet=not args.show_progress,
        model_config=_model_config(args),
        model_widths=_model_widths(args.model_widths),
    )


def _csv_tasks(raw: str) -> tuple[str, ...]:
    return tuple(task.strip() for task in raw.split(",") if task.strip())


def _model_widths(raw: str) -> tuple[tuple[str, int], ...]:
    pairs = []
    for item in raw.split(","):
        if not item.strip():
            continue
        model, separator, width = item.partition("=")
        if not separator or not model.strip() or not width.strip():
            raise ValueError(f"invalid model width: {item!r}")
        pairs.append((model.strip(), int(width)))
    return tuple(pairs)


def _demo(args: argparse.Namespace) -> None:
    seed_everything(args.seed)
    device = select_device(args.device)
    model_config = _model_config(args)
    checkpoint_payload = None
    if args.checkpoint:
        checkpoint_payload = torch.load(args.checkpoint, map_location=device)
        model_config = ModelConfig(**checkpoint_payload["model_config"])

    model = build_model(**asdict(model_config)).to(device)
    if checkpoint_payload is not None:
        model.load_state_dict(checkpoint_payload["model_state"])
    model.eval()

    batch = make_task_batch(args.task, batch_size=args.batch_size, seq_len=args.seq_len, seed=args.seed).to(device)
    with torch.no_grad():
        output = model(batch.input_ids)
    prediction = int(output.logits.argmax(dim=-1)[0].detach().cpu())
    label = int(batch.labels[0].detach().cpu())
    print(f"Task: {args.task}")
    print(f"Input: {batch.input_ids[0].detach().cpu().tolist()}")
    print(f"Label: {label}")
    print(f"Prediction: {prediction}")
    print(f"Mean write strength: {output.memory_stats.mean_write_strength:.3f}")


if __name__ == "__main__":
    main()
