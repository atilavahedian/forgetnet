# ForgetNet

![ForgetNet plastic memory overview](docs/assets/forgetnet-research-banner.png)

ForgetNet is a pure-PyTorch research repo for a contrarian sequence-modeling thesis:

> Long-context models should not only scale attention. They should learn what is worth remembering, when to overwrite stale facts, and when to forget.

The core model combines local causal attention with a fixed-size differentiable memory bank. At every token, the model reads from memory, estimates surprise, and updates memory slots through learned write and erase gates.

This is a serious v1 research implementation, not a state-of-the-art claim. The repo is built to make the idea easy to inspect, train, ablate, and falsify on controlled long-memory tasks.

**Core idea:** keep attention local, make memory bounded, and force the model to learn overwrite/forget behavior under controlled tests.

**Current artifact:** a local Apple Silicon sanity run where a 100-step `changing_facts` checkpoint learns the overwrite task while remaining untrained on the rest of the suite.

## Architecture

```mermaid
flowchart LR
    X["token x_t"] --> E["token + position embedding"]
    E --> A["local causal attention"]
    A --> R["content read from memory M_t"]
    R --> H["fused hidden state h_t"]
    A --> H
    H --> P["prediction head"]
    P --> S["surprise signal"]
    H --> W["learned write / erase gates"]
    S --> W
    W --> M["updated memory M_{t+1}"]
    R --> W
```

The banner above is a conceptual visual; the Mermaid graph and [technical note](docs/technical_note.md) are the source of truth for the implemented data flow and update equations.

## Install

```bash
uv sync
```

ForgetNet chooses devices in this order: CUDA, Apple Silicon MPS, then CPU.

## Quickstart

Train the plastic-memory model:

```bash
uv run forgetnet train --task changing_facts --model forgetnet --steps 100
```

Evaluate all synthetic memory tasks:

```bash
uv run forgetnet eval --task all
```

Evaluate a trained checkpoint:

```bash
uv run forgetnet eval --checkpoint runs/<run>/checkpoint.pt --task all
```

Plot evaluation results:

```bash
uv run forgetnet plot --runs runs/ --output-dir results/
```

Run one interpretable example:

```bash
uv run forgetnet demo --task changing_facts
```

## Models

- `forgetnet`: local attention plus learned read/write/forget memory.
- `tiny_transformer`: compact Transformer baseline with the same answer-head contract.
- `local_transformer`: Transformer baseline restricted to a local attention window.
- `no_forget`: memory writes without learned erase gates.
- `no_surprise`: memory writes without surprise modulation.
- `random_write`: deterministic pseudo-random slot writes.
- `fifo_memory`: round-robin memory writes.

## Synthetic Memory Suite

- `associative_lookup`: read key-value pairs and answer the queried key.
- `changing_facts`: handle overwritten facts where the latest value wins.
- `needle_recall`: recall a sparse relevant pair among distractors.
- `multi_hop`: follow two edges, `A -> B -> C`.
- `length_extrapolation`: train short, evaluate longer associative lookup sequences.

## Outputs

Training writes:

```text
runs/<model-task-timestamp>/
  checkpoint.pt
  metrics.json
```

Evaluation writes:

```text
runs/eval-<timestamp>/
  metrics.json
```

Plotting writes:

```text
results/accuracy_by_task.png
results/plot_data.json
```

The committed `results/` artifacts are a small local sanity run, not a benchmark claim. They compare a fresh ForgetNet eval against a 100-step `changing_facts` checkpoint on Apple Silicon MPS. In that run, the trained checkpoint reached `0.6125` accuracy on `changing_facts` over 160 held-out examples; the other tasks remained low, which is expected because the checkpoint was not trained on the full suite.

![ForgetNet local sanity results](results/accuracy_by_task.png)

## Tests

```bash
uv run pytest
```

The tests cover deterministic data generation, task label correctness, model output contracts, memory bounds, ablation construction, and CLI smoke flows.

## Research Positioning

ForgetNet is inspired by current work on test-time memory, associative memories, and long-context alternatives to full attention, including:

- [Titans: Learning to Memorize at Test Time](https://arxiv.org/abs/2501.00663)
- [Modern Hopfield Networks with Continuous-Time Memories](https://arxiv.org/abs/2502.10122)
- [MemMamba: Rethinking Memory Patterns in State Space Model](https://arxiv.org/abs/2510.03279)

The repo intentionally starts with synthetic tasks because they make memory behavior falsifiable. Real text modeling is a later step, after the write/forget mechanism survives controlled tests.

## Limitations

- The included experiments are small local sanity runs, not benchmark-scale claims.
- The memory update is differentiable hidden state, not persistent weight editing.
- The synthetic tasks are diagnostic and can overstate real-world long-context ability.
- The architecture is designed for inspection and ablation before throughput.
