# Continual Benchmark Protocol

ForgetNet's continual benchmark asks whether bounded plastic memory improves sequential learning and retention under controlled task order, training budget, parameter count, and evaluation examples.

## Checked Pilot

- Models: `forgetnet`, `no_forget`, `local_transformer`
- Seeds: 2027, 2028, 2029
- Task order: associative lookup, changing facts, needle recall
- Training: 100 updates per task, batch size 32, sequence length 40
- Evaluation: five fixed batches per task, paired across every model and seed
- Extra held-out tasks: multi-hop and length extrapolation
- Widths: 64 for memory models, 48 for the local Transformer
- Parameters: 79,936 for memory models and 75,104 for the Transformer
- Device: CPU

```bash
uv run forgetnet benchmark \
  --models forgetnet,no_forget,local_transformer \
  --model-widths forgetnet=64,no_forget=64,local_transformer=48 \
  --seeds 2027,2028,2029 \
  --tasks associative_lookup,changing_facts,needle_recall \
  --eval-tasks associative_lookup,changing_facts,needle_recall,multi_hop,length_extrapolation \
  --steps-per-task 100 \
  --eval-steps 5 \
  --batch-size 32 \
  --seq-len 40 \
  --extrapolate-len 120 \
  --d-model 64 \
  --memory-slots 16 \
  --window-size 8 \
  --max-seq-len 128 \
  --eval-seed 3001 \
  --device cpu
```

## Metric Definitions

For task `j`, let `a(i, j)` be held-out accuracy after training stage `i`. Stage zero is the untrained model, and task `j` is learned at stage `j`.

- Immediate accuracy: `a(j, j)`.
- Final accuracy: `a(T, j)` after the final stage.
- Forgetting: `max_i>=j a(i, j) - a(T, j)`.
- Backward transfer: `a(T, j) - a(j, j)`.
- Final learned-task accuracy: mean final accuracy over the trained tasks.

The aggregate reports population standard deviation across the three seeds and a descriptive normal 95% half-width (`1.96 * std / sqrt(n)`). Paired deltas subtract the ForgetNet value from the comparison model at the same seed before aggregation.

## Interpretation

This pilot is designed to reject weak architectural stories, not establish state of the art. The no-forget ablation and ForgetNet are statistically indistinguishable here, so learned erasure is not supported. ForgetNet's point estimate is above the size-matched local Transformer on final learned-task accuracy, while the Transformer has lower forgetting and much lower wall time; the paired intervals include zero for both accuracy and forgetting.

Three synthetic-task seeds are insufficient for a strong positive claim. A follow-up should increase seeds, tune all models under a declared validation budget, match training FLOPs as well as parameters, and add a task where stale information actively conflicts with later queries.
