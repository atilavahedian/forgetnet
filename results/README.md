# Results

This directory contains compact generated research artifacts.

- `accuracy_by_task.png`: plotted task accuracy from local eval runs.
- `plot_data.json`: the plotted records.

The current sanity run compares:

- a fresh untrained ForgetNet eval across all tasks,
- a 100-step `changing_facts` checkpoint evaluated across all tasks.

These results are included to verify the repo workflow and make the first release inspectable. They are not state-of-the-art claims.

`continual-benchmark/` contains the current three-seed, parameter-matched pilot:

- `benchmark_runs.csv`: one row per model and seed,
- `continual_benchmark.png`: aggregate accuracy and forgetting with descriptive 95% intervals,
- `README.md`: protocol, aggregate values, paired deltas, and the falsification-oriented conclusion.

Full stage matrices and checkpoints are intentionally excluded because they can be regenerated from the documented command in the root README.
