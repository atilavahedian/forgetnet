# Results

This directory contains generated artifacts from a small local sanity run.

- `accuracy_by_task.png`: plotted task accuracy from local eval runs.
- `plot_data.json`: the plotted records.

The current sanity run compares:

- a fresh untrained ForgetNet eval across all tasks,
- a 100-step `changing_facts` checkpoint evaluated across all tasks.

These results are included to verify the repo workflow and make the first release inspectable. They are not state-of-the-art claims.
