# ForgetNet Technical Note

ForgetNet studies whether a small sequence model can use bounded, editable memory instead of scaling full-context attention. The v1 implementation is deliberately compact: it is meant to be readable research code that can be ablated, not a production language model.

## Model

For token `x_t`, the model computes an embedding with position information:

```text
e_t = Emb(x_t) + Pos(t)
```

It then applies causal local attention over the recent window:

```text
a_t = LocalAttention(e_t, e_{t-w:t})
```

The model keeps a memory matrix `M_t` with `S` slots and hidden width `d`. It reads by content similarity:

```text
beta_t = softmax(q(a_t) M_t^T / sqrt(d))
r_t = beta_t M_t
h_t = LayerNorm(a_t + W_r r_t)
```

The model predicts from `h_t`, but it also uses its token prediction confidence to build a surprise signal:

```text
s_t = 1 - p_theta(x_t | h_t)
```

The surprise value modulates memory writes. The write path computes a gate, erase vector, write vector, and slot allocation:

```text
g_t = sigmoid(W_g [h_t; r_t; s_t]) * clamp(s_t, 0.05, 1.0)
e_t = sigmoid(W_e h_t)
v_t = tanh(W_v h_t)
alpha_t = softmax(W_a h_t M_t^T / sqrt(d))
```

Each memory slot is updated with a differentiable erase/write rule:

```text
M_{t+1,i} = tanh(M_{t,i} * (1 - alpha_{t,i} g_t e_t) + alpha_{t,i} g_t v_t)
```

The model returns the final-step logits for the benchmark answer.

## Ablations

The repo includes controlled variants:

- `no_forget`: keeps the write rule but disables erase vectors.
- `no_surprise`: keeps memory writes but removes surprise modulation.
- `random_write`: replaces content-based slot allocation with deterministic pseudo-random writes.
- `fifo_memory`: writes to memory slots in round-robin order.

These ablations are designed to test whether improvement comes from plastic memory specifically, rather than from extra parameters.

## Benchmarks

The benchmark suite uses synthetic tasks because each task has an exact target and a known memory burden.

- Associative lookup tests whether a model can bind keys to values.
- Changing facts tests overwrite behavior; the newest value for a key is correct.
- Needle recall tests sparse relevance among distractors.
- Multi-hop lookup tests whether memory can compose two stored edges.
- Length extrapolation tests whether a model trained on shorter sequences degrades on longer sequences.

## Metrics

Evaluation records:

- task accuracy,
- overwrite accuracy for changing facts,
- sequence length,
- parameter count,
- wall-clock time,
- mean write strength,
- write frequency.

These metrics are intentionally simple. The goal is to make failure obvious before adding harder datasets.

## Future Work

- Train larger sweeps across matched parameter budgets.
- Add real text tasks after synthetic memory behavior is stable.
- Compare against Mamba-style and Hopfield-style memory modules.
- Profile memory update overhead and add a vectorized recurrent path.
- Export small trained models for on-device inference experiments.
