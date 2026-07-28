# Conflict-Localized Memory Protocol

Status: preregistered design; implementation and results not yet run.

This document freezes the hypotheses, benchmark validity gates, comparison
rules, and claim boundaries for ForgetNet's next research iteration. The Git
commit containing this file is the preregistration record. Results produced
after that commit must be reported whether positive, negative, or inconclusive.

## 1. Why the previous pilot cannot test the erase thesis

The checked `equal-update-paired-evaluation-v2` pilot remains useful as an
engineering artifact, but it is not evidence that runtime erasure helps:

1. `changing_facts` puts the replacement value in the last record before the
   query. At the checked sequence length and local-attention window, the answer
   is directly visible without memory or key matching. A fixed tail-position
   heuristic reaches 100% on generated examples.
2. ForgetNet initializes memory inside every `forward` call. The continual
   runner trains different tasks in sequence and therefore measures forgetting
   in model parameters, not selective editing of the runtime memory bank.
3. The surprise signal is high for nearly every token early in training, and
   every separator, query, and noise token is eligible to write. The current
   write-frequency statistic can therefore describe an effectively always-on
   writer as selective.
4. The memory stores one entangled vector per slot. Content addressing compares
   against the same representation that is overwritten, so changing a value can
   also destroy the address used to find it.
5. `multi_hop` can generate invalid labels when distractor collision handling
   shifts one blocked key onto the other blocked key.
6. Baselines are not fully controlled: the memory model has separate token and
   answer heads while the Transformer shares one head, and equal update counts
   are not equal training compute.

The new work separates two questions:

- **Primary: episodic selective memory.** Can a bounded runtime memory replace
  stale values while retaining unrelated facts under conflict and capacity
  pressure?
- **Secondary: parametric continual learning.** Do model weights retain skills
  across training stages? This must remain separately labelled and cannot be
  used as evidence for runtime erasure.

## 2. Proposed contribution

### 2.1 Conflict-Localized Delta Memory (CLDM)

CLDM stores separate key and value states, `K_i` and `V_i`, plus differentiable
usage and recency traces. For a candidate event representation `(k_t, v_t)`,
the key match is

```text
s_i = cosine(k_t, K_i)
m_t = sigmoid(kappa * (max_i(s_i) - theta))
```

where `m_t` estimates whether the event updates an existing key. Matching and
allocation distributions are

```text
a_match = softmax(s / tau_match)
a_free  = softmax(-eta * usage - zeta * recency)
pi      = m_t * a_match + (1 - m_t) * a_free
```

The retrieved value and contradiction score are

```text
r_t = sum_i a_match_i V_i
c_t = (1 - cosine(v_t, r_t)) / 2
```

A learned event gate `p_event` should reject separators, queries, and noise.
The edit magnitude separates novelty from contradiction:

```text
g_t = p_event * ((1 - m_t) + m_t * c_t)
```

Values receive a localized delta replacement,

```text
V_i <- V_i + g_t * pi_i * (v_t - V_i),
```

while keys change rapidly for a newly allocated slot and slowly for a matched
slot. Unlike the original erase-plus-add rule, an update moves the addressed
value toward the candidate rather than distributing unconstrained erase and
write vectors across every slot.

Delta updates, adaptive gates, and key-value memories all have prior art. The
research claim tested here is narrower: whether explicitly decomposing novelty
from contradiction and localizing the delta to the matching key improves the
stale-information/retention tradeoff in a controlled bounded-memory setting.
No priority or global novelty claim will be made without a broader literature
review.

### 2.2 Counterfactual Locality Regularization (CLR)

The benchmark emits paired streams that differ in one designated update while
holding all other events and queries fixed. Let `P` and `P'` be the model's
query distributions for the pair. The ordinary supervised loss teaches the
changed answer. CLR additionally penalizes changes at untouched-key queries:

```text
L_CLR = mean_q_in_untouched 0.5 * (KL(P_q || P'_q) + KL(P'_q || P_q)).
```

This is a direct training signal for edit locality: the changed key should
change its answer while unrelated keys should remain invariant. The full model
must be compared with `CLR=0`; otherwise an architectural gain cannot be
attributed to the conflict-localized memory.

## 3. ConflictStream benchmark

ConflictStream is a task family over explicit events:

```text
SET(key, value)
QUERY(key)
```

Each sequence contains multiple labelled queries. It includes stable keys,
keys updated one or more times, unqueried distractor keys, and a randomized
event schedule. Every confirmatory query must be strictly more than
`local_layers * window_size` tokens after the last relevant `SET`, so stacked
local-attention layers cannot propagate the answer to the query.

The generator records, for every query:

- current value;
- prior stale values;
- stable or updated condition;
- update count;
- lag from the last relevant event;
- active-fact-to-slot load;
- pair identifier and changed key for counterfactual pairs.

### 3.1 Validity gates

The benchmark is invalid unless all gates pass:

1. A deterministic last-write-wins oracle scores 100% on every generated split.
2. A fixed-position and local-tail oracle cannot recover the target above
   chance on conflict queries.
3. Every evaluated relevant event lies outside the full stacked local receptive
   field, not merely one layer's window.
4. Counterfactual pairs differ only in the designated update value and their
   derived labels/metadata.
5. Generation is deterministic for a fixed seed.
6. No key, value, operation, or label leaves the declared vocabulary.

### 3.2 Splits and stress conditions

- IID development: training length, one or two updates per changed key, fact
  load no greater than memory capacity.
- Length OOD: two and four times the training query lag.
- Conflict OOD: two, four, and eight updates per key.
- Capacity: live-fact load at `0.5x`, `1x`, `2x`, and `4x` slot count.
- Novelty control: new facts without contradiction.
- Stable control: untouched keys in streams containing unrelated conflicts.
- Combined worst group: long lag, high conflict, and over-capacity together.

## 4. Outcomes

Primary outcomes are computed over query events, not whole sequences:

- **update accuracy:** current-value accuracy for changed keys;
- **stable accuracy:** accuracy for untouched keys;
- **stale intrusion rate:** fraction of changed-key queries where a prior value
  is predicted;
- **stale probability:** probability mass assigned to prior values;
- **collateral sensitivity:** disagreement and symmetric probability shift at
  paired stable-key queries when only an unrelated update value changes;
- **query NLL and multiclass Brier score.**

Mechanistic and efficiency outcomes:

- write localization: write strength on `SET` values divided by other tokens;
- conflict localization: update mass on the matched slot;
- effective slots used and allocation entropy;
- active/trainable parameters, tokens, optimizer updates, estimated FLOPs,
  wall time, and peak memory.

The parametric continual track must additionally report acquisition gain,
relearning, backward transfer, and initial-normalized retention. Low forgetting
without meaningful acquisition is not a success.

## 5. Baselines and ablations

Required learned baselines:

- original surprise-gated ForgetNet;
- `no_forget`;
- FIFO and pseudo-random allocation;
- a delta-memory ablation without contradiction localization;
- parameter-matched local Transformer with a separate answer head;
- GRU or LSTM with the same answer contract.

Required non-learned controls:

- exact last-write-wins dictionary oracle;
- chance predictor;
- fixed-tail and local-tail leakage probes.

Required CLDM ablations:

- contradiction input zeroed;
- contradiction scores shuffled between paired examples;
- CLR disabled;
- key/value separation removed;
- state reset at each chunk when persistent chunks are evaluated.

If the full model does not beat the zeroed or shuffled conflict control, the
conflict mechanism is unsupported even if total accuracy improves.

## 6. Budget matching and statistics

Two comparisons must be labelled separately:

1. **Equal data:** identical examples, tokens, optimizer updates, schedules,
   and evaluation events.
2. **Equal FLOPs:** adjust update counts after profiling so total training FLOPs
   differ by at most 5%.

Learned models must be within a 5% trainable-parameter ratio for a
parameter-matched claim. Each model receives the same number of development
configurations and the same tuning-data access. Confirmatory configurations are
frozen before confirmatory seeds run.

Development uses five paired seeds and is explicitly exploratory. Confirmation
uses at least ten paired seeds with identical generated streams across models.
Report every seed, mean, median, interquartile range, a paired bootstrap 95%
interval, and a paired sign-flip permutation p-value. Individual examples are
not treated as independent experimental units.

## 7. Acceptance and kill criteria

The primary CLDM claim requires all of the following against the strongest
learned baseline:

1. stable-accuracy noninferiority: paired 95% lower bound above `-0.02`;
2. mean update-accuracy gain of at least `0.05` with paired 95% lower bound
   above zero;
3. stale-probability reduction of at least 20% with paired 95% interval
   excluding zero;
4. parameter ratio and equal-FLOP ratio no greater than 1.05.

The OOD claim additionally requires no worst-group stable- or update-accuracy
regression worse than three points. The efficiency claim is killed if CLDM uses
more than twice the compute of the strongest baseline without a clear
accuracy/stale-intrusion Pareto gain.

Stop before confirmation if the five-seed development study shows either no
positive stale-probability movement or more than three points of stable-key
damage. If the frozen primary endpoint fails, publish the null result; do not
retune on confirmatory seeds.

## 8. Claim boundaries

- This work studies bounded episodic memory, not machine unlearning of model
  weights.
- Synthetic evidence is not a real-text or long-context language-model result.
- A successful development run is not confirmatory evidence.
- A local test or valid implementation is not evidence that CLDM improves the
  scientific outcomes.
- Checked results must preserve raw seed-level records and exact commands.

## 9. Primary literature used to bound the contribution

- Schlag, Irie, and Schmidhuber, *Linear Transformers Are Secretly Fast Weight
  Programmers*, 2021: https://arxiv.org/abs/2102.11174
- Csordas and Schmidhuber, *Improving Differentiable Neural Computers Through
  Memory Masking, De-allocation, and Link Distribution Sharpness Control*,
  2019: https://arxiv.org/abs/1904.10278
- Yang, Kautz, and Hatamizadeh, *Gated Delta Networks*, 2024:
  https://arxiv.org/abs/2412.06464
- Behrouz, Zhong, and Mirrokni, *Titans: Learning to Memorize at Test Time*,
  2024: https://arxiv.org/abs/2501.00663
- Xie, *Learning to Forget: Sleep-Inspired Memory Consolidation for Resolving
  Proactive Interference in Large Language Models*, 2026:
  https://arxiv.org/abs/2603.14517

The protocol deliberately describes CLDM as a repository-specific, falsifiable
combination and decomposition. It does not describe delta updates, key-value
memory, adaptive erasure, or conflict detection themselves as new inventions.
