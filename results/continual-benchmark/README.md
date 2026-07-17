# Parameter-Matched Continual Pilot

This artifact was generated on CPU with `equal-update-paired-evaluation-v2`. It compares three models over seeds 2027, 2028, and 2029, with 100 updates each on associative lookup, changing facts, then needle recall.

| Model | Parameters | Final learned-task accuracy | Mean forgetting | Mean wall time |
| --- | ---: | ---: | ---: | ---: |
| ForgetNet | 79,936 | 0.1285 ± 0.0156 | 0.0333 ± 0.0204 | 17.09 s |
| No forget | 79,936 | 0.1313 ± 0.0255 | 0.0333 ± 0.0217 | 16.22 s |
| Local Transformer | 75,104 | 0.1104 ± 0.0233 | 0.0188 ± 0.0184 | 3.38 s |

Values after `±` are descriptive normal 95% half-widths across three seeds.

Paired deltas relative to ForgetNet:

| Comparison | Accuracy delta | Forgetting delta |
| --- | ---: | ---: |
| No forget − ForgetNet | +0.0028 ± 0.0106 | 0.0000 ± 0.0051 |
| Local Transformer − ForgetNet | −0.0181 ± 0.0339 | −0.0146 ± 0.0310 |

The intervals include zero. This pilot does not support the learned erase gate, and it does not establish an accuracy or retention advantage over the parameter-matched local Transformer. Its purpose is to make that failure visible and reproducible.

See `docs/continual_protocol.md` for the exact command and metric definitions.
