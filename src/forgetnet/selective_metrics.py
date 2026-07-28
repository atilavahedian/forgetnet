"""Auditable metrics and paired inference for the ConflictStream protocol.

The query event is the observational grain, but model comparisons are made only
after aggregating queries to paired seed-level units.  Functions in this module
fail closed on malformed supervision, incomplete counterfactual pairs, missing
seed matches, non-finite values, and undefined acceptance criteria.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Hashable, Iterable, Mapping, Sequence
from dataclasses import asdict, dataclass, is_dataclass
import gzip
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Literal

import numpy as np
import torch
import torch.nn.functional as F

from forgetnet.data import (
    IGNORE_INDEX,
    QUERY_NONE,
    QUERY_STABLE,
    QUERY_UPDATED,
    TaskBatch,
)
from forgetnet.models import ModelOutput


Direction = Literal["higher", "lower"]
PairScope = Literal["stable", "untouched"]

STABLE_NONINFERIORITY_MARGIN = -0.02
UPDATE_MINIMUM_GAIN = 0.05
STALE_MINIMUM_RELATIVE_REDUCTION = 0.20
MAX_MATCHING_RATIO = 1.05


@dataclass(frozen=True)
class QueryRow:
    """One query-event prediction with sufficient statistics for all outcomes."""

    stream_id: int
    pair_id: int
    pair_variant: int
    position: int
    condition: str | int
    query_kind: int
    query_key: int
    changed_key: int
    target: int
    stale_values: tuple[int, ...]
    prediction: int
    correct: int
    confidence: float
    target_probability: float
    stale_probability: float
    stale_intrusion: int
    nll: float
    brier: float
    lag_tokens: int


@dataclass(frozen=True)
class PairLocalityRow:
    """Symmetric prediction change for one aligned counterfactual query."""

    pair_id: int
    position: int
    condition: str | int
    scope: PairScope
    query_kind: int
    query_key: int
    changed_key: int
    correctness_disagreement: float
    target_probability_shift: float
    symmetric_kl: float


def extract_query_rows(
    output: ModelOutput,
    batch: TaskBatch,
    *,
    stream_ids: Sequence[int] | torch.Tensor | None = None,
    condition: str | int = "iid",
) -> list[QueryRow]:
    """Validate a model/batch pair and extract one auditable row per query.

    ``ModelOutput.answer_logits`` must have shape ``[batch, sequence, vocab]``.
    Stale target IDs are deduplicated in their observed order before probability
    mass or intrusion is calculated, so repeated metadata cannot inflate stale
    metrics.
    """

    logits, query_indices = _validated_query_inputs(output, batch)
    batch_indices, positions = query_indices.unbind(dim=1)
    selected_logits = logits[batch_indices, positions].float()
    log_probabilities = F.log_softmax(selected_logits, dim=-1)
    probabilities = log_probabilities.exp()
    targets = batch.answer_targets[batch_indices, positions]
    predictions = probabilities.argmax(dim=-1)
    target_probabilities = probabilities.gather(1, targets.unsqueeze(1)).squeeze(1)
    nll = -log_probabilities.gather(1, targets.unsqueeze(1)).squeeze(1)
    brier = probabilities.square().sum(dim=-1) - 2.0 * target_probabilities + 1.0
    confidence = probabilities.max(dim=-1).values

    resolved_stream_ids = _resolve_stream_ids(stream_ids, logits.shape[0])
    rows: list[QueryRow] = []
    for query_index, (batch_index_tensor, position_tensor) in enumerate(
        zip(batch_indices, positions, strict=True)
    ):
        batch_index = int(batch_index_tensor)
        position = int(position_tensor)
        target = int(targets[query_index])
        stale_values = _unique_stale_values(
            batch.stale_targets[batch_index, position],
            target=target,
        )
        query_kind = int(batch.query_kinds[batch_index, position])
        if query_kind == QUERY_STABLE and stale_values:
            raise ValueError("stable query contains stale targets")
        if query_kind == QUERY_UPDATED and not stale_values:
            raise ValueError("updated query has no stale target")

        stale_probability = float(
            probabilities[query_index, list(stale_values)].sum().detach().cpu()
        ) if stale_values else 0.0
        prediction = int(predictions[query_index])
        stale_intrusion = int(prediction in stale_values)
        lag_tokens = (
            int(batch.lag_tokens[batch_index, position])
            if hasattr(batch, "lag_tokens")
            else -1
        )
        rows.append(
            QueryRow(
                stream_id=resolved_stream_ids[batch_index],
                pair_id=int(batch.pair_ids[batch_index]),
                pair_variant=int(batch.pair_variants[batch_index]),
                position=position,
                condition=condition,
                query_kind=query_kind,
                query_key=int(batch.query_keys[batch_index, position]),
                changed_key=int(batch.changed_keys[batch_index]),
                target=target,
                stale_values=stale_values,
                prediction=prediction,
                correct=int(prediction == target),
                confidence=float(confidence[query_index].detach().cpu()),
                target_probability=float(
                    target_probabilities[query_index].detach().cpu()
                ),
                stale_probability=stale_probability,
                stale_intrusion=stale_intrusion,
                nll=float(nll[query_index].detach().cpu()),
                brier=float(brier[query_index].detach().cpu()),
                lag_tokens=lag_tokens,
            )
        )
    return rows


def summarize_query_rows(
    rows: Sequence[QueryRow | Mapping[str, Any]],
    *,
    ece_bins: int = 15,
) -> dict[str, Any]:
    """Aggregate query rows for one model/seed/condition.

    This function deliberately does not produce confidence intervals.  Its
    output is a single experimental-unit record suitable for
    :func:`paired_seed_summary`.
    """

    records = [_query_mapping(row) for row in rows]
    if not records:
        raise ValueError("cannot summarize an empty query collection")
    records.sort(
        key=lambda row: (
            int(row["stream_id"]),
            int(row["position"]),
            int(row["pair_id"]),
        )
    )
    _validate_serialized_query_rows(records)
    stable = [row for row in records if int(row["query_kind"]) == QUERY_STABLE]
    updated = [row for row in records if int(row["query_kind"]) == QUERY_UPDATED]
    if not stable:
        raise ValueError("query summary has no stable queries")
    if not updated:
        raise ValueError("query summary has no updated queries")

    confidences = [float(row["confidence"]) for row in records]
    correctness = [int(row["correct"]) for row in records]
    return {
        "queries": len(records),
        "stable_queries": len(stable),
        "updated_queries": len(updated),
        "query_accuracy": _mean_field(records, "correct"),
        "stable_accuracy": _mean_field(stable, "correct"),
        "update_accuracy": _mean_field(updated, "correct"),
        "stale_intrusion_rate": _mean_field(updated, "stale_intrusion"),
        "stale_probability": _mean_field(updated, "stale_probability"),
        "query_nll": _mean_field(records, "nll"),
        "stable_nll": _mean_field(stable, "nll"),
        "update_nll": _mean_field(updated, "nll"),
        "query_brier": _mean_field(records, "brier"),
        "stable_brier": _mean_field(stable, "brier"),
        "update_brier": _mean_field(updated, "brier"),
        "ece": expected_calibration_error(
            confidences,
            correctness,
            bins=ece_bins,
        ),
        "aurc": area_under_risk_coverage(confidences, correctness),
    }


def expected_calibration_error(
    confidences: Sequence[float],
    correctness: Sequence[int | bool],
    *,
    bins: int = 15,
) -> float:
    """Return deterministic equal-mass expected calibration error."""

    confidence_array, correctness_array = _validated_calibration_arrays(
        confidences,
        correctness,
    )
    if bins < 1:
        raise ValueError("bins must be positive")
    order = np.argsort(confidence_array, kind="stable")
    partitions = np.array_split(order, min(bins, len(order)))
    return float(
        sum(
            len(partition)
            / len(order)
            * abs(
                float(correctness_array[partition].mean())
                - float(confidence_array[partition].mean())
            )
            for partition in partitions
            if len(partition)
        )
    )


def area_under_risk_coverage(
    confidences: Sequence[float],
    correctness: Sequence[int | bool],
) -> float:
    """Return discrete AURC after sorting predictions by decreasing confidence."""

    confidence_array, correctness_array = _validated_calibration_arrays(
        confidences,
        correctness,
    )
    order = np.argsort(-confidence_array, kind="stable")
    errors = 1.0 - correctness_array[order]
    risks = np.cumsum(errors) / np.arange(1, len(errors) + 1)
    return float(risks.mean())


def extract_pair_locality_rows(
    output: ModelOutput,
    batch: TaskBatch,
    *,
    condition: str | int = "iid",
    scope: PairScope = "stable",
) -> list[PairLocalityRow]:
    """Extract symmetric locality outcomes from complete 0/1 stream pairs.

    ``scope="stable"`` implements the preregistered collateral outcome.
    ``scope="untouched"`` additionally includes updated keys whose event stream
    is unchanged by the designated counterfactual edit.
    """

    if scope not in {"stable", "untouched"}:
        raise ValueError(f"unknown pair locality scope: {scope}")
    logits, _ = _validated_query_inputs(output, batch)
    pair_members = _pair_members(batch)
    log_probabilities = F.log_softmax(logits.float(), dim=-1)
    probabilities = log_probabilities.exp()
    predictions = probabilities.argmax(dim=-1)
    rows: list[PairLocalityRow] = []

    for pair_id in sorted(pair_members):
        left = pair_members[pair_id][0]
        right = pair_members[pair_id][1]
        if int(batch.changed_keys[left]) != int(batch.changed_keys[right]):
            raise ValueError(f"pair {pair_id} has inconsistent changed keys")
        changed_key = int(batch.changed_keys[left])
        left_query_mask = batch.answer_targets[left] != IGNORE_INDEX
        right_query_mask = batch.answer_targets[right] != IGNORE_INDEX
        if not torch.equal(left_query_mask, right_query_mask):
            raise ValueError(f"pair {pair_id} has misaligned query positions")
        if not torch.equal(batch.query_keys[left], batch.query_keys[right]):
            raise ValueError(f"pair {pair_id} has misaligned query keys")
        if not torch.equal(batch.query_kinds[left], batch.query_kinds[right]):
            raise ValueError(f"pair {pair_id} has inconsistent query kinds")

        if scope == "stable":
            eligible = left_query_mask & (batch.query_kinds[left] == QUERY_STABLE)
        else:
            eligible = left_query_mask & (batch.query_keys[left] != changed_key)
        for position_tensor in eligible.nonzero(as_tuple=False).flatten():
            position = int(position_tensor)
            left_target = int(batch.answer_targets[left, position])
            right_target = int(batch.answer_targets[right, position])
            if left_target != right_target:
                raise ValueError(
                    f"pair {pair_id} changes an allegedly untouched query target"
                )
            left_correct = int(predictions[left, position] == left_target)
            right_correct = int(predictions[right, position] == right_target)
            left_target_probability = probabilities[left, position, left_target]
            right_target_probability = probabilities[right, position, right_target]
            symmetric_kl = 0.5 * (
                (
                    probabilities[left, position]
                    * (
                        log_probabilities[left, position]
                        - log_probabilities[right, position]
                    )
                ).sum()
                + (
                    probabilities[right, position]
                    * (
                        log_probabilities[right, position]
                        - log_probabilities[left, position]
                    )
                ).sum()
            )
            rows.append(
                PairLocalityRow(
                    pair_id=pair_id,
                    position=position,
                    condition=condition,
                    scope=scope,
                    query_kind=int(batch.query_kinds[left, position]),
                    query_key=int(batch.query_keys[left, position]),
                    changed_key=changed_key,
                    correctness_disagreement=float(abs(left_correct - right_correct)),
                    target_probability_shift=float(
                        abs(left_target_probability - right_target_probability)
                        .detach()
                        .cpu()
                    ),
                    symmetric_kl=max(0.0, float(symmetric_kl.detach().cpu())),
                )
            )
    if not rows:
        raise ValueError(f"paired batch has no {scope} locality queries")
    return rows


def summarize_pair_locality_rows(
    rows: Sequence[PairLocalityRow | Mapping[str, Any]],
) -> dict[str, Any]:
    """Macro-average locality queries within pairs and then across pairs."""

    if not rows:
        raise ValueError("cannot summarize empty pair locality rows")
    grouped: dict[int, list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        mapping = _mapping(row)
        grouped[int(mapping["pair_id"])].append(mapping)
    metrics = (
        "correctness_disagreement",
        "target_probability_shift",
        "symmetric_kl",
    )
    pair_means = {
        pair_id: {
            metric: _mean_field(pair_rows, metric)
            for metric in metrics
        }
        for pair_id, pair_rows in grouped.items()
    }
    return {
        "pairs": len(grouped),
        "queries": len(rows),
        **{
            metric: float(
                np.mean([pair_summary[metric] for pair_summary in pair_means.values()])
            )
            for metric in metrics
        },
    }


def write_jsonl_gz(
    path: str | Path,
    rows: Iterable[Any],
) -> Path:
    """Write byte-deterministic gzip JSONL using canonical JSON and mtime zero."""

    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        json.dumps(
            _jsonable(row),
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        )
        for row in rows
    ]
    payload = (("\n".join(lines) + "\n") if lines else "").encode("utf-8")
    destination.write_bytes(gzip.compress(payload, compresslevel=9, mtime=0))
    return destination


def read_jsonl_gz(path: str | Path) -> list[dict[str, Any]]:
    """Read and validate a gzip JSONL file written by :func:`write_jsonl_gz`."""

    payload = gzip.decompress(Path(path).read_bytes()).decode("utf-8")
    records: list[dict[str, Any]] = []
    for line_number, line in enumerate(payload.splitlines(), start=1):
        if not line:
            continue
        record = json.loads(line)
        if not isinstance(record, dict):
            raise ValueError(f"JSONL row {line_number} is not an object")
        records.append(record)
    return records


def tensor_sha256(tensors: Mapping[str, torch.Tensor]) -> str:
    """Hash exact tensor names, dtypes, shapes, and canonical contiguous bytes."""

    if not tensors:
        raise ValueError("at least one tensor is required")
    digest = hashlib.sha256()
    for name in sorted(tensors):
        tensor = tensors[name]
        if not isinstance(name, str) or not name:
            raise ValueError("tensor names must be nonempty strings")
        if not isinstance(tensor, torch.Tensor):
            raise TypeError(f"{name!r} is not a tensor")
        if tensor.layout != torch.strided:
            raise ValueError(f"tensor {name!r} must have strided layout")
        canonical = (
            tensor.detach()
            .resolve_conj()
            .resolve_neg()
            .cpu()
            .contiguous()
        )
        raw = canonical.reshape(-1).view(torch.uint8).numpy().tobytes(order="C")
        _hash_part(digest, name.encode("utf-8"))
        _hash_part(digest, str(canonical.dtype).encode("ascii"))
        _hash_part(
            digest,
            json.dumps(list(canonical.shape), separators=(",", ":")).encode("ascii"),
        )
        _hash_part(digest, raw)
    return digest.hexdigest()


def paired_seed_summary(
    candidate: Mapping[Hashable, float],
    baseline: Mapping[Hashable, float],
    *,
    direction: Direction,
    bootstrap_draws: int = 10_000,
    sign_flip_draws: int = 100_000,
    statistics_seed: int = 71_119,
) -> dict[str, Any]:
    """Summarize paired seed metrics with bootstrap CI and sign-flip p-value.

    Benefit deltas are oriented so positive always favors the candidate:
    candidate minus baseline for ``direction="higher"`` and baseline minus
    candidate for ``direction="lower"``.
    """

    if direction not in {"higher", "lower"}:
        raise ValueError(f"unknown metric direction: {direction}")
    units, candidate_values, baseline_values = _paired_arrays(candidate, baseline)
    benefit = (
        candidate_values - baseline_values
        if direction == "higher"
        else baseline_values - candidate_values
    )
    bootstrap = _paired_bootstrap_mean(
        benefit,
        draws=bootstrap_draws,
        seed=statistics_seed,
    )
    sign_flip = _sign_flip_test(
        benefit,
        draws=sign_flip_draws,
        seed=statistics_seed + 1,
    )
    return {
        "n_units": len(units),
        "units": [str(unit) for unit in units],
        "direction": direction,
        "candidate": _describe_values(candidate_values),
        "baseline": _describe_values(baseline_values),
        "benefit_delta": {
            **_describe_values(benefit),
            "ci95": bootstrap["ci95"],
            "bootstrap_draws": bootstrap["draws"],
            "bootstrap_seed": bootstrap["seed"],
        },
        "sign_flip": sign_flip,
    }


def relative_stale_reduction(
    candidate: Mapping[Hashable, float],
    baseline: Mapping[Hashable, float],
    *,
    bootstrap_draws: int = 10_000,
    statistics_seed: int = 71_119,
) -> dict[str, Any]:
    """Return paired bootstrap inference for ``1 - mean(C) / mean(B)``."""

    units, candidate_values, baseline_values = _paired_arrays(candidate, baseline)
    if np.any(candidate_values < 0.0) or np.any(candidate_values > 1.0):
        raise ValueError("candidate stale probabilities must lie in [0, 1]")
    if np.any(baseline_values < 0.0) or np.any(baseline_values > 1.0):
        raise ValueError("baseline stale probabilities must lie in [0, 1]")
    baseline_mean = float(baseline_values.mean())
    if baseline_mean <= 1e-15:
        raise ValueError("relative stale reduction is undefined at zero baseline")
    if bootstrap_draws < 1:
        raise ValueError("bootstrap_draws must be positive")
    rng = np.random.default_rng(statistics_seed)
    reductions = np.empty(bootstrap_draws, dtype=np.float64)
    cursor = 0
    while cursor < bootstrap_draws:
        stop = min(cursor + 65_536, bootstrap_draws)
        indices = rng.integers(0, len(units), size=(stop - cursor, len(units)))
        sampled_baseline = baseline_values[indices].mean(axis=1)
        if np.any(sampled_baseline <= 1e-15):
            raise ValueError("bootstrap sampled a zero stale-probability baseline")
        reductions[cursor:stop] = (
            1.0
            - candidate_values[indices].mean(axis=1) / sampled_baseline
        )
        cursor = stop
    return {
        "n_units": len(units),
        "candidate_mean": float(candidate_values.mean()),
        "baseline_mean": baseline_mean,
        "estimate": 1.0 - float(candidate_values.mean()) / baseline_mean,
        "ci95": [
            float(value)
            for value in np.quantile(reductions, [0.025, 0.975], method="linear")
        ],
        "bootstrap_draws": bootstrap_draws,
        "bootstrap_seed": statistics_seed,
    }


def conflict_acceptance(
    candidate_by_metric: Mapping[str, Mapping[Hashable, float]],
    baseline_by_metric: Mapping[str, Mapping[Hashable, float]],
    *,
    candidate_parameters: int,
    baseline_parameters: int,
    candidate_flops: float,
    baseline_flops: float,
    bootstrap_draws: int = 10_000,
    sign_flip_draws: int = 100_000,
    statistics_seed: int = 71_119,
) -> dict[str, Any]:
    """Evaluate the preregistered primary gates and fail closed on any defect."""

    gate_names = (
        "stable_noninferiority",
        "update_superiority",
        "stale_reduction",
        "parameter_match",
        "flop_match",
    )
    try:
        required = ("stable_accuracy", "update_accuracy", "stale_probability")
        for metric in required:
            if metric not in candidate_by_metric or metric not in baseline_by_metric:
                raise ValueError(f"missing required metric: {metric}")
        stable = paired_seed_summary(
            candidate_by_metric["stable_accuracy"],
            baseline_by_metric["stable_accuracy"],
            direction="higher",
            bootstrap_draws=bootstrap_draws,
            sign_flip_draws=sign_flip_draws,
            statistics_seed=statistics_seed,
        )
        update = paired_seed_summary(
            candidate_by_metric["update_accuracy"],
            baseline_by_metric["update_accuracy"],
            direction="higher",
            bootstrap_draws=bootstrap_draws,
            sign_flip_draws=sign_flip_draws,
            statistics_seed=statistics_seed + 101,
        )
        stale = paired_seed_summary(
            candidate_by_metric["stale_probability"],
            baseline_by_metric["stale_probability"],
            direction="lower",
            bootstrap_draws=bootstrap_draws,
            sign_flip_draws=sign_flip_draws,
            statistics_seed=statistics_seed + 202,
        )
        relative_stale = relative_stale_reduction(
            candidate_by_metric["stale_probability"],
            baseline_by_metric["stale_probability"],
            bootstrap_draws=bootstrap_draws,
            statistics_seed=statistics_seed + 303,
        )
        parameter_ratio = _matching_ratio(
            candidate_parameters,
            baseline_parameters,
            name="parameter",
        )
        flop_ratio = _matching_ratio(
            candidate_flops,
            baseline_flops,
            name="FLOP",
        )
        stable_low = float(stable["benefit_delta"]["ci95"][0])
        update_mean = float(update["benefit_delta"]["mean"])
        update_low = float(update["benefit_delta"]["ci95"][0])
        stale_low = float(stale["benefit_delta"]["ci95"][0])
        relative_low = float(relative_stale["ci95"][0])
        gates = {
            "stable_noninferiority": stable_low > STABLE_NONINFERIORITY_MARGIN,
            "update_superiority": (
                update_mean >= UPDATE_MINIMUM_GAIN and update_low > 0.0
            ),
            "stale_reduction": (
                float(relative_stale["estimate"])
                >= STALE_MINIMUM_RELATIVE_REDUCTION
                and stale_low > 0.0
                and relative_low > 0.0
            ),
            "parameter_match": parameter_ratio <= MAX_MATCHING_RATIO,
            "flop_match": flop_ratio <= MAX_MATCHING_RATIO,
        }
        return {
            "primary_pass": all(gates.values()),
            "gates": gates,
            "thresholds": {
                "stable_ci_low": STABLE_NONINFERIORITY_MARGIN,
                "update_mean_gain": UPDATE_MINIMUM_GAIN,
                "stale_relative_reduction": STALE_MINIMUM_RELATIVE_REDUCTION,
                "maximum_matching_ratio": MAX_MATCHING_RATIO,
            },
            "stable": stable,
            "update": update,
            "stale": stale,
            "relative_stale_reduction": relative_stale,
            "parameter_ratio": parameter_ratio,
            "flop_ratio": flop_ratio,
        }
    except (KeyError, TypeError, ValueError) as error:
        return {
            "primary_pass": False,
            "gates": {name: False for name in gate_names},
            "error": str(error),
        }


def _validated_query_inputs(
    output: ModelOutput,
    batch: TaskBatch,
) -> tuple[torch.Tensor, torch.Tensor]:
    if output.answer_logits is None:
        raise ValueError("selective metrics require per-token answer logits")
    logits = output.answer_logits
    if logits.ndim != 3:
        raise ValueError("answer logits must have shape [batch, sequence, vocab]")
    batch_size, sequence_length, vocab_size = logits.shape
    expected_shape = (batch_size, sequence_length)
    for name in ("answer_targets", "query_kinds", "query_keys"):
        tensor = getattr(batch, name)
        if tuple(tensor.shape) != expected_shape:
            raise ValueError(f"{name} must match answer-logit batch/sequence shape")
    if batch.stale_targets.ndim != 3 or tuple(batch.stale_targets.shape[:2]) != expected_shape:
        raise ValueError("stale_targets must have shape [batch, sequence, stale_slots]")
    for name in ("pair_ids", "pair_variants", "changed_keys"):
        if tuple(getattr(batch, name).shape) != (batch_size,):
            raise ValueError(f"{name} must have shape [batch]")
    if not bool(torch.isfinite(logits).all()):
        raise ValueError("answer logits contain non-finite values")

    query_mask = batch.answer_targets != IGNORE_INDEX
    if not bool(query_mask.any()):
        raise ValueError("batch contains no answer queries")
    if bool((batch.query_kinds[~query_mask] != QUERY_NONE).any()):
        raise ValueError("non-query token has a query kind")
    query_kinds = batch.query_kinds[query_mask]
    if bool(
        ((query_kinds != QUERY_STABLE) & (query_kinds != QUERY_UPDATED)).any()
    ):
        raise ValueError("query token has an unknown query kind")
    targets = batch.answer_targets[query_mask]
    if bool(((targets < 0) | (targets >= vocab_size)).any()):
        raise ValueError("query target is outside the answer vocabulary")
    if bool((batch.query_keys[query_mask] < 0).any()):
        raise ValueError("query token is missing its key")
    if bool((batch.query_keys[query_mask] >= vocab_size).any()):
        raise ValueError("query key is outside the declared vocabulary")

    stale_at_queries = batch.stale_targets[query_mask]
    valid_stale = stale_at_queries != IGNORE_INDEX
    if bool(
        (
            valid_stale
            & ((stale_at_queries < 0) | (stale_at_queries >= vocab_size))
        ).any()
    ):
        raise ValueError("stale target is outside the answer vocabulary")
    if bool((stale_at_queries == targets.unsqueeze(1)).any()):
        raise ValueError("current target is also marked stale")
    valid_pair_variants = (
        (batch.pair_variants == -1)
        | (batch.pair_variants == 0)
        | (batch.pair_variants == 1)
    )
    if not bool(valid_pair_variants.all()):
        raise ValueError("pair variants must be -1, 0, or 1")
    if bool(((batch.pair_ids < 0) != (batch.pair_variants < 0)).any()):
        raise ValueError("pair IDs and variants have inconsistent missingness")
    return logits, query_mask.nonzero(as_tuple=False)


def _unique_stale_values(stale: torch.Tensor, *, target: int) -> tuple[int, ...]:
    seen: set[int] = set()
    unique: list[int] = []
    for value in stale.detach().cpu().tolist():
        value = int(value)
        if value == IGNORE_INDEX:
            continue
        if value == target:
            raise ValueError("current target is also marked stale")
        if value < 0:
            raise ValueError("stale target contains an invalid negative ID")
        if value not in seen:
            seen.add(value)
            unique.append(value)
    return tuple(unique)


def _resolve_stream_ids(
    stream_ids: Sequence[int] | torch.Tensor | None,
    batch_size: int,
) -> list[int]:
    if stream_ids is None:
        return list(range(batch_size))
    if isinstance(stream_ids, torch.Tensor):
        values = stream_ids.detach().cpu().tolist()
    else:
        values = list(stream_ids)
    if len(values) != batch_size:
        raise ValueError("stream_ids must contain one ID per batch row")
    resolved = [int(value) for value in values]
    if len(set(resolved)) != len(resolved):
        raise ValueError("stream_ids must be unique within a batch")
    return resolved


def _pair_members(batch: TaskBatch) -> dict[int, dict[int, int]]:
    members: dict[int, dict[int, int]] = defaultdict(dict)
    for row, (pair_id_raw, variant_raw) in enumerate(
        zip(batch.pair_ids.tolist(), batch.pair_variants.tolist(), strict=True)
    ):
        pair_id = int(pair_id_raw)
        variant = int(variant_raw)
        if pair_id < 0 or variant not in {0, 1}:
            raise ValueError("pair locality requires nonnegative IDs and 0/1 variants")
        if variant in members[pair_id]:
            raise ValueError(f"pair {pair_id} has duplicate variant {variant}")
        members[pair_id][variant] = row
    for pair_id, variants in members.items():
        if set(variants) != {0, 1}:
            raise ValueError(f"pair {pair_id} is missing a 0/1 member")
    return dict(members)


def _validated_calibration_arrays(
    confidences: Sequence[float],
    correctness: Sequence[int | bool],
) -> tuple[np.ndarray, np.ndarray]:
    if len(confidences) != len(correctness) or not confidences:
        raise ValueError("confidence and correctness must be nonempty and aligned")
    confidence_array = np.asarray(confidences, dtype=np.float64)
    correctness_array = np.asarray(correctness, dtype=np.float64)
    if not np.isfinite(confidence_array).all():
        raise ValueError("confidence contains non-finite values")
    if np.any((confidence_array < 0.0) | (confidence_array > 1.0)):
        raise ValueError("confidence must lie in [0, 1]")
    if not np.isin(correctness_array, [0.0, 1.0]).all():
        raise ValueError("correctness must contain only 0/1 values")
    return confidence_array, correctness_array


def _query_mapping(row: QueryRow | Mapping[str, Any]) -> Mapping[str, Any]:
    mapping = _mapping(row)
    required = {
        "stream_id",
        "pair_id",
        "position",
        "query_kind",
        "stale_values",
        "correct",
        "confidence",
        "stale_probability",
        "stale_intrusion",
        "nll",
        "brier",
    }
    missing = required - set(mapping)
    if missing:
        raise ValueError(f"query row is missing fields: {sorted(missing)}")
    return mapping


def _validate_serialized_query_rows(rows: Sequence[Mapping[str, Any]]) -> None:
    numeric_fields = ("confidence", "stale_probability", "nll", "brier")
    for row in rows:
        kind = int(row["query_kind"])
        stale_values = tuple(int(value) for value in row["stale_values"])
        if kind == QUERY_STABLE and stale_values:
            raise ValueError("stable query contains stale targets")
        if kind == QUERY_UPDATED and not stale_values:
            raise ValueError("updated query has no stale target")
        if kind not in {QUERY_STABLE, QUERY_UPDATED}:
            raise ValueError("query row has an unknown query kind")
        if int(row["correct"]) not in {0, 1} or int(row["stale_intrusion"]) not in {0, 1}:
            raise ValueError("query indicators must be 0/1")
        for field in numeric_fields:
            if not math.isfinite(float(row[field])):
                raise ValueError(f"query row {field} is non-finite")
        if not 0.0 <= float(row["confidence"]) <= 1.0:
            raise ValueError("query confidence must lie in [0, 1]")
        if not 0.0 <= float(row["stale_probability"]) <= 1.0 + 1e-7:
            raise ValueError("stale probability must lie in [0, 1]")


def _mapping(row: Any) -> Mapping[str, Any]:
    if is_dataclass(row) and not isinstance(row, type):
        return asdict(row)
    if isinstance(row, Mapping):
        return row
    raise TypeError("row must be a dataclass or mapping")


def _mean_field(rows: Sequence[Mapping[str, Any]], field: str) -> float:
    return float(np.mean([float(row[field]) for row in rows]))


def _jsonable(value: Any) -> Any:
    if is_dataclass(value) and not isinstance(value, type):
        return _jsonable(asdict(value))
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, torch.Tensor):
        if value.ndim == 0:
            return value.detach().cpu().item()
        return value.detach().cpu().tolist()
    return value


def _hash_part(digest: Any, payload: bytes) -> None:
    digest.update(len(payload).to_bytes(8, byteorder="big", signed=False))
    digest.update(payload)


def _paired_arrays(
    candidate: Mapping[Hashable, float],
    baseline: Mapping[Hashable, float],
) -> tuple[list[Hashable], np.ndarray, np.ndarray]:
    if set(candidate) != set(baseline):
        missing_candidate = sorted(set(baseline) - set(candidate), key=repr)
        missing_baseline = sorted(set(candidate) - set(baseline), key=repr)
        raise ValueError(
            "paired units differ; "
            f"missing candidate={missing_candidate}, missing baseline={missing_baseline}"
        )
    if len(candidate) < 2:
        raise ValueError("paired inference requires at least two units")
    units = sorted(candidate, key=repr)
    candidate_values = np.asarray([candidate[unit] for unit in units], dtype=np.float64)
    baseline_values = np.asarray([baseline[unit] for unit in units], dtype=np.float64)
    if not np.isfinite(candidate_values).all() or not np.isfinite(baseline_values).all():
        raise ValueError("paired metrics contain non-finite values")
    return units, candidate_values, baseline_values


def _describe_values(values: np.ndarray) -> dict[str, Any]:
    quartiles = np.quantile(values, [0.25, 0.5, 0.75], method="linear")
    return {
        "mean": float(values.mean()),
        "median": float(quartiles[1]),
        "iqr": [float(quartiles[0]), float(quartiles[2])],
        "values": [float(value) for value in values],
    }


def _paired_bootstrap_mean(
    values: np.ndarray,
    *,
    draws: int,
    seed: int,
) -> dict[str, Any]:
    if draws < 1:
        raise ValueError("bootstrap draws must be positive")
    rng = np.random.default_rng(seed)
    estimates = np.empty(draws, dtype=np.float64)
    cursor = 0
    while cursor < draws:
        stop = min(cursor + 65_536, draws)
        indices = rng.integers(0, len(values), size=(stop - cursor, len(values)))
        estimates[cursor:stop] = values[indices].mean(axis=1)
        cursor = stop
    return {
        "ci95": [
            float(value)
            for value in np.quantile(estimates, [0.025, 0.975], method="linear")
        ],
        "draws": draws,
        "seed": seed,
    }


def _sign_flip_test(
    values: np.ndarray,
    *,
    draws: int,
    seed: int,
) -> dict[str, Any]:
    observed = abs(float(values.mean()))
    tolerance = max(1e-15, observed * 1e-12)
    n_units = len(values)
    if n_units <= 20:
        total = 1 << n_units
        extreme = 0
        bit_positions = np.arange(n_units, dtype=np.uint64)
        for start in range(0, total, 65_536):
            stop = min(start + 65_536, total)
            masks = np.arange(start, stop, dtype=np.uint64)[:, None]
            bits = ((masks >> bit_positions[None, :]) & 1).astype(np.float64)
            signs = 1.0 - 2.0 * bits
            statistics = np.abs(signs @ values / n_units)
            extreme += int(np.count_nonzero(statistics >= observed - tolerance))
        return {
            "method": "exact",
            "draws": total,
            "seed": None,
            "pvalue_two_sided": extreme / total,
        }
    if draws < 1:
        raise ValueError("sign-flip draws must be positive")
    rng = np.random.default_rng(seed)
    extreme = 0
    completed = 0
    while completed < draws:
        count = min(65_536, draws - completed)
        signs = 1.0 - 2.0 * rng.integers(0, 2, size=(count, n_units))
        statistics = np.abs(signs @ values / n_units)
        extreme += int(np.count_nonzero(statistics >= observed - tolerance))
        completed += count
    return {
        "method": "monte_carlo",
        "draws": draws,
        "seed": seed,
        "pvalue_two_sided": (extreme + 1) / (draws + 1),
    }


def _matching_ratio(candidate: float, baseline: float, *, name: str) -> float:
    candidate_value = float(candidate)
    baseline_value = float(baseline)
    if (
        not math.isfinite(candidate_value)
        or not math.isfinite(baseline_value)
        or candidate_value <= 0.0
        or baseline_value <= 0.0
    ):
        raise ValueError(f"{name} counts must be finite and positive")
    return max(candidate_value, baseline_value) / min(candidate_value, baseline_value)


__all__ = [
    "PairLocalityRow",
    "QueryRow",
    "area_under_risk_coverage",
    "conflict_acceptance",
    "expected_calibration_error",
    "extract_pair_locality_rows",
    "extract_query_rows",
    "paired_seed_summary",
    "read_jsonl_gz",
    "relative_stale_reduction",
    "summarize_pair_locality_rows",
    "summarize_query_rows",
    "tensor_sha256",
    "write_jsonl_gz",
]
