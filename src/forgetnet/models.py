from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Literal

import torch
from torch import nn
import torch.nn.functional as F

from forgetnet.data import QUERY_TOKEN, READ_TOKEN, SET_TOKEN


@dataclass(frozen=True)
class MemoryStats:
    final_memory_shape: tuple[int, int, int]
    write_frequency: float
    mean_write_strength: float
    mean_surprise: float


@dataclass(frozen=True)
class ModelOutput:
    logits: torch.Tensor
    memory_stats: MemoryStats
    aux_logits: torch.Tensor | None = None
    answer_logits: torch.Tensor | None = None
    memory_trace: MemoryTrace | None = None
    memory_state: CLDMState | None = None


@dataclass(frozen=True)
class CLDMState:
    keys: torch.Tensor
    values: torch.Tensor
    occupied: torch.Tensor
    usage: torch.Tensor


@dataclass(frozen=True)
class MemoryTrace:
    write_mask: torch.Tensor
    write_gate: torch.Tensor
    conflict: torch.Tensor
    match_probability: torch.Tensor
    slot_index: torch.Tensor
    localization: torch.Tensor
    route_weights: torch.Tensor | None = None


@dataclass(frozen=True)
class CLDMStepTrace:
    write_mask: torch.Tensor
    write_gate: torch.Tensor
    conflict: torch.Tensor
    match_probability: torch.Tensor
    slot_index: torch.Tensor
    localization: torch.Tensor
    route_weights: torch.Tensor


class ConflictLocalizedMemoryCell(nn.Module):
    """A bounded key-value memory with conflict-localized delta replacement."""

    VARIANTS = {
        "cldm",
        "cldm_no_conflict",
        "cldm_no_replace",
        "cldm_soft_route",
    }

    def __init__(
        self,
        d_model: int,
        memory_slots: int,
        variant: str = "cldm",
        match_threshold: float = 0.8,
        match_temperature: float = 0.05,
        conflict_threshold: float = 0.25,
        conflict_temperature: float = 0.05,
    ) -> None:
        super().__init__()
        if variant not in self.VARIANTS:
            raise ValueError(f"unknown CLDM variant: {variant}")
        if memory_slots < 1:
            raise ValueError("memory_slots must be positive")
        self.d_model = d_model
        self.memory_slots = memory_slots
        self.variant = variant
        self.match_threshold = match_threshold
        self.match_temperature = match_temperature
        self.conflict_threshold = conflict_threshold
        self.conflict_temperature = conflict_temperature
        self.write_gate = nn.Linear(d_model * 3 + 2, 1)
        nn.init.zeros_(self.write_gate.weight)
        nn.init.constant_(self.write_gate.bias, 2.0)

    def empty_state(
        self,
        batch_size: int,
        *,
        device: torch.device,
        dtype: torch.dtype,
    ) -> CLDMState:
        return CLDMState(
            keys=torch.zeros(
                batch_size,
                self.memory_slots,
                self.d_model,
                device=device,
                dtype=torch.float32,
            ),
            values=torch.zeros(
                batch_size,
                self.memory_slots,
                self.d_model,
                device=device,
                dtype=dtype,
            ),
            occupied=torch.zeros(
                batch_size,
                self.memory_slots,
                device=device,
                dtype=torch.bool,
            ),
            usage=torch.zeros(
                batch_size,
                self.memory_slots,
                device=device,
                dtype=torch.float32,
            ),
        )

    def read(
        self,
        state: CLDMState,
        query: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        query_fp32 = F.normalize(query.float(), dim=-1, eps=1e-6)
        scores = torch.einsum("bd,bsd->bs", query_fp32, state.keys)
        has_memory = state.occupied.any(dim=-1, keepdim=True)
        masked_scores = scores.masked_fill(~state.occupied, -1e9)
        weights = F.softmax(masked_scores / 0.1, dim=-1)
        weights = torch.where(has_memory, weights, torch.zeros_like(weights))
        read = torch.einsum("bs,bsd->bd", weights.to(state.values.dtype), state.values)
        return read, weights

    def write(
        self,
        state: CLDMState,
        candidate_key: torch.Tensor,
        candidate_value: torch.Tensor,
        write_mask: torch.Tensor,
        *,
        read_weights: torch.Tensor | None = None,
    ) -> tuple[CLDMState, CLDMStepTrace]:
        key = F.normalize(candidate_key.float(), dim=-1, eps=1e-6)
        value = torch.tanh(candidate_value)
        scores = torch.einsum("bd,bsd->bs", key, state.keys)
        has_memory = state.occupied.any(dim=-1)
        masked_scores = scores.masked_fill(~state.occupied, -1e9)
        match_weights = F.softmax(masked_scores / 0.1, dim=-1)
        match_weights = torch.where(
            has_memory.unsqueeze(-1),
            match_weights,
            torch.zeros_like(match_weights),
        )
        max_similarity, match_index = masked_scores.max(dim=-1)
        max_similarity = torch.where(
            has_memory,
            max_similarity,
            torch.full_like(max_similarity, -1.0),
        )
        match_soft = (
            torch.sigmoid(
                (max_similarity - self.match_threshold) / self.match_temperature
            )
            * has_memory.float()
        )
        match_hard = has_memory & (max_similarity >= self.match_threshold)
        match_st = match_hard.float() + match_soft - match_soft.detach()

        free_mask = ~state.occupied
        has_free = free_mask.any(dim=-1)
        first_free = free_mask.float().argmax(dim=-1)
        least_used = state.usage.argmin(dim=-1)
        allocation_index = torch.where(has_free, first_free, least_used)
        match_one_hot = F.one_hot(match_index, self.memory_slots).float()
        allocation_one_hot = F.one_hot(allocation_index, self.memory_slots).float()
        hard_route = torch.where(
            match_hard.unsqueeze(-1),
            match_one_hot,
            allocation_one_hot,
        )
        soft_route = (
            match_st.unsqueeze(-1) * match_weights
            + (1.0 - match_st).unsqueeze(-1) * allocation_one_hot
        )
        if self.variant == "cldm_soft_route":
            route = torch.where(
                match_hard.unsqueeze(-1),
                match_weights,
                allocation_one_hot,
            )
        else:
            route = hard_route + soft_route - soft_route.detach()

        matched_value = torch.einsum(
            "bs,bsd->bd",
            match_weights.to(state.values.dtype),
            state.values,
        )
        value_similarity = F.cosine_similarity(
            value.float(),
            matched_value.float(),
            dim=-1,
            eps=1e-6,
        )
        conflict = 0.5 * (1.0 - value_similarity)
        conflict = conflict * match_st
        conflict_gate = torch.sigmoid(
            (conflict - self.conflict_threshold) / self.conflict_temperature
        )

        gate_features = torch.cat(
            [
                key.to(value.dtype),
                value,
                matched_value,
                match_soft.unsqueeze(-1).to(value.dtype),
                conflict.unsqueeze(-1).to(value.dtype),
            ],
            dim=-1,
        )
        learned_gate = torch.sigmoid(self.write_gate(gate_features)).squeeze(-1)
        if self.variant == "cldm_no_conflict":
            replacement = torch.ones_like(match_st)
        elif self.variant == "cldm_no_replace":
            replacement = 1.0 - match_st
        else:
            replacement = (1.0 - match_st) + match_st * conflict_gate
        event = write_mask.float()
        update_gate = event * learned_gate * replacement

        value_rate = route.to(state.values.dtype).unsqueeze(-1) * update_gate.to(
            state.values.dtype
        ).view(-1, 1, 1)
        new_values = state.values + value_rate * (
            value.unsqueeze(1) - state.values
        )

        novel_event = event * (~match_hard).float()
        key_rate = allocation_one_hot.unsqueeze(-1) * novel_event.view(-1, 1, 1)
        new_keys = state.keys * (1.0 - key_rate) + key.unsqueeze(1) * key_rate
        new_keys = F.normalize(new_keys, dim=-1, eps=1e-6)
        newly_occupied = allocation_one_hot.bool() & novel_event.bool().unsqueeze(-1)
        new_occupied = (state.occupied | newly_occupied).detach()

        if read_weights is None:
            read_weights = torch.zeros_like(state.usage)
        new_usage = (
            0.99 * state.usage
            + read_weights.detach().float()
            + hard_route.detach() * event.unsqueeze(-1)
        ).detach()
        new_state = CLDMState(
            keys=new_keys,
            values=new_values,
            occupied=new_occupied,
            usage=new_usage,
        )
        slot_index = hard_route.argmax(dim=-1)
        slot_index = torch.where(
            write_mask,
            slot_index,
            torch.full_like(slot_index, -1),
        )
        trace = CLDMStepTrace(
            write_mask=write_mask,
            write_gate=update_gate,
            conflict=conflict * event,
            match_probability=match_soft * event,
            slot_index=slot_index,
            localization=route.detach().square().sum(dim=-1) * event,
            route_weights=route,
        )
        return new_state, trace


class ConflictLocalizedMemory(nn.Module):
    """Local attention plus explicit, bounded conflict-localized delta memory."""

    def __init__(
        self,
        vocab_size: int,
        d_model: int = 64,
        memory_slots: int = 16,
        window_size: int = 8,
        max_seq_len: int = 512,
        variant: str = "cldm",
    ) -> None:
        super().__init__()
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.memory_slots = memory_slots
        self.window_size = window_size
        self.max_seq_len = max_seq_len
        self.variant = variant

        self.token_embedding = nn.Embedding(vocab_size, d_model)
        self.position_embedding = nn.Embedding(max_seq_len, d_model)
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.local_out = nn.Linear(d_model, d_model)
        self.controller_query = nn.Linear(d_model, d_model, bias=False)
        self.key_proj = nn.Linear(d_model, d_model, bias=False)
        self.value_proj = nn.Linear(d_model, d_model)
        self.read_out = nn.Linear(d_model, d_model)
        self.norm = nn.LayerNorm(d_model)
        self.token_head = nn.Linear(d_model, vocab_size)
        self.answer_head = nn.Linear(d_model, vocab_size)
        self.memory_cell = ConflictLocalizedMemoryCell(
            d_model=d_model,
            memory_slots=memory_slots,
            variant=variant,
        )
        nn.init.orthogonal_(self.key_proj.weight)

    def forward(
        self,
        input_ids: torch.Tensor,
        *,
        state: CLDMState | None = None,
        trace_mode: Literal["none", "summary", "full"] = "summary",
        return_state: bool = False,
    ) -> ModelOutput:
        if trace_mode not in {"none", "summary", "full"}:
            raise ValueError(f"unknown trace_mode: {trace_mode}")
        batch_size, seq_len = input_ids.shape
        if seq_len > self.max_seq_len:
            raise ValueError(f"seq_len={seq_len} exceeds max_seq_len={self.max_seq_len}")
        semantic = self.token_embedding(input_ids)
        positions = torch.arange(seq_len, device=input_ids.device).unsqueeze(0)
        embeddings = semantic + self.position_embedding(positions)
        memory = state or self.memory_cell.empty_state(
            batch_size,
            device=input_ids.device,
            dtype=semantic.dtype,
        )
        if memory.values.shape[:2] != (batch_size, self.memory_slots):
            raise ValueError("memory state does not match batch size or slot count")

        history: list[torch.Tensor] = []
        answer_logits: list[torch.Tensor] = []
        auxiliary_logits: list[torch.Tensor] = []
        step_traces: list[CLDMStepTrace] = []
        for step in range(seq_len):
            current = embeddings[:, step, :]
            local = self._local_attention(current, history)
            if step >= 2:
                semantic_key = self.key_proj(semantic[:, step - 1, :])
                is_query = (input_ids[:, step - 2] == QUERY_TOKEN) & (
                    input_ids[:, step] == READ_TOKEN
                )
                is_write = input_ids[:, step - 2] == SET_TOKEN
            else:
                semantic_key = self.key_proj(semantic[:, step, :])
                is_query = torch.zeros(batch_size, dtype=torch.bool, device=input_ids.device)
                is_write = torch.zeros(batch_size, dtype=torch.bool, device=input_ids.device)
            query = torch.where(
                is_query.unsqueeze(-1),
                semantic_key,
                self.controller_query(local),
            )
            read, read_weights = self.memory_cell.read(memory, query)
            hidden = self.norm(local + self.read_out(read))
            auxiliary_logits.append(self.token_head(hidden))
            answer_logits.append(self.answer_head(hidden))

            candidate_value = self.value_proj(semantic[:, step, :])
            memory, step_trace = self.memory_cell.write(
                memory,
                semantic_key,
                candidate_value,
                is_write,
                read_weights=read_weights,
            )
            step_traces.append(step_trace)
            history.append(current)
            if len(history) > self.window_size:
                history = history[-self.window_size :]

        answer_sequence = torch.stack(answer_logits, dim=1)
        auxiliary_sequence = torch.stack(auxiliary_logits, dim=1)
        write_gate = torch.stack([trace.write_gate for trace in step_traces], dim=1)
        stats = MemoryStats(
            final_memory_shape=tuple(memory.values.shape),
            write_frequency=float((write_gate > 0.05).float().mean().detach().cpu()),
            mean_write_strength=float(write_gate.mean().detach().cpu()),
            mean_surprise=0.0,
        )
        memory_trace = None
        if trace_mode != "none":
            memory_trace = MemoryTrace(
                write_mask=torch.stack([trace.write_mask for trace in step_traces], dim=1),
                write_gate=write_gate,
                conflict=torch.stack([trace.conflict for trace in step_traces], dim=1),
                match_probability=torch.stack(
                    [trace.match_probability for trace in step_traces], dim=1
                ),
                slot_index=torch.stack([trace.slot_index for trace in step_traces], dim=1),
                localization=torch.stack(
                    [trace.localization for trace in step_traces], dim=1
                ),
                route_weights=(
                    torch.stack([trace.route_weights for trace in step_traces], dim=1)
                    if trace_mode == "full"
                    else None
                ),
            )
        return ModelOutput(
            logits=answer_sequence[:, -1, :],
            answer_logits=answer_sequence,
            aux_logits=auxiliary_sequence,
            memory_stats=stats,
            memory_trace=memory_trace,
            memory_state=memory if return_state else None,
        )

    def _local_attention(
        self,
        current: torch.Tensor,
        history: list[torch.Tensor],
    ) -> torch.Tensor:
        window = torch.stack(history[-self.window_size :] + [current], dim=1)
        query = self.q_proj(current).unsqueeze(1)
        keys = self.k_proj(window)
        values = self.v_proj(window)
        scores = torch.matmul(query, keys.transpose(1, 2)) / math.sqrt(self.d_model)
        weights = F.softmax(scores, dim=-1)
        return self.local_out(torch.matmul(weights, values).squeeze(1))


class ForgetNet(nn.Module):
    """A causal sequence model with local attention and differentiable memory."""

    def __init__(
        self,
        vocab_size: int,
        d_model: int = 64,
        memory_slots: int = 16,
        window_size: int = 8,
        max_seq_len: int = 512,
        variant: str = "forgetnet",
    ) -> None:
        super().__init__()
        if variant not in {"forgetnet", "no_forget", "no_surprise", "random_write", "fifo_memory"}:
            raise ValueError(f"unknown ForgetNet variant: {variant}")
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.memory_slots = memory_slots
        self.window_size = window_size
        self.max_seq_len = max_seq_len
        self.variant = variant

        self.token_embedding = nn.Embedding(vocab_size, d_model)
        self.position_embedding = nn.Embedding(max_seq_len, d_model)
        self.initial_memory = nn.Parameter(torch.randn(memory_slots, d_model) * 0.02)

        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.local_out = nn.Linear(d_model, d_model)

        self.memory_query = nn.Linear(d_model, d_model)
        self.memory_write_query = nn.Linear(d_model, d_model)
        self.write_proj = nn.Linear(d_model, d_model)
        self.erase_proj = nn.Linear(d_model, d_model)
        self.update_gate = nn.Linear(d_model * 2 + 1, d_model)
        self.read_out = nn.Linear(d_model, d_model)
        self.norm = nn.LayerNorm(d_model)

        self.token_head = nn.Linear(d_model, vocab_size)
        self.answer_head = nn.Linear(d_model, vocab_size)

    def forward(self, input_ids: torch.Tensor) -> ModelOutput:
        batch_size, seq_len = input_ids.shape
        if seq_len > self.max_seq_len:
            raise ValueError(f"seq_len={seq_len} exceeds max_seq_len={self.max_seq_len}")

        positions = torch.arange(seq_len, device=input_ids.device).unsqueeze(0)
        embeddings = self.token_embedding(input_ids) + self.position_embedding(positions)
        memory = self.initial_memory.unsqueeze(0).expand(batch_size, -1, -1)

        history: list[torch.Tensor] = []
        answer_logits: list[torch.Tensor] = []
        aux_logits: list[torch.Tensor] = []
        write_strengths: list[torch.Tensor] = []
        surprises: list[torch.Tensor] = []

        for step in range(seq_len):
            current = embeddings[:, step, :]
            local = self._local_attention(current, history)
            read = self._read_memory(local, memory)
            hidden = self.norm(local + self.read_out(read))

            token_logits = self.token_head(hidden)
            answer_logits.append(self.answer_head(hidden))

            surprise = self._causal_surprise(aux_logits, input_ids[:, step])
            memory, write_strength = self._write_memory(hidden, read, memory, surprise, input_ids[:, step], step)
            aux_logits.append(token_logits)
            write_strengths.append(write_strength)
            surprises.append(surprise.mean())

            history.append(current)
            if len(history) > self.window_size:
                history = history[-self.window_size :]

        strengths = torch.stack(write_strengths)
        stats = MemoryStats(
            final_memory_shape=tuple(memory.shape),
            write_frequency=float((strengths > 0.05).float().mean().detach().cpu()),
            mean_write_strength=float(strengths.mean().detach().cpu()),
            mean_surprise=float(torch.stack(surprises).mean().detach().cpu()),
        )
        return ModelOutput(
            logits=answer_logits[-1],
            answer_logits=torch.stack(answer_logits, dim=1),
            aux_logits=torch.stack(aux_logits, dim=1),
            memory_stats=stats,
        )

    def _causal_surprise(
        self,
        previous_logits: list[torch.Tensor],
        current_token_ids: torch.Tensor,
    ) -> torch.Tensor:
        if not previous_logits:
            return torch.ones(
                (current_token_ids.shape[0], 1),
                device=current_token_ids.device,
            )
        with torch.no_grad():
            probabilities = F.softmax(previous_logits[-1], dim=-1)
            return 1.0 - probabilities.gather(1, current_token_ids.unsqueeze(1))

    def _local_attention(self, current: torch.Tensor, history: list[torch.Tensor]) -> torch.Tensor:
        window_tokens = history[-self.window_size :] + [current]
        window = torch.stack(window_tokens, dim=1)
        q = self.q_proj(current).unsqueeze(1)
        k = self.k_proj(window)
        v = self.v_proj(window)
        scores = torch.matmul(q, k.transpose(1, 2)) / math.sqrt(self.d_model)
        weights = F.softmax(scores, dim=-1)
        context = torch.matmul(weights, v).squeeze(1)
        return self.local_out(context)

    def _read_memory(self, hidden: torch.Tensor, memory: torch.Tensor) -> torch.Tensor:
        query = self.memory_query(hidden)
        scores = torch.einsum("bd,bmd->bm", query, memory) / math.sqrt(self.d_model)
        weights = F.softmax(scores, dim=-1)
        return torch.einsum("bm,bmd->bd", weights, memory)

    def _write_memory(
        self,
        hidden: torch.Tensor,
        read: torch.Tensor,
        memory: torch.Tensor,
        surprise: torch.Tensor,
        token_ids: torch.Tensor,
        step: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if self.variant == "no_surprise":
            gate_surprise = torch.ones_like(surprise)
            gate_input_surprise = torch.zeros_like(surprise)
        else:
            gate_surprise = surprise.clamp(0.05, 1.0)
            gate_input_surprise = surprise

        base_gate = torch.sigmoid(self.update_gate(torch.cat([hidden, read, gate_input_surprise], dim=-1)))
        gate = base_gate * gate_surprise
        write_vector = torch.tanh(self.write_proj(hidden))

        if self.variant == "no_forget":
            erase = torch.zeros_like(write_vector)
        else:
            erase = torch.sigmoid(self.erase_proj(hidden))

        slot_weights = self._slot_weights(hidden, memory, token_ids, step)
        write = slot_weights.unsqueeze(-1) * gate.unsqueeze(1) * write_vector.unsqueeze(1)
        erase_amount = slot_weights.unsqueeze(-1) * gate.unsqueeze(1) * erase.unsqueeze(1)
        updated = memory * (1.0 - erase_amount) + write
        return torch.tanh(updated), gate.mean()

    def _slot_weights(
        self,
        hidden: torch.Tensor,
        memory: torch.Tensor,
        token_ids: torch.Tensor,
        step: int,
    ) -> torch.Tensor:
        if self.variant == "fifo_memory":
            slot = step % self.memory_slots
            return F.one_hot(
                torch.full_like(token_ids, fill_value=slot),
                num_classes=self.memory_slots,
            ).float()
        if self.variant == "random_write":
            slots = (token_ids + step * 17) % self.memory_slots
            return F.one_hot(slots, num_classes=self.memory_slots).float()

        query = self.memory_write_query(hidden)
        scores = torch.einsum("bd,bmd->bm", query, memory) / math.sqrt(self.d_model)
        return F.softmax(scores, dim=-1)


class TinyTransformer(nn.Module):
    """A compact Transformer baseline with the same output contract."""

    def __init__(
        self,
        vocab_size: int,
        d_model: int = 64,
        n_layers: int = 2,
        n_heads: int = 4,
        max_seq_len: int = 512,
        window_size: int | None = None,
    ) -> None:
        super().__init__()
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.max_seq_len = max_seq_len
        self.window_size = window_size
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        self.position_embedding = nn.Embedding(max_seq_len, d_model)
        layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_model * 4,
            batch_first=True,
            activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=n_layers)
        self.token_head = nn.Linear(d_model, vocab_size)
        self.answer_head = nn.Linear(d_model, vocab_size)

    def forward(self, input_ids: torch.Tensor) -> ModelOutput:
        batch_size, seq_len = input_ids.shape
        if seq_len > self.max_seq_len:
            raise ValueError(f"seq_len={seq_len} exceeds max_seq_len={self.max_seq_len}")
        positions = torch.arange(seq_len, device=input_ids.device).unsqueeze(0)
        hidden = self.token_embedding(input_ids) + self.position_embedding(positions)
        mask = self._causal_mask(seq_len, input_ids.device)
        encoded = self.encoder(hidden, mask=mask)
        token_logits = self.token_head(encoded)
        answer_logits = self.answer_head(encoded)
        stats = MemoryStats(
            final_memory_shape=(batch_size, 0, self.d_model),
            write_frequency=0.0,
            mean_write_strength=0.0,
            mean_surprise=0.0,
        )
        return ModelOutput(
            logits=answer_logits[:, -1, :],
            answer_logits=answer_logits,
            aux_logits=token_logits,
            memory_stats=stats,
        )

    def _causal_mask(self, seq_len: int, device: torch.device) -> torch.Tensor:
        mask = torch.triu(
            torch.ones(seq_len, seq_len, dtype=torch.bool, device=device),
            diagonal=1,
        )
        if self.window_size is not None:
            far_past = torch.ones(
                seq_len,
                seq_len,
                dtype=torch.bool,
                device=device,
            ).tril(-self.window_size - 1)
            mask = mask | far_past
        return mask


class GRUBaseline(nn.Module):
    """A recurrent baseline with separate token and answer heads."""

    def __init__(self, vocab_size: int, d_model: int = 64) -> None:
        super().__init__()
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        self.recurrent = nn.GRU(d_model, d_model, batch_first=True)
        self.token_head = nn.Linear(d_model, vocab_size)
        self.answer_head = nn.Linear(d_model, vocab_size)

    def forward(self, input_ids: torch.Tensor) -> ModelOutput:
        hidden, _ = self.recurrent(self.token_embedding(input_ids))
        token_logits = self.token_head(hidden)
        answer_logits = self.answer_head(hidden)
        stats = MemoryStats(
            final_memory_shape=(input_ids.shape[0], 1, self.d_model),
            write_frequency=0.0,
            mean_write_strength=0.0,
            mean_surprise=0.0,
        )
        return ModelOutput(
            logits=answer_logits[:, -1, :],
            answer_logits=answer_logits,
            aux_logits=token_logits,
            memory_stats=stats,
        )

def build_model(
    model: str,
    vocab_size: int,
    d_model: int = 64,
    memory_slots: int = 16,
    window_size: int = 8,
    max_seq_len: int = 512,
    n_heads: int = 4,
) -> nn.Module:
    normalized = model.lower().replace("-", "_")
    if normalized in {"forgetnet", "no_forget", "no_surprise", "random_write", "fifo_memory"}:
        return ForgetNet(
            vocab_size=vocab_size,
            d_model=d_model,
            memory_slots=memory_slots,
            window_size=window_size,
            max_seq_len=max_seq_len,
            variant=normalized,
        )
    if normalized in ConflictLocalizedMemoryCell.VARIANTS:
        return ConflictLocalizedMemory(
            vocab_size=vocab_size,
            d_model=d_model,
            memory_slots=memory_slots,
            window_size=window_size,
            max_seq_len=max_seq_len,
            variant=normalized,
        )
    if normalized in {"tiny_transformer", "transformer", "global_transformer"}:
        return TinyTransformer(
            vocab_size=vocab_size,
            d_model=d_model,
            n_heads=n_heads,
            max_seq_len=max_seq_len,
        )
    if normalized == "local_transformer":
        return TinyTransformer(
            vocab_size=vocab_size,
            d_model=d_model,
            n_heads=n_heads,
            max_seq_len=max_seq_len,
            window_size=window_size,
        )
    if normalized == "gru":
        return GRUBaseline(vocab_size=vocab_size, d_model=d_model)
    raise ValueError(f"unknown model: {model}")


def count_parameters(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
