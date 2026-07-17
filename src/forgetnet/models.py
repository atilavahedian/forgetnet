from __future__ import annotations

from dataclasses import dataclass
import math

import torch
from torch import nn
import torch.nn.functional as F


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
        self.head = nn.Linear(d_model, vocab_size)

    def forward(self, input_ids: torch.Tensor) -> ModelOutput:
        batch_size, seq_len = input_ids.shape
        if seq_len > self.max_seq_len:
            raise ValueError(f"seq_len={seq_len} exceeds max_seq_len={self.max_seq_len}")
        positions = torch.arange(seq_len, device=input_ids.device).unsqueeze(0)
        hidden = self.token_embedding(input_ids) + self.position_embedding(positions)
        mask = self._causal_mask(seq_len, input_ids.device)
        encoded = self.encoder(hidden, mask=mask)
        sequence_logits = self.head(encoded)
        stats = MemoryStats(
            final_memory_shape=(batch_size, 0, self.d_model),
            write_frequency=0.0,
            mean_write_strength=0.0,
            mean_surprise=0.0,
        )
        return ModelOutput(
            logits=sequence_logits[:, -1, :],
            aux_logits=sequence_logits,
            memory_stats=stats,
        )

    def _causal_mask(self, seq_len: int, device: torch.device) -> torch.Tensor:
        mask = torch.triu(torch.ones(seq_len, seq_len, dtype=torch.bool, device=device), diagonal=1)
        if self.window_size is not None:
            far_past = torch.ones(seq_len, seq_len, dtype=torch.bool, device=device).tril(-self.window_size - 1)
            mask = mask | far_past
        return mask


def build_model(
    model: str,
    vocab_size: int,
    d_model: int = 64,
    memory_slots: int = 16,
    window_size: int = 8,
    max_seq_len: int = 512,
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
    if normalized in {"tiny_transformer", "transformer", "global_transformer"}:
        return TinyTransformer(vocab_size=vocab_size, d_model=d_model, max_seq_len=max_seq_len)
    if normalized == "local_transformer":
        return TinyTransformer(
            vocab_size=vocab_size,
            d_model=d_model,
            max_seq_len=max_seq_len,
            window_size=window_size,
        )
    raise ValueError(f"unknown model: {model}")


def count_parameters(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
