from __future__ import annotations

import torch
import torch.nn.functional as F

from forgetnet.data import make_task_batch
from forgetnet.experiment import next_token_auxiliary_loss
from forgetnet.models import ForgetNet


def test_auxiliary_loss_trains_predictive_surprise_head() -> None:
    batch = make_task_batch("changing_facts", batch_size=3, seq_len=20, seed=51)
    model = ForgetNet(vocab_size=batch.vocab_size, d_model=24, memory_slots=4)

    output = model(batch.input_ids)
    answer_loss = F.cross_entropy(output.logits, batch.labels)
    auxiliary_loss = next_token_auxiliary_loss(output.aux_logits, batch.input_ids)
    (answer_loss + 0.1 * auxiliary_loss).backward()

    assert auxiliary_loss.item() > 0.0
    assert model.token_head.weight.grad is not None
    assert torch.count_nonzero(model.token_head.weight.grad).item() > 0


def test_auxiliary_loss_uses_next_token_targets() -> None:
    input_ids = torch.tensor([[4, 5, 6]])
    logits = torch.full((1, 3, 8), -10.0)
    logits[0, 0, 5] = 10.0
    logits[0, 1, 6] = 10.0

    loss = next_token_auxiliary_loss(logits, input_ids)

    assert loss.item() < 1e-6
