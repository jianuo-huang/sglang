"""Reference-parity tests for the correctness-first Domino rollout."""

from types import SimpleNamespace
import unittest

import torch
import torch.nn.functional as F
from torch import nn

from sglang.srt.speculative.domino_rollout import DFlashDominoRollout
from sglang.test.ci.ci_register import register_cuda_ci


register_cuda_ci(est_time=20, stage="base-b", runner_config="1-gpu-small")


class _TargetModel(nn.Module):
    def __init__(self, *, vocab_size: int, hidden_size: int, dtype: torch.dtype):
        super().__init__()
        self.embed_tokens = nn.Embedding(vocab_size, hidden_size, dtype=dtype)

    def get_input_embeddings(self):
        return self.embed_tokens


class _LmHead(nn.Module):
    def __init__(self, *, vocab_size: int, hidden_size: int, dtype: torch.dtype):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(vocab_size, hidden_size, dtype=dtype))
        self.shard_indices = SimpleNamespace(
            num_added_elements=0,
            num_org_elements=vocab_size,
            org_vocab_start_index=0,
        )


class _DraftModel(nn.Module):
    def __init__(
        self,
        *,
        vocab_size: int,
        hidden_size: int,
        gru_hidden_size: int,
        emb_dim: int,
        shift_label: bool,
        dtype: torch.dtype,
    ):
        super().__init__()
        self.pure_draft_prefix_len = 1
        self.shift_label = shift_label
        self.prefix_gru = nn.GRU(
            hidden_size,
            gru_hidden_size,
            batch_first=True,
            bias=False,
            dtype=dtype,
        )
        self.embed_proj = nn.Sequential(
            nn.Linear(hidden_size + gru_hidden_size, emb_dim, bias=False, dtype=dtype),
            nn.SiLU(),
            nn.Linear(emb_dim, vocab_size, bias=False, dtype=dtype),
        )


def _reference_rollout(
    *, draft_model, draft_hidden, verified_id, target_model, lm_head, block_size
):
    num_draft = block_size - 1
    hidden = (
        draft_hidden[:, :num_draft]
        if draft_model.shift_label
        else draft_hidden[:, 1:]
    )
    base_logits = F.linear(hidden, lm_head.weight)
    first_draft = torch.argmax(base_logits[:, :1], dim=-1)
    output = [first_draft]

    prefix_ids = torch.cat([verified_id[:, None], first_draft], dim=1)
    _, gru_hidden = draft_model.prefix_gru(target_model.embed_tokens(prefix_ids))
    for step_idx in range(1, num_draft):
        state = gru_hidden.transpose(0, 1)
        bias = draft_model.embed_proj(
            torch.cat([hidden[:, step_idx : step_idx + 1], state], dim=-1)
        )
        next_token = torch.argmax(
            base_logits[:, step_idx : step_idx + 1] + bias,
            dim=-1,
        )
        output.append(next_token)
        if step_idx + 1 < num_draft:
            _, gru_hidden = draft_model.prefix_gru(
                target_model.embed_tokens(next_token),
                gru_hidden,
            )
    return torch.cat(output, dim=1)


@unittest.skipUnless(torch.cuda.is_available(), "Domino rollout requires CUDA")
class TestDFlashDominoRollout(unittest.TestCase):
    def test_matches_checkpoint_reference(self):
        device = torch.device("cuda")
        dtype = torch.bfloat16
        vocab_size = 257
        hidden_size = 32
        block_size = 6

        for shift_label in (True, False):
            with self.subTest(shift_label=shift_label):
                torch.manual_seed(1234)
                target_model = _TargetModel(
                    vocab_size=vocab_size,
                    hidden_size=hidden_size,
                    dtype=dtype,
                ).to(device)
                lm_head = _LmHead(
                    vocab_size=vocab_size,
                    hidden_size=hidden_size,
                    dtype=dtype,
                ).to(device)
                draft_model = _DraftModel(
                    vocab_size=vocab_size,
                    hidden_size=hidden_size,
                    gru_hidden_size=16,
                    emb_dim=8,
                    shift_label=shift_label,
                    dtype=dtype,
                ).to(device)
                for param in list(target_model.parameters()) + list(
                    lm_head.parameters()
                ) + list(draft_model.parameters()):
                    nn.init.normal_(param, std=0.1)
                draft_model.prefix_gru.flatten_parameters()

                draft_hidden = torch.randn(
                    3, block_size, hidden_size, device=device, dtype=dtype
                )
                verified_id = torch.tensor([3, 17, 29], device=device)
                rollout = DFlashDominoRollout(
                    draft_model=draft_model,
                    block_size=block_size,
                )

                expected = _reference_rollout(
                    draft_model=draft_model,
                    draft_hidden=draft_hidden,
                    verified_id=verified_id,
                    target_model=target_model,
                    lm_head=lm_head,
                    block_size=block_size,
                )
                actual = rollout.rollout_draft_block(
                    draft_hidden=draft_hidden,
                    verified_id=verified_id,
                    target_model=target_model,
                    lm_head=lm_head,
                )
                torch.testing.assert_close(actual, expected, rtol=0, atol=0)

    def test_rejects_vocab_mismatch(self):
        device = torch.device("cuda")
        draft_model = _DraftModel(
            vocab_size=127,
            hidden_size=16,
            gru_hidden_size=8,
            emb_dim=4,
            shift_label=True,
            dtype=torch.float32,
        ).to(device)
        target_model = _TargetModel(
            vocab_size=128, hidden_size=16, dtype=torch.float32
        ).to(device)
        lm_head = _LmHead(
            vocab_size=128, hidden_size=16, dtype=torch.float32
        ).to(device)
        rollout = DFlashDominoRollout(draft_model=draft_model, block_size=4)

        with self.assertRaisesRegex(ValueError, "vocabulary mismatch"):
            rollout.rollout_draft_block(
                draft_hidden=torch.randn(1, 4, 16, device=device),
                verified_id=torch.tensor([1], device=device),
                target_model=target_model,
                lm_head=lm_head,
            )


if __name__ == "__main__":
    unittest.main()
