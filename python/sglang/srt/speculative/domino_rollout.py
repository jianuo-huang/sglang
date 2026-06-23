"""Correctness-first Domino draft rollout for DFLASH speculative decoding."""

from __future__ import annotations

import torch
import torch.nn.functional as F


class DFlashDominoRollout:
    """Generate a Domino draft block with the checkpoint's eager reference math.

    The first integration intentionally uses dense TP=1 operations. In
    particular, it preserves the checkpoint's BF16 linear/add ordering instead
    of changing the argmax with fused FP32 accumulation. Kernel fusion, CUDA
    graphs, and tensor parallelism are follow-up performance work.
    """

    def __init__(self, *, draft_model, block_size: int) -> None:
        self.draft_model = draft_model
        self.block_size = int(block_size)
        # Weight loading/casting may replace GRU parameter storages after the
        # model-level load hook ran. Compact them once at final worker setup.
        self.draft_model.prefix_gru.flatten_parameters()

    @torch.inference_mode()
    def rollout_draft_block(
        self,
        *,
        draft_hidden: torch.Tensor,
        verified_id: torch.Tensor,
        target_model,
        lm_head,
    ) -> torch.Tensor:
        """Return ``block_size - 1`` greedy Domino draft tokens.

        SGLang reserves draft slot 0 for the current verified token. For the
        public ``shift_label=True`` checkpoint, hidden slot 0 predicts draft
        slot 1. The first ``pure_draft_prefix_len`` token uses only target
        lm-head logits; later tokens add the sequential GRU-conditioned Domino
        bias exactly as in the checkpoint's reference ``generate`` method.
        """
        bs, total_slots, hidden_size = draft_hidden.shape
        if total_slots != self.block_size:
            raise RuntimeError(
                f"DFLASH Domino expected draft_hidden block dim={self.block_size}, "
                f"got {total_slots}."
            )

        num_draft = self.block_size - 1
        if num_draft <= 0:
            raise RuntimeError(
                f"DFLASH Domino requires block_size > 1, got {self.block_size}."
            )

        draft_model = self.draft_model
        prefix_len = int(draft_model.pure_draft_prefix_len)
        if prefix_len != 1:
            raise NotImplementedError(
                "DFLASH Domino eager rollout currently requires "
                f"pure_draft_prefix_len=1, got {prefix_len}."
            )

        if draft_model.shift_label:
            proposal_hidden = draft_hidden[:, :num_draft, :]
        else:
            proposal_hidden = draft_hidden[:, 1:, :]

        weight = lm_head.weight
        if proposal_hidden.dtype != weight.dtype:
            proposal_hidden = proposal_hidden.to(weight.dtype)

        if hasattr(lm_head, "shard_indices"):
            shard = lm_head.shard_indices
            num_added = int(shard.num_added_elements)
            org_vocab_start = int(shard.org_vocab_start_index)
            target_vocab_size = int(shard.num_org_elements)
            if num_added != 0:
                raise NotImplementedError(
                    "DFLASH Domino rollout does not support added-vocabulary "
                    "lm-head shards yet."
                )
            if org_vocab_start != 0:
                raise RuntimeError(
                    "DFLASH Domino TP=1 expected target org_vocab_start_index=0, "
                    f"got {org_vocab_start}."
                )
        else:
            target_vocab_size = int(weight.shape[0])

        domino_vocab_size = int(draft_model.embed_proj[2].weight.shape[0])
        if target_vocab_size != domino_vocab_size:
            raise ValueError(
                "DFLASH Domino target/draft vocabulary mismatch: "
                f"target_vocab_size={target_vocab_size}, "
                f"domino_vocab_size={domino_vocab_size}."
            )

        base_logits = F.linear(
            proposal_hidden,
            weight[:target_vocab_size],
        )
        first_draft = torch.argmax(base_logits[:, 0, :], dim=-1)
        output_tokens = [first_draft]

        embed_module = target_model.get_input_embeddings()
        prefix_ids = torch.stack([verified_id.to(torch.long), first_draft], dim=1)
        _, gru_hidden = draft_model.prefix_gru(embed_module(prefix_ids))

        for step_idx in range(prefix_len, num_draft):
            state = gru_hidden.transpose(0, 1)
            bias_logits = draft_model.embed_proj(
                torch.cat(
                    [proposal_hidden[:, step_idx : step_idx + 1, :], state],
                    dim=-1,
                )
            )
            step_logits = base_logits[:, step_idx : step_idx + 1, :] + bias_logits
            next_token = torch.argmax(step_logits, dim=-1).squeeze(1)
            output_tokens.append(next_token)

            if step_idx + 1 < num_draft:
                _, gru_hidden = draft_model.prefix_gru(
                    embed_module(next_token).unsqueeze(1),
                    gru_hidden,
                )

        return torch.stack(output_tokens, dim=1)
