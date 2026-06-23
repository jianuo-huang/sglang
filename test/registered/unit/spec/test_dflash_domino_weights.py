"""Unit tests for strict Domino projector weight loading."""

import unittest

import torch
from torch import nn

from sglang.srt.models.dflash import DFlashDraftModel
from sglang.test.ci.ci_register import register_cpu_ci


register_cpu_ci(est_time=5, suite="base-a-test-cpu")


class _TinyDominoModel(DFlashDraftModel):
    def __init__(self):
        nn.Module.__init__(self)
        self.prefix_gru = nn.GRU(
            input_size=4,
            hidden_size=3,
            batch_first=True,
            bias=False,
        )
        self.embed_proj = nn.Sequential(
            nn.Linear(7, 2, bias=False),
            nn.SiLU(),
            nn.Linear(2, 5, bias=False),
        )


class TestDFlashDominoWeights(unittest.TestCase):
    def test_loads_all_projector_weights(self):
        model = _TinyDominoModel()
        loaded = {
            name: torch.randn_like(param) for name, param in model.named_parameters()
        }

        DFlashDraftModel.load_weights(model, loaded.items())

        for name, param in model.named_parameters():
            torch.testing.assert_close(param, loaded[name])

    def test_missing_projector_weight_fails(self):
        model = _TinyDominoModel()
        loaded = {
            name: torch.randn_like(param) for name, param in model.named_parameters()
        }
        loaded.pop("embed_proj.2.weight")

        with self.assertRaisesRegex(
            ValueError, "missing required projector weights.*embed_proj.2.weight"
        ):
            DFlashDraftModel.load_weights(model, loaded.items())

    def test_partial_model_load_after_initial_load(self):
        model = _TinyDominoModel()
        initial = {
            name: torch.randn_like(param) for name, param in model.named_parameters()
        }
        DFlashDraftModel.load_weights(model, initial.items())

        updated_weight = torch.randn_like(model.embed_proj[2].weight)
        DFlashDraftModel.load_weights(
            model,
            [("embed_proj.2.weight", updated_weight)],
        )

        torch.testing.assert_close(model.embed_proj[2].weight, updated_weight)

    def test_partial_update_after_bypass_loader_initialization(self):
        model = _TinyDominoModel()
        initial = {
            name: torch.randn_like(param) for name, param in model.named_parameters()
        }
        with torch.no_grad():
            for name, param in model.named_parameters():
                param.copy_(initial[name])
        model.post_load_weights()

        updated_weight = torch.randn_like(model.embed_proj[2].weight)
        model.load_weights([("embed_proj.2.weight", updated_weight)])

        torch.testing.assert_close(model.embed_proj[2].weight, updated_weight)


if __name__ == "__main__":
    unittest.main()
