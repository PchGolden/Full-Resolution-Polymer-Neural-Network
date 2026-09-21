"""Regression checks for shared training services and MD boundary cases."""

from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch

from frpn.md.models.features import ChainTokenFeaturePlus
from frpn.md.models.model import MultiMolModel
from frpn.md.train import freeze_pretrain_heads
from frpn.training.checkpoints import load_monomer_weights
from frpn.training.pretraining import pretraining_losses, run_pretraining
from frpn.training.targets import fit_label_normalization
from tests.test_chain_rescaling import small_args


class LabelNormalizationTests(unittest.TestCase):
    def test_training_ids_exclude_validation_outliers(self):
        class Dataset:
            ids = [0, 1]

            def __len__(self):
                return 2

            def _get_raw_sample(self, index):
                return {"label": {"density": [2., 2., 9999.][index], "D": [1e-5, 1e-3, 100.][index]}}

        stats = fit_label_normalization(Dataset(), ["density", "D"], ["D"], 1e-16, "d_clamp_min")
        self.assertEqual(stats["mean"], [2., -4.])
        self.assertEqual(stats["std"], [1., 1.])
        self.assertEqual(stats["log10_labels"], ["D"])


class ModelBoundaryTests(unittest.TestCase):
    def test_missing_global_feature_is_masked_per_sample(self):
        layer = ChainTokenFeaturePlus(4, 1, 1)
        _, mask, _ = layer(
            seg_repr=torch.ones(2, 2, 4), seg_dop=torch.ones(2, 2).long(),
            seg_block_id=torch.zeros(2, 2).long(),
            glob_feat=torch.tensor([[300.], [0.]]), glob_feat_mask=torch.tensor([[1.], [0.]]),
            block_feat=torch.zeros(2, 0, 1), block_feat_mask=torch.zeros(2, 0, 1),
            chain_node_seg_id=[torch.tensor([0, 1]), torch.tensor([0, 1])], num_heads=2,
        )
        self.assertTrue(torch.isfinite(mask[0, :, 1, :]).all())
        self.assertTrue(torch.isneginf(mask[1, :, 1, [0, 2, 3]]).all())
        self.assertTrue(torch.isneginf(mask[1, :, [0, 2, 3], 1]).all())
        self.assertTrue((mask[1, :, 1, 1] == 0).all())

    def test_freezing_pretraining_heads_keeps_tied_embedding_trainable(self):
        model = MultiMolModel(small_args("finetune"))
        self.assertIs(model.lm_head.weight, model.embed_tokens.weight)
        freeze_pretrain_heads(model)
        self.assertTrue(model.embed_tokens.weight.requires_grad)
        self.assertFalse(model.lm_head.dense.weight.requires_grad)
        self.assertFalse(model.movement_pred_head.force_proj1.weight.requires_grad)

    def test_monomer_initialization_preserves_prediction_and_chain_weights(self):
        model = MultiMolModel(small_args("chain"))
        original = {name: tensor.clone() for name, tensor in model.state_dict().items()}
        source = {name: torch.full_like(tensor, .25) for name, tensor in original.items()}
        with TemporaryDirectory() as directory:
            checkpoint = Path(directory) / "weights.pt"
            torch.save({"model_state": source}, checkpoint)
            load_monomer_weights(model, checkpoint, SimpleNamespace(rank=1), include_pretrain_heads=False)
        self.assertTrue(torch.equal(model.embed_tokens.weight, source["embed_tokens.weight"]))
        for name, tensor in model.state_dict().items():
            if name.startswith(("chain_", "reg_head.", "movement_pred_head.")):
                self.assertTrue(torch.equal(tensor, original[name]), name)


class PretrainingTests(unittest.TestCase):
    def test_padded_targets_do_not_change_losses_or_gradients(self):
        torch.manual_seed(42)
        logits = torch.randn(2, 3, 5, requires_grad=True)
        positions = torch.randn(2, 3, 3, requires_grad=True)
        batch = {"target_token": torch.tensor([[1, 0, 2], [3, 0, 0]]),
                 "atom_mask": torch.tensor([[1, 1, 1], [1, 0, 0]]),
                 "target_pos": torch.randn(2, 3, 3)}
        first = pretraining_losses(logits, positions, torch.cdist(positions, positions), batch)
        batch["target_pos"][1, 1:] = 1e4
        second = pretraining_losses(logits, positions, torch.cdist(positions, positions), batch)
        for a, b in zip(first, second):
            torch.testing.assert_close(a, b, rtol=0, atol=0)
        sum(second).backward()
        self.assertEqual(positions.grad[1, 1:].abs().sum().item(), 0.)
        self.assertEqual(logits.grad[batch["target_token"] == 0].abs().sum().item(), 0.)

    def test_single_process_pretraining_saves_without_broadcast(self):
        class Model(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.tokens = torch.nn.Linear(3, 5)
                self.offset = torch.nn.Parameter(torch.tensor([.2, -.1, .3]))

            def forward(self, batch):
                positions = batch["src_pos"] + self.offset
                return self.tokens(batch["src_pos"]), positions, torch.cdist(positions, positions)

        class Loader(list):
            @property
            def dataset(self):
                return self

        torch.manual_seed(73)
        model = Model()
        sample = {"src_pos": torch.randn(1, 2, 3), "target_pos": torch.zeros(1, 2, 3),
                  "target_token": torch.tensor([[1, 2]]), "atom_mask": torch.ones(1, 2).long()}
        loader = Loader([sample, sample])
        args = SimpleNamespace(batch_size=1, world_size=1, grad_accum_steps=1, lr=.01, weight_decay=0.,
                               epochs=1, warmup_steps=1, warmup_ratio=.06, amp=False, local_rank=0,
                               distributed=False, rank=0, val_every_steps=1, min_delta_abs=.001,
                               min_delta_rel=.01, grad_clip=1., early_stop_patience=10)
        with TemporaryDirectory() as directory, patch("torch.distributed.broadcast") as broadcast:
            run_pretraining(model, args, loader, None, loader, Path(directory))
            broadcast.assert_not_called()
            checkpoint = torch.load(Path(directory) / "checkpoints/checkpoint_1.pt", weights_only=False)
        self.assertEqual(checkpoint["global_step"], 1)
        self.assertIn("scheduler_state", checkpoint)
        self.assertEqual(checkpoint["args"], vars(args))


if __name__ == "__main__":
    unittest.main()
