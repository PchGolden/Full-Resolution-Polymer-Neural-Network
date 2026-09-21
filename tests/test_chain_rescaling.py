"""CPU regression checks for proportional chain rescaling; no training or data IO.

Run from the GitHub package root:
    python -m unittest discover -s tests -p 'test_chain_rescaling.py' -v
"""
from types import SimpleNamespace
import itertools
import unittest

import torch

torch.set_num_threads(1)
from frpn.linear.models.chain_rescaling import rescale_repeat_counts
from frpn.linear.models.features import ChainTokenFeaturePlus, ChainEdgeFeaturePlus
from frpn.linear.models.model import MultiMolModel


def token_inputs(counts, blocks=None, dim=4):
    counts = torch.tensor(counts, dtype=torch.long)
    b, s = counts.shape
    if blocks is None:
        blocks = [[0] * (s - 1) + [1]] * b
    representation = torch.eye(max(s, dim))[:s, :dim].repeat(b, 1, 1).requires_grad_()
    return dict(seg_repr=representation, seg_dop=counts,
                seg_block_id=torch.tensor(blocks),
                glob_feat=torch.zeros(b, 1), glob_feat_mask=torch.zeros(b, 1),
                block_feat=torch.zeros(b, 2, 1), block_feat_mask=torch.zeros(b, 2, 1),
                num_heads=2)


def small_args(task='stage2_only', anchor=True):
    return SimpleNamespace(main_task=task, encoder_embed_dim=12, encoder_layers=1,
            pair_embed_dim=4, pair_hidden_dim=4, encoder_ffn_embed_dim=24,
            encoder_attention_heads=3, chain_pair_dim=4, chain_pair_hidden_dim=4,
            chain_encoder_layers=1, chain_attention_heads=3, chain_ffn_embed_dim=24,
            num_chain_glob_feat=1, num_chain_block_feat=1, max_chain_dist=32,
            max_chain_tokens=12, max_segments=4, num_seg_smiles_types=8,
            stage2_only_repr='smiles_only', num_tasks=1, fp_nbits=8,
            dropout=0., attention_dropout=0., activation_dropout=0., pair_dropout=0.,
            wo_pair=False, wo_geom_3d=True, wo_triopm=True,
            disable_anchor_symmetry_break=not anchor)


def model_batch(counts):
    inputs = token_inputs(counts)
    b, s = inputs['seg_dop'].shape
    n = s
    return dict(seg_dop=inputs['seg_dop'], seg_block_id=inputs['seg_block_id'],
            seg_valid_mask=(inputs['seg_dop'] > 0).long(),
            seg_smiles_id=torch.arange(1, s + 1).repeat(b, 1),
            chain_glob_feat=torch.zeros(b, 1), chain_glob_mask=torch.zeros(b, 1),
            block_feat=inputs['block_feat'], block_feat_mask=inputs['block_feat_mask'],
            chain_len=inputs['seg_dop'].sum(-1), atom_mask=torch.ones(b, n).long(),
            segment_id=torch.arange(s).repeat(b, 1), src_pos=torch.zeros(b, n, 3),
            pair_type=torch.zeros(b, n, n).long(), src_token=torch.ones(b, n).long(),
            seg_feat=torch.zeros(b, s, 2), seg_feat_mask=torch.ones(b, s, 2),
            base_mask=torch.zeros(b, 2+s+n, 2+s+n), glob_valid_mask=torch.zeros(b, 1).long(),
            atom_feat=torch.ones(b, n, 2).long(), degree=torch.ones(b, n).long(),
            glob_feat=torch.zeros(b, 1), glob_mask=torch.zeros(b, 1),
            shortest_path=torch.zeros(b, n, n).long(), edge_feat=torch.ones(b, n, n, 1).long())


class AllocationTests(unittest.TestCase):
    def test_no_cap_and_boundary(self):
        for cap in (None, -1, 0, 1536, 2000):
            self.assertEqual(rescale_repeat_counts([1024, 512, 0], cap), [1024, 512, 0])
        self.assertEqual(rescale_repeat_counts([], 2), [])
        self.assertEqual(rescale_repeat_counts([0, 0], 2), [0, 0])

    def test_proportions_and_ties(self):
        self.assertEqual(rescale_repeat_counts([2400, 1200]), [1024, 512])
        self.assertEqual(rescale_repeat_counts([2000, 1000, 1000]), [768, 384, 384])
        self.assertEqual(rescale_repeat_counts([5, 5, 5], 4), [2, 1, 1])
        self.assertEqual(rescale_repeat_counts([10**16, 10**16], 1536), [768, 768])

    def test_rare_segments_and_impossible_budget(self):
        self.assertEqual(rescale_repeat_counts([1000000, 1, 0, 1], 1536), [1534, 1, 0, 1])
        self.assertEqual(rescale_repeat_counts([1000000, 1, 1], 3), [1, 1, 1])
        with self.assertRaisesRegex(ValueError, 'cannot retain all'):
            rescale_repeat_counts([5, 5, 5], 2)
        with self.assertRaises(ValueError):
            rescale_repeat_counts([2, -1], 1)

    def test_constrained_rounding_is_optimal(self):
        # Exhaustively compare every feasible integer composition for small cases.
        for values in itertools.product(range(1, 5), repeat=3):
            for cap in range(3, sum(values)):
                result = rescale_repeat_counts(values, cap)
                score = sum((a * sum(values) - cap*n)**2 for a, n in zip(result, values))
                optimum = min(sum((a * sum(values) - cap*n)**2 for a, n in zip(candidate, values))
                    for candidate in itertools.product(range(1, cap-1), repeat=3) if sum(candidate) == cap)
                self.assertEqual(score, optimum)


class ForwardTests(unittest.TestCase):
    def test_default_cap_preserves_both_blocks_before_expansion(self):
        layer = ChainTokenFeaturePlus(4, 1, 1, enable_anchor_symmetry_break=False)
        inp = token_inputs([[2400, 1200]])
        tokens, mask, lengths = layer(**inp, return_chain_lengths=True)
        self.assertEqual(lengths.tolist(), [1536])
        self.assertEqual(tokens.shape, (1, 1540, 4))
        self.assertEqual(mask.shape, (1, 2, 1540, 1540))
        self.assertEqual((tokens[0, 4:, 0] == 1).sum().item(), 1024)
        self.assertEqual((tokens[0, 4:, 1] == 1).sum().item(), 512)
        self.assertTrue(torch.isfinite(mask[0, 0, 2, 4:1028]).all())
        self.assertTrue(torch.isneginf(mask[0, 0, 2, 1028:]).all())
        self.assertTrue(torch.isfinite(mask[0, 0, 3, 1028:]).all())
        tokens[:, 4:].sum().backward()
        self.assertEqual(inp['seg_repr'].grad[0, :, 0].tolist(), [1024., 512.])

    def test_mixed_lengths_pair_padding_and_anchor_gradients(self):
        inp = token_inputs([[2000, 1000, 1000], [1, 1, 0], [0, 0, 0]])
        layer = ChainTokenFeaturePlus(4, 1, 1, max_chain_tokens=12)
        tokens, mask, lengths = layer(**inp, return_chain_lengths=True)
        self.assertEqual(lengths.tolist(), [12, 2, 0])
        self.assertTrue(torch.isfinite(tokens).all())
        self.assertEqual(tokens.shape, (3, 16, 4))
        pair = ChainEdgeFeaturePlus(2, 16)
        with torch.no_grad():
            pair.chain_dist_encoder.weight.copy_(torch.arange(17).view(17, 1).repeat(1, 2))
        # Deliberately wrong stale lengths test use of rescaled lengths.
        bias = pair({'chain_len': torch.tensor([4000, 4000, 4000])}, tokens.new_zeros(3, 16, 16, 2), lengths)
        self.assertEqual(bias[0, 4, 15, 0].item(), 11.)
        self.assertEqual(bias[1, 4, 5, 0].item(), 1.)
        self.assertEqual(bias[1, 4, 6:, :].abs().sum().item(), 0.)
        self.assertTrue(torch.isneginf(mask[1, :, 0, 6:]).all())
        (tokens.sum() + bias.sum()).backward()
        self.assertTrue(torch.isfinite(inp['seg_repr'].grad).all())
        self.assertGreater(layer.topo_mlp[0].weight.grad.abs().sum().item(), 0.)
        self.assertTrue((inp['seg_repr'].grad[0].abs().sum(-1) > 0).all())

    def test_bounded_vs_unbounded_short_chain_same_output(self):
        for anchor in (False, True):
            limited = ChainTokenFeaturePlus(4, 1, 1, max_chain_tokens=12, enable_anchor_symmetry_break=anchor)
            unlimited = ChainTokenFeaturePlus(4, 1, 1, max_chain_tokens=0, enable_anchor_symmetry_break=anchor)
            unlimited.load_state_dict(limited.state_dict(), strict=True)
            inp = token_inputs([[3, 1, 1], [6, 3, 3]])
            torch.manual_seed(71); left = limited(**inp)
            torch.manual_seed(71); right = unlimited(**inp)
            for a, b in zip(left, right):
                self.assertTrue(torch.equal(a, b))

    def test_full_model_chain_and_single_branch_forward_backward(self):
        for task, mode in (('chain', 'smiles_only'), ('stage2_only', 'smiles_only'),
                           ('stage2_only', 'full'), ('stage2_only', 'fp_morgan2048')):
            for anchor in (False, True):
                torch.manual_seed(42)
                args = small_args(task, anchor)
                args.stage2_only_repr = mode
                model = MultiMolModel(args).eval()
                if mode == 'fp_morgan2048':
                    model.fp_table = torch.randn(8, 8)
                # Zero-initialized attention output weights intentionally block
                # upstream gradients at initialization. Use fixed random test
                # weights to exercise the full differentiable prediction path.
                for name, parameter in model.named_parameters():
                    if name.endswith('linear_o.weight'):
                        torch.nn.init.normal_(parameter, std=0.1)
                batch = model_batch([[2400, 1200, 400], [1, 1, 0]])
                outputs = model(batch)
                self.assertEqual(outputs.shape, (2, 1))
                self.assertTrue(torch.isfinite(outputs).all())
                outputs.sum().backward()
                gradient = model.chain_token_feature.chain_cls.weight.grad
                self.assertTrue(torch.isfinite(gradient).all())
                if mode == 'fp_morgan2048':
                    self.assertGreater(model.fp_proj[0].weight.grad.abs().sum().item(), 0.)
                elif task == 'stage2_only':
                    self.assertTrue((model.stage2_only_smiles_embed.weight.grad[1:4].abs().sum(-1) > 0).all())
                else:
                    self.assertGreater(model.embed_tokens.weight.grad.abs().sum().item(), 0.)


if __name__ == '__main__':
    torch.set_num_threads(1)
    unittest.main()
