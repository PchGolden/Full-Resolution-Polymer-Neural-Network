"""Shared monomer encoding without adding state-dict prefixes or parameters."""

from typing import NamedTuple

import torch

from .features import _build_attn_mask, build_padding_only_attn_mask


class MonomerEncoding(NamedTuple):
    nodes: torch.Tensor
    pairs: torch.Tensor
    attention_mask: torch.Tensor
    atom_start: int


class MonomerEncodingMixin:
    """Use the modules owned by each model to encode its monomer representations.

    Constructors stay in the model classes so initialization order, parameter
    ownership and existing checkpoint keys are preserved.
    """

    def _encode_monomers(self, batch, *, pair_dim: int, wo_pair: bool, wo_geom_3d: bool) -> MonomerEncoding:
        atom_mask = batch["atom_mask"]
        segment_ids = batch["segment_id"]
        positions = batch["src_pos"]
        batch_size, atom_count = atom_mask.shape
        atom_start = 2 + batch["seg_feat"].size(1)  # CLS, GLOB, then segment tokens.
        token_count = atom_start + atom_count

        token_features = self.embed_tokens(batch["src_token"])
        nodes = self.atom_feature(batch, token_features)
        if nodes.dtype == torch.float32 and torch.is_autocast_enabled():
            nodes = nodes.to(torch.get_autocast_gpu_dtype())

        if wo_pair:
            attention_mask = build_padding_only_attn_mask(
                atom_mask, batch["seg_valid_mask"], glob_valid=batch["glob_valid_mask"],
                num_heads=self.args.encoder_attention_heads,
            )
        else:
            attention_mask = _build_attn_mask(
                batch["base_mask"], segment_ids, atom_mask, batch["seg_valid_mask"],
                glob_valid=batch["glob_valid_mask"], num_heads=self.args.encoder_attention_heads,
            )

        pairs = nodes.new_zeros(batch_size, token_count, token_count, pair_dim, dtype=nodes.dtype)
        pairs = self.edge_feature(batch, pairs)
        if not wo_geom_3d:
            displacement = positions.unsqueeze(2) - positions.unsqueeze(1)
            distance = displacement.norm(dim=-1)
            bias = self.se3_invariant_kernel(distance.detach(), batch["pair_type"].long())
            same_segment = (segment_ids.unsqueeze(-1) == segment_ids.unsqueeze(-2)).unsqueeze(-1)
            bias.masked_fill_(~same_segment, 0.0)
            pairs[:, atom_start:, atom_start:, :].add_(bias)

        cls_mask = atom_mask.new_ones(batch_size, 1, dtype=torch.bool)
        node_mask = torch.cat([
            cls_mask, batch["glob_valid_mask"].bool(), batch["seg_valid_mask"].bool(), atom_mask.bool(),
        ], dim=1)
        pair_mask = node_mask.unsqueeze(-1) & node_mask.unsqueeze(-2)
        nodes, pairs = self.encoder(
            nodes, pairs, atom_mask=node_mask, pair_mask=pair_mask, attn_mask=attention_mask,
        )
        return MonomerEncoding(nodes, pairs, attention_mask, atom_start)

    def _reconstruct_atoms(self, batch, encoded: MonomerEncoding):
        """Predict masked atoms, denoised positions and their pairwise distances."""
        start = encoded.atom_start
        atom_repr = encoded.nodes[:, start:, :]
        logits = self.lm_head(atom_repr)
        positions = batch["src_pos"]
        displacement = positions.unsqueeze(2) - positions.unsqueeze(1)
        delta = self.movement_pred_head(
            atom_repr, encoded.pairs[:, start:, start:, :],
            encoded.attention_mask[:, :, start:, start:], displacement.detach(),
        )
        predicted_positions = positions + delta
        predicted_distances = torch.cdist(predicted_positions, predicted_positions)
        return logits, predicted_positions, predicted_distances
