"""Shared monomer pair features and additive attention masks."""

import torch
import torch.nn as nn
from torch.nn import Embedding


class EdgeFeaturePlus(nn.Module):

    def __init__(self, pair_dim, num_edge, num_spatial, wo_spd, wo_edge):
        super().__init__()
        self.pair_dim = pair_dim
        self.edge_encoder       = Embedding(num_edge,    pair_dim, padding_idx=0)
        self.shorest_path_encoder  = Embedding(num_spatial, pair_dim, padding_idx=0)
        self.vnode_virtual_distance = Embedding(1,           pair_dim)  # virtual bias t
        self.wo_spd = wo_spd
        self.wo_edge = wo_edge

    # ------------------------------------------------------------------
    def forward(self, batched_data, graph_attn_bias):

        shortest_path = batched_data["shortest_path"]          # [B,N,N]
        edge_input    = batched_data["edge_feat"]              # [B,N,N,K]
        n_seg         = batched_data["seg_feat"].size(1)
        N_atom        = shortest_path.size(-1)
        special_len   = 1 + 1 + n_seg                          # CLS+GLOB+SEG
        B             = graph_attn_bias.size(0)

        # ------ 1. atom - atom bias: shortest path + edge features ------
        if self.wo_spd and self.wo_edge:
            pass
        else:
            if self.wo_spd:
                atom_bias = self.edge_encoder(edge_input).mean(-2)     # [B,N,N,D]
            elif self.wo_edge:
                atom_bias = self.shorest_path_encoder(shortest_path)   # [B,N,N,D]
            else:
                atom_bias = self.shorest_path_encoder(shortest_path) + self.edge_encoder(edge_input).mean(-2)     # [B,N,N,D]
            graph_attn_bias[:, special_len:, special_len:, :] = atom_bias

        # ------ 2. Any pair involving a special token -> use t bias ------
        t = self.vnode_virtual_distance.weight.view(1, 1, self.pair_dim)
        graph_attn_bias[:, :special_len, :, :] = t             # row
        graph_attn_bias[:, :, :special_len, :] = t             # col

        return graph_attn_bias


def _build_attn_mask(base_mask, seg_id, atom_mask, seg_valid, glob_valid, *, num_heads: int):
    """Construct the *additive* attention mask (-inf for disallowed positions).

    Args:
        base_mask (Tensor): user-provided base mask, shape [B, T, T]
        seg_id (Tensor): segment ids for each atom [B, N]
        atom_mask (Tensor): padding mask for atoms [B, N]
        seg_valid (Tensor): which segments are valid [B, S]
        glob_valid (Tensor): [B, 1] indicating if GLOB token is valid
        num_heads (int): number of attention heads (for expansion)

    Returns:
        Tensor: [B, num_heads, T, T] additive mask (0 for allowed, -inf for block)
    """
    B, S = seg_valid.shape
    N = seg_id.shape[1]

    special_len = 2 + S  # CLS + GLOB + SEG
    T = special_len + N
    device = base_mask.device

    attn_mask = base_mask.clone()
    padding_msk = attn_mask == float("-inf")

    seg_rows_all = torch.arange(S, device=device) + 2
    atom_rows = torch.arange(N, device=device) + special_len

    for b in range(B):
        seg_rows = seg_rows_all[seg_valid[b] == 1]

        # 1. Mask all segments by default
        attn_mask[b, seg_rows_all, :] = float("-inf")
        attn_mask[b, :, seg_rows_all] = float("-inf")

        # 2. Allow CLS <-> valid segments
        attn_mask[b, seg_rows, 0] = 0
        attn_mask[b, 0, seg_rows] = 0

        # 3. Allow GLOB <-> valid segments only if glob_valid[b] == 1
        if glob_valid[b] == 1:
            attn_mask[b, seg_rows, 1] = 0
            attn_mask[b, 1, seg_rows] = 0
        else:
            # Block GLOB token completely
            attn_mask[b, 1, :] = float("-inf")
            attn_mask[b, :, 1] = float("-inf")

        # 4. segment <-> own atoms
        for idx_s, row in zip(torch.nonzero(seg_valid[b]).flatten(), seg_rows):
            idx_atoms = atom_rows[seg_id[b] == idx_s]
            attn_mask[b, row, idx_atoms] = 0
            attn_mask[b, idx_atoms, row] = 0
            attn_mask[b, row, row] = 0  # self

        # 5. padding atoms (block both row and column)
        pad_atoms = atom_rows[atom_mask[b] == 0]
        attn_mask[b, pad_atoms, :] = float("-inf")
        attn_mask[b, :, pad_atoms] = float("-inf")

    # 6. Ensure diagonal is 0 (self-attention allowed unless already -inf)
    diag_idx = torch.arange(T, device=device)
    attn_mask[:, diag_idx, diag_idx] = 0.0

    return attn_mask.unsqueeze(1).float()  # [B, num_heads, T, T]


def build_padding_only_attn_mask(atom_mask, seg_valid, glob_valid, *,
                                 num_heads: int, device=None):
    B, N = atom_mask.shape
    S = seg_valid.shape[1]
    device = atom_mask.device if device is None else device

    cls_valid  = torch.ones(B, 1, dtype=torch.bool, device=device)          # CLS=1
    glob_valid = glob_valid.view(B, 1).bool().to(device)                    # GLOB��{0,1}
    seg_valid  = seg_valid.bool().to(device)                                # S seg tokens
    atom_valid = atom_mask.bool().to(device)                                # N atom tokens

    token_valid = torch.cat([cls_valid, glob_valid, seg_valid, atom_valid], dim=1)
    T = token_valid.size(1)

    valid_pair = token_valid.unsqueeze(-1) & token_valid.unsqueeze(-2)      # outer AND

    attn = torch.zeros(B, 1, T, T, device=device, dtype=torch.float32)
    attn = attn.masked_fill(~valid_pair.unsqueeze(1), float("-inf"))

    eye = torch.eye(T, dtype=torch.bool, device=device).view(1, 1, T, T)
    attn = torch.where(eye, torch.zeros_like(attn), attn)

    attn = attn.expand(-1, num_heads, -1, -1).contiguous()
    return attn
