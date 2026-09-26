"""Monomer and polymer token features specific to this workflow."""

from typing import Optional
from torch.nn import Embedding
import torch
import torch.nn as nn

# Retain the established feature-module imports for checkpoint tooling.
from .chain_rescaling import rescale_repeat_counts
from frpn.common.layers import (
    NonLinear,
    Linear,
    DropPath,
    OuterProduct,
    TriangleMultiplication,
    Transition,
    permute_final_dims,
)
from frpn.common.geometry import (
    SE3InvariantKernel,
    gaussian,
    GaussianKernel,
)
from frpn.common.features import (
    EdgeFeaturePlus,
    _build_attn_mask,
    build_padding_only_attn_mask,
)
from frpn.common.heads import (
    MovementPredictionHead,
    MaskLMHead,
)


class AtomFeaturePlus(nn.Module):

    def __init__(
        self,
        num_atom: int,
        num_degree: int,
        hidden_dim: int,
        wo_node: bool = False,
        wo_atom_feat: Optional[list[int]] = None,
        num_glob_feat: int = 2,
        num_seg_feat: int = 2,
    ):
        super().__init__()
        # Atom categories, degree and the virtual CLS node.
        self.atom_encoder   = Embedding(num_atom,   hidden_dim, padding_idx=0)
        self.degree_encoder = Embedding(num_degree, hidden_dim, padding_idx=0)
        self.vnode_encoder  = Embedding(1,          hidden_dim)  # [CLS] token
        self.wo_node = wo_node
        self.wo_atom_feat = wo_atom_feat
        # --- Additional: projection layers for numeric global/segment features ---
        self.num_glob_feat = num_glob_feat
        self.num_seg_feat  = num_seg_feat
        self.glob_projs = nn.ModuleList(
            [Linear(1, hidden_dim, bias=True, init="glorot") for _ in range(num_glob_feat)]
        )
        self.seg_projs = nn.ModuleList(
            [Linear(1, hidden_dim, bias=True, init="glorot") for _ in range(num_seg_feat)]
        )

    # ------------------------------ forward ------------------------------
    def forward(self, batched_data, token_feat):

        x           = batched_data["atom_feat"]          # [B, N, F_a]
        degree      = batched_data["degree"]             # [B, N]
        seg_id      = batched_data["segment_id"]         # [B, N]
        glob_f      = batched_data["glob_feat"]          # [B, G]
        glob_mask   = batched_data["glob_mask"]          # [B, G]
        chain_glob_f = batched_data.get("chain_glob_feat", None)
        chain_glob_m = batched_data.get("chain_glob_mask", None)
        glob_valid  = batched_data["glob_valid_mask"]
        seg_f       = batched_data["seg_feat"]           # [B, S, F_s]
        seg_f_mask  = batched_data["seg_feat_mask"]      # [B, S, F_s]
        seg_valid   = batched_data["seg_valid_mask"]     # [B, S]

        B, N_atom = x.shape[:2]
        S         = seg_f.size(1)
        D         = token_feat.size(-1)
        dtype     = token_feat.dtype
        device    = x.device

        # ---------- 1. Atom nodes ----------
        if self.wo_node:
            atom_vec = token_feat
        else:
            atom_emb = self.atom_encoder(x)

            if self.wo_atom_feat is not None:
                atom_emb[:, :, self.wo_atom_feat, :] = 0.0
            atom_vec = atom_emb.sum(dim=-2)

            if self.wo_atom_feat is None or 1 not in self.wo_atom_feat:
                atom_vec = atom_emb.sum(dim=-2) + self.degree_encoder(degree)

            atom_vec = atom_vec + token_feat

        # ---------- 2. GLOB token ----------
        glob_vec = torch.zeros(B, D, device=device)


        for i in range(self.num_glob_feat):
            if i < glob_f.size(1):
                val = glob_f[:, i].unsqueeze(-1)
                gmsk = glob_mask[:, i].unsqueeze(-1)
            elif chain_glob_f is not None and chain_glob_m is not None and i < chain_glob_f.size(1):
                # Fairness bridge: allow Stage-1 to consume coexistence stored in chain globals.
                val = chain_glob_f[:, i].unsqueeze(-1)
                gmsk = chain_glob_m[:, i].unsqueeze(-1)
            else:
                val = torch.zeros(B, 1, device=device, dtype=glob_f.dtype)
                gmsk = torch.zeros(B, 1, device=device, dtype=glob_mask.dtype)
            glob_vec += gmsk * self.glob_projs[i](val)
        # glob_vec=0 if entire row is None.
        glob_vec = glob_vec * glob_valid
        glob_vec = glob_vec.unsqueeze(1)
        # ---------- 3. SEG tokens ----------
        seg_vec = torch.zeros(B, S, D, device=device)
        for j in range(self.num_seg_feat):
            val  = seg_f[:, :, j].unsqueeze(-1)          # [B,S,1]
            smsk = seg_f_mask[:, :, j].unsqueeze(-1)     # [B,S,1]
            seg_vec += smsk * self.seg_projs[j](val)
        # seg_valid
        seg_vec = seg_vec * seg_valid.unsqueeze(-1)      # [B,S,D]

        # ---------- 4. CLS ----------
        cls_vec = self.vnode_encoder.weight.unsqueeze(0).repeat(B, 1, 1)

        # ---------- 5. Concat ----------
        graph_node_feature = torch.cat(
            [cls_vec, glob_vec, seg_vec, atom_vec], dim=1
        ).type(dtype)                                    # [B, 1+1+S+N, D]
        return graph_node_feature


class ChainTokenFeaturePlus(nn.Module):

    def __init__(
        self,
        embed_dim: int,
        num_glob_feat: int,
        num_block_feat: int,
        max_chain_tokens: int = 1536,
        enable_anchor_symmetry_break: bool = True,
    ):
        super().__init__()

        self.embed_dim = embed_dim
        self.num_glob_feat = num_glob_feat
        self.num_block_feat = num_block_feat
        self.max_chain_tokens = max_chain_tokens
        self.enable_anchor_symmetry_break = enable_anchor_symmetry_break

        # ---- CLS token ----
        self.chain_cls = nn.Embedding(1, embed_dim)

        # ---- Global feature projections (like AtomFeaturePlus) ----
        self.glob_projs = nn.ModuleList(
            [nn.Linear(1, embed_dim, bias=True) for _ in range(num_glob_feat)]
        )

        # ---- Block feature projections (shared across blocks) ----
        self.block_projs = nn.ModuleList(
            [nn.Linear(1, embed_dim, bias=True) for _ in range(num_block_feat)]
        )

        self.topo_mlp = nn.Sequential(
            nn.Linear(1, embed_dim),
            nn.SiLU(),
            nn.Linear(embed_dim, embed_dim),
        )

        self.topo_bias = nn.Parameter(torch.zeros(embed_dim))

    def forward(
        self,
        seg_repr,          # [B, S, D]
        seg_dop,           # [B, S]
        seg_block_id,      # [B, S]
        glob_feat,         # [B, G]
        glob_feat_mask,    # [B, G]
        block_feat,        # [B, 2, F]
        block_feat_mask,   # [B, 2, F]
        num_heads: int,
        return_chain_lengths: bool = False,
    ):
        """
        Returns:
            chain_tokens : [B, T, D]
            attn_mask    : [B, H, T, T]
            chain_lengths: [B], optionally; repeat counts after rescaling
        """
        B, S, D = seg_repr.shape
        device = seg_repr.device

        chain_tokens_list = []
        chain_block_list = []
        chain_lengths_list = []

        # ---------- build unpadded sequences ----------
        for b in range(B):
            tokens = []
            blocks = []

            # CLS
            cls_tok = self.chain_cls.weight.unsqueeze(0)   # [1, 1, D]
            cls_tok = cls_tok.repeat(1, 1, 1).squeeze(0)   # [1, D]
            tokens.append(cls_tok)
            blocks.append(-1)

            # GLOB
            glob_vec = torch.zeros(D, device=device)

            # glob_feat[b] : Tensor [G]
            # glob_feat_mask[b] : Tensor [G]
            feat_b = glob_feat[b]
            mask_b = glob_feat_mask[b]

            for i in range(self.num_glob_feat):
                if mask_b[i].item() == 1:
                    glob_vec += self.glob_projs[i](feat_b[i].view(1, 1)).view(-1)

            tokens.append(glob_vec.view(1, D))
            blocks.append(-1)

            # BLOCK0 / BLOCK1
            block_feat_b = block_feat[b]           # Tensor [2, F]
            block_feat_mask_b = block_feat_mask[b] # Tensor [2, F]

            for blk in range(2):
                blk_vec = torch.zeros(D, device=device)
                for j in range(self.num_block_feat):
                    if block_feat_mask_b[blk, j].item() == 1:
                        blk_vec += self.block_projs[j](block_feat_b[blk, j].view(1, 1)).view(-1)
                tokens.append(blk_vec.view(1, D))
                blocks.append(blk)

            # Rescale all block/monomer counts together BEFORE allocating tokens.
            # This preserves both blocks and their composition under the budget.
            seg_block_id_b = seg_block_id[b]  # Tensor [S]
            counts = [
                max(0, int(seg_dop[b, s])) if int(seg_block_id_b[s]) in (0, 1) else 0
                for s in range(S)
            ]
            counts = rescale_repeat_counts(counts, self.max_chain_tokens)
            chain_lengths_list.append(sum(counts))

            # REPEAT tokens (random interleave per block, as for uncapped chains)
            for blk in range(2):
                seg_idx = [
                    s for s in range(S)
                    if int(seg_block_id_b[s]) == blk and counts[s] > 0
                ]
                if not seg_idx:
                    continue

                reps = []
                for s in seg_idx:
                    n = counts[s]
                    reps.append(seg_repr[b][s].unsqueeze(0).repeat(n, 1))
                reps = torch.cat(reps, dim=0)

                perm = torch.randperm(reps.size(0), device=device)
                reps = reps[perm]

                tokens.append(reps)
                blocks.extend([blk] * reps.size(0))

            tokens = torch.cat(tokens, dim=0)   # [T_b, D]
            blocks = torch.tensor(blocks, device=device)

            if self.enable_anchor_symmetry_break:
                # ===== Semantic Rewrite (junction-based) =====
                T_b = tokens.size(0)

                # ===== find junction index (last A repeat token) =====
                repeat_start = 4
                repeat_blocks = blocks[repeat_start:]

                # indices in full token space
                repeat_indices = torch.arange(repeat_start, T_b, device=device)

                A_repeat_mask = repeat_blocks == 0
                if A_repeat_mask.any():
                    junction_idx = repeat_indices[A_repeat_mask].max()
                else:
                    # degenerate case: no A block (should not happen in AB)
                    junction_idx = repeat_start

                # compute signed distance to junction
                idx = torch.arange(T_b, device=device)
                repeat_mask = blocks >= 0
                repeat_mask[:4] = False

                L_chain = repeat_mask.sum().float().clamp(min=1.0)

                dist = (idx - junction_idx).float() / (L_chain + 1e-6)
                dist = dist.unsqueeze(-1)

                # mask out non-repeat tokens (CLS/GLOB/BLOCK)
                rewrite_mask = torch.zeros(T_b, 1, device=device)
                rewrite_mask[4:] = 1.0   # only REPEAT tokens get rewritten

                # topological field
                topo_field = self.topo_mlp(dist) * rewrite_mask  # [T_b, D]

                # semantic rewrite (affine modulation)
                tokens = tokens * (1.0 + topo_field) + self.topo_bias * rewrite_mask

            chain_tokens_list.append(tokens)
            chain_block_list.append(blocks)

        # ---------- padding ----------
        max_T = max(t.size(0) for t in chain_tokens_list)

        chain_tokens = torch.zeros(B, max_T, D, device=device)
        chain_valid = torch.zeros(B, max_T, dtype=torch.bool, device=device)

        for b in range(B):
            T_b = chain_tokens_list[b].size(0)
            chain_tokens[b, :T_b] = chain_tokens_list[b]
            chain_valid[b, :T_b] = 1

        # ---------- attention mask ----------
        attn_mask = torch.full(
            (B, max_T, max_T),
            float("-inf"),
            device=device,
        )

        for b in range(B):
            valid = chain_valid[b]
            blocks = chain_block_list[b]

            idx = torch.nonzero(valid).flatten()

            # self
            attn_mask[b].masked_fill_(torch.eye(max_T, device=device).bool(), 0.0)

            # CLS & GLOB
            for t in [0, 1]:
                attn_mask[b, t, idx] = 0.0
                attn_mask[b, idx, t] = 0.0

            # BLOCK <-> own REPEAT
            for k in [0, 1]:
                blk_row = 2 + k
                T_b = blocks.size(0)
                rep_idx = torch.nonzero(blocks == k).flatten()
                attn_mask[b, blk_row, rep_idx] = 0.0
                attn_mask[b, rep_idx, blk_row] = 0.0

            # BLOCK <-> BLOCK
            attn_mask[b, 2, 3] = 0.0
            attn_mask[b, 3, 2] = 0.0
            rep_start = 4
            rep_valid = valid.clone()
            rep_valid[:rep_start] = False
            rep_idx = torch.nonzero(rep_valid).flatten()
            if rep_idx.numel() > 0:
                attn_mask[b][rep_idx[:, None], rep_idx[None, :]] = 0.0

        attn_mask = attn_mask.unsqueeze(1).repeat(1, num_heads, 1, 1)

        if return_chain_lengths:
            lengths = torch.tensor(chain_lengths_list, device=device, dtype=torch.long)
            return chain_tokens, attn_mask, lengths
        return chain_tokens, attn_mask


class ChainEdgeFeaturePlus(nn.Module):

    def __init__(self, pair_dim, max_chain_dist):
        super().__init__()
        self.pair_dim = pair_dim
        self.max_chain_dist = max_chain_dist
        self.chain_dist_encoder = nn.Embedding(
             max_chain_dist + 1, pair_dim, padding_idx=0
        )
        self.vnode_virtual_distance = nn.Embedding(1, pair_dim)

    def forward(self, batched_data, graph_attn_bias, chain_lengths=None):
        special_len = 4
        # The model supplies actual per-sample repeat counts after rescaling.
        chain_len = batched_data["chain_len"] if chain_lengths is None else chain_lengths
        B = graph_attn_bias.size(0)
        device = graph_attn_bias.device

        for b in range(B):
            available = graph_attn_bias.size(1) - special_len
            L = min(int(chain_len[b]), int(available))
            if L == 0:
                continue
            idx = torch.arange(L, device=device)
            dist = torch.abs(idx[:, None] - idx[None, :])
            dist = torch.clamp(dist, max=self.max_chain_dist)
            graph_attn_bias[b,
                special_len:special_len+L,
                special_len:special_len+L,
                :
            ] = self.chain_dist_encoder(dist)

        t = self.vnode_virtual_distance.weight.view(1, 1, self.pair_dim)
        graph_attn_bias[:, :special_len, :, :] = t
        graph_attn_bias[:, :, :special_len, :] = t
        return graph_attn_bias
