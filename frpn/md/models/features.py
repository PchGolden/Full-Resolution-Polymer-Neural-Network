"""Monomer and polymer token features specific to this workflow."""

from typing import Optional
from torch.nn import Embedding
import torch
import torch.nn as nn

# Retain the established feature-module imports for checkpoint tooling.
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
            val  = glob_f[:, i].unsqueeze(-1)            # [B,1]
            gmsk = glob_mask[:, i].unsqueeze(-1)         # [B,1]
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
        enable_topology_symmetry_break: bool = True,
    ):
        super().__init__()

        self.embed_dim = embed_dim
        self.num_glob_feat = num_glob_feat
        self.num_block_feat = num_block_feat
        self.enable_topology_symmetry_break = bool(enable_topology_symmetry_break)

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

        # Legacy (AB junction) symmetry breaking: scalar signed distance to junction.
        self.topo_mlp = nn.Sequential(
            nn.Linear(1, embed_dim),
            nn.SiLU(),
            nn.Linear(embed_dim, embed_dim),
        )

        # Topology-driven symmetry breaking: use distances to two anchors.
        # Kept separate to avoid breaking legacy checkpoint shapes.
        self.topo_mlp_anchor2 = nn.Sequential(
            nn.Linear(2, embed_dim),
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
        block_feat,        # [B, Bmax, F]
        block_feat_mask,   # [B, Bmax, F]
        chain_node_seg_id=None,
        chain_topo_dist=None,
        num_heads: int = 8,
    ):
        """
        Returns:
            chain_tokens : [B, T, D]
            attn_mask    : [B, H, T, T]
            meta         : Per-sample lengths, repeat offsets and topology distances
                           used to construct chain pair features.
        """
        B, S, D = seg_repr.shape
        device = seg_repr.device

        # number of block tokens is determined by preprocessing (Bmax can be 0)
        if block_feat is None or block_feat_mask is None:
            Bmax = 0
        else:
            if not isinstance(block_feat, torch.Tensor) or not isinstance(block_feat_mask, torch.Tensor):
                raise TypeError("block_feat and block_feat_mask must be torch.Tensor when provided")
            if block_feat.dim() != 3 or block_feat_mask.dim() != 3:
                raise ValueError(f"block_feat/block_feat_mask must be 3D, got {block_feat.shape} and {block_feat_mask.shape}")
            Bmax = int(block_feat.size(1))

        chain_tokens_list = []
        chain_block_list = []  # legacy only
        chain_len_list = []
        topo_dist_list = []
        block_valid_list = []

        def _as_1d_long(x, device: torch.device) -> torch.Tensor:
            if x is None:
                return torch.empty(0, dtype=torch.long, device=device)
            if isinstance(x, torch.Tensor):
                return x.to(device=device, dtype=torch.long).view(-1)
            # python list / numpy
            return torch.tensor(list(x), dtype=torch.long, device=device).view(-1)

        def _as_2d_long(x, device: torch.device) -> torch.Tensor:
            if x is None:
                return torch.empty(0, 0, dtype=torch.long, device=device)
            if isinstance(x, torch.Tensor):
                return x.to(device=device, dtype=torch.long)
            return torch.tensor(x, dtype=torch.long, device=device)

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

            # BLOCK tokens (variable count, pad-able like stage-1 SEG)
            block_valid = torch.zeros(Bmax, dtype=torch.bool, device=device)
            if Bmax > 0:
                block_feat_b = block_feat[b]                 # [Bmax, F]
                block_feat_mask_b = block_feat_mask[b]       # [Bmax, F]
                block_valid = (block_feat_mask_b.sum(dim=-1) > 0)

                for blk in range(Bmax):
                    blk_vec = torch.zeros(D, device=device)
                    for j in range(self.num_block_feat):
                        if j >= int(block_feat_b.size(1)):
                            break
                        if block_feat_mask_b[blk, j].item() == 1:
                            blk_vec += self.block_projs[j](block_feat_b[blk, j].view(1, 1)).view(-1)
                    # if a block is invalid, keep it zero; masking happens in attn_mask
                    if not bool(block_valid[blk].item()):
                        blk_vec.zero_()
                    tokens.append(blk_vec.view(1, D))
                    blocks.append(blk)

            # =========================
            # New path: topology-driven node sequence
            # =========================
            if chain_node_seg_id is not None:
                node_seg = chain_node_seg_id[b] if isinstance(chain_node_seg_id, (list, tuple)) else chain_node_seg_id
                node_seg = _as_1d_long(node_seg, device)
                L = int(node_seg.numel())

                if L > 0:
                    node_tok = seg_repr[b].index_select(0, node_seg)  # [L, D]
                    tokens.append(node_tok)

                # Symmetry breaking (topology-derived) — optional
                topo = None
                if chain_topo_dist is not None:
                    topo_b = chain_topo_dist[b] if isinstance(chain_topo_dist, (list, tuple)) else chain_topo_dist
                    topo = _as_2d_long(topo_b, device)
                    if (
                        self.enable_topology_symmetry_break
                        and topo.dim() == 2
                        and topo.size(0) == L
                        and topo.size(1) == L
                        and L > 0
                    ):
                        dist_sum = topo.float().sum(dim=1)
                        anchor1 = int(torch.argmin(dist_sum).item())
                        d_from_a1 = topo[anchor1].float()
                        anchor2 = int(torch.argmax(d_from_a1).item())

                        d1 = topo[:, anchor1].float()
                        d2 = topo[:, anchor2].float()
                        denom = torch.maximum(d1.max(), d2.max()).clamp(min=1.0)
                        topo_feat = torch.stack([d1 / denom, d2 / denom], dim=-1)  # [L, 2]

                        topo_field = self.topo_mlp_anchor2(topo_feat)  # [L, D]
                        rewrite_mask = torch.ones(L, 1, device=device)
                        # Apply only to node tokens (after CLS/GLOB)
                        if L > 0:
                            tokens[-1] = tokens[-1] * (1.0 + topo_field) + self.topo_bias * rewrite_mask

                tokens = torch.cat(tokens, dim=0)  # [T_b, D]
                chain_tokens_list.append(tokens)
                chain_len_list.append(L)
                topo_dist_list.append(topo if topo is not None else None)
                block_valid_list.append(block_valid)
                continue

            # REPEAT tokens (random interleave per block)
            for blk in range(Bmax):
                seg_block_id_b = seg_block_id[b]  # Tensor [S]
                seg_dop_b = seg_dop[b]            # Tensor [S]

                seg_idx = [
                    s for s in range(S)
                    if int(seg_block_id_b[s]) == blk and int(seg_dop_b[s]) > 0
                ]
                if not seg_idx:
                    continue

                reps = []
                for s in seg_idx:
                    n = int(seg_dop[b][s])
                    reps.append(seg_repr[b][s].unsqueeze(0).repeat(n, 1))
                reps = torch.cat(reps, dim=0)

                perm = torch.randperm(reps.size(0), device=device)
                reps = reps[perm]

                tokens.append(reps)
                blocks.extend([blk] * reps.size(0))

            tokens = torch.cat(tokens, dim=0)   # [T_b, D]
            blocks = torch.tensor(blocks, device=device)
            # ===== Semantic Rewrite (junction-based) =====
            T_b = tokens.size(0)

            # ===== find junction index (last A repeat token) =====
            repeat_start = 2 + Bmax
            repeat_blocks = blocks[repeat_start:]

            # indices in full token space
            repeat_indices = torch.arange(repeat_start, T_b, device=device)

            # Only apply the legacy AB junction rewrite when there are exactly 2 blocks.
            if Bmax == 2:
                A_repeat_mask = repeat_blocks == 0
                if A_repeat_mask.any():
                    junction_idx = repeat_indices[A_repeat_mask].max()
                else:
                    junction_idx = repeat_start
            else:
                junction_idx = repeat_start

            # compute signed distance to junction
            idx = torch.arange(T_b, device=device)
            repeat_mask = blocks >= 0
            repeat_mask[:repeat_start] = False

            L_chain = repeat_mask.sum().float().clamp(min=1.0)

            dist = (idx - junction_idx).float() / (L_chain + 1e-6)
            dist = dist.unsqueeze(-1)

            # mask out non-repeat tokens (CLS/GLOB/BLOCK)
            rewrite_mask = torch.zeros(T_b, 1, device=device)
            rewrite_mask[repeat_start:] = 1.0   # only REPEAT tokens get rewritten

            # topological field
            topo_field = self.topo_mlp(dist) * rewrite_mask  # [T_b, D]

            # semantic rewrite (affine modulation)
            if Bmax == 2:
                tokens = tokens * (1.0 + topo_field) + self.topo_bias * rewrite_mask

            chain_tokens_list.append(tokens)
            chain_block_list.append(blocks)
            # Preserve the legacy count, which includes nonnegative block IDs.
            chain_len_list.append(int((blocks >= 0).sum().item()))
            topo_dist_list.append(None)
            block_valid_list.append(block_valid)

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
            idx = torch.nonzero(valid).flatten()

            # allow all valid tokens to attend each other (then selectively block GLOB below)
            if idx.numel() > 0:
                attn_mask[b][idx[:, None], idx[None, :]] = 0.0

            # ensure diagonal is 0
            attn_mask[b].masked_fill_(torch.eye(max_T, device=device).bool(), 0.0)

            # If chain-level global feature is completely absent, block GLOB like stage-1.
            if glob_feat_mask is not None:
                mask_b = glob_feat_mask[b]
                if isinstance(mask_b, torch.Tensor):
                    glob_valid = float(mask_b.sum().item()) > 0.0
                else:
                    glob_valid = float(sum(mask_b)) > 0.0
                if not glob_valid:
                    attn_mask[b, 1, :] = float("-inf")
                    attn_mask[b, :, 1] = float("-inf")
                    attn_mask[b, 1, 1] = 0.0

            # Block invalid block tokens (pad them out like stage-1 SEG)
            if Bmax > 0 and b < len(block_valid_list):
                bv = block_valid_list[b]
                for blk in range(Bmax):
                    if not bool(bv[blk].item()):
                        t = 2 + blk
                        attn_mask[b, t, :] = float("-inf")
                        attn_mask[b, :, t] = float("-inf")
                        attn_mask[b, t, t] = 0.0

        attn_mask = attn_mask.unsqueeze(1).repeat(1, num_heads, 1, 1)

        # Pad topology distance matrices (repeat-token space) if provided in new mode.
        max_L = int(max(chain_len_list)) if chain_len_list else 0
        topo_padded = None
        if any(t is not None for t in topo_dist_list) and max_L > 0:
            topo_padded = torch.zeros(B, max_L, max_L, dtype=torch.long, device=device)
            for b in range(B):
                topo = topo_dist_list[b]
                L = int(chain_len_list[b])
                if topo is None or L == 0:
                    continue
                topo_padded[b, :L, :L] = topo

        meta = {
            "chain_len": torch.tensor(chain_len_list, dtype=torch.long, device=device),
            "repeat_start": 2 + Bmax,
            "topo_dist": topo_padded,
        }

        return chain_tokens, attn_mask, meta


class ChainEdgeFeaturePlus(nn.Module):

    def __init__(self, pair_dim, max_chain_dist):
        super().__init__()
        self.pair_dim = pair_dim
        self.max_chain_dist = max_chain_dist
        self.chain_dist_encoder = nn.Embedding(
             max_chain_dist + 1, pair_dim, padding_idx=0
        )
        self.vnode_virtual_distance = nn.Embedding(1, pair_dim)

    def forward(
        self,
        batched_data,
        graph_attn_bias,
        *,
        repeat_start: int = 4,
        chain_len: Optional[torch.Tensor] = None,
        topo_dist: Optional[torch.Tensor] = None,
    ):
        """Fill Stage-2 pair bias for repeat/node tokens.

        Args:
            repeat_start: index where repeat/node tokens begin in the stage-2 sequence.
            chain_len: Long[B] number of repeat/node tokens per sample.
            topo_dist: Optional Long[B, max_L, max_L] topology distances in repeat-token space.
                If None, falls back to linear |i-j| distance.
        """
        if chain_len is None:
            chain_len = batched_data["chain_len"]

        B = graph_attn_bias.size(0)
        device = graph_attn_bias.device

        for b in range(B):
            L = int(chain_len[b])
            if L <= 0:
                continue

            if topo_dist is not None:
                dist = topo_dist[b, :L, :L].to(device=device)
            else:
                idx = torch.arange(L, device=device)
                dist = torch.abs(idx[:, None] - idx[None, :])

            dist = torch.clamp(dist, max=self.max_chain_dist).long()
            graph_attn_bias[b,
                repeat_start:repeat_start + L,
                repeat_start:repeat_start + L,
                :
            ] = self.chain_dist_encoder(dist)

        t = self.vnode_virtual_distance.weight.view(1, 1, self.pair_dim)
        graph_attn_bias[:, :repeat_start, :, :] = t
        graph_attn_bias[:, :, :repeat_start, :] = t
        return graph_attn_bias
