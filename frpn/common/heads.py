"""Masked-atom and coordinate reconstruction heads."""

from torch import Tensor
import torch
import torch.nn as nn
import torch.nn.functional as F
from .layers import Linear


class MovementPredictionHead(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        pair_dim: int,
        num_head: int,
    ):
        super().__init__()
        self.layer_norm = nn.LayerNorm(embed_dim)
        self.embed_dim = embed_dim
        self.q_proj = Linear(embed_dim, embed_dim, bias=False, init="glorot")
        self.k_proj = Linear(embed_dim, embed_dim, bias=False, init="glorot")
        self.v_proj = Linear(embed_dim, embed_dim, bias=False, init="glorot")
        self.num_head = num_head
        self.scaling = (embed_dim // num_head) ** -0.5
        self.force_proj1 = Linear(embed_dim, 1, init="final", bias=False)
        self.linear_bias = Linear(pair_dim, num_head)
        self.pair_layer_norm = nn.LayerNorm(pair_dim)
        self.dropout = 0.1

    def zero_init(self):
        nn.init.zeros_(self.force_proj1.weight)

    def forward(
        self,
        query: Tensor,
        pair: Tensor,
        attn_mask: Tensor,
        delta_pos: Tensor,
    ) -> Tensor:
        bsz, n_node, _ = query.size()
        query = self.layer_norm(query)
        q = (
            self.q_proj(query).view(bsz, n_node, self.num_head, -1).transpose(1, 2)
            * self.scaling
        )
        k = self.k_proj(query).view(bsz, n_node, self.num_head, -1).transpose(1, 2)
        v = self.v_proj(query).view(bsz, n_node, self.num_head, -1).transpose(1, 2)
        attn = q @ k.transpose(-1, -2)  # [bsz, head, n, n]
        pair = self.pair_layer_norm(pair)
        bias = self.linear_bias(pair).permute(0, 3, 1, 2).contiguous()
        attn = attn + bias
        attn = attn + attn_mask
        attn = F.softmax(attn, dim=-1)
        attn = F.dropout(attn, p=self.dropout, training=self.training)
        attn_probs = attn.view(bsz, self.num_head, n_node, n_node)
        rot_attn_probs = attn_probs.unsqueeze(-1) * delta_pos.unsqueeze(1).type_as(
            attn_probs
        )  # [bsz, head, n, n, 3]
        rot_attn_probs = rot_attn_probs.permute(0, 1, 4, 2, 3)
        x = rot_attn_probs @ v.unsqueeze(2)  # [bsz, head , 3, n, d]
        x = x.permute(0, 3, 2, 1, 4).contiguous().view(bsz, n_node, 3, -1)
        cur_force = self.force_proj1(x).view(bsz, n_node, 3)
        return cur_force


class MaskLMHead(nn.Module):
    """Head for masked language modeling."""

    def __init__(self, embed_dim, output_dim, weight=None):
        super().__init__()
        self.dense = nn.Linear(embed_dim, embed_dim)
        self.layer_norm = nn.LayerNorm(embed_dim)
        self.act = nn.GELU()
        if weight is None:
            weight = nn.Linear(embed_dim, output_dim, bias=False).weight
        self.weight = weight
        self.bias = nn.Parameter(torch.zeros(output_dim))

    def forward(self, features, masked_tokens=None, **kwargs):
        # Only project the masked tokens while training,
        # saves both memory and computation
        if masked_tokens is not None:
            features = features[masked_tokens, :]

        x = self.layer_norm(features)
        x = self.dense(x)
        x = self.act(x)
        # project back to size of vocabulary with bias
        x = F.linear(x, self.weight) + self.bias
        return x
