"""Distance kernels for within-monomer geometric attention bias."""

from torch.nn import Embedding
import torch
import torch.nn as nn
from .layers import NonLinear


class SE3InvariantKernel(nn.Module):

    def __init__(
        self,
        pair_dim,
        num_pair,
        num_kernel,
        std_width=1.0,
        start=0.0,
        stop=9.0,
    ):
        super(SE3InvariantKernel, self).__init__()
        self.num_kernel = num_kernel

        self.gaussian = GaussianKernel(
            self.num_kernel,
            num_pair,
            std_width=std_width,
            start=start,
            stop=stop,
        )
        self.out_proj = NonLinear(self.num_kernel, pair_dim)

    def forward(self, dist, node_type_edge):
        edge_feature = self.gaussian(
            dist,
            node_type_edge.long(),
        )
        edge_feature = self.out_proj(edge_feature)
        return edge_feature


def gaussian(x, mean, std):
    pi = 3.14159
    a = (2 * pi) ** 0.5
    return torch.exp(-0.5 * (((x - mean) / std) ** 2)) / (a * std)


class GaussianKernel(nn.Module):
    def __init__(self, K=128, num_pair=512, std_width=1.0, start=0.0, stop=9.0):
        super().__init__()
        self.K = K
        std_width = std_width
        start = start
        stop = stop
        mean = torch.linspace(start, stop, K)
        self.std = (std_width * (mean[1] - mean[0]))
        self.register_buffer("mean", mean)
        self.mul = Embedding(num_pair, 1, padding_idx=0)
        self.bias = Embedding(num_pair, 1, padding_idx=0)
        nn.init.constant_(self.bias.weight, 0)
        nn.init.constant_(self.mul.weight, 1.0)

    def forward(self, x, atom_pair):
        mul = self.mul(atom_pair).abs().squeeze(-1)
        bias = self.bias(atom_pair).squeeze(-1)
        x = mul * x + bias
        x = x.unsqueeze(-1).expand(-1, -1, -1, self.K)
        mean = self.mean.float().view(-1)
        return gaussian(x.float(), mean, self.std)
