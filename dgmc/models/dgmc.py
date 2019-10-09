import torch
from torch.nn import Sequential as Seq, Linear as Lin, ReLU
from torch_scatter import scatter_add
from torch_geometric.utils import to_dense_batch
from torch_geometric.nn.inits import reset
from pykeops.torch import LazyTensor


def masked_softmax(src, mask, dim=-1):
    out = src.masked_fill(~mask, float('-inf'))
    out = torch.softmax(out, dim=dim)
    out = out.masked_fill(~mask, 0)
    return out


def to_sparse(x, mask):
    return x[mask]


def to_dense(x, mask):
    out = x.new_zeros(tuple(mask.size()) + (x.size(-1), ))
    return out.masked_scatter_(mask.unsqueeze(-1), x)


class DGMC(torch.nn.Module):
    r"""The Deep Graph Matching Conensus module which first matches nodes
    locally via a graph neural network :math:`\Psi_{\theta_1}`, and then
    updates correspondence scores iteratively by reaching for neighborhood
    consensus via a second graph neural network :math:`\Psi_{\theta_2}`.

    Args:
        psi_1 (torch.nn.Module): The first GNN :math:`\Psi_{\theta_1}` which
            takes in node features :obj:`x`, edge connectivity :`edge_index`,
            and optional edge features :`edge_attr`.
        psi_2 (torch.nn.Module): The second GNN :math:`\Psi_{\theta_2}` which
            takes in node features :obj:`x`, edge connectivity :`edge_index`,
            and optional edge features :`edge_attr`.
            :obj:`psi_2` needs to hold the attributes :obj:`in_channels` and
            :obj:`out_channels` indicating the dimensionality of randomly drawn
            node indicator functions and the output dimensionality of
            :obj:`psi_2` respectively.
        num_steps (int): Number of consensus iterations.
        k (int, optional): The sparsity parameter. If set to :obj:`-1`, will
            not sparsify initial correspondence rankings. (default: :obj:`-1`)
        detach (bool, optional): If set to :obj:`True`, will detach the
            computation of :math:`\Psi_{\theta_1}` from the current graph.
            (default: :obj:`False`)
        backend (str, optional): Specifies the map-reduce scheme used in KeOps.
            (default: :obj:`auto`)
    """
    def __init__(self, psi_1, psi_2, num_steps, k=-1, detach=False,
                 backend='auto'):
        super(DGMC, self).__init__()

        self.psi_1 = psi_1
        self.psi_2 = psi_2
        self.num_steps = num_steps
        self.k = k
        self.detach = detach
        self.backend = backend

        self.mlp = Seq(
            Lin(psi_2.out_channels, psi_2.out_channels),
            ReLU(),
            Lin(psi_2.out_channels, 1),
        )

        self.reset_parameters()

    def reset_parameters(self):
        self.psi_1.reset_parameters()
        self.psi_2.reset_parameters()
        reset(self.mlp)

    def top_k(self, x_s, x_t):
        x_s, x_t = LazyTensor(x_s.unsqueeze(-2)), LazyTensor(x_t.unsqueeze(-3))
        S_ij = (-x_s * x_t).sum(dim=-1)
        return S_ij.argKmin(self.k, dim=1, backend=self.backend)

    def forward(self, x_s, edge_index_s, edge_attr_s, batch_s, x_t,
                edge_index_t, edge_attr_t, batch_t, y=None):
        r""""""
        h_s = self.psi_1(x_s, edge_index_s, edge_attr_s)
        h_t = self.psi_1(x_t, edge_index_t, edge_attr_t)

        h_s, h_t = (h_s.detach(), h_t.detach()) if self.detach else (h_s, h_t)

        h_s, h_s_mask = to_dense_batch(h_s, batch_s, fill_value=float('inf'))
        h_t, h_t_mask = to_dense_batch(h_t, batch_t, fill_value=float('-inf'))

        assert h_s.size(0) == h_t.size(0), 'Encountered unequal batch-sizes'
        (B, N_s, F), N_t, R = h_s.size(), h_s.size(1), self.psi_2.in_channels

        if self.k < 1:
            # ------ Dense variant ------ #
            S_hat = h_s @ h_t.transpose(-1, -2)  # [B, N_s, N_t, F]
            S_mask = h_s_mask.view(B, N_s, 1) & h_t_mask.view(B, 1, N_t)
            S_0 = masked_softmax(S_hat, S_mask, dim=-1)

            for _ in range(self.num_steps):
                S = masked_softmax(S_hat, S_mask, dim=-1)
                r_s = torch.randn((B, N_s, R), dtype=h_s.dtype,
                                  device=h_s.device)
                r_t = S.transpose(-1, -2) @ r_s

                r_s, r_t = to_sparse(r_s, h_s_mask), to_sparse(r_t, h_t_mask)
                o_s = self.psi_2(r_s, edge_index_s, edge_attr_s)
                o_t = self.psi_2(r_t, edge_index_t, edge_attr_t)
                o_s, o_t = to_dense(o_s, h_s_mask), to_dense(o_t, h_t_mask)

                D = o_s.view(B, N_s, 1, R) - o_t.view(B, 1, N_t, R)
                S_hat = S_hat + self.mlp(D).squeeze(-1).masked_fill(~S_mask, 0)

            S_L = masked_softmax(S_hat, S_mask, dim=-1)

            return S_0, S_L
        else:
            # ------ Sparse variant ------ #
            S_idx = self.top_k(h_s, h_t)  # [B, N_s, k]
            if self.training and y is not None:
                # TODO: Include gt as index
                pass
            tmp_s = h_s.view(B, N_s, 1, F)
            tmp_t = h_t.view(B, 1, N_t, F).expand(-1, N_s, -1, -1)
            idx = S_idx.view(B, N_s, self.k, 1).expand(-1, -1, -1, F)
            S_hat = (tmp_s * torch.gather(tmp_t, -2, idx)).sum(dim=-1)
            S_0 = S_hat.softmax(dim=-1)

            for _ in range(self.num_steps):
                S = S_hat.softmax(dim=-1)
                r_s = torch.randn((B, N_s, R), dtype=h_s.dtype,
                                  device=h_s.device)

                tmp_t = r_s.view(B, N_s, 1, R) * S.view(B, N_s, self.k, 1)
                tmp_t = tmp_t.view(B, N_s * self.k, R)
                idx = S_idx.view(B, N_s * self.k, 1)
                r_t = scatter_add(tmp_t, idx, dim=1, dim_size=N_t)

                r_s, r_t = to_sparse(r_s, h_s_mask), to_sparse(r_t, h_t_mask)
                o_s = self.psi_2(r_s, edge_index_s, edge_attr_s)
                o_t = self.psi_2(r_t, edge_index_t, edge_attr_t)
                o_s, o_t = to_dense(o_s, h_s_mask), to_dense(o_t, h_t_mask)

                o_t = o_t.view(B, 1, N_t, R).expand(-1, N_s, -1, -1)
                idx = S_idx.view(B, N_s, self.k, 1).expand(-1, -1, -1, R)
                D = o_s.view(B, N_s, 1, R) - torch.gather(o_t, -2, idx)
                S_hat = S_hat + self.mlp(D).squeeze(-1)

            S_L = S_hat.softmax(dim=-1)

            return S_0, S_L, S_idx

    def __repr__(self):
        return ('{}(\n'
                '    psi_1={},\n'
                '    psi_2={},\n'
                '    num_steps={}, k={}\n)').format(self.__class__.__name__,
                                                    self.psi_1, self.psi_2,
                                                    self.num_steps, self.k)
