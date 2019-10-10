import torch
from torch.nn import Sequential as Seq, Linear as Lin, ReLU
from torch_scatter import scatter_add
from torch_geometric.utils import to_dense_batch
from torch_geometric.nn.inits import reset
from pykeops.torch import LazyTensor

EPS = 1e-8


def masked_softmax(src, mask, dim=-1):
    out = src.masked_fill(~mask, float('-inf'))
    out = torch.softmax(out, dim=dim)
    out = out.masked_fill(~mask, 0)
    return out


def to_sparse(x, mask):
    return x[mask]


def to_dense(x, mask):
    out = x.new_zeros(tuple(mask.size()) + (x.size(-1), ))
    out[mask] = x
    return out


class DGMC(torch.nn.Module):
    r"""The Deep Graph Matching Consensus module which first matches nodes
    locally via a graph neural network :math:`\Psi_{\theta_1}`, and then
    updates correspondence scores iteratively by reaching for neighborhood
    consensus via a second graph neural network :math:`\Psi_{\theta_2}`.

    Args:
        psi_1 (torch.nn.Module): The first GNN :math:`\Psi_{\theta_1}` which
            takes in node features :obj:`x`, edge connectivity :`edge_index`,
            and optional edge features :`edge_attr` and computes node
            embeddings.
        psi_2 (torch.nn.Module): The second GNN :math:`\Psi_{\theta_2}` which
            takes in node features :obj:`x`, edge connectivity :`edge_index`,
            and optional edge features :`edge_attr` and validates for
            neighborhood consensus.
            :obj:`psi_2` needs to hold the attributes :obj:`in_channels` and
            :obj:`out_channels` indicating the dimensionality of randomly drawn
            node indicator functions and the output dimensionality of
            :obj:`psi_2`, respectively.
        num_steps (int): Number of consensus iterations.
        k (int, optional): Sparsity parameter. If set to :obj:`-1`, will
            not sparsify initial correspondence rankings. (default: :obj:`-1`)
        detach (bool, optional): If set to :obj:`True`, will detach the
            computation of :math:`\Psi_{\theta_1}` from the current computation
            graph. (default: :obj:`False`)
    """
    def __init__(self, psi_1, psi_2, num_steps, k=-1, detach=False):
        super(DGMC, self).__init__()

        self.psi_1 = psi_1
        self.psi_2 = psi_2
        self.num_steps = num_steps
        self.k = k
        self.detach = detach
        self.backend = 'auto'

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

    def __top_k__(self, x_s, x_t):
        r"""Memory-efficient top-k correspondence computation."""
        x_s, x_t = LazyTensor(x_s.unsqueeze(-2)), LazyTensor(x_t.unsqueeze(-3))
        S_ij = (-x_s * x_t).sum(dim=-1)
        return S_ij.argKmin(self.k, dim=2, backend=self.backend)

    def __append_gt__(self, S_idx, s_mask, y):
        r"""Appends the ground-truth values in :obj:`y` to the index tensor
        :obj:`S_idx`."""
        (B, N_s), (row, col), k = s_mask.size(), y, S_idx.size(-1)

        gt_mask = (S_idx[s_mask][row] != col.view(-1, 1)).all(dim=-1)

        sparse_mask = gt_mask.new_zeros((s_mask.sum(), ))
        sparse_mask[row] = gt_mask

        dense_mask = sparse_mask.new_zeros((B, N_s))
        dense_mask[s_mask] = sparse_mask
        last_entry = torch.zeros(k, dtype=torch.bool, device=gt_mask.device)
        last_entry[-1] = 1
        dense_mask = dense_mask.view(B, N_s, 1) * last_entry.view(1, 1, k)

        return S_idx.masked_scatter(dense_mask, col[gt_mask])

    def forward(self, x_s, edge_index_s, edge_attr_s, batch_s, x_t,
                edge_index_t, edge_attr_t, batch_t, y=None):
        r"""
        Args:
            x_s (Tensor): Source graph node features of shape
                :obj:`[batch_size * num_nodes, F]`.
            edge_index_s (LongTensor): Source graph edge connectivity of shape
                :obj:`[2, num_edges]`.
            edge_attr_s (Tensor): Source graph edge features of shape
                :obj:`[num_edges, D]`. Can be :obj:`None`.
            batch_s (LongTensor): Source graph batch vector of shape
                :obj:`[batch_size * num_nodes]` indicating node to graph
                assignment. Can be :obj:`None` in case operating on single
                graphs.
            x_t (Tensor): Target graph node features of shape
                :obj:`[batch_size * num_nodes, F]`.
            edge_index_t (LongTensor): Target graph edge connectivity of shape
                :obj:`[2, num_edges]`.
            edge_attr_t (Tensor): Target graph edge features of shape
                :obj:`[num_edges, D]`. Can be :obj:`None`.
            batch_t (LongTensor): Target graph batch vector of shape
                :obj:`[batch_size * num_nodes]` indicating node to graph
                assignment. Can be :obj:`None` in case operating on single
                graphs.
            y (LongTensor, optional): Ground-truth matchings of shape
                :obj:`[2, num_ground_truths]` to include ground-truth values
                when training against sparse correspondences. Ground-truths
                are only used in case the model is in training mode.
                (default: :obj:`None`)

        Returns:
            Initial and refined correspondence matrices :obj:`(S_0, S_L)`
            of shapes :obj:`[batch_size * num_nodes, num_nodes]`. The
            correspondence matrix are either given as dense or sparse matrices.
        """
        h_s = self.psi_1(x_s, edge_index_s, edge_attr_s)
        h_t = self.psi_1(x_t, edge_index_t, edge_attr_t)

        h_s, h_t = (h_s.detach(), h_t.detach()) if self.detach else (h_s, h_t)

        h_s, s_mask = to_dense_batch(h_s, batch_s, fill_value=float('inf'))
        h_t, t_mask = to_dense_batch(h_t, batch_t, fill_value=float('-inf'))

        assert h_s.size(0) == h_t.size(0), 'Encountered unequal batch-sizes'
        (B, N_s, F), N_t, R = h_s.size(), h_t.size(1), self.psi_2.in_channels

        if self.k < 1:
            # ------ Dense variant ------ #
            S_hat = h_s @ h_t.transpose(-1, -2)  # [B, N_s, N_t, F]
            S_mask = s_mask.view(B, N_s, 1) & t_mask.view(B, 1, N_t)
            S_0 = masked_softmax(S_hat, S_mask, dim=-1)[s_mask]

            for _ in range(self.num_steps):
                S = masked_softmax(S_hat, S_mask, dim=-1)
                r_s = torch.randn((B, N_s, R), dtype=h_s.dtype,
                                  device=h_s.device)
                r_t = S.transpose(-1, -2) @ r_s

                r_s, r_t = to_sparse(r_s, s_mask), to_sparse(r_t, t_mask)
                o_s = self.psi_2(r_s, edge_index_s, edge_attr_s)
                o_t = self.psi_2(r_t, edge_index_t, edge_attr_t)
                o_s, o_t = to_dense(o_s, s_mask), to_dense(o_t, t_mask)

                D = o_s.view(B, N_s, 1, R) - o_t.view(B, 1, N_t, R)
                S_hat = S_hat + self.mlp(D).squeeze(-1).masked_fill(~S_mask, 0)

            S_L = masked_softmax(S_hat, S_mask, dim=-1)[s_mask]

            return S_0, S_L
        else:
            # ------ Sparse variant ------ #
            S_idx = self.__top_k__(h_s, h_t)  # [B, N_s, k]

            if self.training and y is not None:
                S_idx = self.__append_gt__(S_idx, s_mask, y)

            tmp_s = h_s.view(B, N_s, 1, F)
            idx = S_idx.view(B, N_s * self.k, 1).expand(-1, -1, F)
            tmp_t = torch.gather(h_t.view(B, N_t, F), -2, idx)
            S_hat = (tmp_s * tmp_t.view(B, N_s, self.k, F)).sum(dim=-1)
            S_0 = S_hat.softmax(dim=-1)[s_mask]

            for _ in range(self.num_steps):
                S = S_hat.softmax(dim=-1)
                r_s = torch.randn((B, N_s, R), dtype=h_s.dtype,
                                  device=h_s.device)

                tmp_t = r_s.view(B, N_s, 1, R) * S.view(B, N_s, self.k, 1)
                tmp_t = tmp_t.view(B, N_s * self.k, R)
                idx = S_idx.view(B, N_s * self.k, 1)
                r_t = scatter_add(tmp_t, idx, dim=1, dim_size=N_t)

                r_s, r_t = to_sparse(r_s, s_mask), to_sparse(r_t, t_mask)
                o_s = self.psi_2(r_s, edge_index_s, edge_attr_s)
                o_t = self.psi_2(r_t, edge_index_t, edge_attr_t)
                o_s, o_t = to_dense(o_s, s_mask), to_dense(o_t, t_mask)

                o_t = o_t.view(B, 1, N_t, R).expand(-1, N_s, -1, -1)
                idx = S_idx.view(B, N_s, self.k, 1).expand(-1, -1, -1, R)
                D = o_s.view(B, N_s, 1, R) - torch.gather(o_t, -2, idx)
                S_hat = S_hat + self.mlp(D).squeeze(-1)

            S_L = S_hat.softmax(dim=-1)[s_mask]
            S_idx = S_idx[s_mask]

            # Convert sparse layout to `torch.sparse_coo_tensor`.
            row = torch.arange(x_s.size(0), device=S_idx.device)
            row = row.view(-1, 1).repeat(1, self.k)
            idx = torch.stack([row.view(-1), S_idx.view(-1)], dim=0)
            size = torch.Size([x_s.size(0), N_t])

            S_sparse_0 = torch.sparse_coo_tensor(
                idx, S_0.view(-1), size, requires_grad=S_0.requires_grad)
            S_sparse_0.__idx__ = S_idx
            S_sparse_0.__val__ = S_0

            S_sparse_L = torch.sparse_coo_tensor(
                idx, S_L.view(-1), size, requires_grad=S_L.requires_grad)
            S_sparse_L.__idx__ = S_idx
            S_sparse_L.__val__ = S_L

            return S_sparse_0, S_sparse_L

    def loss(self, S, y):
        r"""Computes the negative log-likelihood loss on the correspondence
        matrix.

        Args:
            S (Tensor): Sparse or dense correspondence matrix of shape
                :obj:`[batch_size * num_nodes, num_nodes]`.
            y (LongTensor): Ground-truth matchings of shape
                :obj:`[2, num_ground_truths]`.
        """
        if not S.is_sparse:
            val = S[y[0], y[1]]
        else:
            assert S.__idx__ is not None and S.__val__ is not None
            mask = S.__idx__[y[0]] == y[1].view(-1, 1)
            val = S.__val__[[y[0]]][mask]
        return -torch.log(val + EPS).mean()

    def acc(self, S, y):
        r"""Computes the accuracy of correspondence predictions.

        Args:
            S (Tensor): Sparse or dense correspondence matrix of shape
                :obj:`[batch_size * num_nodes, num_nodes]`.
            y (LongTensor): Ground-truth matchings of shape
                :obj:`[2, num_ground_truths]`.
        """
        if not S.is_sparse:
            pred = S[y[0]].argmax(dim=-1)
        else:
            assert S.__idx__ is not None and S.__val__ is not None
            pred = S.__idx__[y[0], S.__val__[y[0]].argmax(dim=-1)]

        correct = (pred == y[1]).sum().item()
        return correct / y.size(1)

    def hits_at(self, k, S, y):
        r"""Computes the Hits@k of correspondence predictions.

        Args:
            k (int): The top-k predictions to consider.
            S (Tensor): Sparse or dense correspondence matrix of shape
                :obj:`[batch_size * num_nodes, num_nodes]`.
            y (LongTensor): Ground-truth matchings of shape
                :obj:`[2, num_ground_truths]`.
        """
        if not S.is_sparse:
            pred = S[y[0]].argsort(dim=-1, descending=True)[:, :k]
        else:
            assert S.__idx__ is not None and S.__val__ is not None
            perm = S.__val__[y[0]].argsort(dim=-1, descending=True)[:, :k]
            pred = torch.gather(S.__idx__[y[0]], -1, perm)

        correct = (pred == y[1].view(-1, 1)).sum().item()
        return correct / y.size(1)

    def __repr__(self):
        return ('{}(\n'
                '    psi_1={},\n'
                '    psi_2={},\n'
                '    num_steps={}, k={}\n)').format(self.__class__.__name__,
                                                    self.psi_1, self.psi_2,
                                                    self.num_steps, self.k)
