import torch
from dgmc.models import DGMC, GIN

from torch_geometric.datasets import KarateClub

data = KarateClub()[0]
x, e, N = data.x, data.edge_index, data.num_nodes
psi_1 = GIN(data.num_node_features, 16, num_layers=2)
psi_2 = GIN(8, 8, num_layers=2, batch_norm=False)


def test_dgmc():
    model = DGMC(psi_1, psi_2, num_steps=1)
    assert model.__repr__() == (
        'DGMC(\n'
        '    psi_1=GIN(34, 16, num_layers=2, batch_norm=True, cat=True, '
        'lin=True),\n'
        '    psi_2=GIN(8, 8, num_layers=2, batch_norm=False, cat=True, '
        'lin=True),\n'
        '    num_steps=1, k=-1\n)')

    torch.manual_seed(12345)
    S1_0, S1_L = model(x, e, None, None, x, e, None, None)
    model.k = data.num_nodes  # Test a sparse "dense" variant.
    torch.manual_seed(12345)
    y = torch.stack([torch.arange(N), torch.arange(N)], dim=0)
    S2_0, S2_L, S2_idx = model(x, e, None, None, x, e, None, None, y)

    assert torch.allclose(torch.gather(S1_0, -1, S2_idx), S2_0)
    assert torch.allclose(torch.gather(S1_L, -1, S2_idx), S2_L)


def test_append_gt():
    model = DGMC(psi_1, psi_2, num_steps=1)

    S_idx = torch.tensor([[[0, 1], [1, 2]], [[1, 2], [0, 1]]])
    s_mask = torch.tensor([[True, False], [True, True]])
    y = torch.tensor([[0, 1], [0, 0]])

    S_idx = model.__append_gt__(S_idx, s_mask, y)
    assert S_idx.tolist() == [[[0, 1], [1, 2]], [[1, 0], [0, 1]]]
