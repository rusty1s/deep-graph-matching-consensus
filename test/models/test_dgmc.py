import torch
from dgmc.models import DGMC, GIN

from torch_geometric.datasets import KarateClub


def test_dgmc():
    data = KarateClub()[0]
    x, e = data.x, data.edge_index

    psi_1 = GIN(data.num_node_features, 16, num_layers=2)

    psi_2 = GIN(8, 8, num_layers=2, batch_norm=False)

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
    model.k = data.num_nodes
    torch.manual_seed(12345)
    S2_0, S2_L, S2_idx = model(x, e, None, None, x, e, None, None)

    assert torch.allclose(torch.gather(S1_0, -1, S2_idx), S2_0)
    assert torch.allclose(torch.gather(S1_L, -1, S2_idx), S2_L)
