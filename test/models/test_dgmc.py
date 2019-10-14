import torch
from torch_geometric.datasets import KarateClub
from torch_geometric.data import Batch
from dgmc.models import DGMC, GIN

data = KarateClub()[0]
N = data.num_nodes
psi_1 = GIN(data.num_node_features, 16, num_layers=2)
psi_2 = GIN(8, 8, num_layers=2, batch_norm=False)


def set_seed():
    torch.manual_seed(12345)


def test_dgmc_repr():
    model = DGMC(psi_1, psi_2, num_steps=1)
    assert model.__repr__() == (
        'DGMC(\n'
        '    psi_1=GIN(34, 16, num_layers=2, batch_norm=True, cat=True, '
        'lin=True),\n'
        '    psi_2=GIN(8, 8, num_layers=2, batch_norm=False, cat=True, '
        'lin=True),\n'
        '    num_steps=1, k=-1\n)')
    model.reset_parameters()


def test_dgmc_on_single_graphs():
    set_seed()
    model = DGMC(psi_1, psi_2, num_steps=1)
    x, e = data.x, data.edge_index
    y = torch.stack([torch.arange(N), torch.arange(N)], dim=0)

    set_seed()
    S1_0, S1_L = model(x, e, None, None, x, e, None, None)
    loss1 = model.loss(S1_0, y)
    loss1.backward()
    acc1 = model.acc(S1_0, y)
    hits1_1 = model.hits_at(1, S1_0, y)
    hits1_10 = model.hits_at(10, S1_0, y)
    hits1_all = model.hits_at(data.num_nodes, S1_0, y)

    set_seed()
    model.k = data.num_nodes  # Test a sparse "dense" variant.
    y = torch.stack([torch.arange(N), torch.arange(N)], dim=0)
    S2_0, S2_L = model(x, e, None, None, x, e, None, None, y)
    loss2 = model.loss(S2_0, y)
    loss2.backward()
    acc2 = model.acc(S2_0, y)
    hits2_1 = model.hits_at(1, S2_0, y)
    hits2_10 = model.hits_at(10, S2_0, y)
    hits2_all = model.hits_at(data.num_nodes, S2_0, y)

    assert S1_0.size() == (data.num_nodes, data.num_nodes)
    assert S1_L.size() == (data.num_nodes, data.num_nodes)
    assert torch.allclose(S1_0, S2_0.to_dense())
    assert torch.allclose(S1_L, S2_L.to_dense())
    assert torch.allclose(loss1, loss2)
    assert acc1 == acc2 == hits1_1 == hits2_1
    assert hits1_1 <= hits1_10 == hits2_10 <= hits1_all
    assert hits1_all == hits2_all == 1.0


def test_dgmc_on_multiple_graphs():
    set_seed()
    model = DGMC(psi_1, psi_2, num_steps=1)

    batch = Batch.from_data_list([data, data])
    x, e, b = batch.x, batch.edge_index, batch.batch

    set_seed()
    S1_0, S1_L = model(x, e, None, b, x, e, None, b)
    assert S1_0.size() == (batch.num_nodes, data.num_nodes)
    assert S1_L.size() == (batch.num_nodes, data.num_nodes)

    set_seed()
    model.k = data.num_nodes  # Test a sparse "dense" variant.
    S2_0, S2_L = model(x, e, None, b, x, e, None, b)

    assert torch.allclose(S1_0, S2_0.to_dense())
    assert torch.allclose(S1_L, S2_L.to_dense())


def test_dgmc_append_gt():
    model = DGMC(psi_1, psi_2, num_steps=1)

    S_idx = torch.tensor([[[0, 1], [1, 2]], [[1, 2], [0, 1]]])
    s_mask = torch.tensor([[True, False], [True, True]])
    y = torch.tensor([[0, 1], [0, 0]])

    S_idx = model.__append_gt__(S_idx, s_mask, y)
    assert S_idx.tolist() == [[[0, 1], [1, 2]], [[1, 0], [0, 1]]]
