import torch
from torch_geometric.data import Data
from dgmc.utils import PairDataset


def test_pair_dataset():
    x = torch.randn(10, 16)
    edge_index = torch.randint(x.size(0), (2, 30), dtype=torch.long)
    y = torch.randperm(x.size(0))
    data = Data(x=x, edge_index=edge_index, y=y)

    dataset = PairDataset([data, data], [data, data], sample=True)
    assert dataset.__repr__() == (
        'PairDataset([Data(edge_index=[2, 30], x=[10, 16], y=[10]), '
        'Data(edge_index=[2, 30], x=[10, 16], y=[10])], ['
        'Data(edge_index=[2, 30], x=[10, 16], y=[10]), '
        'Data(edge_index=[2, 30], x=[10, 16], y=[10])], sample=True)')
    assert len(dataset) == 2
    pair = dataset[0]
    assert len(pair) == 5
    assert torch.allclose(pair.x_s, x)
    assert pair.edge_index_s.tolist() == edge_index.tolist()
    assert torch.allclose(pair.x_t, x)
    assert pair.edge_index_t.tolist() == edge_index.tolist()
    assert pair.y.tolist() == torch.arange(x.size(0)).tolist()

    dataset = PairDataset([data, data], [data, data], sample=False)
    assert dataset.__repr__() == (
        'PairDataset([Data(edge_index=[2, 30], x=[10, 16], y=[10]), '
        'Data(edge_index=[2, 30], x=[10, 16], y=[10])], ['
        'Data(edge_index=[2, 30], x=[10, 16], y=[10]), '
        'Data(edge_index=[2, 30], x=[10, 16], y=[10])], sample=False)')
    assert len(dataset) == 4
    pair = dataset[0]
    assert len(pair) == 5
    assert torch.allclose(pair.x_s, x)
    assert pair.edge_index_s.tolist() == edge_index.tolist()
    assert torch.allclose(pair.x_t, x)
    assert pair.edge_index_t.tolist() == edge_index.tolist()
    assert pair.y.tolist() == torch.arange(x.size(0)).tolist()
