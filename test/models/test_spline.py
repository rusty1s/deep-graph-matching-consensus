import torch
from dgmc.models import SplineCNN


def test_spline():
    conv = SplineCNN(16, 32, dim=3, num_layers=2, cat=True, lin=True,
                     dropout=0.5)
    assert conv.__repr__() == ('SplineCNN(16, 32, dim=3, num_layers=2, '
                               'cat=True, lin=True, dropout=0.5)')

    x = torch.randn(100, 16)
    edge_index = torch.randint(100, (2, 400), dtype=torch.long)
    edge_attr = torch.rand((400, 3))
    out = conv(x, edge_index, edge_attr)
    assert out.size() == (100, 32)
