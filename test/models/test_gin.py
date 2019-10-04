import torch
from dgmc.models import GIN


def test_gin():
    conv = GIN(16, 32, num_layers=2, batch_norm=True, cat=True, lin=True)
    assert conv.__repr__() == ('GIN(16, 32, num_layers=2, batch_norm=True, '
                               'cat=True, lin=True)')

    x = torch.randn(100, 16)
    edge_index = torch.randint(100, (2, 400), dtype=torch.long)
    out = conv(x, edge_index)
    assert out.size() == (100, 32)
