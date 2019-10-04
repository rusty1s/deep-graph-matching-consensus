import torch
from dgmc.models import RelCNN


def test_rel():
    conv = RelCNN(16, 32, num_layers=2, batch_norm=True, cat=True, lin=True,
                  dropout=0.5)
    assert conv.__repr__() == ('RelCNN(16, 32, num_layers=2, batch_norm=True, '
                               'cat=True, lin=True, dropout=0.5)')

    x = torch.randn(100, 16)
    edge_index = torch.randint(100, (2, 400), dtype=torch.long)
    out = conv(x, edge_index)
    assert out.size() == (100, 32)
