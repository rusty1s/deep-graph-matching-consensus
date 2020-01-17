import torch
from dgmc.models import MLP


def test_mlp():
    model = MLP(16, 32, num_layers=2, batch_norm=True, dropout=0.5)
    assert model.__repr__() == ('MLP(16, 32, num_layers=2, batch_norm=True'
                                ', dropout=0.5)')

    x = torch.randn(100, 16)
    out = model(x)
    assert out.size() == (100, 32)
