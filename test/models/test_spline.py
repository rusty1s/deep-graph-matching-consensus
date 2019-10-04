from itertools import product

import torch
from dgmc.models import SplineCNN


def test_spline():
    model = SplineCNN(16, 32, dim=3, num_layers=2, cat=True, lin=True,
                      dropout=0.5)
    assert model.__repr__() == ('SplineCNN(16, 32, dim=3, num_layers=2, '
                                'cat=True, lin=True, dropout=0.5)')

    x = torch.randn(100, 16)
    edge_index = torch.randint(100, (2, 400), dtype=torch.long)
    edge_attr = torch.rand((400, 3))
    for cat, lin in product([False, True], [False, True]):
        model = SplineCNN(16, 32, 3, 2, cat, lin, 0.5)
        out = model(x, edge_index, edge_attr)
        assert out.size() == (100, 16 + 2 * 32 if not lin and cat else 32)
        assert out.size() == (100, model.out_channels)
