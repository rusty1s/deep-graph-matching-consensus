from itertools import product

import torch
from dgmc.models import GIN


def test_gin():
    model = GIN(16, 32, num_layers=2, batch_norm=True, cat=True, lin=True)
    assert model.__repr__() == ('GIN(16, 32, num_layers=2, batch_norm=True, '
                                'cat=True, lin=True)')

    x = torch.randn(100, 16)
    edge_index = torch.randint(100, (2, 400), dtype=torch.long)
    for cat, lin in product([False, True], [False, True]):
        model = GIN(16, 32, 2, True, cat, lin)
        out = model(x, edge_index)
        assert out.size() == (100, 16 + 2 * 32 if not lin and cat else 32)
