from itertools import product

import torch
from dgmc.models import RelCNN


def test_rel():
    model = RelCNN(16, 32, num_layers=2, batch_norm=True, cat=True, lin=True,
                   dropout=0.5)
    assert model.__repr__() == ('RelCNN(16, 32, num_layers=2, batch_norm=True'
                                ', cat=True, lin=True, dropout=0.5)')
    assert model.convs[0].__repr__() == 'RelConv(16, 32)'

    x = torch.randn(100, 16)
    edge_index = torch.randint(100, (2, 400), dtype=torch.long)
    for cat, lin in product([False, True], [False, True]):
        model = RelCNN(16, 32, 2, True, cat, lin, 0.5)
        out = model(x, edge_index)
        assert out.size() == (100, 16 + 2 * 32 if not lin and cat else 32)
        assert out.size() == (100, model.out_channels)
