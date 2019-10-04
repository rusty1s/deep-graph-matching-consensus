import torch
from torch.nn import Sequential as Seq, Linear as Lin, ReLU, BatchNorm1d as BN
from torch_geometric.nn import GINConv


def MLP(in_channels, out_channels, batch_norm=True):
    return Seq(
        Lin(in_channels, out_channels),
        ReLU(),
        BN(out_channels) if batch_norm else lambda x: x,
        Lin(out_channels, out_channels),
        ReLU(),
        BN(out_channels) if batch_norm else lambda x: x,
    )


class GIN(torch.nn.Module):
    def __init__(self, in_channels, out_channels, num_layers, batch_norm=True,
                 cat=True, lin=True):
        super(GIN, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_layers = num_layers
        self.batch_norm = batch_norm
        self.cat = cat
        self.lin = lin

        self.convs = torch.nn.ModuleList()
        for _ in range(num_layers):
            mlp = MLP(in_channels, out_channels, batch_norm)
            self.convs.append(GINConv(mlp, train_eps=True))
            in_channels = out_channels

        if self.cat:
            in_channels = self.in_channels + num_layers * out_channels
        else:
            in_channels = out_channels

        if self.lin:
            self.final = Lin(self.in_channels + num_layers * out_channels,
                             out_channels)

        self.reset_parameters()

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()
        if self.lin:
            self.final.reset_parameters()

    def forward(self, x, edge_index):
        xs = [x]
        for conv in self.convs:
            xs += [conv(xs[-1], edge_index)]

        x = torch.cat(xs, dim=-1) if self.cat else xs[-1]
        x = self.final(x) if self.lin else x
        return x

    def __repr__(self):
        return ('{}({}, {}, num_layers={}, batch_norm={}, cat={}, '
                'lin={})').format(self.__class__.__name__, self.in_channels,
                                  self.out_channels, self.num_layers,
                                  self.batch_norm, self.cat, self.lin)
