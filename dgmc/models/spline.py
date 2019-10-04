import torch
from torch.nn import Linear as Lin
import torch.nn.functional as F
from torch_geometric.nn import SplineConv


class SplineCNN(torch.nn.Module):
    def __init__(self, in_channels, out_channels, dim, num_layers, cat=True,
                 lin=True, dropout=0.0):
        super(SplineCNN, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.dim = dim
        self.num_layers = num_layers
        self.cat = cat
        self.lin = lin
        self.dropout = dropout

        self.convs = torch.nn.ModuleList()
        for _ in range(num_layers):
            conv = SplineConv(in_channels, out_channels, dim, kernel_size=5)
            self.convs.append(conv)
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

    def forward(self, x, edge_index, edge_attr):
        xs = [x]
        for conv in self.convs:
            xs += [F.relu(conv(xs[-1], edge_index, edge_attr))]

        x = torch.cat(xs, dim=-1) if self.cat else xs[-1]
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.final(x) if self.lin else x
        return x
