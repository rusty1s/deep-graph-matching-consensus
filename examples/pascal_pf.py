import os.path as osp
import random

import argparse
import torch
from torch_geometric.data import Data, DataLoader
import torch_geometric.transforms as T
from torch_geometric.datasets import PascalPF

from dgmc.models import DGMC, SplineCNN

parser = argparse.ArgumentParser()
parser.add_argument('--dim', type=int, default=128)
parser.add_argument('--rnd_dim', type=int, default=32)
parser.add_argument('--num_layers', type=int, default=2)
parser.add_argument('--num_steps', type=int, default=10)
parser.add_argument('--lr', type=float, default=0.001)
parser.add_argument('--batch_size', type=int, default=64)
parser.add_argument('--epochs', type=int, default=200)
args = parser.parse_args()


class RandomGraphDataset(torch.utils.data.Dataset):
    def __init__(self, min_inliers, max_inliers, min_outliers, max_outliers,
                 min_scale=0.9, max_scale=1.2, noise=0.05, transform=None):

        self.min_inliers = min_inliers
        self.max_inliers = max_inliers
        self.min_outliers = min_outliers
        self.max_outliers = max_outliers
        self.min_scale = min_scale
        self.max_scale = max_scale
        self.noise = noise
        self.transform = transform

    def __len__(self):
        return 1024 * 64

    def __getitem__(self, idx):
        num_inliers = random.randint(self.min_inliers, self.max_inliers)
        num_outliers = random.randint(self.min_outliers, self.max_outliers)

        pos_s = 2 * torch.rand((num_inliers, 2)) - 1
        pos_t = pos_s + self.noise * torch.randn_like(pos_s)

        y_s = torch.arange(pos_s.size(0))
        y_t = torch.arange(pos_t.size(0))

        pos_s = torch.cat([pos_s, 3 - torch.rand((num_outliers, 2))], dim=0)
        pos_t = torch.cat([pos_t, 3 - torch.rand((num_outliers, 2))], dim=0)

        data_s = Data(pos=pos_s, y_index=y_s)
        data_t = Data(pos=pos_t, y=y_t)

        if self.transform is not None:
            data_s = self.transform(data_s)
            data_t = self.transform(data_t)

        data = Data(num_nodes=pos_s.size(0))
        for key in data_s.keys:
            data['{}_s'.format(key)] = data_s[key]
        for key in data_t.keys:
            data['{}_t'.format(key)] = data_t[key]

        return data


transform = T.Compose([
    T.Constant(),
    T.KNNGraph(k=8),
    T.Cartesian(),
])
train_dataset = RandomGraphDataset(30, 60, 0, 20, transform=transform)
train_loader = DataLoader(train_dataset, args.batch_size, shuffle=True,
                          follow_batch=['x_s', 'x_t'])

device = 'cuda' if torch.cuda.is_available() else 'cpu'
psi_1 = SplineCNN(1, args.dim, 2, args.num_layers, cat=False, dropout=0.0)
psi_2 = SplineCNN(args.rnd_dim, args.rnd_dim, 2, args.num_layers, cat=True,
                  lin=False, dropout=0.0)
model = DGMC(psi_1, psi_2, num_steps=args.num_steps).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)


def train():
    model.train()

    total_loss = 0
    total_examples = total_correct = 0
    for i, data in enumerate(train_loader):
        optimizer.zero_grad()
        data = data.to(device)
        S_0, S_L = model(data.x_s, data.edge_index_s, data.edge_attr_s,
                         data.x_s_batch, data.x_t, data.edge_index_t,
                         data.edge_attr_t, data.x_t_batch)
        y = torch.stack([data.y_index_s, data.y_t], dim=0)
        pred = S_L[y[0]].argmax(dim=1).eq(y[1])
        total_correct += pred.sum().item()
        total_examples += pred.numel()
        loss = model.loss(S_0, y)
        loss = model.loss(S_L, y) + loss if model.num_steps > 0 else loss
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        if (i + 1) % 10 == 0:
            print(i + 1, len(train_loader), total_loss / 10,
                  total_correct / total_examples)
            total_loss = 0


@torch.no_grad()
def test(category):
    model.eval()

    path = osp.join(osp.dirname(osp.realpath(__file__)), '..', 'data', 'PF')
    test_dataset = PascalPF(path, category, transform=transform)

    correct = num_examples = 0
    for pair in test_dataset.pairs:
        data_s, data_t = test_dataset[pair[0]], test_dataset[pair[1]]
        data_s, data_t = data_s.to(device), data_t.to(device)
        S_0, S_L = model(data_s.x, data_s.edge_index, data_s.edge_attr, None,
                         data_t.x, data_t.edge_index, data_t.edge_attr, None)
        y = torch.arange(data_s.num_nodes, device=device)
        y = torch.stack([y, y], dim=0)
        correct += model.acc(S_L, y, reduction='sum')
        num_examples += y.size(1)
    print('Acc', test_dataset.category, correct / num_examples)
    return correct / num_examples, len(test_dataset.pairs)


train()
for category in PascalPF.categories:
    test(category)
