import copy
import os.path as osp

import argparse
import torch
import torch_geometric.transforms as T
from torch_geometric.datasets import WILLOWObjectClass as WILLOW
from torch_geometric.datasets import PascalVOCKeypoints as PascalVOC
from torch_geometric.data import DataLoader

from dgmc.utils import ValidPairDataset, PairDataset
from dgmc.models import DGMC, SplineCNN

parser = argparse.ArgumentParser()
parser.add_argument('--isotropic', action='store_true')
parser.add_argument('--dim', type=int, default=256)
parser.add_argument('--rnd_dim', type=int, default=128)
parser.add_argument('--num_layers', type=int, default=2)
parser.add_argument('--num_steps', type=int, default=10)
parser.add_argument('--lr', type=float, default=0.001)
parser.add_argument('--batch_size', type=int, default=512)
parser.add_argument('--pre_epochs', type=int, default=15)
parser.add_argument('--epochs', type=int, default=15)
parser.add_argument('--runs', type=int, default=20)
parser.add_argument('--test_samples', type=int, default=100)
args = parser.parse_args()

pre_filter1 = lambda d: d.num_nodes > 0  # noqa
pre_filter2 = lambda d: d.num_nodes > 0 and d.name[:4] != '2007'  # noqa

transform = T.Compose([
    T.Delaunay(),
    T.FaceToEdge(),
    T.Distance() if args.isotropic else T.Cartesian(),
])

path = osp.join('..', 'data', 'PascalVOC-WILLOW')
pretrain_datasets = []
for category in PascalVOC.categories:
    dataset = PascalVOC(
        path, category, train=True, transform=transform, pre_filter=pre_filter2
        if category in ['car', 'motorbike'] else pre_filter1)
    pretrain_datasets += [ValidPairDataset(dataset, dataset, sample=True)]
pretrain_dataset = torch.utils.data.ConcatDataset(pretrain_datasets)
pretrain_loader = DataLoader(pretrain_dataset, args.batch_size, shuffle=True,
                             follow_batch=['x_s', 'x_t'])

path = osp.join('..', 'data', 'WILLOW')
datasets = [WILLOW(path, cat, transform) for cat in WILLOW.categories]

device = 'cuda' if torch.cuda.is_available() else 'cpu'
psi_1 = SplineCNN(dataset.num_node_features, args.dim,
                  dataset.num_edge_features, args.num_layers, cat=False,
                  dropout=0.5)
psi_2 = SplineCNN(args.rnd_dim, args.rnd_dim, dataset.num_edge_features,
                  args.num_layers, cat=True, dropout=0.0)
model = DGMC(psi_1, psi_2, num_steps=args.num_steps).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)


def generate_voc_y(y_col):
    y_row = torch.arange(y_col.size(0), device=device)
    return torch.stack([y_row, y_col], dim=0)


def pretrain():
    model.train()

    total_loss = 0
    for data in pretrain_loader:
        optimizer.zero_grad()
        data = data.to(device)
        S_0, S_L = model(data.x_s, data.edge_index_s, data.edge_attr_s,
                         data.x_s_batch, data.x_t, data.edge_index_t,
                         data.edge_attr_t, data.x_t_batch)
        y = generate_voc_y(data.y)
        loss = model.loss(S_0, y)
        loss = model.loss(S_L, y) + loss if model.num_steps > 0 else loss
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * (data.x_s_batch.max().item() + 1)

    return total_loss / len(pretrain_loader.dataset)


print('Pretraining model on PascalVOC...')
for epoch in range(1, args.pre_epochs + 1):
    loss = pretrain()
    print(f'Epoch: {epoch:02d}, Loss: {loss:.4f}')
state_dict = copy.deepcopy(model.state_dict())
print('Done!')


def generate_y(num_nodes, batch_size):
    row = torch.arange(num_nodes * batch_size, device=device)
    col = row[:num_nodes].view(1, -1).repeat(batch_size, 1).view(-1)
    return torch.stack([row, col], dim=0)


def train(train_loader, optimizer):
    model.train()

    total_loss = 0
    for data in train_loader:
        optimizer.zero_grad()
        data = data.to(device)
        S_0, S_L = model(data.x_s, data.edge_index_s, data.edge_attr_s,
                         data.x_s_batch, data.x_t, data.edge_index_t,
                         data.edge_attr_t, data.x_t_batch)
        num_graphs = data.x_s_batch.max().item() + 1
        y = generate_y(num_nodes=10, batch_size=num_graphs)
        loss = model.loss(S_0, y)
        loss = model.loss(S_L, y) + loss if model.num_steps > 0 else loss
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * num_graphs

    return total_loss / len(train_loader.dataset)


@torch.no_grad()
def test(test_dataset):
    model.eval()

    test_loader1 = DataLoader(test_dataset, args.batch_size, shuffle=True)
    test_loader2 = DataLoader(test_dataset, args.batch_size, shuffle=True)

    correct = num_examples = 0
    while (num_examples < args.test_samples):
        for data_s, data_t in zip(test_loader1, test_loader2):
            data_s, data_t = data_s.to(device), data_t.to(device)
            _, S_L = model(data_s.x, data_s.edge_index, data_s.edge_attr,
                           data_s.batch, data_t.x, data_t.edge_index,
                           data_t.edge_attr, data_t.batch)
            y = generate_y(num_nodes=10, batch_size=data_t.num_graphs)
            correct += model.acc(S_L, y, reduction='sum')
            num_examples += y.size(1)

            if num_examples >= args.test_samples:
                return correct / num_examples


def run(i, datasets):
    datasets = [dataset.shuffle() for dataset in datasets]
    train_datasets = [dataset[:20] for dataset in datasets]
    test_datasets = [dataset[20:] for dataset in datasets]
    train_datasets = [
        PairDataset(train_dataset, train_dataset, sample=False)
        for train_dataset in train_datasets
    ]
    train_dataset = torch.utils.data.ConcatDataset(train_datasets)
    train_loader = DataLoader(train_dataset, args.batch_size, shuffle=True,
                              follow_batch=['x_s', 'x_t'])

    model.load_state_dict(state_dict)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    for epoch in range(1, 1 + args.epochs):
        train(train_loader, optimizer)

    accs = [100 * test(test_dataset) for test_dataset in test_datasets]

    print(f'Run {i:02d}:')
    print(' '.join([category.ljust(13) for category in WILLOW.categories]))
    print(' '.join([f'{acc:.2f}'.ljust(13) for acc in accs]))

    return accs


accs = [run(i, datasets) for i in range(1, 1 + args.runs)]
print('-' * 14 * 5)
accs, stds = torch.tensor(accs).mean(dim=0), torch.tensor(accs).std(dim=0)
print(' '.join([category.ljust(13) for category in WILLOW.categories]))
print(' '.join([f'{a:.2f} ± {s:.2f}'.ljust(13) for a, s in zip(accs, stds)]))
