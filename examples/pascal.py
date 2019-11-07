import os.path as osp

import argparse
import torch
from torch_geometric.datasets import PascalVOCKeypoints
import torch_geometric.transforms as T
from torch_geometric.data import DataLoader

from dgmc.utils import PairDataset
from dgmc.models import DGMC, SplineCNN

parser = argparse.ArgumentParser()
parser.add_argument('--category', type=str, required=True)
parser.add_argument('--cartesian', type=str, default='True')
parser.add_argument('--dim', type=int, default=128)
parser.add_argument('--rnd_dim', type=int, default=32)
parser.add_argument('--num_layers', type=int, default=2)
parser.add_argument('--num_steps', type=int, default=10)
parser.add_argument('--lr', type=float, default=0.001)
parser.add_argument('--batch_size', type=int, default=512)
parser.add_argument('--epochs', type=int, default=200)
args = parser.parse_args()
args.cartesian = True if args.cartesian == 'True' else False

pre_filter = lambda data: data.pos.size(0) >= 3  # noqa
transform = T.Compose([
    T.Delaunay(),
    T.FaceToEdge(),
    T.Cartesian() if args.cartesian else T.Distance(),
])

path = osp.join(osp.dirname(osp.realpath(__file__)), '..', 'data', 'PascalVOC')
train_dataset = PascalVOCKeypoints(path, args.category, train=True,
                                   transform=transform, pre_filter=pre_filter)
test_dataset = PascalVOCKeypoints(path, args.category, train=False,
                                  transform=transform, pre_filter=pre_filter)

train_pair_dataset = PairDataset(train_dataset, train_dataset, sample=True)
test_pair_dataset = PairDataset(test_dataset, train_dataset, sample=False)

train_loader = DataLoader(train_pair_dataset, args.batch_size, shuffle=True,
                          follow_batch=['x_s', 'x_t'])
test_loader = DataLoader(test_pair_dataset, args.batch_size, shuffle=False,
                         follow_batch=['x_s', 'x_t'])

device = 'cuda' if torch.cuda.is_available() else 'cpu'
psi_1 = SplineCNN(train_dataset.num_node_features, args.dim,
                  train_dataset.num_edge_features, args.num_layers, cat=False,
                  dropout=0.5)
psi_2 = SplineCNN(args.rnd_dim, args.rnd_dim, train_dataset.num_edge_features,
                  args.num_layers, cat=True, lin=False, dropout=0.0)
model = DGMC(psi_1, psi_2, num_steps=args.num_steps).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)


def generate_y(y_col):
    y_row = torch.arange(y_col.size(0), device=device)
    return torch.stack([y_row, y_col], dim=0)


def train():
    model.train()

    total_loss = 0
    for data in train_loader:
        optimizer.zero_grad()
        data = data.to(device)
        S_0, S_L = model(data.x_s, data.edge_index_s, data.edge_attr_s,
                         data.x_s_batch, data.x_t, data.edge_index_t,
                         data.edge_attr_t, data.x_t_batch)
        y = generate_y(data.y)
        loss = model.loss(S_0, y)
        loss = model.loss(S_L, y) + loss if model.num_steps > 0 else loss
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * (data.x_s_batch.max().item() + 1)

    return total_loss / len(train_loader.dataset)


@torch.no_grad()
def test():
    model.eval()

    correct = num_examples = 0
    for data in test_loader:
        data = data.to(device)
        S_0, S_L = model(data.x_s, data.edge_index_s, data.edge_attr_s,
                         data.x_s_batch, data.x_t, data.edge_index_t,
                         data.edge_attr_t, data.x_t_batch)
        y = generate_y(data.y)
        correct += model.acc(S_L, y, reduction='sum')
        num_examples += y.size(1)

    return correct / num_examples


model.num_steps = 0
for epoch in range(1, args.epochs + 1):
    if epoch == 51:
        model.num_steps = args.num_steps
    loss = train()
    if epoch % 25 == 0:
        acc = 100 * test()
        print('Epoch: {:03d}, Loss: {:.4f}, Acc: {:.4f}'.format(
            epoch, loss, acc))
