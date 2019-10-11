import os.path as osp

import argparse
import torch
import torch_geometric.transforms as T
from torch_geometric.datasets import WILLOWObjectClass
from torch_geometric.data import DataLoader, Batch
from dgmc.models import DGMC, SplineCNN

parser = argparse.ArgumentParser()
parser.add_argument('--category', type=str, required=True)
parser.add_argument('--cartesian', type=str, default='True')
parser.add_argument('--dim', type=int, default=64)
parser.add_argument('--rnd_dim', type=int, default=64)
parser.add_argument('--num_layers', type=int, default=2)
parser.add_argument('--num_steps', type=int, default=20)
parser.add_argument('--lr', type=float, default=0.001)
parser.add_argument('--batch_size', type=int, default=512)
parser.add_argument('--epochs', type=int, default=200)
parser.add_argument('--runs', type=int, default=20)
args = parser.parse_args()
args.cartesian = True if args.cartesian == 'True' else False

device = 'cuda' if torch.cuda.is_available() else 'cpu'
path = osp.join(osp.dirname(osp.realpath(__file__)), '..', 'data', 'WILLOW')
dataset = WILLOWObjectClass(path, category=args.category)
dataset.transform = T.Compose([
    T.Delaunay(),
    T.FaceToEdge(),
    T.Cartesian() if args.cartesian else T.Distance(),
])

psi_1 = SplineCNN(dataset.num_node_features, args.dim,
                  dataset.num_edge_features, args.num_layers, cat=False,
                  dropout=0.5)
psi_2 = SplineCNN(args.rnd_dim, args.rnd_dim, dataset.num_edge_features,
                  args.num_layers, cat=True, dropout=0.0)
model = DGMC(psi_1, psi_2, num_steps=args.num_steps).to(device)


def generate_y(num_nodes, batch_size):
    row = torch.arange(num_nodes * batch_size, device=device)
    col = row[:num_nodes].view(1, -1).repeat(batch_size, 1).view(-1)
    return torch.stack([row, col], dim=0)


def train(train_loader, optimizer):
    model.train()

    total_loss = 0
    for data_s, data_t in zip(train_loader, train_loader):
        optimizer.zero_grad()
        data_s, data_t = data_s.to(device), data_t.to(device)
        S_0, S_L = model(data_s.x, data_s.edge_index, data_s.edge_attr,
                         data_s.batch, data_t.x, data_t.edge_index,
                         data_t.edge_attr, data_t.batch)
        y = generate_y(num_nodes=10, batch_size=data_t.num_graphs)
        loss = model.loss(S_0, y)
        loss = model.loss(S_L, y) + loss if model.num_steps > 0 else loss
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * data_t.num_graphs

    return total_loss / len(train_loader.dataset)


def test(test_dataset, train_loader):
    model.eval()

    correct = num_examples = 0
    for data_s in test_dataset:
        for data_t in train_loader:
            data_s = Batch.from_data_list([data_s] * data_t.num_graphs)
            data_s, data_t = data_s.to(device), data_t.to(device)
            with torch.no_grad():
                _, S_L = model(data_s.x, data_s.edge_index, data_s.edge_attr,
                               data_s.batch, data_t.x, data_t.edge_index,
                               data_t.edge_attr, data_t.batch)
            y = generate_y(num_nodes=10, batch_size=data_t.num_graphs)
            correct += model.acc(S_L, y, norm=False)
            num_examples += y.size(1)

    return correct / num_examples


def run(i, dataset):
    dataset = dataset.shuffle()
    train_dataset, test_dataset = dataset[:20], dataset[20:]

    train_loader = DataLoader(train_dataset, args.batch_size, shuffle=True)

    model.reset_parameters()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    for epoch in range(1, 1 + args.epochs):
        train(train_loader, optimizer)
    acc = 100 * test(test_dataset, train_loader)
    print('Run: {:02d}, Acc: {:.2f}'.format(i, acc))
    return acc


accs = [run(i, dataset) for i in range(1, 1 + args.runs)]
acc, std = torch.tensor(accs).mean().item(), torch.tensor(accs).std().item()
print('Final: {:.2f} ± {:.2f}'.format(acc, std))
