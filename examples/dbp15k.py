import os.path as osp

import argparse
import torch
from torch_geometric.datasets import DBP15K

from dgmc.models import DGMC, RelCNN

parser = argparse.ArgumentParser()
parser.add_argument('--category', type=str, required=True)
parser.add_argument('--dim', type=int, default=256)
parser.add_argument('--rnd_dim', type=int, default=32)
parser.add_argument('--num_layers', type=int, default=3)
parser.add_argument('--num_steps', type=int, default=10)
parser.add_argument('--k', type=int, default=10)
args = parser.parse_args()


class SumEmbedding(object):
    def __call__(self, data):
        data.x1, data.x2 = data.x1.sum(dim=1), data.x2.sum(dim=1)
        return data


device = 'cuda' if torch.cuda.is_available() else 'cpu'
path = osp.join('..', 'data', 'DBP15K')
data = DBP15K(path, args.category, transform=SumEmbedding())[0].to(device)

psi_1 = RelCNN(data.x1.size(-1), args.dim, args.num_layers, batch_norm=False,
               cat=True, lin=True, dropout=0.5)
psi_2 = RelCNN(args.rnd_dim, args.rnd_dim, args.num_layers, batch_norm=False,
               cat=True, lin=True, dropout=0.0)
model = DGMC(psi_1, psi_2, num_steps=None, k=args.k).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)


def train():
    model.train()
    optimizer.zero_grad()

    _, S_L = model(data.x1, data.edge_index1, None, None, data.x2,
                   data.edge_index2, None, None, data.train_y)

    loss = model.loss(S_L, data.train_y)
    loss.backward()
    optimizer.step()
    return loss


@torch.no_grad()
def test():
    model.eval()

    _, S_L = model(data.x1, data.edge_index1, None, None, data.x2,
                   data.edge_index2, None, None)

    hits1 = model.acc(S_L, data.test_y)
    hits10 = model.hits_at_k(10, S_L, data.test_y)

    return hits1, hits10


print('Optimize initial feature matching...')
model.num_steps = 0
for epoch in range(1, 201):
    if epoch == 101:
        print('Refine correspondence matrix...')
        model.num_steps = args.num_steps
        model.detach = True

    loss = train()

    if epoch % 10 == 0 or epoch > 100:
        hits1, hits10 = test()
        print((f'{epoch:03d}: Loss: {loss:.4f}, Hits@1: {hits1:.4f}, '
               f'Hits@10: {hits10:.4f}'))
