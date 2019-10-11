import torch
import random
from torch_geometric.data import Dataset, Data


class MyData(Data):
    def __inc__(self, key, value):
        if key == 'edge_index_s':
            return self.x_s.size(0)
        elif key == 'edge_index_t':
            return self.x_t.size(0)
        else:
            return 0


class PairDataset(Dataset):
    def __init__(self, dataset_s, dataset_t, sample=False):
        self.dataset_s = dataset_s
        self.dataset_t = dataset_t
        self.sample = sample
        super(PairDataset, self).__init__('/tmp/')
        self.pairs, self.cumdeg = self.__compute_pairs__()

    def _download(self):
        pass

    def _process(self):
        pass

    def __compute_pairs__(self):
        num_classes = max(self.dataset_s.data.y.max().item() + 1,
                          self.dataset_t.data.y.max().item() + 1)

        y_s = torch.zeros((len(self.dataset_s), num_classes), dtype=torch.bool)
        y_t = torch.zeros((len(self.dataset_t), num_classes), dtype=torch.bool)

        for i, data in enumerate(self.dataset_s):
            y_s[i, data.y] = 1
        for i, data in enumerate(self.dataset_t):
            y_t[i, data.y] = 1

        y_s = y_s.view(len(self.dataset_s), 1, num_classes)
        y_t = y_t.view(1, len(self.dataset_t), num_classes)

        pairs = ((y_s * y_t).sum(dim=-1) == y_s.sum(dim=-1)).nonzero()
        cumdeg = pairs[:, 0].bincount().cumsum(dim=0)

        return pairs.tolist(), [0] + cumdeg.tolist()

    def __len__(self):
        return len(self.cumdeg) - 1 if self.sample else len(self.pairs)

    def __getitem__(self, idx):
        if self.sample:
            data_s = self.dataset_s[idx]
            i = random.randint(self.cumdeg[idx], self.cumdeg[idx + 1] - 1)
            data_t = self.dataset_t[self.pairs[i][1]]
        else:
            data_s = self.dataset_s[self.pairs[idx][0]]
            data_t = self.dataset_t[self.pairs[idx][1]]

        y = data_s.y.new_full((data_t.y.max().item() + 1, ), -1)
        y[data_t.y] = torch.arange(data_t.num_nodes)
        y = y[data_s.y]

        return MyData(
            x_s=data_s.x,
            edge_index_s=data_s.edge_index,
            edge_attr_s=data_s.edge_attr,
            x_t=data_t.x,
            edge_index_t=data_t.edge_index,
            edge_attr_t=data_t.edge_attr,
            y=y,
            num_nodes=None,
        )

    def __repr__(self):
        return '{}({}, {})'.format(self.__class__.__name__, self.dataset_s,
                                   self.dataset_t)
