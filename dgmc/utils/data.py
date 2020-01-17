import re
from itertools import chain

import torch
import random
from torch_geometric.data import Data


class PairData(Data):  # pragma: no cover
    def __inc__(self, key, value):
        if bool(re.search('index_s', key)):
            return self.x_s.size(0)
        if bool(re.search('index_t', key)):
            return self.x_t.size(0)
        else:
            return 0


class PairDataset(torch.utils.data.Dataset):
    r"""Combines two datasets, a source dataset and a target dataset, by
    building pairs between separate dataset examples.

    Args:
        dataset_s (torch.utils.data.Dataset): The source dataset.
        dataset_t (torch.utils.data.Dataset): The target dataset.
        sample (bool, optional): If set to :obj:`True`, will sample exactly
            one target example for every source example instead of holding the
            product of all source and target examples. (default: :obj:`False`)
    """
    def __init__(self, dataset_s, dataset_t, sample=False):
        self.dataset_s = dataset_s
        self.dataset_t = dataset_t
        self.sample = sample

    def __len__(self):
        return len(self.dataset_s) if self.sample else len(
            self.dataset_s) * len(self.dataset_t)

    def __getitem__(self, idx):
        if self.sample:
            data_s = self.dataset_s[idx]
            data_t = self.dataset_t[random.randint(0, len(self.dataset_t) - 1)]
        else:
            data_s = self.dataset_s[idx // len(self.dataset_t)]
            data_t = self.dataset_t[idx % len(self.dataset_t)]

        return PairData(
            x_s=data_s.x,
            edge_index_s=data_s.edge_index,
            edge_attr_s=data_s.edge_attr,
            x_t=data_t.x,
            edge_index_t=data_t.edge_index,
            edge_attr_t=data_t.edge_attr,
            num_nodes=None,
        )

    def __repr__(self):
        return '{}({}, {}, sample={})'.format(self.__class__.__name__,
                                              self.dataset_s, self.dataset_t,
                                              self.sample)


class ValidPairDataset(torch.utils.data.Dataset):
    r"""Combines two datasets, a source dataset and a target dataset, by
    building valid pairs between separate dataset examples.
    A pair is valid if each node class in the source graph also exists in the
    target graph.

    Args:
        dataset_s (torch.utils.data.Dataset): The source dataset.
        dataset_t (torch.utils.data.Dataset): The target dataset.
        sample (bool, optional): If set to :obj:`True`, will sample exactly
            one target example for every source example instead of holding the
            product of all source and target examples. (default: :obj:`False`)
    """
    def __init__(self, dataset_s, dataset_t, sample=False):
        self.dataset_s = dataset_s
        self.dataset_t = dataset_t
        self.sample = sample
        self.pairs, self.cumdeg = self.__compute_pairs__()

    def __compute_pairs__(self):
        num_classes = 0
        for data in chain(self.dataset_s, self.dataset_t):
            num_classes = max(num_classes, data.y.max().item() + 1)

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
        return len(self.dataset_s) if self.sample else len(self.pairs)

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

        return PairData(
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
        return '{}({}, {}, sample={})'.format(self.__class__.__name__,
                                              self.dataset_s, self.dataset_t,
                                              self.sample)
