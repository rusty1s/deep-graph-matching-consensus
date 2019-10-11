<h1 align="center">Deep Graph Matching Consensus</h1>

<img width="100%" src="https://raw.githubusercontent.com/rusty1s/deep-graph-matching-consensus/master/figures/overview.png?token=ABU7ZATGTDHG6DPI3ANYBUS5TXPOQ" />

--------------------------------------------------------------------------------

This is a PyTorch implementation of **Deep Graph Matching Consensus**, as described in our paper:

Matthias Fey, Jan E. Lenssen, Christopher Morris, Jonathan Masci, Nils M. Kriege: [Deep Graph Matching Consensus](https://arxiv.org/abs/) *(CoRR 2019)*

## Installation

```
$ python setup.py install
```

## Requirements

* **[PyTorch](https://pytorch.org/get-started/locally/)** (>=1.2.0)
* **[PyTorch Geometric](https://github.com/rusty1s/pytorch_geometric)** (>=1.3.2)
* **[KeOps](https://github.com/getkeops/keops)** (>=1.1.0)

## Running examples

We provide training and evaluation procedures for the [WILLOW-ObjectClass](https://pytorch-geometric.readthedocs.io/en/latest/modules/datasets.html#torch_geometric.datasets.WILLOWObjectClass) dataset, the [PascalVOC with Berkely annotations](https://pytorch-geometric.readthedocs.io/en/latest/modules/datasets.html#torch_geometric.datasets.PascalVOCKeypoints) dataset, and the [DBP15K](https://pytorch-geometric.readthedocs.io/en/latest/modules/datasets.html#torch_geometric.datasets.DBP15K) dataset, *e.g.*:

```
$ cd examples/
$ python willow.py --category=Duck
$ python pascal.py --category=Aeroplane
$ python dbp15k.py --category=zh_en
```

<p align="center">
  <img height="250px" src="https://raw.githubusercontent.com/rusty1s/deep-graph-matching-consensus/master/figures/best_motorbike.png?token=ABU7ZAX225LWQP4BLVMFOXK5TXPRO" />
  <img height="250px" src="https://raw.githubusercontent.com/rusty1s/deep-graph-matching-consensus/master/figures/best_car.png?token=ABU7ZAS4IBROSQUHHM6JRPS5TXRMG" />
  <img height="250px" src="https://raw.githubusercontent.com/rusty1s/deep-graph-matching-consensus/master/figures/best_duck.png?token=ABU7ZASBUISFO6UKAVIEAWC5TXRK6" />
  <img height="250px" src="https://raw.githubusercontent.com/rusty1s/deep-graph-matching-consensus/master/figures/worst_duck.png?token=ABU7ZAX6JXQS4NPYBK4Q32C5TXRNE" />
</p>

## Cite

Please cite [our paper](https://arxiv.org/abs/) if you use this code in your own work:

```
@article{Fey/etal/2019,
  title={Deep Graph Matching Consensus},
  author={Fey, M. and Lenssen, J. E. and Morris, C. and Masci, J. and Kriege, N. M.},
  journal={CoRR},
  volume={abs/},
  year={2019},
}
```

## Running tests

```
$ python setup.py test
```
