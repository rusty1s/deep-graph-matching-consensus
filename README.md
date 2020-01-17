[build-image]: https://travis-ci.org/rusty1s/deep-graph-matching-consensus.svg?branch=master
[build-url]: https://travis-ci.org/rusty1s/deep-graph-matching-consensus
[docs-image]: https://readthedocs.org/projects/deep-graph-matching-consensus/badge/?version=latest
[docs-url]: https://deep-graph-matching-consensus.readthedocs.io/en/latest/?badge=latest
[coverage-image]: https://codecov.io/gh/rusty1s/deep-graph-matching-consensus/branch/master/graph/badge.svg
[coverage-url]: https://codecov.io/github/rusty1s/deep-graph-matching-consensus?branch=master

<h1 align="center">Deep Graph Matching Consensus</h1>

<img width="100%" src="https://raw.githubusercontent.com/rusty1s/deep-graph-matching-consensus/master/figures/overview.png?token=ABU7ZATGTDHG6DPI3ANYBUS5TXPOQ" />

--------------------------------------------------------------------------------

[![Build Status][build-image]][build-url]
[![Docs Status][docs-image]][docs-url]
[![Code Coverage][coverage-image]][coverage-url]

**[Documentation](https://deep-graph-matching-consensus.readthedocs.io)**

This is a PyTorch implementation of **Deep Graph Matching Consensus**, as described in our paper:

Matthias Fey, Jan E. Lenssen, Christopher Morris, Jonathan Masci, Nils M. Kriege: [Deep Graph Matching Consensus](https://openreview.net/forum?id=HyeJf1HKvS) *(ICLR 2020)*

## Requirements

* **[PyTorch](https://pytorch.org/get-started/locally/)** (>=1.2.0)
* **[PyTorch Geometric](https://github.com/rusty1s/pytorch_geometric)** (master)
* **[KeOps](https://github.com/getkeops/keops)** (>=1.1.0)

## Installation

```
$ python setup.py install
```

Head over to our [documentation](https://deep-graph-matching-consensus.readthedocs.io) for a detailed overview of the `DGMC` module.

## Running examples

We provide training and evaluation procedures for the [PascalVOC with Berkely annotations](https://pytorch-geometric.readthedocs.io/en/latest/modules/datasets.html#torch_geometric.datasets.PascalVOCKeypoints), the [WILLOW-ObjectClass](https://pytorch-geometric.readthedocs.io/en/latest/modules/datasets.html#torch_geometric.datasets.WILLOWObjectClass) dataset, the [PascalPF](https://pytorch-geometric.readthedocs.io/en/latest/modules/datasets.html#torch_geometric.datasets.PascalPF) dataset, and the [DBP15K](https://pytorch-geometric.readthedocs.io/en/latest/modules/datasets.html#torch_geometric.datasets.DBP15K) dataset.
Experiments can be run via:

```
$ cd examples/
$ python pascal.py
$ python willow.py
$ python pascal_pf.py
$ python dbp15k.py --category=zh_en
```

<p align="center">
  <img height="250px" src="https://raw.githubusercontent.com/rusty1s/deep-graph-matching-consensus/master/figures/best_motorbike.png?token=ABU7ZAX225LWQP4BLVMFOXK5TXPRO" />
  <img height="250px" src="https://raw.githubusercontent.com/rusty1s/deep-graph-matching-consensus/master/figures/best_car.png?token=ABU7ZAS4IBROSQUHHM6JRPS5TXRMG" />
  <img height="250px" src="https://raw.githubusercontent.com/rusty1s/deep-graph-matching-consensus/master/figures/best_duck.png?token=ABU7ZASBUISFO6UKAVIEAWC5TXRK6" />
  <img height="250px" src="https://raw.githubusercontent.com/rusty1s/deep-graph-matching-consensus/master/figures/worst_duck.png?token=ABU7ZAX6JXQS4NPYBK4Q32C5TXRNE" />
</p>

## Cite

Please cite [our paper](https://openreview.net/forum?id=HyeJf1HKvS) if you use this code in your own work:

```
@inproceedings{Fey/etal/2020,
  title={Deep Graph Matching Consensus},
  author={Fey, M. and Lenssen, J. E. and Morris, C. and Masci, J. and Kriege, N. M.},
  booktitle={International Conference on Learning Representations (ICLR)},
  year={2020},
}
```

## Running tests

```
$ python setup.py test
```
