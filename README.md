# Deep Graph Matching Consensus

<object data="https://github.com/rusty1s/deep-graph-matching-consensus/raw/master/figures/overview.pdf" type="application/pdf" width="700px" height="700px" />

--------------------------------------------------------------------------------

This is a PyTorch implementation of Deep Graph Matching Consensus, as described in our paper:

Matthias Fey, Jan E. Lenssen, Christopher Morris, Jonathan Masci, Nils M. Kriege: [Deep Graph Matching Consensus](https://arxiv.org/abs/) (CoRR 2019)

## Installation

```
$ python setup.py install
```

## Requirements

* **[PyTorch](https://pytorch.org/get-started/locally/)** (>=1.2.0)
* **[PyTorch Geometric](https://github.com/rusty1s/pytorch_geometric)** (>=1.3.2)

## Running examples

We provide training and evaluation procedures for the [WILLOW-ObjectClass]() dataset, the [PascalVOC with Berekely Annotations]() dataset, and the [DBP15K]() dataset, *e.g.*:

```
$ cd examples/
$ python willow.py --category=Duck
```



## Cite

Please cite [our paper](https://arxiv.org/abs/) if you use this code in your own work:

```
@article{Fey/etal/2019,
  title={Deep Graph Matching Consensus},
  author={Fey, Matthias and Lenssen, Jan E. and Morris, Christopher and Masci, Jonathan, Kriege, Nils M.},
  journal={CoRR},
  volume={abs/},
  year={2019},
}
```

## Running tests

```
python setup.py test
```
