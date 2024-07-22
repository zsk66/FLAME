# FLAME-master
This repo holds the source code and scripts for reproducing the key experiments of our paper:

**"On ADMM in Heterogeneous Federated Learning: Personalization, Robustness, and Fairness".**

Authors: Shengkun Zhu, Jinshan Zeng, Sheng Wang, Yuan Sun, Xiaodong Li, Yuan Yao, Zhiyong Peng.

This repository is built based on PyTorch.

## Datasets and Models
| Datesets | # of samples | ref. | Models |
| :----: | :----: | :----: | :----: |
Mnist | 70,000 | [LeCun et al.](https://citeseerx.ist.psu.edu/document?repid=rep1&type=pdf&doi=4cccb7c5b2d59bc0b86914340c81b26dd4835140) | MLP
Fmnist | 70,000 | [Xiao et al.](https://arxiv.org/pdf/1708.07747.pdf) | MLP |
Mmnist | 58,954 | [Kaggle](https://www.kaggle.com/datasets/andrewmvd/medical-mnist) | CNN1
Cifar10 | 60,000 | [Krizhevsky et al.](http://www.cs.utoronto.ca/~kriz/learning-features-2009-TR.pdf) | CNN
Femnist | 382,705 | [Leaf](https://leaf.cmu.edu/) | CNN2
## Start

The default values for various parameters parsed to the experiment are given in `options.py`. Details are given on some of those parameters:
* `framework:` five personalized federated learning frameworks.

* `partition:` six data partitioning schemes.

* `num_users:` number of users.

* `q:` number of data shards of each user.

* `model:` SVM, MLP, MLR, CNN for choices.

* `dataset:` four datasets for choices.

* `strategy:` client selection strategy.

* `frac_candidates:` fraction of clients candidates, c/m in our paper.

* `frac:` fraction of clients, s/m in our paper.

* `optimizer:` type of optimizer, default sgd.

* `momentum:` sgd momentum, default 0.

* `epochs:` number of communication rounds.

* `local_ep:` the number of local iterations.

* `local_bs:` local batch size.

* `lr:` learning rate.

* `mu:` hyperparameter in regularization term.

* `Lambda:` hyperparameter in Moreau envelope.

* `rho:` hyperparameter in penalty term.

* `iid:` data distribution, 0 for non-iid.

* `seed:` random seed.

* `eta:` learning rate for the global model in pFedMe.

* `eta2:` learning rate for the global model in ditto.
