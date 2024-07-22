# FLAME-master
This repo holds the source code and scripts for reproducing the key experiments of our paper:

**"Personalized Federated Learning via ADMM with Moreau Envelope".**

Author: Shengkun Zhu, Jinshan Zeng, Sheng Wang, Yuan Sun, Zhiyong Peng.
## Datasets and Models
| Datesets | # of samples | ref. | Models |
| :----: | :----: | :----: | :----: |
Synthetic | 5k-10k/device | [Li et al.](https://proceedings.mlsys.org/paper_files/paper/2020/file/1f5fe83998a09396ebe6477d9475ba0c-Paper.pdf) | SVM
Mnist | 70,000 | [LeCun et al.](https://citeseerx.ist.psu.edu/document?repid=rep1&type=pdf&doi=4cccb7c5b2d59bc0b86914340c81b26dd4835140) | MLR
Fmnist | 70,000 | [Xiao et al.](https://arxiv.org/pdf/1708.07747.pdf) | MLP |
Mmnist | 58,954 | [Kaggle](https://www.kaggle.com/datasets/andrewmvd/medical-mnist) | CNN

## Start

The default values for various parameters parsed to the experiment are given in `options.py`. Details are given on some of those parameters:
* `framework:` two personalied federated learnign frameworks, i.e., FLAME and pFedMe.

* `num_users:` number of users.

* `q:` number of data shards of each user.

* `model:` SVM, MLP, MLR, CNN for choices.

* `dataset:` four datasets for choices.

* `strategy:` client selection strategy.

* `frac_candidates:` fraction of clients candidates, c/m in our paper.

* `frac:` fraction of clients, s/m in our paper.

* `optimizer:` type of optimizer, default sgd.

* `momentum:` sgd momentum, default 0.

* `epoches:` number of communication rounds.

* `local_ep:` number of local iterations.

* `local_bs:` local batch size.

* `lr:` learning rate.

* `mu:` hyperparameter in regularization term.

* `Lambda:` hyperparameter in Moreau envelope.

* `rho:` hyperparameter in penalty term.

* `iid:` data distribution, 0 for non iid.

* `seed:` random seed.

* `eta:` learning rate for global model in pFedMe.
