# FastSRM

[![CircleCI](https://circleci.com/gh/hugorichard/FastSRM.svg?style=svg)](https://circleci.com/gh/hugorichard/FastSRM)

Implementation of FastSRM algorithms.


The model of probabilistic SRM is given by:


<!-- <img src="https://latex.codecogs.com/svg.image?\forall i \in \{1, \dots, n}, \enspace X_i = W_i S + n_i" /> -->

<img src="https://latex.codecogs.com/svg.image?\begin{align*}&\forall&space;i&space;\in&space;\{1,&space;\dots,&space;n\},&space;\enspace&space;X_i&space;=&space;W_i&space;\mathbf{s}&space;&plus;&space;\mathbf{n}_i&space;\\&\text{where&space;}&space;\mathbf{n}_i&space;\sim&space;\mathcal{N}(0,&space;\sigma&space;I)&space;\text{&space;and&space;}&space;\\&\mathbf{s}&space;\sim&space;\mathcal{N}(0,&space;\mathrm{diag}(\Sigma))\end{align*}&space;" title="\begin{align*}&\forall i \in \{1, \dots, n\}, \enspace X_i = W_i \mathbf{s} + \mathbf{n}_i \\&\text{where } \mathbf{n}_i \sim \mathcal{N}(0, \sigma I) \text{ and } \\&\mathbf{s} \sim \mathcal{N}(0, \mathrm{diag}(\Sigma))\end{align*} " />

```math
\forall i \in \{1, \dots, n}, \enspace X_i = W_i S + n_i
```
where 
* `X_i \in \mathbb{R}^{v \times n}` is the data of subject `i`
* `W_i \in \mathbb{R}^{v \times k}` is the basis of subject `i`
* `S \in \mathbb{R}^{k \times n}` is the shared response assumed to be sampled from a centered Gaussian with covariance `\Sigma \in \mathbb{R}^{k \times k}`
* `n_i` is  the subject specific noise assumed to be sampled from a centered Gaussian with covariance `\sigma_i I` where `I \in \mathbb{R}^{v, v}` is the identity matrix

When the number of features in X_i is 


See https://arxiv.org/pdf/1909.12537.pdf

Install
---------

`pip install fastsrm`

Usage
--------
```python
# Input data X: neuroimaging data 
# X is a np array of shape (n_subjects, n_sessions)
# X[i, j] is a path to a np array of shape (n_voxels, n_timeframes)
from fastsrm.identifiable_srm import IdentifiableFastSRM
srm = IdentifiableFastSRM(n_components=5, temp_dir="./", n_jobs=5)
S = srm.fit_transform(X) # Shared response: np array of shape (n_components, n_timeframes)
W = srm.basis_list # Shared response: np array of shape (n_components, n_timeframes)
Sigma = srm.source_covariance # (Diagonal) Covariance of the shared response: np array of shape (n_components,)
sigmas = srm.noise_variance # Variance of the noise: np array of shape (n_subjects)

```
Documentation
--------------

https://hugorichard.github.io/FastSRM/
