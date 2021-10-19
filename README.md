# FastSRM

[![CircleCI](https://circleci.com/gh/hugorichard/FastSRM.svg?style=svg)](https://circleci.com/gh/hugorichard/FastSRM)

Implementation of FastSRM algorithms.


The model of probabilistic SRM is given by:

<img src="https://latex.codecogs.com/svg.image?X_i&space;=&space;W_i&space;\mathbf{s}&space;&plus;&space;\mathbf{n}_i&space;" title="X_i = W_i \mathbf{s} + \mathbf{n}_i " />

where 
* <img src="https://latex.codecogs.com/svg.image?X_i&space;\in&space;\mathbb{R}^{v,&space;n}&space;" title="X_i \in \mathbb{R}^{v, n} " /> is the data of subject <img src="https://latex.codecogs.com/svg.image?i&space;" title="i " />
* <img src="https://latex.codecogs.com/svg.image?W_i&space;\in&space;\mathbb{R}^{v&space;\times&space;k}&space;" title="W_i \in \mathbb{R}^{v \times k} " /> is the basis of subject <img src="https://latex.codecogs.com/svg.image?i&space;" title="i " />
* <img src="https://latex.codecogs.com/svg.image?S&space;\in&space;\mathbb{R}^{k&space;\times&space;n}" title="S \in \mathbb{R}^{k \times n}" /> is the shared response assumed to be sampled from a centered Gaussian with covariance <img src="https://latex.codecogs.com/svg.image?\Sigma&space;\in&space;\mathbb{R}^{k&space;\times&space;k}" title="\Sigma \in \mathbb{R}^{k \times k}" />
* <img src="https://latex.codecogs.com/svg.image?\mathbf{n}_i" title="\mathbf{n}_i" /> is  the noise in subject <img src="https://latex.codecogs.com/svg.image?i&space;" title="i " /> assumed to be sampled from a centered Gaussian with covariance <img src="https://latex.codecogs.com/svg.image?\sigma_i&space;I" title="\sigma_i I" /> where <img src="https://latex.codecogs.com/svg.image?I&space;\in&space;\mathbb{R}^{v,&space;v}" title="I \in \mathbb{R}^{v, v}" /> is the identity matrix

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
