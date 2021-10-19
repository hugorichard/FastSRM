# FastSRM

[![CircleCI](https://circleci.com/gh/hugorichard/FastSRM.svg?style=svg)](https://circleci.com/gh/hugorichard/FastSRM)

Standalone implementation of FastSRM.

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
S = srm.fit_transform(X) # Shared response np array of shape (n_components, n_timeframes)
W = srm.basis_list # Shared response np array of shape (n_components, n_timeframes)
Sigma = srm.source_covariance # Covariance of the shared response
sigmas = srm.noise_variance # Variance of the noise
```
Documentation
--------------

https://hugorichard.github.io/FastSRM/
