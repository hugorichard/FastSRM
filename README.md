# FastSRM

[![CircleCI](https://circleci.com/gh/hugorichard/FastSRM.svg?style=svg)](https://circleci.com/gh/hugorichard/FastSRM)

Standalone implementation of FastSRM.

See https://arxiv.org/pdf/1909.12537.pdf

Install
---------

`pip install fastsrm`

Usage
--------

When the input data is a list of m subjects containing arrays of shape (n_voxels, n_timeframes) with n_timeframes >> n_voxels.
```python
# Input data X: neuroimaging data 
# X is a np array of shape (n_subjects, n_sessions)
# X[i, j] is a path to a np array of shape (n_voxels, n_timeframes)
from 
srm = IdentifiableSRM(


```


Documentation
--------------

https://hugorichard.github.io/FastSRM/
