# FastSRM

[![CircleCI](https://circleci.com/gh/hugorichard/FastSRM.svg?style=svg)](https://circleci.com/gh/hugorichard/FastSRM)

Implementation of FastSRM algorithms.


The model of probabilistic SRM is given by:

<img src="https://latex.codecogs.com/svg.image?\mathbf{x}_i&space;=&space;W_i&space;\mathbf{s}&space;&plus;&space;\mathbf{n}_i" title="\mathbf{x}_i = W_i \mathbf{s} + \mathbf{n}_i" />

where 
* <img src="https://latex.codecogs.com/svg.image?\mathbf{x}_i&space;\in&space;\mathbb{R}^v" title="\mathbf{x}_i \in \mathbb{R}^v" /> is the data of subject <img src="https://latex.codecogs.com/svg.image?i&space;" title="i " />
* <img src="https://latex.codecogs.com/svg.image?W_i&space;\in&space;\mathbb{R}^{v&space;\times&space;k}&space;" title="W_i \in \mathbb{R}^{v \times k} " /> is the basis of subject <img src="https://latex.codecogs.com/svg.image?i&space;" title="i " />
* <img src="https://latex.codecogs.com/svg.image?\mathbf{s}&space;\in&space;\mathbb{R}^k" title="\mathbf{s} \in \mathbb{R}^k" /> is the shared response (or sources) assumed to be sampled from a centered Gaussian with covariance <img src="https://latex.codecogs.com/svg.image?\Sigma&space;\in&space;\mathbb{R}^{k&space;\times&space;k}" title="\Sigma \in \mathbb{R}^{k \times k}" />
* <img src="https://latex.codecogs.com/svg.image?\mathbf{n}_i&space;\in&space;\mathbb{R}^v" title="\mathbf{n}_i \in \mathbb{R}^v" /> is  the noise in subject <img src="https://latex.codecogs.com/svg.image?i&space;" title="i " /> assumed to be sampled from a centered Gaussian with covariance <img src="https://latex.codecogs.com/svg.image?\sigma_i&space;I" title="\sigma_i I" /> where <img src="https://latex.codecogs.com/svg.image?I&space;\in&space;\mathbb{R}^{v,&space;v}" title="I \in \mathbb{R}^{v, v}" /> is the identity matrix. We call <img src="https://latex.codecogs.com/svg.image?\sigma_i" title="\sigma_i" /> the noise variance of subject <img src="https://latex.codecogs.com/svg.image?i&space;" title="i " />.

In practice we observe n samples of <img src="https://latex.codecogs.com/svg.image?\mathbf{x}_i" title="\mathbf{x}_i" />. When the number of samples is much lower than the number of features v, the SRM model can be fitted efficiently. This is what this repository provides. We also assume that the covariance of the shared response is diagonal to obtain identifiability.

Install
---------
Clone the repository

`git clone https://github.com/hugorichard/FastSRM`

In the cloned directory execute

`pip install -e .`

Requirements:

```
scipy>=0.18.0
numpy>=1.12
scikit-learn>=0.23
joblib>=1.1.0
matplotlib>=2.0.0
pytest>=6.2.5
```

To run the experiments on identifiability, brainiak (https://github.com/brainiak/brainiak) and nilearn (https://nilearn.github.io/stable/index.html) are necessary.

Usage
--------
In many neuroscience datasets, the samples are split into sessions. Therefore, for each subject we will have several sessions that can be time-wise concatenated to obtain all samples.
In `IdentifiableFastSRM` 


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

Run experiments
---------------

## Synthetic experiment

1. In the `experiments` folder run 
`ipython synthetic.py`

2. Go in the `plotting` folder and run
`ipython plot_synthetic_gradients.py`

The figure is now available in the `figures` folder under the name
`synthetic_gradient.pdf`


## Timesegment matching experiment

1. Go in the experiments folder and download the sherlock dataset by running:
`ipython get_sherlock.py`

2. Go in the `experiments` folder and run 
`bash timesegment.bash` if you have access to a cluster with slurm otherwise run `bash timesegment_local.bash`

4. Go in the `plotting` folder and run
`ipython plot_timesegment.py`

The figure is now available in the `figures` folder under the name
`timesegment_matching_sherlock.pdf`


## Identifiability experiment

1. Go in the experiments folder and download the sherlock dataset by running:
`ipython get_sherlock.py`

2. Download and install Brainiak (see https://github.com/brainiak/brainiak) 

3. Go in the `experiments` folder and run 
`bash identifiability.bash` if you have access to a cluster with slurm otherwise run `bash identifiability_local.bash`

4. Go in the `plotting` folder and run
`ipython plot_identifiability.py`

The figure is now available in the `figures` folder under the name
`identifiability.pdf`

Documentation
--------------

https://hugorichard.github.io/FastSRM/
