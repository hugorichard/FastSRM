from fastsrm.srm import projection, detsrm, probsrm
import numpy as np
import scipy
from fastsrm.utils import error_source_rotation, error_source


dim = 10, 10, 10
m, v, k, n = 10, np.prod(dim), 2, 300

rng = np.random.RandomState(0)
Sigma = rng.dirichlet(np.ones(k), 1).flatten()
print(Sigma)
S = np.sqrt(Sigma)[:, None] * rng.randn(k, n)
W = np.array([projection(rng.randn(v, k)) for i in range(m)])
sigmas = 0.1 * rng.rand(m)
N = np.array([np.sqrt(sigmas[i]) * rng.randn(v, n) for i in range(m)])
X = np.array([W[i].dot(S) + N[i] for i in range(m)])


def test_error_source_rotation():
    A = np.random.laplace(0, 1, size=(2, 300))
    W = projection(np.random.randn(2, 2))
    error_source_rotation(W.dot(A), A) < 0.1


def test_detsrm():
    W_, S_ = detsrm(X, k, n_iter=10, random_state=0)
    assert error_source_rotation(S_, S) < 0.2


def test_probsrm():
    W_, S_ = probsrm(X, k, n_iter=10, random_state=0)[:2]
    assert np.sum(error_source(S_, S)) < 0.1



