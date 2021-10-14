import tempfile

import numpy as np
import pytest
from numpy.testing import assert_allclose

from fastsrm.fastsrm import fastsrm, safe_load, svd_reduce
from fastsrm.srm import detsrm, probsrm
from fastsrm.utils import error_source, generate_data

n_voxels = 10
n_subjects = 5
n_components = 3  # number of components used for SRM model


@pytest.mark.parametrize("method", ["prob", "det"])
def test_equivalence_fastsrm_srm(method):
    X = np.array([np.random.randn(1000, 100) for _ in range(10)])
    if method == "prob":
        W, S, sigmas, Sigmas = probsrm(X, 20, random_state=0)
    else:
        W, S = detsrm(X, 20, random_state=0)
    W2, S2 = fastsrm([[x] for x in X], 20, random_state=0, method=method)[:2]
    assert_allclose(S, S2)
    assert_allclose(W, W2)


def apply_input_format(X, input_format):
    if input_format == "array":
        n_sessions = len(X[0])
        XX = [
            [np.load(X[i, j]) for j in range(len(X[i]))] for i in range(len(X))
        ]
    elif input_format == "list_of_array":
        XX = [[x] for x in X]
        n_sessions = 1
    else:
        XX = X
        n_sessions = len(X[0])
    return XX, n_sessions


@pytest.mark.parametrize(
    "input_format, tempdir, n_jobs, n_timeframes, method",
    [
        ("array", True, 1, [50], "prob"),
        ("list_of_list", False, 1, [25, 24], "det",),
        ("list_of_list", False, 2, [25, 25], "prob",),
        ("list_of_array", True, 2, [25, 24], "prob"),
        ("list_of_array", False, 2, [25, 25], "det"),
        ("list_of_array", True, 1, [25, 25], "det"),
        ("list_of_array", True, 2, [25, 24], "prob"),
    ],
)
def test_fastsrm_correctness(
    input_format, tempdir, n_jobs, n_timeframes, method
):
    n_voxels = 1000
    n_components = 2
    with tempfile.TemporaryDirectory() as datadir:
        np.random.seed(0)
        X, W, S = generate_data(
            n_voxels,
            n_timeframes,
            n_subjects,
            n_components,
            datadir,
            0.001,
            input_format,
        )
        XX, n_sessions = apply_input_format(X, input_format)

        if tempdir:
            temp_dir = datadir
        else:
            temp_dir = None

        W, S_ = fastsrm(
            X,
            n_components=n_components,
            verbose=True,
            n_iter=200,
            random_state=0,
        )[:2]

        # Check that the decomposition works
        for i in range(n_subjects):
            assert (
                np.sum(error_source(S_, W[i].T.dot(np.column_stack(XX[i]))))
                < 0.05
            )

def test_svd_reduce():
    n_subjects = 4
    n_timeframes = [10, 11]
    n_voxels = 100

    X = [
        [np.random.rand(n_voxels, n_t) for n_t in n_timeframes]
        for _ in range(n_subjects)
    ]
    XX = [np.concatenate(X[i], axis=1) for i in range(n_subjects)]

    X_red_svds = []
    for i in range(n_subjects):
        U, S, V = np.linalg.svd(XX[i])
        X_red_svds.append(S.reshape(-1, 1) * V)
    X_red = svd_reduce(X, 1)

    for i in range(n_subjects):
        np.testing.assert_array_almost_equal(np.abs(X_red[i]), np.abs(X_red_svds[i]))
