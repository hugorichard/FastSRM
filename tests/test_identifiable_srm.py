import os
from glob import glob
from sklearn.base import clone
import tempfile
import numpy as np
import pytest
from numpy.testing import assert_array_almost_equal
from sklearn.exceptions import NotFittedError
from fastsrm.identifiable_srm import IdentifiableFastSRM, create_temp_dir
from fastsrm.srm import safe_load
from fastsrm.utils import (
    generate_data,
    apply_aggregate,
    apply_input_format,
)

n_voxels = 300
n_subjects = 5
n_components = 2  # number of components used for SRM model
n_timeframes = [100, 101]


def test_isrm_class():
    n_jobs = 1
    with tempfile.TemporaryDirectory() as datadir:
        np.random.seed(0)

        np.random.seed(0)
        paths, W, S = generate_data(
            n_voxels, n_timeframes, n_subjects, n_components, datadir=datadir,
        )

        srm = IdentifiableFastSRM(
            n_components=n_components,
            n_iter=10,
            temp_dir=datadir,
            n_jobs=n_jobs,
        )

        # Raises an error because model is not fitted yet
        with pytest.raises(NotFittedError):
            srm.transform(paths)

        srm.fit(paths)

        # An error can occur if temporary directory already exists
        with pytest.raises(
            ValueError,
            match=(
                "Path %s already exists. When a model "
                "is used, filesystem should be "
                r"cleaned by using the .clean\(\) "
                "method" % srm.temp_dir
            ),
        ):
            # Error can occur if the filesystem is uncleaned
            create_temp_dir(srm.temp_dir)
            create_temp_dir(srm.temp_dir)

        shared_response = srm.transform(paths)

        # Raise error when wrong index
        with pytest.raises(
            ValueError,
            match=(
                "subjects_indexes should be either "
                "a list, an array or None but "
                "received type <class 'int'>"
            ),
        ):
            srm.transform(paths, subjects_indexes=1000)

        with pytest.raises(
            ValueError,
            match=(
                "subjects_indexes should be either "
                "a list, an array or None but "
                "received type <class 'int'>"
            ),
        ):
            srm.inverse_transform(shared_response, subjects_indexes=1000)

        with pytest.raises(
            ValueError,
            match=(
                "sessions_indexes should be either "
                "a list, an array or None but "
                "received type <class 'int'>"
            ),
        ):
            srm.inverse_transform(shared_response, sessions_indexes=1000)

        with pytest.raises(
            ValueError,
            match=(
                "Input data imgs has len 5 whereas "
                "subject_indexes has len 1. "
                "The number of basis used to compute "
                "the shared response should be equal to "
                "the number of subjects in imgs"
            ),
        ):
            srm.transform(paths, subjects_indexes=[0])

        with pytest.raises(
            ValueError,
            match=(
                "Index 1 of subjects_indexes has value 8 "
                "whereas value should be between 0 and 4"
            ),
        ):
            srm.transform(paths[:2], subjects_indexes=[0, 8])

        with pytest.raises(
            ValueError,
            match=(
                "Index 1 of sessions_indexes has value 8 "
                "whereas value should be between 0 and 1"
            ),
        ):
            srm.inverse_transform(shared_response, sessions_indexes=[0, 8])

        # Check behavior of .clean
        assert os.path.exists(srm.temp_dir_)
        srm.clean()
        assert not os.path.exists(srm.temp_dir_)


@pytest.mark.parametrize(
    "input_format, tempdir, n_jobs, n_timeframes, aggregate, method",
    [
        ("array", True, 1, [25, 25], "mean", "prob"),
        ("list_of_list", False, 1, [25, 24], None, "det"),
        ("list_of_list", False, 1, [25, 25], None, "prob"),
        ("list_of_array", True, 1, [25, 24], None, "det"),
        ("list_of_array", True, 1, [25, 25], None, "det"),
        ("list_of_array", True, 1, [25, 25], None, "prob"),
        ("list_of_array", True, 2, [25, 24], "mean", "det"),
    ],
)
def test_fastsrm_class_correctness(
    input_format, tempdir, n_jobs, n_timeframes, aggregate, method
):
    with tempfile.TemporaryDirectory() as datadir:
        np.random.seed(0)
        X, W, S = generate_data(
            n_voxels,
            n_timeframes,
            n_subjects,
            n_components,
            datadir,
            1e-3,
            input_format,
        )

        XX, n_sessions = apply_input_format(X, input_format)

        if tempdir:
            temp_dir = datadir
        else:
            temp_dir = None

        srm = IdentifiableFastSRM(
            n_components=n_components,
            n_iter=1000,
            tol=1e-7,
            temp_dir=temp_dir,
            n_jobs=n_jobs,
            method=method,
            aggregate=aggregate,
        )

        # Check that there is no difference between fit_transform
        # and fit then transform
        shared_response_fittransform = apply_aggregate(
            srm.fit_transform(X), aggregate, input_format
        )
        prev_basis = srm.basis_list
        basis = [safe_load(b) for b in srm.basis_list]
        shared_response_raw = srm.transform(X)
        shared_response = apply_aggregate(
            shared_response_raw, aggregate, input_format
        )

        for j in range(n_sessions):
            assert_array_almost_equal(
                shared_response_fittransform[j], shared_response[j]
            )

        # Check that the decomposition works
        for i in range(n_subjects):
            for j in range(n_sessions):
                assert_array_almost_equal(
                    basis[i].dot(shared_response[j]), XX[i][j], 1
                )

        # Check that if we use all subjects but one if gives almost the
        # same shared response
        shared_response_partial_raw = srm.transform(
            X[1:5], subjects_indexes=list(range(1, 5))
        )

        shared_response_partial = apply_aggregate(
            shared_response_partial_raw, aggregate, input_format
        )
        for j in range(n_sessions):
            assert_array_almost_equal(
                shared_response_partial[j], shared_response[j], 1
            )

        # Check that if we add 2 times the same subject we
        # obtain the same decomposition

        srm.add_subjects(X[:1], shared_response_raw)
        assert_array_almost_equal(
            safe_load(srm.basis_list[0]), safe_load(srm.basis_list[-1]), 4
        )


@pytest.mark.parametrize(
    "input_format, tempdir, n_jobs, n_timeframes, aggregate, method",
    [
        ("array", True, 1, [25, 25], "mean", "prob"),
        ("list_of_list", False, 1, [25, 24], None, "det"),
        ("list_of_array", True, 1, [25, 24], "mean", "det"),
    ],
)
def test_class_srm_inverse_transform(
    input_format, tempdir, n_jobs, n_timeframes, aggregate, method
):

    with tempfile.TemporaryDirectory() as datadir:
        X, W, S = generate_data(
            n_voxels,
            n_timeframes,
            n_subjects,
            n_components,
            datadir,
            1e-5,
            input_format,
        )

        if tempdir:
            temp_dir = datadir
        else:
            temp_dir = None

        srm = IdentifiableFastSRM(
            n_components=n_components,
            n_iter=10,
            temp_dir=temp_dir,
            n_jobs=n_jobs,
            aggregate=aggregate,
            method=method,
        )

        srm.fit(X)
        shared_response_raw = srm.transform(X)
        # Check inverse transform
        if input_format == "list_of_array":
            reconstructed_data = srm.inverse_transform(
                shared_response_raw, subjects_indexes=[0, 2]
            )
            for i, ii in enumerate([0, 2]):
                assert_array_almost_equal(reconstructed_data[i], X[ii], 2)

            reconstructed_data = srm.inverse_transform(
                shared_response_raw, subjects_indexes=None
            )
            for i in range(len(X)):
                assert_array_almost_equal(reconstructed_data[i], X[i], 2)
        else:
            reconstructed_data = srm.inverse_transform(
                shared_response_raw,
                sessions_indexes=[1],
                subjects_indexes=[0, 2],
            )
            for i, ii in enumerate([0, 2]):
                for j, jj in enumerate([1]):
                    assert_array_almost_equal(
                        reconstructed_data[i][j], safe_load(X[ii][jj]), 2
                    )

            reconstructed_data = srm.inverse_transform(
                shared_response_raw,
                subjects_indexes=None,
                sessions_indexes=None,
            )

            for i in range(len(X)):
                for j in range(len(X[i])):
                    assert_array_almost_equal(
                        reconstructed_data[i][j], safe_load(X[i][j]), 2
                    )


@pytest.mark.parametrize("tempdir", (True, False))
@pytest.mark.parametrize("method", ("prob", "det"))
def test_addsubs_wo_fit(tempdir, method):

    with tempfile.TemporaryDirectory() as datadir:
        X, W, S = generate_data(
            n_voxels,
            [24, 25],
            n_subjects,
            n_components,
            datadir,
            1e-4,
            "list_of_list",
        )

        if tempdir:
            temp_dir = datadir
        else:
            temp_dir = None

        srm = IdentifiableFastSRM(
            n_components=n_components,
            n_iter=10,
            temp_dir=temp_dir,
            method=method,
        )

        srm.add_subjects(X, S)

        for i in range(len(W)):
            assert_array_almost_equal(safe_load(srm.basis_list[i]), W[i], 2)


def test_ncomponents():
    X_train = [np.random.rand(100, 20) for _ in range(3)]
    X_test = [np.random.rand(100, 5) for _ in range(3)]

    srm = IdentifiableFastSRM(n_components=10, verbose=False)
    srm.fit(X_train)
    srm.transform(X_test)




@pytest.mark.parametrize("method", ("prob", "det"))
def test_use_pca(method):
    for i in range(20):
        X_train = [np.random.rand(100, 10) for _ in range(3)]
        srm = IdentifiableFastSRM(
            n_components=5,
            use_pca=False,
            method=method,
        )
        A = srm.fit_transform(X_train)

        srm2 = IdentifiableFastSRM(
            n_components=5,
            use_pca=True,
            method=method
        )
        B = srm2.fit_transform(X_train)
        np.testing.assert_array_almost_equal(A, B, 4)


@pytest.mark.parametrize("method", ("prob", "det"))
def test_random_state(method):
    n_voxels = 500
    n_timeframes = 10
    n_subjects = 4
    X_train = [np.random.rand(n_voxels, n_timeframes) for _ in range(n_subjects)]
    srm = IdentifiableFastSRM(
        n_components=5,
        method=method,
        random_state=None,
    )
    srm2 = IdentifiableFastSRM(
        n_components=5,
        method=method,
        random_state=0,
    )
    srm3 = IdentifiableFastSRM(
        n_components=5,
        method=method,
        random_state=0,
    )
    S1 = srm.fit_transform(X_train)
    S2 = srm2.fit_transform(X_train)
    S3 = srm3.fit_transform(X_train)
    np.testing.assert_allclose(S2, S3)

    for S in [S2, S3]:
        with pytest.raises(AssertionError,):
            np.testing.assert_allclose(S, S1)


def test_temp_files():
    n_voxels = 500
    n_timeframes = 10
    n_subjects = 4
    X_train = [np.random.rand(n_voxels, n_timeframes) for _ in range(n_subjects)]

    X_train2 = [np.random.rand(n_voxels, n_timeframes) for _ in range(n_subjects)]
    srm = IdentifiableFastSRM(
        n_components=5,
        n_iter=1,
        random_state=None,
        temp_dir="./temp",
    )
    srm.fit(X_train)
    srm2 = clone(srm)
    srm2.fit(X_train2)
    assert srm2.basis_list[0] != srm.basis_list[0]
    assert srm2.temp_dir_ != srm.temp_dir_
    srm2.clean()
    srm.clean()
    assert len(glob("./temp/*")) == 0
