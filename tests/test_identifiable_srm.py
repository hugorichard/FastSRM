import os
import tempfile
from time import time
import numpy as np
import pytest
from numpy.testing import assert_array_almost_equal
from sklearn.exceptions import NotFittedError
from scipy import signal
from fastsrm.identifiable_srm import IdentifiableFastSRM
from fastsrm.fastsrm import create_temp_dir, safe_load
from picard import picard
from fastsrm.utils import (
    generate_data,
    apply_aggregate,
    apply_input_format,
    to_path,
)
from fastsrm.utils import align_basis
import matplotlib.pyplot as plt

n_voxels = 500
n_subjects = 5
n_components = 3  # number of components used for SRM model
n_timeframes = [100, 101]


def generate_decorr_friendly_data(
    n_voxels,
    n_timeframes,
    n_subjects,
    datadir,
    noise_level=0.1,
    input_format="array",
):
    n_sessions = len(n_timeframes)
    cumsum_timeframes = np.cumsum([0] + n_timeframes)
    slices_timeframes = [
        slice(cumsum_timeframes[i], cumsum_timeframes[i + 1])
        for i in range(n_sessions)
    ]

    W = [
        np.linalg.svd(
            np.random.rand(n_voxels, n_components), full_matrices=False
        )[0].T
        for _ in range(n_subjects)
    ]

    Ss = []
    for j in range(n_sessions):
        Sj = np.random.rand(n_components, n_timeframes[j])
        Sj = Sj - np.mean(Sj, axis=1, keepdims=True)
        Sj = np.linalg.svd(Sj)[0].T.dot(Sj)
        Ss.append(Sj)

    # Generate fake data
    X = []
    for subject in range(n_subjects):
        X_ = []
        for session in range(n_sessions):
            S_s = Ss[session]
            data = W[subject].T.dot(S_s)
            X_.append(data)
        X.append(X_)

    # create paths such that paths[i, j] contains data
    # of subject i during session j
    if datadir is not None:
        paths = to_path(X, datadir)

    if input_format == "array":
        return paths, W, Ss

    elif input_format == "list_of_list":
        return X, W, Ss

    elif input_format == "list_of_array":
        return (
            [
                np.concatenate([X[i][j].T for j in range(n_sessions)]).T
                for i in range(n_subjects)
            ],
            W,
            Ss,
        )
    else:
        raise ValueError("Wrong input_format")


def generate_ica_friendly_data(
    n_voxels,
    n_timeframes,
    n_subjects,
    datadir,
    noise_level=0.1,
    input_format="array",
):
    n_sessions = len(n_timeframes)
    cumsum_timeframes = np.cumsum([0] + n_timeframes)
    slices_timeframes = [
        slice(cumsum_timeframes[i], cumsum_timeframes[i + 1])
        for i in range(n_sessions)
    ]

    time = np.linspace(0, 24, n_voxels * n_subjects)

    w1 = np.sin(2 * time)  # Signal 1 : sinusoidal signal
    w2 = np.sign(np.sin(3 * time))  # Signal 2 : square signal
    w3 = signal.sawtooth(2 * np.pi * time)  # Signal 3: saw tooth signal

    W = np.c_[w1, w2, w3]
    W += 0.1 * np.random.normal(size=W.shape)  # Add noise

    W = np.split(W, n_subjects, axis=0)
    W = [w - np.mean(w, axis=0, keepdims=True) for w in W]

    def ortho(Wi):
        K, U, W_i = picard(Wi.T, centering=False)
        a = np.sqrt(W_i.dot(W_i.T)[0, 0])
        W_i = W_i / a
        return W_i.T

    W = [ortho(W[i]).T for i in range(n_subjects)]
    S = np.random.multivariate_normal(
        np.zeros(3), np.diag([5, 2, 1]), size=(int(np.sum(n_timeframes)))
    ).T
    S = S - np.mean(S, axis=1, keepdims=True)

    # Generate fake data
    X = []
    for subject in range(n_subjects):
        Q = W[subject].T
        X_ = []
        for session in range(n_sessions):
            S_s = S[:, slices_timeframes[session]]
            S_s = S_s - np.mean(S_s, axis=1, keepdims=True)
            data = Q.dot(S_s)
            noise = np.random.rand(*data.shape)
            noise = noise - np.mean(noise, axis=1, keepdims=True)
            noise = noise - np.mean(noise, axis=0, keepdims=True)
            data = data + noise_level * noise
            X_.append(data)
        X.append(X_)

    # create paths such that paths[i, j] contains data
    # of subject i during session j
    if datadir is not None:
        paths = to_path(X, datadir)

    S = [
        (S[:, s] - np.mean(S[:, s], axis=1, keepdims=True))
        for s in slices_timeframes
    ]

    if input_format == "array":
        return paths, W, S

    elif input_format == "list_of_list":
        return X, W, S

    elif input_format == "list_of_array":
        return (
            [
                np.concatenate([X[i][j].T for j in range(n_sessions)]).T
                for i in range(n_subjects)
            ],
            W,
            S,
        )
    else:
        raise ValueError("Wrong input_format")


def test_voxelcentered():
    X = [np.random.rand(n_voxels, n_timeframes[0]) for _ in range(3)]
    srm = IdentifiableFastSRM(identifiability="ica")
    with pytest.raises(
        ValueError,
        match=(
            "Input data should be voxel-centered when identifiability = ica"
        ),
    ):
        srm.fit(X)


@pytest.mark.parametrize("identifiability", ("ica", "decorr", None))
def test_fastsrm_class(identifiability):
    n_jobs = 1
    with tempfile.TemporaryDirectory() as datadir:
        np.random.seed(0)

        np.random.seed(0)
        paths, W, S = generate_ica_friendly_data(
            n_voxels, n_timeframes, n_subjects, datadir, 0
        )

        atlas = np.arange(1, n_voxels + 1)
        srm = IdentifiableFastSRM(
            identifiability=identifiability,
            n_subjects_ica=n_subjects,
            atlas=atlas,
            n_components=n_components,
            n_iter=10,
            temp_dir=datadir,
            low_ram=True,
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
        assert os.path.exists(srm.temp_dir)
        srm.clean()
        assert not os.path.exists(srm.temp_dir)


@pytest.mark.parametrize(
    "input_format, low_ram, tempdir, atlas, n_jobs, n_timeframes, aggregate",
    [
        # ("array", True, True, None, 1, [25, 25], "mean"),
        (
            "list_of_list",
            False,
            False,
            np.arange(1, n_voxels + 1),
            1,
            [25, 24],
            None,
        ),
        # ("list_of_list", False, False, np.arange(1, n_voxels + 1), 2, [25, 25
        #                                                                ], None),
        # ("list_of_array", True, True, np.eye(n_voxels), 2, [25, 24], None),
        # ("list_of_array", True, True, np.eye(n_voxels), 2, [25, 25], None),
        # ("list_of_array", True, True, np.eye(n_voxels), 1, [25, 25], None),
        # ("list_of_array", False, True, None, 2, [25, 24], "mean")
    ],
)
@pytest.mark.parametrize("identifiability", ("ica", "decorr", None))
def test_fastsrm_class_correctness(
    input_format,
    low_ram,
    tempdir,
    atlas,
    n_jobs,
    n_timeframes,
    aggregate,
    identifiability,
):
    with tempfile.TemporaryDirectory() as datadir:
        np.random.seed(0)
        X, W, S = generate_ica_friendly_data(
            n_voxels, n_timeframes, n_subjects, datadir, 0, input_format
        )

        XX, n_sessions = apply_input_format(X, input_format)

        if tempdir:
            temp_dir = datadir
        else:
            temp_dir = None

        srm = IdentifiableFastSRM(
            identifiability=identifiability,
            n_subjects_ica=n_subjects,
            atlas=atlas,
            n_components=n_components,
            n_iter=1000,
            tol=1e-7,
            temp_dir=temp_dir,
            low_ram=low_ram,
            n_jobs=n_jobs,
            aggregate=aggregate,
        )

        # Check that there is no difference between fit_transform
        # and fit then transform
        shared_response_fittransform = apply_aggregate(
            srm.fit_transform(X), aggregate, input_format
        )
        prev_basis = srm.basis_list
        # we need to align both basis though...
        srm.fit(X)
        srm.basis_list = align_basis(srm.basis_list, prev_basis)

        basis = [safe_load(b) for b in srm.basis_list]
        shared_response_raw = srm.transform(X)
        shared_response = apply_aggregate(
            shared_response_raw, aggregate, input_format
        )

        for j in range(n_sessions):
            assert_array_almost_equal(
                shared_response_fittransform[j], shared_response[j], 1
            )

        # Check that the decomposition works
        for i in range(n_subjects):
            for j in range(n_sessions):
                assert_array_almost_equal(
                    shared_response[j].T.dot(basis[i]), XX[i][j].T
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
                shared_response_partial[j], shared_response[j]
            )

        # Check that if we perform add 2 times the same subject we
        # obtain the same decomposition
        srm.add_subjects(X[:1], shared_response_raw)
        assert_array_almost_equal(
            safe_load(srm.basis_list[0]), safe_load(srm.basis_list[-1])
        )


@pytest.mark.parametrize(
    "input_format, low_ram, tempdir, atlas, n_jobs, n_timeframes, aggregate",
    [
        ("array", True, True, None, 1, [25, 25], "mean"),
        (
            "list_of_list",
            False,
            False,
            np.arange(1, n_voxels + 1),
            1,
            [25, 24],
            None,
        ),
        ("list_of_array", False, True, None, 1, [25, 24], "mean"),
    ],
)
@pytest.mark.parametrize("identifiability", ("ica", "decorr", None))
def test_class_srm_inverse_transform(
    input_format,
    low_ram,
    tempdir,
    atlas,
    n_jobs,
    n_timeframes,
    aggregate,
    identifiability,
):

    with tempfile.TemporaryDirectory() as datadir:
        X, W, S = generate_ica_friendly_data(
            n_voxels, n_timeframes, n_subjects, datadir, 0, input_format
        )


        if tempdir:
            temp_dir = datadir
        else:
            temp_dir = None

        srm = IdentifiableFastSRM(
            identifiability=identifiability,
            n_subjects_ica=n_subjects,
            atlas=atlas,
            n_components=n_components,
            n_iter=10,
            temp_dir=temp_dir,
            low_ram=low_ram,
            n_jobs=n_jobs,
            aggregate=aggregate,
        )

        srm.fit(X)
        shared_response_raw = srm.transform(X)
        # Check inverse transform
        if input_format == "list_of_array":
            reconstructed_data = srm.inverse_transform(
                shared_response_raw, subjects_indexes=[0, 2]
            )
            for i, ii in enumerate([0, 2]):
                assert_array_almost_equal(reconstructed_data[i], X[ii])

            reconstructed_data = srm.inverse_transform(
                shared_response_raw, subjects_indexes=None
            )
            for i in range(len(X)):
                assert_array_almost_equal(reconstructed_data[i], X[i])
        else:
            reconstructed_data = srm.inverse_transform(
                shared_response_raw,
                sessions_indexes=[1],
                subjects_indexes=[0, 2],
            )
            for i, ii in enumerate([0, 2]):
                for j, jj in enumerate([1]):
                    assert_array_almost_equal(
                        reconstructed_data[i][j], safe_load(X[ii][jj])
                    )

            reconstructed_data = srm.inverse_transform(
                shared_response_raw,
                subjects_indexes=None,
                sessions_indexes=None,
            )

            for i in range(len(X)):
                for j in range(len(X[i])):
                    assert_array_almost_equal(
                        reconstructed_data[i][j], safe_load(X[i][j])
                    )


@pytest.mark.parametrize("tempdir", (True, False))
@pytest.mark.parametrize("identifiability", ("ica", "decorr", None))
def test_addsubs_wo_fit(tempdir, identifiability):

    with tempfile.TemporaryDirectory() as datadir:
        X, W, S = generate_data(
            n_voxels,
            [24, 25],
            n_subjects,
            n_components,
            datadir,
            0,
            "list_of_list",
        )

        if tempdir:
            temp_dir = datadir
        else:
            temp_dir = None

        srm = IdentifiableFastSRM(
            identifiability=identifiability,
            n_subjects_ica=n_subjects,
            n_components=n_components,
            n_iter=10,
            temp_dir=temp_dir,
        )

        srm.add_subjects(X, S)

        for i in range(len(W)):
            assert_array_almost_equal(safe_load(srm.basis_list[i]), W[i])


def test_recover_ica_basis():
    X, W, S = generate_ica_friendly_data(
        n_voxels, n_timeframes, n_subjects, None, 0, "list_of_list"
    )
    srm = IdentifiableFastSRM(n_components=3, identifiability="ica")
    srm.fit(X)

    srm.basis_list = align_basis(srm.basis_list, W)
    for i in range(len(srm.basis_list)):
        assert_array_almost_equal(srm.basis_list[i], W[i], 2)


def test_recover_decorr_basis():
    n_voxels = 50
    n_timeframes = [100, 101]
    n_subjects = 4

    X, W, S = generate_decorr_friendly_data(
        n_voxels, n_timeframes, n_subjects, None, 0, "list_of_list"
    )
    srm = IdentifiableFastSRM(
        n_components=3, identifiability="decorr", n_iter=10, aggregate="mean"
    )
    S_pred = srm.fit_transform(X)

    S_pred_full = np.concatenate(S_pred, axis=1)
    S_full = np.concatenate(S, axis=1)

    srm.basis_list = align_basis(srm.basis_list, W)
    for i in range(len(srm.basis_list)):
        assert_array_almost_equal(srm.basis_list[i], W[i])


def test_convergence():
    n_voxels = 500
    n_timeframes = [100, 101]
    n_subjects = 4

    X = [
        [np.random.rand(n_voxels, n_t) for n_t in n_timeframes]
        for _ in range(n_subjects)
    ]

    srm = IdentifiableFastSRM(atlas = np.arange(n_voxels), n_components=3, tol=1e-9, n_iter=1000, n_iter_reduced=1000)
    srm.fit(X)
    assert srm.grads[0][-1] < 1e-5
    assert srm.grads[1][-1] < 1e-5

    tot_loss = np.concatenate([srm.losses[0], srm.losses[1]])
    diff_tot_loss = tot_loss[:-1] - tot_loss[1:]

    assert np.prod(diff_tot_loss > 0) == 1


def test_memory():
    n_voxels = 500
    n_timeframes = [100, 101]
    n_subjects = 4

    with tempfile.TemporaryDirectory() as datadir:
        X, W, S = generate_ica_friendly_data(
            n_voxels, n_timeframes, n_subjects, datadir, 0
        )

        dts = []
        for (low_ram, tempdir, n_jobs, aggregate, identifiability) in [
            (True, True,1, "mean", "decorr"),
            (False, False, 2, None, "ica"),
            (True, True, 1, "mean", None),
            (False, False, 1, None, None),
        ]:
            if tempdir:
                temp_dir = datadir
            else:
                temp_dir = None

            srm = IdentifiableFastSRM(
                identifiability=identifiability,
                n_subjects_ica=n_subjects,
                atlas=np.arange(n_voxels),
                n_components=n_components,
                n_iter=100,
                n_iter_reduced=100,
                temp_dir=temp_dir,
                low_ram=low_ram,
                tol=0,
                n_jobs=n_jobs,
                aggregate=aggregate,
                memory=datadir + "/memory"
            )
            t0 = time()
            srm.fit(X)
            t1 = time()

            dts.append(t1 - t0)

            shared_response_raw = srm.transform(X)

            # Check inverse transform
            reconstructed_data = srm.inverse_transform(
                shared_response_raw,
                sessions_indexes=[1],
                subjects_indexes=[0, 2],
            )
            for i, ii in enumerate([0, 2]):
                for j, jj in enumerate([1]):
                    assert_array_almost_equal(
                        reconstructed_data[i][j], safe_load(X[ii][jj])
                    )

            reconstructed_data = srm.inverse_transform(
                shared_response_raw,
                subjects_indexes=None,
                sessions_indexes=None,
            )

            for i in range(len(X)):
                for j in range(len(X[i])):
                    assert_array_almost_equal(
                        reconstructed_data[i][j], safe_load(X[i][j])
                    )

            srm.clean()

    for i in range(len(dts) -1):
        assert dts[0] > dts[i+1]

def test_ncomponents():
    X_train = [np.random.rand(100, 20) for _ in range(3)]
    X_test = [np.random.rand(100, 5) for _ in range(3)]

    srm = IdentifiableFastSRM(n_components=10, verbose=False)
    srm.fit(X_train)
    srm.transform(X_test)
