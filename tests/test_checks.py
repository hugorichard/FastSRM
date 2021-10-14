import os
import tempfile

import numpy as np
import pytest
from numpy.testing import assert_array_almost_equal
from sklearn.exceptions import NotFittedError

from fastsrm.check_inputs import check_imgs, check_shared_response
from fastsrm.identifiable_srm import create_temp_dir

from numpy.testing import assert_allclose
from fastsrm.fastsrm import (
    fastsrm,
    svd_reduce,
    safe_load,
)
from fastsrm.srm import probsrm, detsrm
from fastsrm.identifiable_srm import IdentifiableFastSRM


from fastsrm.utils import generate_data


def test_generated_data():
    with tempfile.TemporaryDirectory() as datadir:

        # We authorize different timeframes for different sessions
        # but they should be the same across subject
        n_voxels = 10
        n_timeframes = [25, 24]
        n_subjects = 2
        n_components = 3  # number of components used for SRM model
        n_sessions = len(n_timeframes)

        np.random.seed(0)
        paths, W, S = generate_data(
            n_voxels, n_timeframes, n_subjects, n_components, datadir
        )

        # Test if generated data has the good shape
        for subject in range(n_subjects):
            for session in range(n_sessions):
                assert np.load(paths[subject, session]).shape == (
                    n_voxels,
                    n_timeframes[session],
                )

        # Test if generated basis have good shape
        assert len(W) == n_subjects
        for w in W:
            assert w.shape == (n_voxels, n_components)

        assert len(S) == n_sessions
        for j, s in enumerate(S):
            assert s.shape == (n_components, n_timeframes[j])


empty_list_error = "%s is a list of length 0 which is not valid"
array_type_error = "%s should be of type np.ndarray but is of type %s"
array_2axis_error = "%s must have exactly 2 axes but has %i axes"


def test_check_imgs():
    with pytest.raises(
        ValueError,
        match=(
            r"Since imgs is a list, it should be a list of list "
            r"of arrays or a list "
            r"of arrays but imgs\[0\] has type <class 'str'>"
        ),
    ):
        check_imgs(["bla"])

    with pytest.raises(
        ValueError,
        match=(
            "Input imgs should either be a list or an array but "
            "has type <class 'str'>"
        ),
    ):
        check_imgs("bla")

    with pytest.raises(ValueError, match=empty_list_error % "imgs"):
        check_imgs([])

    with pytest.raises(
        ValueError,
        match=r"imgs\[1\] has type <class 'str'> whereas imgs\[0\] has "
        "type <class 'int'>. This is inconsistent.",
    ):
        check_imgs([0, "bla"])

    with pytest.raises(ValueError, match=empty_list_error % r"imgs\[0\]"):
        check_imgs([[]])

    with pytest.raises(
        ValueError,
        match=(
            r"imgs\[1\] has length 1 whereas imgs\[0\] has length 2."
            " All subjects should have the same number of sessions."
        ),
    ):
        check_imgs([["a", "a"], ["a"]])

    with pytest.raises(
        ValueError,
        match=array_type_error % (r"imgs\[0\]\[0\]", r"<class 'str'>"),
    ):
        check_imgs([["bka"]])

    with pytest.raises(
        ValueError, match=array_2axis_error % (r"imgs\[0\]\[0\]", 1)
    ):
        check_imgs([[np.random.rand(5)]])

    with pytest.raises(
        ValueError, match=array_2axis_error % (r"imgs\[0\]", 1)
    ):
        check_imgs([np.random.rand(5)])

    with pytest.raises(
        ValueError,
        match=(r"path_or_array must be either a string or an array"),
    ):
        check_imgs(np.random.rand(5, 3))

    with pytest.raises(ValueError, match=array_2axis_error % (r"imgs", 1)):
        check_imgs(np.random.rand(5))

    with pytest.raises(
        ValueError, match=("The number of subjects should be greater than 1")
    ):
        check_imgs([np.random.rand(5, 3)])

    with pytest.raises(
        ValueError,
        match=(
            "Subject 1 Session 0 does not have the same number "
            "of timeframes as Subject 0 Session 0"
        ),
    ):
        check_imgs([np.random.rand(10, 5), np.random.rand(10, 10)])

    with pytest.raises(
        ValueError,
        match=(
            "Subject 1 Session 0 does not have the same number "
            "of voxels as Subject 0 Session 0"
        ),
    ):
        check_imgs([np.random.rand(10, 5), np.random.rand(20, 5)])

    with pytest.raises(
        ValueError,
        match=(
            "Total number of timeframes is shorter than number "
            "of components \(8 > 5\)"
        ),
    ):
        check_imgs(
            [np.random.rand(10, 5), np.random.rand(10, 5)], n_components=8
        )


def test_check_shared():
    n_subjects = 2
    n_sessions = 2
    input_shapes = np.zeros((n_subjects, n_sessions, 2))
    input_shapes[0, 0, 0] = 10
    input_shapes[0, 0, 1] = 3
    input_shapes[0, 1, 0] = 10
    input_shapes[0, 1, 1] = 2
    input_shapes[1, 0, 0] = 10
    input_shapes[1, 0, 1] = 3
    input_shapes[1, 1, 0] = 10
    input_shapes[1, 1, 1] = 2

    shared_list_list = [
        [np.array([[1, 2, 3], [4, 5, 6]]), np.array([[1, 2], [4, 5]]),],
        [np.array([[2, 3, 4], [5, 6, 7]]), np.array([[2, 3], [5, 6]]),],
    ]

    shared_list_subjects = [
        np.array([[1, 2, 3, 1, 2], [4, 5, 6, 4, 5]]),
        np.array([[2, 3, 4, 2, 3], [5, 6, 7, 5, 6]]),
    ]

    shared_list_sessions = [
        np.array([[1.5, 2.5, 3.5], [4.5, 5.5, 6.5]]),
        np.array([[1.5, 2.5], [4.5, 5.5]]),
    ]

    shared_array = np.array(
        [[1.5, 2.5, 3.5, 1.5, 2.5], [4.5, 5.5, 6.5, 4.5, 5.5]]
    )

    with pytest.raises(
        ValueError,
        match=(
            r"shared_response should be either a list or an "
            "array but is of type <class 'str'>"
        ),
    ):
        check_shared_response("bla")

    with pytest.raises(
        ValueError,
        match=(
            r"shared_response is a list but shared_response\[0\] "
            "is neither a list or an array. This is invalid."
        ),
    ):
        check_shared_response(["bla", "bli"])

    with pytest.raises(
        ValueError,
        match=(
            r"shared_response\[0\] is a list but shared_response\[1\] "
            "is not a list this is incompatible"
        ),
    ):
        check_shared_response(
            [[np.random.rand(2, 2)], np.array([1])], aggregate=None
        )

    with pytest.raises(
        ValueError,
        match=(
            r"shared_response\[1\] has len 1 whereas "
            r"shared_response\[0\] has len 2. They should "
            "have same len"
        ),
    ):
        check_shared_response(
            [
                [np.random.rand(2, 2), np.random.rand(2, 2)],
                [np.random.rand(2, 2)],
            ],
            aggregate=None,
        )

    with pytest.raises(
        ValueError,
        match=(
            "Number of timeframes in input images during session 0 "
            "does not match the number of timeframes during session "
            r"0 of shared_response \(2 != 3\)"
        ),
    ):
        check_shared_response(
            [np.random.rand(2, 2), np.random.rand(2, 2)],
            aggregate="mean",
            input_shapes=input_shapes,
        )

    with pytest.raises(
        ValueError,
        match=(
            "Number of components in shared_response "
            "during session 0 is different than "
            "the number of components of the "
            r"model \(2 != 4\)"
        ),
    ):
        check_shared_response(np.random.rand(2, 10), n_components=4)

    with pytest.raises(
        ValueError,
        match=(
            "self.aggregate has value 'mean' but shared "
            "response is a list of list. "
            "This is incompatible"
        ),
    ):
        added_session, reshaped_shared = check_shared_response(
            shared_list_list,
            aggregate="mean",
            n_components=2,
            input_shapes=input_shapes,
        )

    added_session, reshaped_shared = check_shared_response(
        shared_list_subjects,
        aggregate=None,
        n_components=2,
        input_shapes=input_shapes,
    )
    assert added_session
    assert_array_almost_equal(
        np.array(reshaped_shared), shared_array.reshape(1, 2, 5)
    )

    added_session, reshaped_shared = check_shared_response(
        shared_list_sessions,
        aggregate="mean",
        n_components=2,
        input_shapes=input_shapes,
    )
    assert not added_session
    for j in range(len(reshaped_shared)):
        assert_array_almost_equal(reshaped_shared[j], shared_list_sessions[j])

    added_session, reshaped_shared = check_shared_response(shared_array,)

    assert added_session
    assert_array_almost_equal(
        np.array(reshaped_shared), shared_array.reshape(1, 2, 5)
    )


def test_error_small_n_voxels():
    X = [np.random.rand(10, 12) for _ in range(2)]

    srm = IdentifiableFastSRM(n_components=12, n_iter=10)

    with pytest.raises(
        ValueError,
        match=("Number of components is larger " "than the number "),
    ):
        srm.fit(X)
