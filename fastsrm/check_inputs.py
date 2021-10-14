import numpy as np


def assert_valid_index(indexes, max_value, name_indexes):
    """
    Check that indexes are between 0 and max_value and number
    of indexes is less than max_value
    """
    for i, ind_i in enumerate(indexes):
        if ind_i < 0 or ind_i >= max_value:
            raise ValueError(
                "Index %i of %s has value %i "
                "whereas value should be between 0 and %i"
                % (i, name_indexes, ind_i, max_value - 1)
            )


def check_indexes(indexes, name):
    if not (indexes is None or isinstance(indexes, (list, np.ndarray))):
        raise ValueError(
            "%s should be either a list, an array or None but received type %s"
            % (name, type(indexes))
        )


def _check_shared_response_list_of_list(
    shared_response, n_components, input_shapes
):
    # Check that shared_response is indeed a list of list of arrays
    n_subjects = len(shared_response)
    n_sessions = None
    for i in range(len(shared_response)):
        if not isinstance(shared_response[i], list):
            raise ValueError(
                "shared_response[0] is a list but "
                "shared_response[%i] is not a list "
                "this is incompatible." % i
            )
        assert_non_empty_list(shared_response[i], "shared_response[%i]" % i)
        if n_sessions is None:
            n_sessions = len(shared_response[i])
        elif n_sessions != len(shared_response[i]):
            raise ValueError(
                "shared_response[%i] has len %i whereas "
                "shared_response[0] has len %i. They should "
                "have same length"
                % (i, len(shared_response[i]), len(shared_response[0]))
            )
        for j in range(len(shared_response[i])):
            assert_array_2axis(
                shared_response[i][j], "shared_response[%i][%i]" % (i, j)
            )

    return _check_shared_response_list_sessions(
        [
            np.mean([shared_response[i][j] for i in range(n_subjects)], axis=0)
            for j in range(n_sessions)
        ],
        n_components,
        input_shapes,
    )


def _check_shared_response_list_sessions(
    shared_response, n_components, input_shapes
):
    for j in range(len(shared_response)):
        assert_array_2axis(shared_response[j], "shared_response[%i]" % j)
        if input_shapes is not None:
            if shared_response[j].shape[1] != input_shapes[0][j][1]:
                raise ValueError(
                    "Number of timeframes in input images during "
                    "session %i does not match the number of "
                    "timeframes during session %i "
                    "of shared_response (%i != %i)"
                    % (
                        j,
                        j,
                        shared_response[j].shape[1],
                        input_shapes[0, j, 1],
                    )
                )
        if n_components is not None:
            if shared_response[j].shape[0] != n_components:
                raise ValueError(
                    "Number of components in "
                    "shared_response during session %i is "
                    "different than "
                    "the number of components of the model (%i != %i)"
                    % (j, shared_response[j].shape[0], n_components)
                )
    return shared_response


def _check_shared_response_list_subjects(
    shared_response, n_components, input_shapes
):
    for i in range(len(shared_response)):
        assert_array_2axis(shared_response[i], "shared_response[%i]" % i)

    return _check_shared_response_array(
        np.mean(shared_response, axis=0), n_components, input_shapes
    )


def _check_shared_response_array(shared_response, n_components, input_shapes):
    assert_array_2axis(shared_response, "shared_response")
    if input_shapes is None:
        new_input_shapes = None
    else:
        n_subjects, n_sessions, _ = input_shapes.shape
        new_input_shapes = np.zeros((n_subjects, 1, 2))
        new_input_shapes[:, 0, 0] = input_shapes[:, 0, 0]
        new_input_shapes[:, 0, 1] = np.sum(input_shapes[:, :, 1], axis=1)
    return _check_shared_response_list_sessions(
        [shared_response], n_components, new_input_shapes
    )


def check_shared_response(
    shared_response, aggregate="mean", n_components=None, input_shapes=None
):
    """
    Check that shared response has valid input and turn it into
    a session-wise shared response

    Returns
    -------
    added_session: bool
        True if an artificial sessions was added to match the list of
        session input type for shared_response
    reshaped_shared_response: list of arrays
        shared response (reshaped to match the list of session input)
    """
    # Depending on aggregate and shape of input we infer what to do
    if isinstance(shared_response, list):
        assert_non_empty_list(shared_response, "shared_response")
        if isinstance(shared_response[0], list):
            if aggregate == "mean":
                raise ValueError(
                    "self.aggregate has value 'mean' but "
                    "shared response is a list of list. This is "
                    "incompatible"
                )
            return (
                False,
                _check_shared_response_list_of_list(
                    shared_response, n_components, input_shapes
                ),
            )
        elif isinstance(shared_response[0], np.ndarray):
            if aggregate == "mean":
                return (
                    False,
                    _check_shared_response_list_sessions(
                        shared_response, n_components, input_shapes
                    ),
                )
            else:
                return (
                    True,
                    _check_shared_response_list_subjects(
                        shared_response, n_components, input_shapes
                    ),
                )
        else:
            raise ValueError(
                "shared_response is a list but "
                "shared_response[0] is neither a list "
                "or an array. This is invalid."
            )
    elif isinstance(shared_response, np.ndarray):
        return (
            True,
            _check_shared_response_array(
                shared_response, n_components, input_shapes
            ),
        )
    else:
        raise ValueError(
            "shared_response should be either "
            "a list or an array but is of type %s" % type(shared_response)
        )


def get_safe_shape(path_or_array):
    """
    Get shape of an array of saved array
    """
    if isinstance(path_or_array, np.ndarray):
        return path_or_array.shape
    elif isinstance(path_or_array, str):
        return get_shape(path_or_array)
    else:
        raise ValueError("path_or_array must be either a string or an array")


def get_shape(path):
    """Get shape of saved np array
    Parameters
    ----------
    path: str
        path to np array
    """
    f = open(path, "rb")
    version = np.lib.format.read_magic(f)
    shape, fortran_order, dtype = np.lib.format._read_array_header(f, version)
    f.close()
    return shape


def assert_array_2axis(array, name_array):
    """Check that input is an np array with 2 axes

    Parameters
    ----------
    array: np array
    name_array: str
        Name of the array
    """

    if not isinstance(array, np.ndarray):
        raise ValueError(
            "%s should be of type "
            "np.ndarray but is of type %s" % (name_array, type(array))
        )

    if len(array.shape) != 2:
        raise ValueError(
            "%s must have exactly 2 axes "
            "but has %i axes" % (name_array, len(array.shape))
        )


def assert_non_empty_list(input_list, list_name):
    """
    Check that input list is not empty
    Parameters
    ----------
    input_list: list
    list_name: str
        Name of the list
    """
    if len(input_list) == 0:
        raise ValueError(
            "%s is a list of length 0 which is not valid" % list_name
        )


def _check_imgs_list(imgs):
    """
    Checks that imgs is a non empty list of elements of the same type

    Parameters
    ----------

    imgs : list
    """
    # Check the list is non empty
    assert_non_empty_list(imgs, "imgs")

    # Check that all input have same type
    for i in range(len(imgs)):
        if not isinstance(imgs[i], type(imgs[0])):
            raise ValueError(
                "imgs[%i] has type %s whereas "
                "imgs[%i] has type %s. "
                "This is inconsistent." % (i, type(imgs[i]), 0, type(imgs[0]))
            )


def _check_imgs_list_list(imgs):
    """
    Check input images if they are list of list of arrays

    Parameters
    ----------

    imgs : list of list of array of shape [n_voxels, n_components]
            imgs is a list of list of arrays where element i, j of
            the array is a numpy array of shape [n_voxels, n_timeframes] that
            contains the data of subject i collected during session j.
            n_timeframes and n_voxels are assumed to be the same across
            subjects
            n_timeframes can vary across sessions
            Each voxel's timecourse is assumed to have mean 0 and variance 1

    Returns
    -------
    shapes: array
        Shape of input images
    """
    n_subjects = len(imgs)

    # Check that the number of session is not 0
    assert_non_empty_list(imgs[0], "imgs[%i]" % 0)

    # Check that the number of sessions is the same for all subjects
    n_sessions = None
    for i in range(len(imgs)):
        if n_sessions is None:
            n_sessions = len(imgs[i])
        if n_sessions != len(imgs[i]):
            raise ValueError(
                "imgs[%i] has length %i whereas imgs[%i] "
                "has length %i. All subjects should have "
                "the same number of sessions."
                % (i, len(imgs[i]), 0, len(imgs[0]))
            )

    shapes = np.zeros((n_subjects, n_sessions, 2))
    # Run array-level checks
    for i in range(len(imgs)):
        for j in range(len(imgs[i])):
            assert_array_2axis(imgs[i][j], "imgs[%i][%i]" % (i, j))
            shapes[i, j, :] = imgs[i][j].shape

    return shapes


def _check_imgs_list_array(imgs):
    """
    Check input images if they are list of arrays.
    In this case returned images are a list of list of arrays
    where element i,j of the array is a numpy array of
    shape [n_voxels, n_timeframes] that contains the data of subject i
    collected during session j.

    Parameters
    ----------

    imgs : array of str, shape=[n_subjects, n_sessions]
            imgs is a list of arrays where element i of the array is
            a numpy array of shape [n_voxels, n_timeframes] that contains the
            data of subject i (number of sessions is implicitly 1)
            n_timeframes and n_voxels are assumed to be the same across
            subjects
            n_timeframes can vary across sessions
            Each voxel's timecourse is assumed to have mean 0 and variance 1

    Returns
    -------
    shapes: array
        Shape of input images
    new_imgs: list of list of array of shape [n_voxels, n_components]
    """
    n_subjects = len(imgs)
    n_sessions = 1
    shapes = np.zeros((n_subjects, n_sessions, 2))
    new_imgs = []
    for i in range(len(imgs)):
        assert_array_2axis(imgs[i], "imgs[%i]" % i)
        shapes[i, 0, :] = imgs[i].shape
        new_imgs.append([imgs[i]])

    return new_imgs, shapes


def _check_imgs_array(imgs):
    """Check input image if it is an array

    Parameters
    ----------
    imgs : array of str, shape=[n_subjects, n_sessions]
            Element i, j of the array is a path to the data of subject i
            collected during session j.
            Data are loaded with numpy.load and expected
            shape is [n_voxels, n_timeframes]
            n_timeframes and n_voxels are assumed to be the same across
            subjects
            n_timeframes can vary across sessions
            Each voxel's timecourse is assumed to have mean 0 and variance 1

    Returns
    -------
    shapes : array
        Shape of input images
    """
    assert_array_2axis(imgs, "imgs")
    n_subjects, n_sessions = imgs.shape

    shapes = np.zeros((n_subjects, n_sessions, 2))
    for i in range(n_subjects):
        for j in range(n_sessions):
            shapes[i, j, :] = get_safe_shape(imgs[i, j])
    return shapes


def _check_shapes(
    shapes, n_components=None, ignore_nsubjects=False, ignore_ncomponents=False
):
    """Check that number of voxels is the same for each subjects. Number of
    timeframes can vary between sessions but must be consistent across
    subjects

    Parameters
    ----------
    shapes : array of shape (n_subjects, n_sessions, 2)
        Array of shapes of input images
    """
    n_subjects, n_sessions, _ = shapes.shape

    if n_subjects <= 1 and not ignore_nsubjects:
        raise ValueError("The number of subjects should be greater than 1")

    n_timeframes_list = [None] * n_sessions
    n_voxels = None
    for n in range(n_subjects):
        for m in range(n_sessions):
            if n_timeframes_list[m] is None:
                n_timeframes_list[m] = shapes[n, m, 1]

            if n_voxels is None:
                n_voxels = shapes[n, m, 0]

            if n_timeframes_list[m] != shapes[n, m, 1]:
                raise ValueError(
                    "Subject %i Session %i does not have the "
                    "same number of timeframes "
                    "as Subject %i Session %i" % (n, m, 0, m)
                )

            if n_voxels != shapes[n, m, 0]:
                raise ValueError(
                    "Subject %i Session %i"
                    " does not have the same number of voxels as "
                    "Subject %i Session %i." % (n, m, 0, 0)
                )

    if not ignore_ncomponents:
        if n_components > n_voxels:
            raise ValueError(
                "Number of components is larger than "
                "the number of voxels (%i > %i)" % (n_components, n_voxels)
            )

        if n_components > np.sum(n_timeframes_list):
            raise ValueError(
                "Total number of timeframes is shorter than number "
                "of components (%i > %i)"
                % (n_components, np.sum(n_timeframes_list))
            )


def check_imgs(
    imgs, n_components=None, ignore_nsubjects=False, ignore_ncomponents=False
):
    """
    Check input images

    Parameters
    ----------

    imgs : array of str, shape=[n_subjects, n_sessions]
            Element i, j of the array is a path to the data of subject i
            collected during session j.
            Data are loaded with numpy.load and expected
            shape is [n_voxels, n_timeframes]
            n_timeframes and n_voxels are assumed to be the same across
            subjects
            n_timeframes can vary across sessions
            Each voxel's timecourse is assumed to have mean 0 and variance 1

            imgs can also be a list of list of arrays where element i, j of
            the array is a numpy array of shape [n_voxels, n_timeframes] that
            contains the data of subject i collected during session j.

            imgs can also be a list of arrays where element i of the array is
            a numpy array of shape [n_voxels, n_timeframes] that contains the
            data of subject i (number of sessions is implicitly 1)

    Returns
    -------
    reshaped_input: bool
        True if input had to be reshaped to match the
        n_subjects, n_sessions input
    new_imgs: list of list of array or np array
        input imgs reshaped if it is a list of arrays so that it becomes a
        list of list of arrays
    shapes: array
        Shape of input images
    """
    reshaped_input = False
    new_imgs = imgs
    if isinstance(imgs, list):
        _check_imgs_list(imgs)
        if isinstance(imgs[0], list):
            shapes = _check_imgs_list_list(imgs)
        elif isinstance(imgs[0], np.ndarray):
            new_imgs, shapes = _check_imgs_list_array(imgs)
            reshaped_input = True
        else:
            raise ValueError(
                "Since imgs is a list, it should be a list of list "
                "of arrays or a list of arrays but imgs[0] has type %s"
                % type(imgs[0])
            )
    elif isinstance(imgs, np.ndarray):
        shapes = _check_imgs_array(imgs)
    else:
        raise ValueError(
            "Input imgs should either be a list or an array but has type %s"
            % type(imgs)
        )

    _check_shapes(shapes, n_components, ignore_nsubjects, ignore_ncomponents)

    return reshaped_input, new_imgs, shapes
