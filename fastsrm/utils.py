# utilities mainly used for tests
import os
import scipy
import numpy as np
from scipy.optimize import linear_sum_assignment
from fastsrm.srm import projection
from fastsrm.check_inputs import get_safe_shape


def extract_slices(img):
    """
    Extract slices from images shapes

    Parameters
    -----------
    imgs: list of n_sessions arrays of shape\
        (n_voxels, n_timeframes)

    Returns
    --------
    slices: list of slices
    """
    slices = []
    t_i = 0
    for i in range(len(img)):
        n_voxels, n_timeframes = get_safe_shape(img[i])
        slices.append(slice(t_i, t_i + n_timeframes))
        t_i = t_i + n_timeframes
    return slices


def apply_aggregate(shared_response, aggregate, input_format):
    if aggregate is None:
        if input_format == "list_of_array":
            return [np.mean(shared_response, axis=0)]
        else:
            return [
                np.mean(
                    [
                        shared_response[i][j]
                        for i in range(len(shared_response))
                    ],
                    axis=0,
                )
                for j in range(len(shared_response[0]))
            ]
    else:
        if input_format == "list_of_array":
            return [shared_response]
        else:
            return shared_response


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


def to_path(X, dirpath):
    """
    Save list of list of array to path and returns the path_like array
    Parameters
    ----------
    X: list of list of array
        input data
    dirpath: str
        dirpath
    Returns
    -------
    paths: array of str
        path arrays where all data are stored
    """
    paths = []
    for i, sessions in enumerate(X):
        sessions_path = []
        for j, session in enumerate(sessions):
            pth = "%i_%i" % (i, j)
            np.save(os.path.join(dirpath, pth), session)
            sessions_path.append(os.path.join(dirpath, pth + ".npy"))
        paths.append(sessions_path)
    return np.array(paths)


def generate_data(
    n_voxels,
    n_timeframes,
    n_subjects,
    n_components,
    datadir,
    noise_level=0.1,
    input_format="array",
    seed=0,
):
    rng = np.random.RandomState(seed)
    n_sessions = len(n_timeframes)
    cumsum_timeframes = np.cumsum([0] + n_timeframes)
    slices_timeframes = [
        slice(cumsum_timeframes[i], cumsum_timeframes[i + 1])
        for i in range(n_sessions)
    ]

    n = np.sum(n_timeframes)
    k = n_components
    v = n_voxels
    m = n_subjects

    Sigma = rng.dirichlet(np.ones(k), 1).flatten()
    S = np.sqrt(Sigma)[:, None] * rng.randn(k, n)
    Us, Ds, Vs = np.linalg.svd(S, full_matrices=False)
    S = Ds[:, None] * Vs
    W = np.array([projection(rng.randn(v, k)).dot(Us.T) for i in range(m)])
    sigmas = noise_level * rng.rand(m)
    N = np.array([np.sqrt(sigmas[i]) * rng.randn(v, n) for i in range(m)])
    X = np.array([W[i].dot(S) + N[i] for i in range(m)])

    # Cut data
    X = [[x[:, slices] for slices in slices_timeframes] for x in X]

    # create paths such that paths[i, j] contains data
    # of subject i during session j
    if datadir is not None:
        paths = to_path(X, datadir)

    # Cut sources
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
                np.column_stack([X[i][j] for j in range(n_sessions)])
                for i in range(n_subjects)
            ],
            W,
            S,
        )
    else:
        raise ValueError("Wrong input_format")


# Match score
def solve_hungarian(recov, source):
    """
    Compute maximum correlations between true indep components and estimated components

    Parameters
    ----------------

    recov: np.array shape (n_timeframes, n_components)
    Array with the recovered sources (n_timeframes, n_components)

    source: np.array
    Array with the true sources

    Returns
    ----------------

    CorMat[ii].mean(): float
    Maximum correlation between true indep components and estimated components,
    averaged across all components.

    CorMat: np.array
    n_dimensions X n_dimensions matrix; all correlations among true and recovered sources

    ii: tuple
    Tuple with the matched indices maximising the correlations

    """
    Ncomp = source.shape[1]
    CorMat = (np.abs(np.corrcoef(recov.T, source.T)))[:Ncomp, Ncomp:]
    ii = linear_sum_assignment(-1 * CorMat)
    return CorMat[ii].mean(), CorMat, ii


def align_sign(recov, source):
    for i in range(len(source)):
        # and sign here
        mult = []
        for sign in np.sum(source[i] * recov[i], axis=1):
            if sign < 0:
                mult.append(-1)
            else:
                mult.append(1)
        mult = np.diag(np.array(mult))
    return [mult.dot(w) for w in recov]


def align_basis(recov, source, return_index=False):
    # Let us align components here
    _, ib = solve_hungarian(
        np.concatenate(source, axis=1).T, np.concatenate(recov, axis=1).T
    )[2]

    if return_index:
        return align_sign([w[ib] for w in recov], source), ib
    else:
        return align_sign([w[ib] for w in recov], source)


def hungarian(M):
    u, order = scipy.optimize.linear_sum_assignment(-abs(M))
    vals = M[u, order]
    return order, np.sign(vals)


def error_dot(M):
    order, _ = hungarian(M)
    return 1 - M[np.arange(M.shape[0]), order]


def error_source(S1, S2):
    S1_ = S1 - np.mean(S1, axis=1, keepdims=True)
    S1_ = S1_ / np.linalg.norm(S1_, axis=1, keepdims=True)

    S2_ = S2 - np.mean(S2, axis=1, keepdims=True)
    S2_ = S2_ / np.linalg.norm(S2_, axis=1, keepdims=True)
    return error_dot(np.abs(S1_.dot(S2_.T)))


def corr(x, y):
    return np.sum(x * y) / np.sqrt(np.sum(x * x) * np.sum(y * y))


def error_source_rotation(S1, S2):
    S1_ = S1.copy()
    S2_ = S2.copy()
    S1_ = S1_ / np.linalg.norm(S1_, axis=1, keepdims=True)
    S2_ = S2_ / np.linalg.norm(S2_, axis=1, keepdims=True)
    return np.linalg.norm(projection(S2_.dot(S1_.T)).dot(S1_) - S2_)


def time_segment_matching(
    data, win_size=10, verbose=True, l=None,
):
    """
    data list of n_voxels, n_timeframes data
    l: np array of shape n_subjects, n_components:
    noise level
    """
    if l is None:
        l = np.ones((len(data), len(data[0])))

    # Pull out shape information
    n_subjs = len(data)
    (n_features, n_TR) = data[0].shape  # Voxel/feature by timepoint

    # How many segments are there (account for edges)
    n_seg = n_TR - win_size + 1

    # mysseg prediction prediction
    train_data = np.zeros((n_features * win_size, n_seg))

    # train data
    # n_features x n_timeframes
    # Ex: n_features=2, n_timeframes=6, windows_size=2
    # data_i   train_data_i
    # X1Y1       X1 X2 X3 X4 # first
    # X2Y2       Y1 Y2 Y3 Y4
    # X3Y3       X2 X3 X4 X5 # second
    # X4Y4       Y2 Y3 Y4 Y5
    # X5Y5  ->   X3 X4 X5 X6 # last
    # X6Y6  ->   Y3 Y4 Y5 Y6

    # Concatenate the data across participants
    for ppt_counter in range(n_subjs):
        for window_counter in range(win_size):
            train_data[
                window_counter
                * n_features : (window_counter + 1)
                * n_features,
                :,
            ] += data[ppt_counter][:, window_counter : window_counter + n_seg]

    # Iterate through the participants, leaving one out
    accuracy = np.zeros(shape=n_subjs)
    for ppt_counter in range(n_subjs):

        # Preset
        test_data = np.zeros((n_features * win_size, n_seg))
        for window_counter in range(win_size):
            test_data[
                window_counter
                * n_features : (window_counter + 1)
                * n_features,
                :,
            ] = data[ppt_counter][:, window_counter : (window_counter + n_seg)]

        normalized_test_data = np.zeros((n_features * win_size, n_seg))
        for window_counter in range(win_size):
            normalized_test_data[
                window_counter
                * n_features : (window_counter + 1)
                * n_features,
                :,
            ] = data[ppt_counter][
                :, window_counter : (window_counter + n_seg)
            ] / (
                l[ppt_counter].reshape(-1, 1) ** 2
            )

        # Take this participant data away
        train_ppts = stats.zscore((train_data - test_data), axis=0, ddof=1)
        test_ppts = stats.zscore(normalized_test_data, axis=0, ddof=1)

        # Correlate the two data sets
        corr_mtx = test_ppts.T.dot(train_ppts)

        # If any segments have a correlation difference less than the window size and they aren't the same segments then set the value to negative infinity
        for seg_1 in range(n_seg):
            for seg_2 in range(n_seg):
                if abs(seg_1 - seg_2) < win_size and seg_1 != seg_2:
                    corr_mtx[seg_1, seg_2] = -np.inf

        # Find the segement with the max value
        rank = np.argmax(corr_mtx, axis=1)

        # Find the number of segments that were matched for this participant
        accuracy[ppt_counter] = sum(rank == range(n_seg)) / float(n_seg)

        # Print accuracy
        if verbose:
            print(
                "Accuracy for subj %d is: %0.2f"
                % (ppt_counter, accuracy[ppt_counter])
            )

    if verbose:
        print(
            "The average accuracy among all subjects is {0:f} +/- {1:f}".format(
                np.mean(accuracy), np.std(accuracy)
            )
        )
    return accuracy


def load_and_concat(paths):
    """
    Take list of path and yields input data for ProbSRM
    Parameters
    ----------
    paths
    Returns
    -------
    X
    """
    X = []
    for i in range(len(paths)):
        X_i = np.concatenate(
            [np.load(paths[i, j]) for j in range(len(paths[i]))], axis=1
        )
        X.append(X_i)
    return X
