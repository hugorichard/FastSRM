# author Hugo Richard

import numpy as np
from scipy import stats
from sklearn.model_selection import KFold
from fastsrm.identifiable_srm import IdentifiableFastSRM


def time_segment_matching(
        data,
        win_size=10,
):
    """
    This does subjects wise time segment matching (like in SRM paper)
    Parameters
    ----------
    data: list of np array of shape n_voxels, n_timeframes
        Input subject specific shared response
        data[i] is the shared response of subject i
    win_size: int
        Length of time segment to recover
    Returns
    -------
    accuracy: np array of shape n_subjects
    leave-one out accuracy among subjects
    """
    # Pull out shape information
    n_subjs = len(data)
    (n_features, n_TR) = data[0].shape  # Voxel/feature by timepoint

    # How many segments are there (account for edges)
    n_seg = n_TR - win_size

    # mysseg prediction prediction
    train_data = np.zeros((n_features * win_size, n_seg))

    # Concatenate the data across participants
    for ppt_counter in range(n_subjs):
        for window_counter in range(win_size):
            train_data[window_counter * n_features:(window_counter + 1) *
                       n_features, :, ] += data[ppt_counter][:, window_counter:
                                                             window_counter +
                                                             n_seg]

    # Iterate through the participants, leaving one out
    accuracy = np.zeros(shape=n_subjs)
    for ppt_counter in range(n_subjs):

        # Preset
        test_data = np.zeros((n_features * win_size, n_seg))

        for window_counter in range(win_size):
            test_data[window_counter * n_features:(window_counter + 1) *
                      n_features, :, ] = data[ppt_counter][:, window_counter:(
                          window_counter + n_seg)]

        # Take this participant data away
        train_ppts = stats.zscore((train_data - test_data), axis=0, ddof=1)
        test_ppts = stats.zscore(test_data, axis=0, ddof=1)

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
        print("Accuracy for subj %d is: %0.2f" %
              (ppt_counter, accuracy[ppt_counter]))

    print("The average accuracy among all subjects is {0:f} +/- {1:f}".format(
        np.mean(accuracy), np.std(accuracy)))
    return accuracy


def cross_validated_timesegment_matching(algo, paths, win_size):
    """
    This does subjects wise time segment matching (like in SRM paper) cross validated
    Parameters
    ----------
    algo: IdentifiableSRM instance
    paths: str np array of shape (n_subjects, n_sessions)
        paths[i, j] is the path to masked data
        np.load(paths[i, j]) is of shape n_voxels, n_components
    win_size: int
        Length of time segment to recover
    """
    cv = KFold(n_splits=5, shuffle=False)
    n_sessions = len(paths[0])
    cv_scores = []
    for i, (sessions_train,
            sessions_test) in enumerate(cv.split(np.arange(n_sessions))):
        algo.fit(paths[:, sessions_train])
        shared_response = algo.transform(paths[:, sessions_test])
        shared_response = [np.concatenate(s, axis=1) for s in shared_response]
        cv_scores.append(time_segment_matching(shared_response, win_size=win_size))
    return np.array(cv_scores)


# Usage example
# srm = IdentifiableFastSRM(
#     identifiability="ica",
#     n_components=20,
#     verbose=True,
#     n_jobs=1,
#     n_iter_reduced=10000,
#     tol=1e-12,
#     aggregate=None,
# )
# cv_scores = cross_validated_timesegment_matching(srm, input_paths)


def stack_t(S, win):
    """
    Utility function that stacks time segments"""
    S_ = np.copy(S)
    n_components = S_.shape[0]
    S__ = np.zeros((n_components * win, len(S_[0]) - win))
    for i in range(len(S_[0]) - win):
        S__[:, i] = np.concatenate([S_[:, k] for k in range(i, i + win)])
    S__ = S__ - np.mean(S__, axis=0, keepdims=True)
    S__ = S__ / np.linalg.norm(S__, axis=0, keepdims=True)
    return S__


def repeated_sessions_timesegment_matching(data,
                                           repeated_data,
                                           other_data,
                                           win_size=50):
    """
    This does timesegment matching between repeated sessions
    (sample segment is data, target segment
    is the corresponding segment in repeated_data).
    Possible targets are target timesegment and
    timesegments in repeated_data and other data
    non overlapping with the target.
    Parameters
    ----------
    data: list of np array of shape n_components, n_timeframes
        Subject-specific shared response of some sessions
    repeated_data: list of np array of shape n_components,n_timeframes
        Subject-specific shared response of some sessions (same stimuli as in data)
        length of repeated sessions should match.
    other_data: list of np array of shape n_components,n_timeframes
        Subject-specific shared response on other sessions
    win_size: int
        Length of time segment to recover
    Returns
    -------
    accuracy: np array of shape n_subjects, n_components, n_timesegments
        leave one subject out accuracy per subjects per components
    """
    import matplotlib.pyplot as plt
    assert len(data) == len(repeated_data) == len(other_data)
    accuracy = []
    for i in range(len(data)):
        n_components, n_timeframes = data[i].shape
        assert (n_components, n_timeframes) == repeated_data[i].shape
        assert n_components == other_data[i].shape[0]
        n_t = len(data[i][0])
        S = np.concatenate([data[i], repeated_data[i], other_data[i]], axis=1)
        accuracy_ = []
        for comp in range(n_components):
            S_ = [S[comp]]
            S__ = stack_t(S_, win_size)
            C = S__.T.dot(S__)
            C = C[:n_t - win_size]
            C = C[:, n_t:]
            for i in range(len(C)):
                for j in range(max(0, i - win_size), min(n_t, i + win_size)):
                    if i != j:
                        C[i, j] = 0
            A = np.argmax(C, axis=1)
            accuracy_.append(np.array(np.arange(n_t - win_size) == A))
        accuracy.append(accuracy_)
    return np.array(accuracy)


# Usage example
# srm = IdentifiableFastSRM(
#     identifiability="ica",
#     n_components=20,
#     verbose=True,
#     n_jobs=1,
#     n_iter_reduced=10000,
#     tol=1e-12,
#     aggregate=None,
# )
# S = srm.fit_transform(paths)
# data = [s[0] for s in S]
# repeated_data = [s[10] for s in S]
# other_data = [
#     np.concatenate([s[i] for i in [1, 2, 3, 4, 5, 6, 7, 8, 9, 11]], axis=1)
#     for s in S
# ]
# accuracy = repeated_sessions_timesegment_matching(data, repeated_data, other_data, win_size=50)
