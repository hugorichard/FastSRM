# utilities mainly used for tests
import os
import numpy as np
from scipy.optimize import linear_sum_assignment


def apply_aggregate(shared_response, aggregate, input_format):
    if aggregate is None:
        if input_format == "list_of_array":
            return [np.mean(shared_response, axis=0)]
        else:
            return [
                np.mean([
                    shared_response[i][j] for i in range(len(shared_response))
                ],
                        axis=0) for j in range(len(shared_response[0]))
            ]
    else:
        if input_format == "list_of_array":
            return [shared_response]
        else:
            return shared_response


def apply_input_format(X, input_format):
    if input_format == "array":
        n_sessions = len(X[0])
        XX = [[np.load(X[i, j]) for j in range(len(X[i]))]
              for i in range(len(X))]
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


def generate_data(n_voxels,

                  n_timeframes,
                  n_subjects,
                  n_components,
                  datadir,
                  noise_level=0.1,
                  input_format="array"):
    n_sessions = len(n_timeframes)
    cumsum_timeframes = np.cumsum([0] + n_timeframes)
    slices_timeframes = [
        slice(cumsum_timeframes[i], cumsum_timeframes[i + 1])
        for i in range(n_sessions)
    ]

    # Create a Shared response S with K = 3
    theta = np.linspace(-4 * np.pi, 4 * np.pi, int(np.sum(n_timeframes)))
    z = np.linspace(-2, 2, int(np.sum(n_timeframes)))
    r = z**2 + 1
    x = r * np.sin(theta)
    y = r * np.cos(theta)

    S = np.vstack((x, y, z))

    # Generate fake data
    W = []
    X = []
    for subject in range(n_subjects):
        Q, R = np.linalg.qr(np.random.random((n_voxels, n_components)))
        W.append(Q.T)
        X_ = []
        for session in range(n_sessions):
            S_s = S[:, slices_timeframes[session]]
            S_s = S_s - np.mean(S_s, axis=1, keepdims=True)
            noise = noise_level * np.random.random(
                (n_voxels, n_timeframes[session]))
            noise = noise - np.mean(noise, axis=1, keepdims=True)
            data = Q.dot(S_s) + noise
            X_.append(data)
        X.append(X_)

    # create paths such that paths[i, j] contains data
    # of subject i during session j
    if datadir is not None:
        paths = to_path(X, datadir)

    S = [(S[:, s] - np.mean(S[:, s], axis=1, keepdims=True))
         for s in slices_timeframes]

    if input_format == "array":
        return paths, W, S

    elif input_format == "list_of_list":
        return X, W, S

    elif input_format == "list_of_array":
        return [
            np.concatenate([X[i][j].T for j in range(n_sessions)]).T
            for i in range(n_subjects)
        ], W, S
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
        np.concatenate(source, axis=1).T,
        np.concatenate(recov, axis=1).T)[2]

    if return_index:
        return align_sign([w[ib] for w in recov], source), ib
    else:
        return align_sign([w[ib] for w in recov], source)
