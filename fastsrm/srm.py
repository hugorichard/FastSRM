from sklearn.utils import check_random_state
import numpy as np
from time import time


def safe_load(data):
    """If data is an array returns data else returns np.load(data)"""
    if isinstance(data, np.ndarray):
        if len(data.shape) == 2:
            return data
    if isinstance(data, np.ndarray) or isinstance(data, list):
        if isinstance(data[0], str):
            return np.concatenate([np.load(d) for d in data], axis=1)
    else:
        return np.load(data)


def projection(X):
    """Perform projection onto Stiefel manifold."""
    U, D, V = np.linalg.svd(X, full_matrices=False)
    return U.dot(V)


def detsrm(
    X,
    n_components,
    n_iter=100,
    random_state=None,
    verbose=False,
    tol=1e-5,
    callback=None,
):
    """Perform Deterministic SRM on numpy arrays.
    To be used when data hold in memory and the number
    of features is not much larger than the number
    of samples (in which case fastsrm is preferable)

    Parameters
    -----------

    X : np array of shape (n_views, n_features, n_samples)
        Input data

    n_components : int
        Number of timecourses of the shared coordinates

    n_iter : int
        Number of iterations to perform

    random_state: int, RandomState instance or None, default=None
        Controls the randomness of the initialization.

    verbose : bool
        If True, logs are enabled. If False, logs are disabled.

    tol: float
        Stops if the norm of the gradient falls below tolerance value

    callback: function or None
        At each iteration calls callback(S, gnorm, it, t0) where `S` is the
        current estimate of the shared response, `gnorm` is the current
        `gradient norm`, `it` is the current iteration and `t0` the current
        time. The result is saved in a list `record`.
        If callback is None, nothing is done.

    Returns
    --------

    W : np array of shape \
    (n_views, n_features, n_components)
        Subject specific basis

    S : np array of shape (n_components, n_timeframes)
        Shared response

    records : list of shape (n_iter,)
        The recorded information from callback.
        Only returned if callback is not None

    """
    rng = check_random_state(random_state)
    X = np.array([safe_load(x) for x in X])
    m, v, n = X.shape
    # Init
    S = rng.randn(n_components, n)
    W = []
    for i in range(m):
        W.append(projection(X[i].dot(S.T)))
    W = np.array(W)
    Y = np.array([W[i].T.dot(X[i]) for i in range(m)])
    record = []
    for it in range(n_iter):
        gnorm = np.max(np.abs(m * S - np.sum(Y, axis=0)))
        if verbose:
            print("it: %i Gradient: %.5f" % (it, gnorm))
        if gnorm < tol:
            break
        S = np.mean(Y, axis=0)
        for i in range(m):
            Wi = projection(X[i].dot(S.T))
            Y[i] = Wi.T.dot(X[i])
            W[i] = Wi
        if callback is not None:
            t0 = time()
            record.append(callback(S, gnorm, it, t0))

    if gnorm > tol:
        if verbose:
            print(
                "DetSRM did not converge. Current gradient norm is %.6f"
                % gnorm
            )
    if callback is not None:
        return W, S, record
    return W, S


def probsrm(
    X,
    n_components,
    n_iter=100,
    random_state=None,
    corrective_factor=1,
    verbose=False,
    tol=1e-5,
    callback=None,
):
    """Perform Probabilistic SRM on numpy arrays.
    To be used when data hold in memory and the number
    of features is not much larger than the number
    of samples (in which case fastsrm is preferable)

    Parameters
    -----------

    X : np array of shape (n_views, n_features, n_samples)
        Input data

    n_components : int
        Number of timecourses of the shared coordinates

    n_iter : int
        Number of iterations to perform

    random_state: int, RandomState instance or None, default=None
        Controls the randomness of the initialization.

    verbose : bool
        If True, logs are enabled. If False, logs are disabled.

    tol : float
        Stops if the norm of the gradient falls below tolerance value

    callback : function or None
        At each iteration calls callback(S, gnorm, it, t0) where `S` is the
        current estimate of the shared response, `gnorm` is the current
        `gradient norm`, `it` is the current iteration and `t0` the current
        time. The result is saved in a list `record`.
        If callback is None, nothing is done.

    Returns
    --------

    W : np array of shape \
    (n_views, n_features, n_components)
        Subject specific basis

    S : np array of shape (n_components, n_timeframes)
        Shared response

    sigmas : np array of shape (n_views,)
        Noise variance

    Sigma : np array of shape (n_components,)
        (Diagonal) Shared response covariance

    records : list of shape (n_iter,)
        The recorded information from callback.
        Only returned if callback is not None

    """
    X = np.array([safe_load(x) for x in X])
    rng = check_random_state(random_state)
    m, v, n = X.shape
    # Init

    # Init
    S = rng.randn(n_components, n)
    W = []
    for i in range(m):
        W.append(projection(X[i].dot(S.T)))
    W = np.array(W)
    sigmas = np.ones(m)
    Sigma = np.ones(n_components)
    normX = [np.sum(x ** 2) for x in X]
    likelihood = np.inf
    vap = v / corrective_factor
    record = []
    for it in range(n_iter):
        V = 1 / (np.sum(1 / sigmas) + 1 / Sigma)
        S = V[:, None] * np.sum(
            [W[i].T.dot(X[i]) / sigmas[i] for i in range(m)], axis=0
        )
        likelihood_prec = likelihood
        likelihood = (
            vap / 2 * np.sum(np.log(sigmas))
            + 1 / 2 * np.sum(np.log(Sigma))
            - 1 / 2 * np.sum(np.log(V))
            + 1 / 2 * np.sum([normX[i] / sigmas[i] for i in range(m)]) / n
            - 1 / 2 * np.sum(1 / V[:, None] * S ** 2) / n
        )
        if verbose:
            print(
                "it: %i Loss: %.5f Diff: %.5f"
                % (it, likelihood, likelihood_prec - likelihood)
            )
        diff = np.abs(likelihood_prec - likelihood)
        if np.abs(likelihood_prec - likelihood) < tol:
            break
        normS = np.sum(S ** 2)
        for i in range(m):
            XST = X[i].dot(S.T)
            W[i] = projection(XST)
            norm = normX[i] + normS - 2 * np.trace(XST.T.dot(W[i]))
            sigmas[i] = 1 / vap * (norm / n + np.sum(V ** 2))
        Sigma = V + np.diag(S.dot(S.T)) / n
        if callback is not None:
            t0 = time()
            record.append(callback(S, diff, it, t0))
    if diff > tol:
        print("ProbSRM did not converge. Current diff is %.6f" % diff)

    if callback is not None:
        return W, S, sigmas, Sigma, record
    return W, S, sigmas, Sigma
