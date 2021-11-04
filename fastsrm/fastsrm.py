import os
from fastsrm.srm import detsrm, probsrm, projection, safe_load
import numpy as np
from fastsrm.check_inputs import get_safe_shape
from fastsrm.check_inputs import check_imgs
from fastsrm.utils import extract_slices
from joblib import Parallel, delayed


def svd_reduce(imgs, n_jobs):
    """Reduce data using svd.
    Work done in parallel across subjects.

    Parameters
    ----------

    imgs : array of str, shape=[n_subjects, n_sessions]
        Element i, j of the array is a path to the data of subject i
        collected during session j.
        Data are loaded with numpy.load and expected shape is
        [n_voxels, n_timeframes]
        n_timeframes and n_voxels are assumed to be the same across subjects
        n_timeframes can vary across sessions
        Each voxel's timecourse is assumed to have mean 0 and variance 1

        imgs can also be a list of list of arrays where element i, j of
        the array is a numpy array of shape [n_voxels, n_timeframes] that
        contains the data of subject i collected during session j.

        imgs can also be a list of arrays where element i of the array is
        a numpy array of shape [n_voxels, n_timeframes] that contains the
        data of subject i (number of sessions is implicitly 1)

    n_jobs : integer, optional, default=1
        The number of CPUs to use to do the computation.
         -1 means all CPUs, -2 all CPUs but one, and so on.

    Returns
    -------
    reduced_data : np array shape=(n_subjects, n_timeframes, n_timeframes)
    """

    def svd_i(img):
        n_voxels = get_safe_shape(img[0])[0]
        slices = []
        t_i = 0
        for i in range(len(img)):
            n_voxels, n_timeframes = get_safe_shape(img[i])
            slices.append(slice(t_i, t_i + n_timeframes))
            t_i = t_i + n_timeframes
        total_timeframes = t_i

        # First compute X^TX
        C = np.zeros((total_timeframes, total_timeframes))
        for i in range(len(img)):
            Xi = safe_load(img[i])
            slice_i = slices[i]
            C[slice_i, slice_i] = Xi.T.dot(Xi) / 2
            for j in range(i + 1, len(img)):
                Xj = safe_load(img[j])
                slice_j = slices[j]
                C[slice_i, slice_j] = Xi.T.dot(Xj)
                del Xj
            del Xi
        C = C + C.T

        # Then compute SVD
        V, S, Vt = np.linalg.svd(C)
        X_reduced = (
            np.sqrt(S.reshape(-1, 1)) * Vt
        )  # X_reduced = np.diag(np.sqrt(S)).dot(V)
        return X_reduced

    X = Parallel(n_jobs=n_jobs)(delayed(svd_i)(img) for img in imgs)
    return X


def save(U, path):
    """Save data if path is not None.
    Returns:
    path if path is not None
    input data if path is None
    """
    if path is not None:
        np.save(path, U)
        U = path
    return U


def fastsrm(
    imgs,
    n_components,
    n_jobs=1,
    verbose=False,
    n_iter=100,
    tol=1e-3,
    method="prob",
    temp_dir=None,
    random_state=None,
    callback=None,
):
    """Performs an SRM decomposition on
    input data with a number of features
    much larger than the number of samples:
    Reduces the data by PCA and apply an SRM
    algorithm one reduced data.

    Parameters
    ----------

    imgs : array of str, shape=[n_subjects, n_sessions] or \
list of list of arrays or list of arrays
            Element i, j of the array is a path to the data of subject i \
collected during session j. Data are loaded with numpy.load and expected \
shape is [n_voxels, n_timeframes] n_timeframes and n_voxels are assumed \
to be the same across subjects n_timeframes can vary across sessions. \
Each voxel's timecourse is assumed to have mean 0 and variance 1
            imgs can also be a list of list of arrays where element i, j \
of the array is a numpy array of shape [n_voxels, n_timeframes] \
that contains the data of subject i collected during session j.
            imgs can also be a list of arrays where element i \
of the array is a numpy array of shape [n_voxels, n_timeframes] \
that contains the data of subject i (number of sessions is implicitly 1)

    n_components : int
        Number of timecourses of the shared coordinates

    n_jobs : int, optional, default=1
        The number of CPUs to use to do the computation. \
-1 means all CPUs, -2 all CPUs but one, and so on.

    verbose : bool or "warn"
        If True, logs are enabled. If False, logs are disabled. \
If "warn" only warnings are printed.

    n_iter : int
        Number of iterations to perform

    tol: float
        Stops if the norm of the gradient falls below tolerance value

    method : str, default="prob"
        if "prob", uses Probabilistic SRM
        if "det", uses Deterministic SRM

    temp_dir : str or None
        Path to dir where temporary results are stored. If None \
temporary results will be stored in memory. This can results in memory \
errors when the number of subjects and/or sessions is large

    random_state: int, RandomState instance or None, default=None
        Controls the randomness of the initialization.

    callback : function or None
        At each iteration calls callback(S, gnorm, it, t0) where `S` is the
        current estimate of the shared response, `gnorm` is the current
        `gradient norm`, `it` is the current iteration and `t0` the current
        time. The result is saved in a list `record`.
        If callback is None, nothing is done.

    Returns
    -------

    W : list of n_subjects numpy array of shape (n_voxels, n_components) \
        or path to np array of shape (n_voxels, n_components)
        Subject specific basis

    S : np array of shape (n_components, n_timeframes)
        Shared response

    sigmas: np array of shape (n_subjects,)
        Noise variance (only returned if method == "probsrm")

    Sigma: np array of shape (n_components,)
        (Diagonal) source covariance \
        (only returned if method == "probsrm")

    records : list of shape (n_iter,)
        The recorded information from callback.
        Only returned if callback is not None

    """
    # Get the number of voxels and timeframes
    reshaped_input, imgs_, shapes = check_imgs(imgs, n_components=n_components)
    n_voxels = shapes[0, 0, 0]
    n_timeframes = np.sum(shapes[0, :, 1])

    if verbose is True:
        print("[FastSRM.fit] Reducing data using online svd")
    Xred = svd_reduce(imgs_, n_jobs=n_jobs)

    if verbose is True:
        print("[FastSRM.fit] Finds shared " "response using reduced data")
    if method == "prob":
        ressrm = probsrm(
            Xred,
            n_components,
            n_iter,
            random_state,
            corrective_factor=n_timeframes / n_voxels,
            verbose=verbose,
            tol=tol,
            callback=callback,
        )

    if method == "det":
        ressrm = detsrm(
            Xred, n_components, n_iter, random_state, verbose=verbose, tol=tol,
        )

    S = ressrm[1]

    if method not in ["det", "prob"]:
        raise ValueError("method %s does not exists" % method)

    if verbose is True:
        print("[FastSRM.fit] Finds basis " "using reduced data")

    slices = extract_slices(imgs_[0])
    W = []
    for subject_number, img in enumerate(imgs_):
        w = None
        for i, x in enumerate(img):
            if w is None:
                w = safe_load(x).dot(S[:, slices[i]].T)
            else:
                w += safe_load(x).dot(S[:, slices[i]].T)
        if temp_dir is not None:
            path_w = os.path.join(temp_dir, "basis_%i.npy" % subject_number)
        else:
            path_w = None
        W.append(save(projection(w), path_w))
    return [W] + list(ressrm)[1:]
