"""Fast Shared Response Model (FastSRM)

The implementation is based on the following publications:

.. [Richard2019] "Fast Shared Response Model for fMRI data"
    H. Richard, L. Martin, A. Pinho, J. Pillow, B. Thirion, 2019
    https://arxiv.org/pdf/1909.12537.pdf
"""

# Author: Hugo Richard

import os
import uuid
import numpy as np
from joblib import Parallel, delayed, Memory

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.exceptions import NotFittedError
from fastsrm.fastsrm import (
    _reduced_space_compute_shared_response,
    _compute_subject_basis,
    check_atlas,
    check_imgs,
    create_temp_dir,
    reduce_data,
    _compute_and_save_corr_mat,
    _compute_and_save_subject_basis,
    check_indexes,
    assert_valid_index,
    _compute_shared_response_online,
    check_shared_response,
    assert_array_2axis,
    safe_load,
    get_safe_shape,
)
from picard import picard
import warnings
from sklearn.utils import check_random_state


def svd_reduce(imgs, n_jobs, verbose):
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

    temp_dir : str or None
        path to dir where temporary results are stored
        if None temporary results will be stored in memory. This
        can results in memory errors when the number of subjects
        and / or sessions is large

    low_ram : bool
        if True and temp_dir is not None, reduced_data will be saved on disk
        this increases the number of IO but reduces memory complexity when
         the number of subject and/or sessions is large

    Returns
    -------
    reduced_data: np array shape=[n_subjects, n_sessions]
        Element i, j of the array is a path to the reduced data of subject i session j.
        np.load(reduced_data[i, j]) has shape [n_timeframes, n_timeframes] first is dimension.
        If low_ram is False then reduced data is a list of list (n_subjects, n_sessions)
        of arrays of shape (n_timeframes, n_timeframes)

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
        X_reduced = [X_reduced[:, i] for i in slices]
        return X_reduced

    X = Parallel(n_jobs=n_jobs, verbose=verbose)(
        delayed(svd_i)(img) for img in imgs
    )
    return X


def clean_temp_dir(temp_dir, memory):
    if temp_dir is not None and memory is None:
        if os.path.exists(temp_dir):
            for root, dirs, files in os.walk(temp_dir, topdown=False):
                for name in files:
                    os.remove(os.path.join(root, name))
            os.rmdir(temp_dir)


def apply_rotation(basis, rotation, temp_dir):
    """
    Apply rotation to matrix
    Parameters
    ----------
    basis: array of shape [n_components, n_voxels]

    rotation: array of shape [n_components, n_components]

    temp_dir : str or None
        Path to dir where temporary results are stored. If None \
temporary results will be stored in memory. This can results in memory \
errors when the number of subjects and/or sessions is large

    Returns
    -------
    rotated_basis: array of shape [n_components, n_voxels]
        rotated_basis = rotation.dot(basis)
    """
    if isinstance(basis, np.ndarray):
        return rotation.dot(basis)
    else:
        name = basis.split(".npy")[0]
        np.save(name, rotation.dot(safe_load(basis)))
        return basis


def _compute_basis_subject_online(sessions, shared_response_list):
    """Computes subject's basis with shared response fixed

    Parameters
    ----------

    sessions : array of str
        Element i of the array is a path to the data
        collected during session i.
        Data are loaded with numpy.load and expected shape is
         [n_timeframes, n_voxels]
        n_timeframes and n_voxels are assumed to be the same across subjects
        n_timeframes can vary across sessions
        Each voxel's timecourse is assumed to have mean 0 and variance 1

    shared_response_list : list of array, element i has
    shape=[n_timeframes, n_components]
        shared response, element i is the shared response during session i

    Returns
    -------

    basis: array, shape=[n_components, n_voxels]
        basis
    """

    basis_i = None
    i = 0
    for session in sessions:
        data = safe_load(session).T
        if basis_i is None:
            basis_i = shared_response_list[i].T.dot(data)
        else:
            basis_i += shared_response_list[i].T.dot(data)
        i += 1
        del data
    return _compute_subject_basis(basis_i)


def ica_find_rotation(basis, n_subjects_ica, random_state):
    """
    Finds rotation r such that
    r.dot(srm.basis_list[0]) is the appropriate basis

    Parameters
    ----------

    basis: list of array

    n_subjects_ica: int
        Number of randomly selected subject used to fit ica

    transpose: bool
        if False: basis[i] has shape [n_components, n_voxels]
        if True: basis[i] has shape [n_voxels, n_components]
    """
    if n_subjects_ica == 0:
        raise ValueError(
            "ICA is used to find optimal rotation but \
        n_subjects_ica == 0. Please set a positive value for n_subjects_ica"
        )

    if n_subjects_ica is None:
        index = np.arange(len(basis))
        n_subjects_ica = len(basis)
    else:
        index = random_state.choice(
            np.arange(len(basis)), size=n_subjects_ica, replace=False
        )

    used_basis = []
    for i in index:
        basis_i = safe_load(basis[i])
        used_basis.append(basis_i)

    used_basis = np.concatenate(used_basis, axis=1) / np.sqrt(n_subjects_ica)

    n_features, n_samples = used_basis.shape
    used_basis = used_basis * np.sqrt(n_samples)
    K, W, Y = picard(used_basis, whiten=False, max_iter=1000)
    return W


def decorr_find_rotation(shared_response):
    """
    Finds rotation r such that
    r.dot(W) is the appropriate basis

    Parameters
    ----------

    shared_response: np array of shape [n_components, n_timeframes]
    """
    U, S, V = np.linalg.svd(shared_response, full_matrices=False)

    def get_sign(u):
        max_abs_cols = np.argmax(np.abs(u), axis=0)
        signs = np.sign(u[max_abs_cols, range(u.shape[1])])
        return signs

    sign = get_sign(S * V.T)
    U = sign * U
    return U.T


def fast_srm(
    reduced_data_list,
    n_iter,
    n_components,
    save_basis,
    tol,
    verbose,
    temp_dir,
    init,
    transpose,
    random_state,
):
    """Computes shared response and basis in reduced space

    Parameters
    ----------

    reduced_data_list : array, shape=[n_subjects, n_sessions]
    or array, shape=[n_subjects, n_sessions, n_timeframes, n_supervoxels]
        Element i, j of the array is a path to the data of subject i
        collected during session j.
        Data are loaded with numpy.load and expected
        shape is [n_timeframes, n_supervoxels]
        or Element i, j of the array is the data in array of
        shape=[n_timeframes, n_supervoxels]
        n_timeframes and n_supervoxels are
         assumed to be the same across subjects
        n_timeframes can vary across sessions
        Each voxel's timecourse is assumed to have mean 0 and variance 1

    n_iter : int
        Number of iterations performed

    n_components : int or None
        number of components

    Returns
    -------

    shared_response_list : list of array, element i has
     shape=[n_timeframes, n_components]
        shared response, element i is the shared response during session i
    """
    n_subjects = len(reduced_data_list)
    n_sessions = len(reduced_data_list[0])

    if init is None:
        shared_response = []
        for j in range(n_sessions):
            if transpose:
                n_voxels, n_timeframes = get_safe_shape(
                    reduced_data_list[0][j]
                )
            else:
                n_timeframes, n_voxels = get_safe_shape(
                    reduced_data_list[0][j]
                )

            shared_response.append(
                random_state.rand(n_timeframes, n_components)
            )
    else:
        shared_response = init

    reduced_basis = [None] * n_subjects
    grads = []
    losses = []
    for iteration in range(n_iter):
        for n in range(n_subjects):
            cov = None
            for m in range(n_sessions):
                if transpose:
                    data_nm = safe_load(reduced_data_list[n][m]).T
                else:
                    data_nm = safe_load(reduced_data_list[n][m])

                if cov is None:
                    cov = shared_response[m].T.dot(data_nm)
                else:
                    cov += shared_response[m].T.dot(data_nm)

            if save_basis and temp_dir is not None:
                path = os.path.join(temp_dir, "basis_%i" % n)
                np.save(path, _compute_subject_basis(cov))
                reduced_basis[n] = path + ".npy"
            else:
                reduced_basis[n] = _compute_subject_basis(cov)

        shared_response_new = _reduced_space_compute_shared_response(
            reduced_data_list, reduced_basis, n_components, transpose
        )

        _, n_voxels = get_safe_shape(reduced_basis[0])
        S_new = np.concatenate(shared_response_new, axis=0)
        S = np.concatenate(shared_response, axis=0)

        grad_norm = np.sum((S - S_new) ** 2) / (np.prod(S.shape))

        shared_response = shared_response_new
        loss = -np.sum(S_new ** 2) / (np.prod(S_new.shape))

        grads.append(grad_norm)
        if verbose:
            print(
                "iteration: %i grad_norm: %.5e loss: %.5e"
                % (iteration, grad_norm, loss)
            )

        if grad_norm < tol:
            break
        losses.append(loss)

    if grad_norm > tol:
        warnings.warn(
            "Convergence warning: ISRM did not converge. You should increase "
            "the number of iterations. Gradient norm is %.5e" % grad_norm
        )

    return reduced_basis, shared_response, grads, losses


class IdentifiableFastSRM(BaseEstimator, TransformerMixin):
    """SRM decomposition using a very low amount of memory and \
computational power thanks to the use of an atlas \
as described in [Richard2019]_.

    Given multi-subject data, factorize it as a shared response S \
among all subjects and an orthogonal transform (basis) W per subject:

    .. math:: X_i \\approx W_i S, \\forall i=1 \\dots N

    Parameters
    ----------

    atlas :  array, shape=[n_supervoxels, n_voxels] or array,\
shape=[n_voxels] or str or None, default=None
        Probabilistic or deterministic atlas on which to project the data. \
Deterministic atlas is an array of shape [n_voxels,] \
where values range from 1 \
to n_supervoxels. Voxels labelled 0 will be ignored. If atlas is a str the \
corresponding array is loaded with numpy.load and expected shape \
is (n_voxels,) for a deterministic atlas and \
(n_supervoxels, n_voxels) for a probabilistic atlas.

    n_components : int
        Number of timecourses of the shared coordinates

    n_iter : int
        Number of iterations to perform

    n_iter_reduced : int
        Number of iterations to perform on reduced data

    temp_dir : str or None
        Path to dir where temporary results are stored. If None \
temporary results will be stored in memory. This can results in memory \
errors when the number of subjects and/or sessions is large

    low_ram : bool
        If True and temp_dir is not None, reduced_data will be saved on \
disk. This increases the number of IO but reduces memory complexity when \
the number of subject and/or sessions is large

    n_jobs : int, optional, default=1
        The number of CPUs to use to do the computation. \
-1 means all CPUs, -2 all CPUs but one, and so on.

    verbose : bool or "warn"
        If True, logs are enabled. If False, logs are disabled. \
If "warn" only warnings are printed.

    aggregate: str or None, default="mean"
        If "mean", shared_response is the mean shared response \
from all subjects. If None, shared_response contains all \
subject-specific responses in shared space

    identifiability: str or None
        Possible values:
        - None: yields the SRM solution without modification
        - "ica": performs a linear ICA on spatial maps
        - "decorr" (default): shared response has diagonal covariance\
(diagonal values are sorted)

    n_subjects_ica: int
        Number of randomly selected subject used to fit ica
        (only used if identifiability = "ica")

    tol: float
        Stops if the norm of the gradient falls below tolerance value

    memory: str
        Path to the caching directory.
        Used to cache the fitting process.
        By default, no caching is done.

    use_pca: bool
        If True use pca as a preprocessing step.
        By default this is used when n_iter > 1

    Attributes
    ----------

    `basis_list`: list of array, element i has \
shape=[n_components, n_voxels] or list of str
        - if basis is a list of array, element i is the basis of subject i
        - if basis is a list of str, element i is the path to the basis \
of subject i that is loaded with np.load yielding an array of \
shape [n_components, n_voxels].

        Note that any call to the clean method erases this attribute

    Note
    -----

        **References:**
        H. Richard, L. Martin, A. Pinho, J. Pillow, B. Thirion, 2019: \
Fast shared response model for fMRI data (https://arxiv.org/pdf/1909.12537.pdf)

    """

    def __init__(
        self,
        atlas=None,
        n_components=20,
        n_iter=1,
        n_iter_reduced=10000,
        temp_dir=None,
        low_ram=False,
        n_jobs=1,
        verbose="warn",
        aggregate="mean",
        identifiability="ica",
        n_subjects_ica=None,
        tol=1e-12,
        memory=None,
        use_pca=True,
        random_state=0,
    ):

        self.n_jobs = n_jobs
        self.verbose = verbose
        self.n_components = n_components
        self.n_iter = n_iter
        self.atlas = atlas
        self.n_subjects_ica = n_subjects_ica
        self.identifiability = identifiability
        self.tol = tol
        self.memory = memory
        self.n_iter_reduced = n_iter_reduced
        self.random_state = check_random_state(random_state)

        if atlas is not None:
            self.use_pca = False
        else:
            self.use_pca = use_pca

        if aggregate is not None and aggregate != "mean":
            raise ValueError("aggregate can have only value mean or None")

        self.aggregate = aggregate

        self.basis_list = []

        if temp_dir is None:
            if self.verbose == "warn" or self.verbose is True:
                warnings.warn(
                    "temp_dir has value None. "
                    "All basis (spatial maps) and reconstructed "
                    "data will therefore be kept in memory."
                    "This can lead to memory errors when the "
                    "number of subjects "
                    "and/or sessions is large."
                )

            self.temp_dir = None
            self.low_ram = False

        if self.verbose == "warn":
            self.verbose = False

        if temp_dir is not None:
            if memory is not None:
                self.temp_dir = os.path.join(
                    memory, "fastsrm" + str(uuid.uuid4())
                )
            else:
                self.temp_dir = os.path.join(
                    temp_dir, "fastsrm" + str(uuid.uuid4())
                )

            self.low_ram = low_ram

    def clean(self):
        """This erases temporary files and basis_list attribute to \
free memory. This method should be called when fitted model \
is not needed anymore.
        """
        clean_temp_dir(self.temp_dir, self.memory)
        self.basis_list = []

    def fit(self, imgs):
        """Computes basis across subjects from input imgs

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

        Returns
        -------
        self : object
           Returns the instance itself. Contains attributes listed \
at the object level.
        """
        mem = Memory(self.memory)

        @mem.cache(
            ignore=[
                "n_jobs",
                "low_ram",
                "temp_dir",
                "verbose",
                "memory",
                "use_pca",
            ]
        )
        def _fit(
            imgs,
            atlas,
            n_jobs,
            low_ram,
            temp_dir,
            n_components,
            verbose,
            memory,
            n_iter,
            tol,
            n_iter_reduced,
            use_pca,
            random_state,
        ):
            if verbose is True:
                print("[FastSRM.fit] Checking input atlas")
            atlas_shape = check_atlas(atlas, n_components)

            if verbose is True:
                print("[FastSRM.fit] Checking input images")

            reshaped_input, imgs_, shapes = check_imgs(
                imgs, n_components=n_components, atlas_shape=atlas_shape
            )

            clean_temp_dir(temp_dir, memory)
            create_temp_dir(temp_dir)

            if atlas is None and use_pca is False:
                grads_reduced = []
                losses_reduced = []

                if verbose is True:
                    print("[FastSRM.fit] Finding basis")
                (
                    basis,
                    shared_response_list,
                    grads_full,
                    losses_full,
                ) = fast_srm(
                    imgs_,
                    n_iter=n_iter,
                    n_components=n_components,
                    save_basis=True,
                    tol=tol,
                    verbose=verbose,
                    temp_dir=temp_dir,
                    init=None,
                    transpose=True,
                    random_state=random_state,
                )
            else:
                if use_pca:
                    if verbose is True:
                        print("[FastSRM.fit] Reducing data using online svd")
                    reduced_data = svd_reduce(
                        imgs_, n_jobs=n_jobs, verbose=verbose
                    )
                    transpose = True

                elif atlas is not None:
                    if verbose is True:
                        print("[FastSRM.fit] Reducing data using input atlas")

                    reduced_data = reduce_data(
                        imgs_,
                        atlas=atlas,
                        n_jobs=n_jobs,
                        low_ram=low_ram,
                        temp_dir=temp_dir,
                    )
                    transpose = False

                if verbose is True:
                    print(
                        "[FastSRM.fit] Finds shared "
                        "response using reduced data"
                    )
                (
                    _,
                    shared_response_list,
                    grads_reduced,
                    losses_reduced,
                ) = fast_srm(
                    reduced_data,
                    n_iter=n_iter_reduced,
                    n_components=n_components,
                    save_basis=False,
                    tol=tol,
                    verbose=verbose,
                    temp_dir=temp_dir,
                    init=None,
                    transpose=transpose,
                    random_state=random_state,
                )

                if verbose is True:
                    print(
                        "[FastSRM.fit] Finds basis using "
                        "full data and shared response"
                    )

                (
                    basis,
                    shared_response_list,
                    grads_full,
                    losses_full,
                ) = fast_srm(
                    imgs_,
                    n_iter=n_iter,
                    n_components=n_components,
                    save_basis=True,
                    tol=tol,
                    verbose=verbose,
                    temp_dir=temp_dir,
                    init=shared_response_list,
                    transpose=True,
                    random_state=random_state,
                )

            grads = [grads_reduced, grads_full]
            losses = [losses_reduced, losses_full]
            return grads, losses, basis, shared_response_list, temp_dir

        grads, losses, basis, shared_response_list, temp_dir = _fit(
            imgs,
            self.atlas,
            self.n_jobs,
            self.low_ram,
            self.temp_dir,
            self.n_components,
            self.verbose,
            self.memory,
            self.n_iter,
            self.tol,
            self.n_iter_reduced,
            self.use_pca,
            self.random_state,
        )

        if temp_dir != self.temp_dir:
            clean_temp_dir(self.temp_dir, None)
            create_temp_dir(self.temp_dir)

        # Basis can either be a list of path or arrays depending on tempdir
        # If memory is used we don't know in which form they are so we correct this here
        if self.memory is not None:
            # If basis is already an array nothing to do
            # If basis is not an array but we want an array we load it
            if self.temp_dir is None and not isinstance(basis[0], np.ndarray):
                basis = [safe_load(basis[i]) for i in range(len(basis))]
            # If we don't want an array we save the basis where we want
            if self.temp_dir is not None:
                for i in range(len(basis)):
                    basis_i = safe_load(basis[i])
                    path = os.path.join(self.temp_dir, "basis_%i" % i)
                    np.save(path, basis_i)
                    basis[i] = path + ".npy"

        if self.identifiability == "ica":
            # Compute rotation matrix and apply
            r = ica_find_rotation(
                basis,
                n_subjects_ica=self.n_subjects_ica,
                random_state=self.random_state,
            )
            basis = Parallel(n_jobs=self.n_jobs)(
                delayed(apply_rotation)(b, r, self.temp_dir) for b in basis
            )

        if self.identifiability == "decorr":
            shared = np.concatenate(shared_response_list, axis=0).T
            r = decorr_find_rotation(shared)
            basis = Parallel(n_jobs=self.n_jobs)(
                delayed(apply_rotation)(b, r, self.temp_dir) for b in basis
            )

        self.grads = grads
        self.losses = losses
        self.basis_list = basis
        return self

    def fit_transform(self, imgs, subjects_indexes=None):
        """Computes basis across subjects and shared response from input imgs
        return shared response.

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

        subjects_indexes : list or None:
            if None imgs[i] will be transformed using basis_list[i]. \
Otherwise imgs[i] will be transformed using basis_list[subjects_index[i]]

        Returns
        --------
        shared_response : list of arrays, list of list of arrays or array
            - if imgs is a list of array and self.aggregate="mean": shared \
response is an array of shape (n_components, n_timeframes)
            - if imgs is a list of array and self.aggregate=None: shared \
response is a list of array, element i is the projection of data of \
subject i in shared space.
            - if imgs is an array or a list of list of array and \
self.aggregate="mean": shared response is a list of array, \
element j is the shared response during session j
            - if imgs is an array or a list of list of array and \
self.aggregate=None: shared response is a list of list of array, \
element i, j is the projection of data of subject i collected \
during session j in shared space.
        """
        self.fit(imgs)
        return self.transform(imgs, subjects_indexes=subjects_indexes)

    def transform(self, imgs, subjects_indexes=None):
        """From data in imgs and basis from training data,
        computes shared response.

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

        subjects_indexes : list or None:
            if None imgs[i] will be transformed using basis_list[i]. \
Otherwise imgs[i] will be transformed using basis[subjects_index[i]]

        Returns
        --------
        shared_response : list of arrays, list of list of arrays or array
            - if imgs is a list of array and self.aggregate="mean": shared \
response is an array of shape (n_components, n_timeframes)
            - if imgs is a list of array and self.aggregate=None: shared \
response is a list of array, element i is the projection of data of \
subject i in shared space.
            - if imgs is an array or a list of list of array and \
self.aggregate="mean": shared response is a list of array, \
element j is the shared response during session j
            - if imgs is an array or a list of list of array and \
self.aggregate=None: shared response is a list of list of array, \
element i, j is the projection of data of subject i collected \
during session j in shared space.
         """
        aggregate = self.aggregate
        if self.basis_list == []:
            raise NotFittedError("The model fit has not been run yet.")

        atlas_shape = check_atlas(self.atlas, self.n_components)
        reshaped_input, imgs, shapes = check_imgs(
            imgs,
            n_components=self.n_components,
            atlas_shape=atlas_shape,
            ignore_nsubjects=True,
            ignore_ncomponents=True,
        )
        check_indexes(subjects_indexes, "subjects_indexes")
        if subjects_indexes is None:
            subjects_indexes = np.arange(len(imgs))
        else:
            subjects_indexes = np.array(subjects_indexes)

        # Transform specific checks
        if len(subjects_indexes) < len(imgs):
            raise ValueError(
                "Input data imgs has len %i whereas "
                "subject_indexes has len %i. "
                "The number of basis used to compute "
                "the shared response should be equal "
                "to the number of subjects in imgs"
                % (len(imgs), len(subjects_indexes))
            )

        assert_valid_index(
            subjects_indexes, len(self.basis_list), "subjects_indexes"
        )

        shared_response = _compute_shared_response_online(
            imgs,
            self.basis_list,
            self.temp_dir,
            self.n_jobs,
            subjects_indexes,
            aggregate,
        )

        # If shared response has only 1 session we need to reshape it
        if reshaped_input:
            if aggregate == "mean":
                shared_response = shared_response[0]
            if aggregate is None:
                shared_response = [
                    shared_response[i][0] for i in range(len(subjects_indexes))
                ]

        return shared_response

    def inverse_transform(
        self, shared_response, subjects_indexes=None, sessions_indexes=None,
    ):
        """From shared response and basis from training data
        reconstruct subject's data

        Parameters
        ----------

        shared_response : list of arrays, list of list of arrays or array
            - if imgs is a list of array and self.aggregate="mean": shared \
response is an array of shape (n_components, n_timeframes)
            - if imgs is a list of array and self.aggregate=None: shared \
response is a list of array, element i is the projection of data of \
subject i in shared space.
            - if imgs is an array or a list of list of array and \
self.aggregate="mean": shared response is a list of array, \
element j is the shared response during session j
            - if imgs is an array or a list of list of array and \
self.aggregate=None: shared response is a list of list of array, \
element i, j is the projection of data of subject i collected \
during session j in shared space.

        subjects_indexes : list or None
            if None reconstructs data of all subjects used during train. \
Otherwise reconstructs data of subjects specified by subjects_indexes.

        sessions_indexes : list or None
            if None reconstructs data of all sessions. \
Otherwise uses reconstructs data of sessions specified by sessions_indexes.

        Returns
        -------
        reconstructed_data: list of list of arrays or list of arrays
            - if reconstructed_data is a list of list : element i, j is \
the reconstructed data for subject subjects_indexes[i] and \
session sessions_indexes[j] as an np array of shape n_voxels, \
n_timeframes
            - if reconstructed_data is a list : element i is the \
reconstructed data for subject \
subject_indexes[i] as an np array of shape n_voxels, n_timeframes

        """
        added_session, shared = check_shared_response(
            shared_response, self.aggregate, n_components=self.n_components
        )
        n_subjects = len(self.basis_list)
        n_sessions = len(shared)

        for j in range(n_sessions):
            assert_array_2axis(shared[j], "shared_response[%i]" % j)

        check_indexes(subjects_indexes, "subjects_indexes")
        check_indexes(sessions_indexes, "sessions_indexes")

        if subjects_indexes is None:
            subjects_indexes = np.arange(n_subjects)
        else:
            subjects_indexes = np.array(subjects_indexes)

        assert_valid_index(subjects_indexes, n_subjects, "subjects_indexes")

        if sessions_indexes is None:
            sessions_indexes = np.arange(len(shared))
        else:
            sessions_indexes = np.array(sessions_indexes)

        assert_valid_index(sessions_indexes, n_sessions, "sessions_indexes")

        data = []
        for i in subjects_indexes:
            data_ = []
            basis_i = safe_load(self.basis_list[i])
            if added_session:
                data.append(basis_i.T.dot(shared[0]))
            else:
                for j in sessions_indexes:
                    data_.append(basis_i.T.dot(shared[j]))
                data.append(data_)
        return data

    def add_subjects(self, imgs, shared_response):
        """ Add subjects to the current fit. Each new basis will be \
appended at the end of the list of basis (which can \
be accessed using self.basis)

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

        shared_response : list of arrays, list of list of arrays or array
            - if imgs is a list of array and self.aggregate="mean": shared \
response is an array of shape (n_components, n_timeframes)
            - if imgs is a list of array and self.aggregate=None: shared \
response is a list of array, element i is the projection of data of \
subject i in shared space.
            - if imgs is an array or a list of list of array and \
self.aggregate="mean": shared response is a list of array, \
element j is the shared response during session j
            - if imgs is an array or a list of list of array and \
self.aggregate=None: shared response is a list of list of array, \
element i, j is the projection of data of subject i collected \
during session j in shared space.
        """
        if self.basis_list == []:
            self.clean()
            create_temp_dir(self.temp_dir)

        atlas_shape = check_atlas(self.atlas, self.n_components)
        reshaped_input, imgs, shapes = check_imgs(
            imgs,
            n_components=self.n_components,
            atlas_shape=atlas_shape,
            ignore_nsubjects=True,
        )

        _, shared_response_list = check_shared_response(
            shared_response,
            n_components=self.n_components,
            aggregate=self.aggregate,
            input_shapes=shapes,
        )

        # we need to transpose shared_response_list to be consistent with
        # other functions
        shared_response_list = [
            shared_response_list[j].T for j in range(len(shared_response_list))
        ]

        if self.n_jobs == 1:
            basis = []
            for i, sessions in enumerate(imgs):
                basis_i = _compute_basis_subject_online(
                    sessions, shared_response_list
                )
                if self.temp_dir is None:
                    basis.append(basis_i)
                else:
                    path = os.path.join(
                        self.temp_dir, "basis_%i" % (len(self.basis_list) + i)
                    )
                    np.save(path, basis_i)
                    basis.append(path + ".npy")
                del basis_i
        else:
            if self.temp_dir is None:
                basis = Parallel(n_jobs=self.n_jobs)(
                    delayed(_compute_basis_subject_online)(
                        sessions, shared_response_list
                    )
                    for sessions in imgs
                )
            else:
                Parallel(n_jobs=self.n_jobs)(
                    delayed(_compute_and_save_corr_mat)(
                        imgs[i][j], shared_response_list[j], self.temp_dir
                    )
                    for j in range(len(imgs[0]))
                    for i in range(len(imgs))
                )

                basis = Parallel(n_jobs=self.n_jobs)(
                    delayed(_compute_and_save_subject_basis)(
                        len(self.basis_list) + i, sessions, self.temp_dir
                    )
                    for i, sessions in enumerate(imgs)
                )

        self.basis_list += basis
