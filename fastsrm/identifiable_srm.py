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

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils import check_random_state
from fastsrm.check_inputs import (
    check_imgs,
    check_indexes,
    assert_valid_index,
    check_shared_response,
    assert_array_2axis,
)

from sklearn.exceptions import NotFittedError
from fastsrm.srm import probsrm, detsrm, safe_load
from fastsrm.fastsrm import fastsrm, projection
from fastsrm.utils import extract_slices


def create_temp_dir(temp_dir):
    """
    This check whether temp_dir exists and creates dir otherwise
    """
    if temp_dir is None:
        return None

    if not os.path.exists(temp_dir):
        os.makedirs(temp_dir)
    else:
        raise ValueError(
            "Path %s already exists. "
            "When a model is used, filesystem should be cleaned "
            "by using the .clean() method" % temp_dir
        )


def clean_temp_dir(temp_dir):
    if temp_dir is not None:
        if os.path.exists(temp_dir):
            for root, dirs, files in os.walk(temp_dir, topdown=False):
                for name in files:
                    os.remove(os.path.join(root, name))
            os.rmdir(temp_dir)


class IdentifiableFastSRM(BaseEstimator, TransformerMixin):
    """SRM decomposition using a very low amount of memory and \
computational power thanks to the use of an atlas \
as described in [Richard2019].

    Given multi-subject data, factorize it as a shared response S \
among all subjects and an orthogonal transform (basis) W per subject:

    .. math:: X_i \\approx W_i S, \\forall i=1 \\dots N

    Parameters
    ----------

    method : str, default="prob"
        if "prob", uses Probabilistic SRM
        if "det", uses Deterministic SRM

    n_components : int
        Number of timecourses of the shared coordinates

    n_iter : int
        Number of iterations to perform

    temp_dir : str or None
        Path to dir where temporary results are stored. If None \
temporary results will be stored in memory. This can results in memory \
errors when the number of subjects and/or sessions is large

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

    tol: float
        Stops if the norm of the gradient falls below tolerance value

    use_pca: bool or "auto"
        If True use pca as a preprocessing step.
        If "auto", it is set to True
        if n_iter > 1 and n_voxels > 10 * n_timeframes.

    random_state: int, RandomState instance or None, default=None
        Controls the randomness of the initialization.

    Attributes
    ----------

    `basis_list`: list of array, element i has \
shape=[n_components, n_voxels] or list of str
        - if basis is a list of array, element i is the basis of subject i
        - if basis is a list of str, element i is the path to the basis \
of subject i that is loaded with np.load yielding an array of \
shape [n_components, n_voxels].


    `noise_variance` : np array of shape (n_views,)
        Noise variance

    `source_covariance` : np array of shape (n_components,)
        (Diagonal) Shared response covariance

        Note that any call to the clean method erases this attribute

    Note
    -----

        **References:**
        H. Richard, L. Martin, A. Pinho, J. Pillow, B. Thirion, 2019: \
Fast shared response model for fMRI data (https://arxiv.org/pdf/1909.12537.pdf)

    """

    def __init__(
        self,
        method="prob",
        n_components=20,
        n_iter=1,
        temp_dir=None,
        n_jobs=1,
        verbose="warn",
        aggregate="mean",
        tol=1e-12,
        use_pca="auto",
        random_state=0,
    ):

        self.n_jobs = n_jobs
        self.verbose = verbose
        self.n_components = n_components
        self.n_iter = n_iter
        self.tol = tol
        self.random_state = check_random_state(random_state)
        self.temp_dir = temp_dir
        self.temp_dir_ = None
        self.use_pca = use_pca

        if method != "prob" and method != "det":
            raise ValueError("method can only have value prob or det")
        self.method = method

        if aggregate is not None and aggregate != "mean":
            raise ValueError("aggregate can have only value mean or None")
        self.aggregate = aggregate

        self.basis_list = []

        if self.verbose == "warn":
            self.verbose = False

    def clean(self):
        """This erases temporary files and basis_list attribute to \
free memory. This method should be called when fitted model \
is not needed anymore.
        """
        clean_temp_dir(self.temp_dir_)
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
        if self.verbose is True:
            print("[FastSRM.fit] Checking input images")

        reshaped_input, imgs_, shapes = check_imgs(
            imgs, n_components=self.n_components
        )
        n_voxels_ = shapes[0, 0, 0]
        n_timeframes_ = np.sum(shapes[0, :, 1])
        if self.use_pca == "auto" and n_voxels_ < 10 * n_timeframes_:
            use_pca_ = True
        else:
            use_pca_ = self.use_pca

        if self.temp_dir is not None:
            if self.temp_dir_ is None:
                self.temp_dir_ = os.path.join(
                    self.temp_dir, "fastsrm" + str(uuid.uuid4())
                )
            else:
                clean_temp_dir(self.temp_dir_)
            create_temp_dir(self.temp_dir_)

        if use_pca_:
            if self.method == "det":
                W, S = fastsrm(
                    imgs_,
                    self.n_components,
                    self.n_jobs,
                    self.verbose,
                    self.n_iter,
                    self.tol,
                    "det",
                    self.temp_dir_,
                    self.random_state,
                )
            elif self.method == "prob":
                W, S, sigmas, Sigma = fastsrm(
                    imgs_,
                    self.n_components,
                    self.n_jobs,
                    self.verbose,
                    self.n_iter,
                    self.tol,
                    "prob",
                    self.temp_dir_,
                    self.random_state,
                )
        else:
            X = [
                np.column_stack([safe_load(xi) for xi in img])
                for img in imgs_
            ]
            if self.method == "det":
                W, S = detsrm(
                    X,
                    self.n_components,
                    self.n_iter,
                    self.random_state,
                    self.verbose,
                    self.tol,
                )
            elif self.method == "prob":
                W, S, sigmas, Sigma = probsrm(
                    X,
                    self.n_components,
                    self.n_iter,
                    self.random_state,
                    1,
                    self.verbose,
                    self.tol,
                )
                self.noise_variance = sigmas
                self.source_covariance = Sigma

        self.basis_list = W
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
        if self.basis_list == []:
            raise NotFittedError("The model fit has not been run yet.")

        reshaped_input, imgs, shapes = check_imgs(
            imgs,
            n_components=self.n_components,
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

        S = []
        if self.aggregate == "mean":
            for j in range(len(imgs[0])):
                Sj = None
                for i, iw in enumerate(subjects_indexes):
                    if Sj is None:
                        Sj = safe_load(self.basis_list[iw]).T.dot(
                            safe_load(imgs[i][j])
                        )
                    else:
                        Sj += safe_load(self.basis_list[iw]).T.dot(
                            safe_load(imgs[i][j])
                        )
                S.append(Sj / len(subjects_indexes))
        else:
            for i, iw in enumerate(subjects_indexes):
                S.append(
                    [
                        safe_load(self.basis_list[iw]).T.dot(safe_load(imgij))
                        for imgij in imgs[i]
                    ]
                )

        shared_response = S
        # If shared response has only 1 session we need to reshape it
        if reshaped_input:
            if self.aggregate == "mean":
                shared_response = shared_response[0]
            if self.aggregate is None:
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
                data.append(basis_i.dot(shared[0]))
            else:
                for j in sessions_indexes:
                    data_.append(basis_i.dot(shared[j]))
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
            create_temp_dir(self.temp_dir_)

        reshaped_input, imgs, shapes = check_imgs(
            imgs,
            n_components=self.n_components,
            ignore_nsubjects=True,
            ignore_ncomponents=True,
        )

        _, shared_response_list = check_shared_response(
            shared_response,
            aggregate=self.aggregate,
            n_components=self.n_components,
            input_shapes=shapes,
        )

        basis = []
        for img in imgs:
            basis.append(
                projection(
                    np.sum(
                        [
                            safe_load(img[j]).dot(shared_response_list[j].T)
                            for j in range(len(shared_response_list))
                        ],
                        axis=0,
                    )
                )
            )
        self.basis_list += basis
