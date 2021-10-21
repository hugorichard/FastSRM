from time import time
import sys
from cogspaces.datasets import fetch_mask
import os
from fastsrm2.fastsrm import (
    fastsrm,
    reduce_all,
    probsrm,
    detsrm,
    reduce_optimal,
    reduce_rena,
    reduce_randomproj,
)
from memory_profiler import memory_usage
import numpy as np
from joblib import Parallel, delayed, dump, load
from sklearn.model_selection import KFold

from fastsrm_age.timesegments import time_segment_matching
from fastsrm_age.utils import load_and_concat
import string
import random

data_dir = "/storage/store/work/hrichard/"
storage_dir = "/storage/store2/work/hrichard/"
mask = fetch_mask("/storage/store/work/hrichard/")


def random_string(length):
    # choose from all lowercase letter
    letters = string.ascii_lowercase
    result_str = "".join(random.choice(letters) for i in range(length))
    return result_str


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


def do_srm(config, n_components, algo):
    result_directory = os.path.join(
        storage_dir,
        "fastsrm_age",
        "timesegment",
        "results_sm_None_rm_hv_False",
    )
    os.makedirs(result_directory, exist_ok=True)
    print("Start experiment with config %s" % config)
    if config == "forrest":
        n_subjects = 19
        n_sessions = 7
    elif config == "gallant":
        n_subjects = 12
        n_sessions = 12
    elif config == "sherlock":
        n_subjects = 16
        n_sessions = 5
    elif config == "raiders":
        n_subjects = 11
        n_sessions = 10
    paths = np.array(
        [
            [
                os.path.join(
                    data_dir,
                    "masked_data",
                    "smoothing_None",
                    "rm_hv_confounds_False",
                    "%s/subject_%i_session_%i.npy" % (config, i, j),
                )
                for j in range(n_sessions)
            ]
            for i in range(n_subjects)
        ]
    )
    cv = KFold(n_splits=5, shuffle=False)
    for i_ses, (sessions_train, sessions_test) in enumerate(
        cv.split(np.arange(n_sessions))
    ):
        print(
            "../results/cv_accuracy_timesegment_matching-%i-%i-%s-%s"
            % (i_ses, n_components, algo, config)
        )
        to_clean = []
        if os.path.exists(
            "../results/cv_accuracy_timesegment_matching-%i-%i-%s-%s.npy"
            % (i_ses, n_components, algo, config),
        ):
            print(
                "Done: %s"
                % "../results/cv_accuracy_timesegment_matching-%i-%i-%s-%s"
                % (i_ses, n_components, algo, config)
            )
            continue
        paths_train = paths[:, sessions_train]
        t0 = time()
        if algo == "detsrm":
            X_train = load_and_concat(paths_train)
            W = detsrm(
                X_train,
                n_components,
                n_iter=1000,
                random_state=0,
                tol=1e-2,
                verbose=True,
            )[0]
        if algo == "probsrm":
            X_train = load_and_concat(paths_train)
            W = probsrm(
                X_train,
                n_components,
                n_iter=1000,
                random_state=0,
                tol=1e-2,
                verbose=True,
            )[0]
        if "fastsrm" in algo:
            algo_name, method, func_name, n_regions = algo.split("_")
            if func_name == "pca":
                func = reduce_optimal
            if func_name == "rena":
                func = reduce_rena
            if func_name == "proj":
                func = reduce_randomproj

            paths_atlas = [
                "/storage/store2/work/hrichard/temp_srm/U%i_%i_%s_%s_timesegmentmaching_%s.npy"
                % (isub, i_ses, random_string(10), config, algo)
                for isub in range(n_subjects)
            ]

            paths_basis = [
                "/storage/store2/work/hrichard/temp_srm/W%i_%i_%s_%s_timesegmentmaching_%s.npy"
                % (isub, i_ses, random_string(10), config, algo)
                for isub in range(n_subjects)
            ]

            W = fastsrm(
                paths_train,
                n_components,
                n_iter=1000,
                method=method,
                n_regions=n_regions,
                mask=mask,
                func=func,
                random_state=0,
                paths_basis=paths_basis,
                paths_atlas=paths_atlas,
                tol=1e-2,
            )[0]

            to_clean += paths_basis + paths_atlas
        dt = time() - t0
        np.save(
            "../results/fit_time_timesegmentmatching-%i-%i-%s-%s.npy"
            % (i_ses, n_components, algo, config),
            dt,
        )

        paths_test = paths[:, sessions_test]
        X_test = load_and_concat(paths_test)
        shared_response = [W[i].T.dot(X_test[i]) for i in range(n_subjects)]
        cv_scores = time_segment_matching(shared_response, win_size=9)
        np.save(
            "../results/cv_accuracy_timesegment_matching-%i-%i-%s-%s"
            % (i_ses, n_components, algo, config),
            cv_scores,
        )

        # clean paths
        for p in to_clean:
            os.system("rm %s" % p)


def algos_k(k):
    algos = []
    for method in ["prob", "det"]:
        # algos.append("fastsrm_%s_%s_%s" % (method, "rena", "1n"))
        # algos.append("fastsrm_%s_%s_%s" % (method, "proj", "5n"))
        # algos.append("fastsrm_%s_%s_%i" % (method, "pca", k + 1))
        algos.append("fastsrm_%s_%s_%s" % (method, "pca", "2n"))
    return algos


datasets = ["forrest", "sherlock"]
Parallel(n_jobs=8, verbose=True)(
    delayed(do_srm)(dataset, n_components, algo)
    for n_components in [5, 10, 20, 50]
    for algo in algos_k(n_components)
    for dataset in datasets
)
