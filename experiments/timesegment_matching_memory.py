from time import time
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


def random_string(length):
    # choose from all lowercase letter
    letters = string.ascii_lowercase
    result_str = "".join(random.choice(letters) for i in range(length))
    return result_str


data_dir = "/storage/store/work/hrichard/"
storage_dir = "/storage/store2/work/hrichard/"
mask = fetch_mask("/storage/store/work/hrichard/")


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


def do_srm(config, n_components, algo, session):
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
    for i, (sessions_train, sessions_test) in enumerate(
        cv.split(np.arange(n_sessions))
    ):
        if i != session:
            continue
        paths_train = paths[:, sessions_train]
        # if algo == "brainiakdetsrm":
        #     X_train = load_and_concat(paths_train)
        #     W = (
        #         DetSRM(n_iter=2, features=n_components, rand_seed=0)
        #         .fit(X_train)
        #         .w_
        #     )
        # if algo == "brainiakprobsrm":
        #     X_train = load_and_concat(paths_train)
        #     W = (
        #         SRM(n_iter=2, features=n_components, rand_seed=0)
        #         .fit(X_train)
        #         .w_
        #     )
        if algo == "detsrm":
            X_train = load_and_concat(paths_train)
            W = detsrm(X_train, n_components, n_iter=2, random_state=0)[0]
        if algo == "probsrm":
            X_train = load_and_concat(paths_train)
            W = probsrm(X_train, n_components, n_iter=2, random_state=0)[0]

        if "fastsrm" in algo:
            algo_name, method, func_name, n_regions = algo.split("_")

            if "n" not in n_regions:
                n_regions = int(n_regions)

            if func_name == "pca":
                func = reduce_optimal
            if func_name == "rena":
                func = reduce_rena
            if func_name == "proj":
                func = reduce_randomproj

            paths_atlas = [
                "/storage/store2/work/hrichard/temp_srm/U%i_%i_%s_%s_timesegmentmaching_memory_%s.npy"
                % (isub, i, random_string(10), config, algo)
                for isub in range(n_subjects)
            ]

            paths_basis = [
                "/storage/store2/work/hrichard/temp_srm/W%i_%i_%s_%s_timesegmentmaching_memory_%s.npy"
                % (isub, i, random_string(10), config, algo)
                for isub in range(n_subjects)
            ]

            fastsrm(
                paths_train,
                n_components,
                n_iter=2,
                method=method,
                n_regions=n_regions,
                mask=mask,
                func=func,
                random_state=0,
                paths_basis=paths_basis,
                paths_atlas=paths_atlas,
            )[0]

            # clean paths
            for p in paths_basis:
                os.system("rm %s" % p)

            for p in paths_atlas:
                os.system("rm %s" % p)


def do_memory_srm(config, n_components, algo, session):
    # if os.path.exists(
    #     "../results/memory_usage_timesegmentmatching-%i-%i-%s-%s.npy"
    #     % (session, n_components, algo, config)
    # ):
    if False:

        print(
            "Done: ../results/memory_usage_timesegmentmatching-%i-%i-%s-%s.npy"
            % (session, n_components, algo, config)
        )
        return None
    else:
        print(
            "Doing: ../results/memory_usage_timesegmentmatching-%i-%i-%s-%s.npy"
            % (session, n_components, algo, config)
        )

    memory = np.max(
        memory_usage(lambda: do_srm(config, n_components, algo, session))
    )
    np.save(
        "../results/memory_usage_timesegmentmatching-%i-%i-%s-%s.npy"
        % (session, n_components, algo, config),
        memory,
    )


def algos_k(k):
    algos = []
    # algos = ["probsrm", "detsrm"]
    for method in ["prob", "det"]:
        # algos.append("fastsrm_%s_%s_%s" % (method, "rena", "1n"))
        # algos.append("fastsrm_%s_%s_%s" % (method, "proj", "5n"))
        # algos.append("fastsrm_%s_%s_%i" % (method, "pca", k + 1))
        algos.append("fastsrm_%s_%s_%s" % (method, "pca", "2n"))
    return algos


import sys

config, n_components = sys.argv[1:]
print(n_components, config)
for algo in algos_k(int(n_components)):
    for session in np.arange(5):
        do_memory_srm(config, int(n_components), algo, session)

# for dataset in ["sherlock"]:
#     do_srm(dataset, n_components)

# for n_components in [5, 10, 20, 50, 50]:
#     for dataset in ["forrest", "sherlock", "raiders", "gallant"]:
#         do_srm(dataset, n_components)
