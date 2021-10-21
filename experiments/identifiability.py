from time import time
import sys
from cogspaces.datasets import fetch_mask
import os
from fastsrm.fastsrm import fastsrm
from memory_profiler import memory_usage
import numpy as np
from joblib import Parallel, delayed, dump, load
from sklearn.model_selection import ShuffleSplit

from fastsrm_age.timesegments import time_segment_matching
from fastsrm_age.utils import load_and_concat
import string
import random
from brainiak.funcalign.srm import SRM

data_dir = "/storage/store/work/hrichard/"
storage_dir = "/storage/store3/work/hrichard/"


def do_srm(config, n_components, algo):
    result_directory = os.path.join(storage_dir, "fastsrm", "identifiability",)
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
    cv = ShuffleSplit(n_splits=10, train_size=0.5, random_state=0)
    for i_sub, subs in enumerate(cv.split(np.arange(n_subjects))):
        for is_train, c_sub in enumerate(subs):
            print(is_train, i_sub, c_sub)
            time_dir = os.path.join(result_directory, "time",)
            os.makedirs(time_dir, exist_ok=True)
            temp_dir = os.path.join(
                result_directory,
                "temp",
                "%i-%i-%s-%i-%s"
                % (i_sub, is_train, config, n_components, algo),
            )
            os.makedirs(temp_dir, exist_ok=True)
            t0 = time()
            if algo == "fast":
                S = fastsrm(
                    paths[c_sub],
                    n_components=n_components,
                    n_jobs=10,
                    verbose=True,
                    n_iter=500,
                    method="prob",
                    temp_dir=temp_dir,
                    tol=0,
                    random_state=0,
                )[1]

            if algo == "brainiak":
                srm = SRM(
                    n_iter=500, features=n_components, rand_seed=0
                )
                srm.fit(
                    [
                        np.column_stack([np.load(XX) for XX in X])
                        for X in paths[c_sub]
                    ]
                )
                S = srm.s_
            dt = time() - t0
            save_time = os.path.join(
                time_dir,
                "%i-%i-%s-%i-%s"
                % (i_sub, is_train, config, n_components, algo),
            )
            np.save(save_time, dt)

            save_dir = os.path.join(result_directory, "shared", algo)
            save_path = os.path.join(
                save_dir,
                "%i-%i-%s-%i-%s.npy"
                % (i_sub, is_train, config, n_components, algo),
            )
            np.save(save_path, S)


datasets = ["forrest", "sherlock"]
algos = ["brainiak", "fast"]
Parallel(n_jobs=4, verbose=True)(
    delayed(do_srm)(dataset, 10, algo)
    for algo in algos
    for dataset in datasets
)
