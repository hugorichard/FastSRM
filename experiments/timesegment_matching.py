from time import time
import sys
import os
from fastsrm.srm import (
    probsrm,
    detsrm,
)
from fastsrm.fastsrm import fastsrm
import numpy as np
from joblib import Parallel, delayed
from sklearn.model_selection import KFold

from fastsrm.utils import time_segment_matching
from fastsrm.utils import load_and_concat
from memory_profiler import memory_usage


os.makedirs("./results", exist_ok=True)


def do_find_basis(config, algo, paths_train, n_components, i_ses):
    if algo == "det":
        X_train = load_and_concat(paths_train)
        W = detsrm(
            X_train,
            n_components,
            n_iter=1000,
            random_state=0,
            tol=1e-2,
            verbose=True,
        )[0]
    if algo == "prob":
        X_train = load_and_concat(paths_train)
        W = probsrm(
            X_train,
            n_components,
            n_iter=1000,
            random_state=0,
            tol=1e-2,
            verbose=True,
        )[0]
    if "fastprob" in algo:
        os.makedirs(
            "./tempprob/%i-%i-%s-%s" % (i_ses, n_components, algo, config),
            exist_ok=True,
        )
        W = fastsrm(
            paths_train,
            n_components,
            n_iter=1000,
            method="prob",
            random_state=0,
            tol=1e-2,
            temp_dir="./tempprob/%i-%i-%s-%s"
            % (i_ses, n_components, algo, config),
        )[0]

    if "fastdet" in algo:
        os.makedirs(
            "./tempdet/%i-%i-%s-%s" % (i_ses, n_components, algo, config),
            exist_ok=True,
        )
        W = fastsrm(
            paths_train,
            n_components,
            n_iter=1000,
            method="det",
            random_state=0,
            tol=1e-2,
            temp_dir="./tempdet/%i-%i-%s-%s"
            % (i_ses, n_components, algo, config),
        )[0]

    np.save(
        "./results/basis_timesegmentmatching-%i-%i-%s-%s.npy"
        % (i_ses, n_components, algo, config),
        W,
    )


def do_srm(n_components, algo):
    config = "sherlock"
    n_subjects = 16
    n_sessions = 5
    paths = np.array(
        [
            [
                "./data/masked_sherlock/subject_%i_session_%i.npy" % (i, j)
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
            "./results/cv_accuracy_timesegment_matching-%i-%i-%s-%s"
            % (i_ses, n_components, algo, config)
        )
        if os.path.exists(
            "./results/cv_accuracy_timesegment_matching-%i-%i-%s-%s.npy"
            % (i_ses, n_components, algo, config),
        ):
            print(
                "Done: %s"
                % "./results/cv_accuracy_timesegment_matching-%i-%i-%s-%s"
                % (i_ses, n_components, algo, config)
            )
            continue
        paths_train = paths[:, sessions_train]
        t0 = time()
        memory = np.max(
            memory_usage(
                lambda: do_find_basis(
                    config, algo, paths_train, n_components, i_ses
                )
            )
        )
        dt = time() - t0
        np.save(
            "./results/memory_usage_timesegmentmatching-%i-%i-%s-%s.npy"
            % (i_ses, n_components, algo, config),
            memory,
        )
        np.save(
            "./results/fit_time_timesegmentmatching-%i-%i-%s-%s.npy"
            % (i_ses, n_components, algo, config),
            dt,
        )
        W = np.load(
            "./results/basis_timesegmentmatching-%i-%i-%s-%s.npy"
            % (i_ses, n_components, algo, config)
        )

        paths_test = paths[:, sessions_test]
        X_test = load_and_concat(paths_test)
        print(W[0])
        shared_response = [W[i].T.dot(X_test[i]) for i in range(n_subjects)]
        cv_scores = time_segment_matching(shared_response, win_size=9)
        np.save(
            "./results/cv_accuracy_timesegment_matching-%i-%i-%s-%s"
            % (i_ses, n_components, algo, config),
            cv_scores,
        )


if __name__ == "__main__":
    n_components = int(sys.argv[1])
    algo = sys.argv[2]
    print(n_components, algo)
    do_srm(n_components, algo)
