from glob import glob
import numpy as np
import os
from time import time
from fastsrm2.fastsrm import probsrm, fastsrm
from memory_profiler import memory_usage
from joblib import dump

import string
import random


def random_string(length):
    """Generates a random string"""
    # choose from all lowercase letter
    letters = string.ascii_lowercase
    result_str = "".join(random.choice(letters) for i in range(length))
    return result_str


path = glob("/storage/store/work/hrichard/masked_data/camcan/func/*.npy")
n_subjects = len(path)


def do_memory_srm():
    """Memory for srm."""
    dump(
        probsrm(
            path, 10, n_iter=1000, random_state=0, tol=1e-2, verbose=True,
        ),
        "/storage/store2/work/hrichard/temp_srm/res_probsrm_camcan",
    )


def do_memory_fastsrm():
    """Memory for FastSRM."""
    dump(
        fastsrm(
            path,
            10,
            method="prob",
            n_iter=1000,
            tol=1e-2,
            random_state=0,
            paths_basis=[
                "/storage/store2/work/hrichard/temp_srm/W%i_%s_camcan.npy"
                % (isub, random_string(10))
                for isub in range(n_subjects)
            ],
            paths_atlas=[
                "/storage/store2/work/hrichard/temp_srm/U%i_%s_camcan.npy"
                % (isub, random_string(10))
                for isub in range(n_subjects)
            ],
        ),
        "/storage/store2/work/hrichard/temp_srm/res_fastsrm_camcan",
    )


# t0 = time()
# memory = np.max(memory_usage(do_memory_srm))
# dt = time() - t0
# np.save("../results/camcan_time_srm.npy", dt)
# np.save("../results/camcan_memory_srm.npy", memory)
t0 = time()
memory = np.max(memory_usage(do_memory_fastsrm))
dt = time() - t0
np.save("../results/camcan_time_fastsrm.npy", dt)
np.save("../results/camcan_memory_fastsrm.npy", memory)
