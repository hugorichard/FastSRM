from fastsrm2.srm import projection, detsrm, probsrm
import nibabel as nib
import matplotlib.pyplot as plt
from memory_profiler import memory_usage
import numpy as np
from time import time
from fastsrm2.fastsrm import (
    fastsrm,
    reduce_rena,
    reduce_optimal,
    reduce_randomproj,
)
from fastsrm2.exp_utils import reg_error
import numpy as np
from joblib import delayed, Parallel


dim = (30, 30, 30)
m, v, k, n = 10, np.prod(dim), 50, 1000


def generate_data(seed):
    rng = np.random.RandomState(seed)
    Sigma = rng.dirichlet(np.ones(k), 1).flatten()
    S = np.sqrt(Sigma)[:, None] * rng.randn(k, n)
    W = np.array([projection(rng.randn(v, k)) for i in range(m)])
    sigmas = 0.1 * rng.rand(m)
    N = np.array([sigmas[i] * rng.randn(v, n) for i in range(m)])
    X = np.array([W[i].dot(S) + N[i] for i in range(m)])
    S_true = S

    def to_niimgs(Xt, dim):
        X = np.copy(Xt).T
        from nilearn.masking import _unmask_from_to_3d_array
        import nibabel

        p = np.prod(dim)
        assert len(dim) == 3
        assert X.shape[-1] <= p
        mask = np.zeros(p).astype(np.bool)
        mask[: X.shape[-1]] = 1
        assert mask.sum() == X.shape[1]
        mask = mask.reshape(dim)
        X = np.rollaxis(
            np.array([_unmask_from_to_3d_array(x, mask) for x in X]),
            0,
            start=4,
        )
        affine = np.eye(4)
        return (
            nibabel.Nifti1Image(X, affine),
            nibabel.Nifti1Image(mask.astype(np.float), affine),
        )

    _, mask = to_niimgs(X[0], dim)

    for i, x in enumerate(X):
        np.save("../synthetic_data/subject_%i_seed_%i.npy" % (i, seed), x)
    nib.save(mask, "../synthetic_data/mask.nii.gz")


def do_algo(algo, X, mask, it, rng=0):
    t0 = time()
    if algo == "detsrm":
        S = detsrm(X, k, n_iter=it, random_state=rng)[1]
    if algo == "probsrm":
        S = probsrm(X, k, n_iter=it, random_state=rng)[1]
    if "fastsrm" in algo:
        algo_name, method, func_name, n_regions = algo.split("_")
        if func_name == "pca":
            func = reduce_optimal
        if func_name == "rena":
            func = reduce_rena
        if func_name == "proj":
            func = reduce_randomproj

        if "n" not in n_regions:
            n_regions = int(n_regions)

        S = fastsrm(
            X,
            k,
            n_iter=it,
            paths_atlas=["../temp/W_%i.npy" % i for i in range(10)],
            paths_basis=["../temp/U_%i.npy" % i for i in range(10)],
            method=method,
            n_regions=n_regions,
            mask=mask,
            func=func,
            random_state=rng,
        )[1]


# generate_data(0)

X = ["../synthetic_data/subject_%i_seed_%i.npy" % (i, 0) for i in range(10)]
mask = "../synthetic_data/mask.nii.gz"
algos = ["probsrm"]
method = "prob"
algos.append("fastsrm_%s_%s_%s" % (method, "rena", "1n"))
algos.append("fastsrm_%s_%s_%s" % (method, "proj", "5n"))
algos.append("fastsrm_%s_%s_%i" % (method, "pca", k + 1))
algos.append("fastsrm_%s_%s_%s" % (method, "pca", "2n"))

for algo in algos:
    data = memory_usage(lambda: do_algo(algo, X, mask, 30, 0))
    plt.plot(data, label=algo)
plt.legend()
plt.show()
# algos = ["probsrm", "detsrm", "brainiakdetsrm", "brainiakprobsrm"]
# # for method in ["prob", "det"]:
# #     for func_name in ["pca", "rena", "proj"]:
# #         for n_regions in [str(n), str(k)]:
# #             algos.append("fastsrm_%s_%s_%s" % (method, func_name, n_regions))
# for method in ["prob"]:
#     for func_name in ["pca", "rena", "proj"]:
#         for n_regions in [str(int(2 * n))]:
#             algos.append("fastsrm_%s_%s_%s" % (method, func_name, n_regions))
#     algos.append("fastsrm_%s_%s_%s" % (method, "pca", str(2 * k)))
# algos = []
# for method in ["prob"]:
#     for func_name in ["proj"]:
#         for n_regions in [str(int(20 * n))]:
#             algos.append("fastsrm_%s_%s_%s" % (method, func_name, n_regions))

# for method in ["prob"]:
#     for func_name in ["proj"]:
#         for n_regions in [str(int(v))]:
#             algos.append("fastsrm_%s_%s_%s" % (method, func_name, n_regions))

# algos = []
# for method in ["prob"]:
#     for func_name in ["proj"]:
#         for n_regions in [str(int(50 * n))]:
#             algos.append("fastsrm_%s_%s_%s" % (method, func_name, n_regions))

# # algos = []
# for method in ["prob"]:
#     for func_name in ["proj"]:
#         for n_regions in [str(int(100 * n))]:
#             algos.append("fastsrm_%s_%s_%s" % (method, func_name, n_regions))


# iters = np.arange(1, 41, 5)
# seeds = np.arange(30)
# res = Parallel(n_jobs=10, verbose=10)(
#     delayed(do_expe)(it, seed, algo)
#     for it in iters
#     for seed in seeds
#     for algo in algos
# )
# a, b, c = len(iters), len(seeds), len(algos)
# res = np.array(res)
# res = res.reshape((a, b, c, -1))
# for a, algo in enumerate(algos):
#     np.save("../results/synthetic_%s.npy" % algo, res[:, :, a, :])
