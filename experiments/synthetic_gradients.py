from fastsrm2.srm import projection, detsrm, probsrm
import os
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

# from brainiak.funcalign.srm import SRM, DetSRM


dim = (50, 50, 50)
m, v, k, n = 10, np.prod(dim), 50, 1000


def do_expe(it, seed, algo):
    rng = np.random.RandomState(seed)
    Sigma = rng.dirichlet(np.ones(k), 1).flatten()
    S = np.sqrt(Sigma)[:, None] * rng.randn(k, n)
    W = np.array([projection(rng.randn(v, k)) for i in range(m)])
    sigmas = 0.1 * rng.rand(m)
    N = np.array([sigmas[i] * rng.randn(v, n) for i in range(m)])
    X = np.array([W[i].dot(S) + N[i] for i in range(m)])
    S_true = S
    t_init = time()

    def callback(source, gnorm, current_iter, current_time):
        return (
            float(reg_error(np.copy(S_true), np.copy(source))),
            float(current_time - t_init),
            current_iter,
            seed,
            algo,
            float(gnorm),
        )

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

    if algo == "detsrm":
        S = detsrm(
            X, k, n_iter=it, random_state=rng, callback=callback, tol=-1,
        )[-1]
    if algo == "probsrm":
        S = probsrm(
            X, k, n_iter=it, random_state=rng, callback=callback, tol=-1,
        )[-1]
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

        try:
            S = fastsrm(
                X,
                k,
                n_iter=it,
                method=method,
                n_regions=n_regions,
                mask=mask,
                func=func,
                random_state=rng,
                callback=callback,
                tol=-1,
            )[-1]
        except:
            print("-----------ERROR--------------")
            print(seed)
            print(algo)
            raise ValueError("Error seed: %i, algo: %s" % (seed, algo))
    return np.array(S)


algos = []
# algos = ["probsrm", "detsrm"]
for method in ["prob", "det"]:
    # algos.append("fastsrm_%s_%s_%s" % (method, "rena", "1n"))
    # algos.append("fastsrm_%s_%s_%s" % (method, "proj", "5n"))
    # algos.append("fastsrm_%s_%s_%i" % (method, "pca", k + 1))
    algos.append("fastsrm_%s_%s_%s" % (method, "pca", "2n"))

seeds = np.arange(30)
iters = 100
for algo in algos:
    # if os.path.exists("../results/synthetic_grad_%s.npy" % algo):
    #     print("Done: %s" % algo)
    #     continue
    res = Parallel(n_jobs=5, verbose=10)(
        delayed(do_expe)(iters, seed, algo) for seed in seeds
    )
    res = np.array(res)
    res = res.reshape((len(seeds), iters, -1))
    np.save("../results/synthetic_grad_%s.npy" % algo, res)
