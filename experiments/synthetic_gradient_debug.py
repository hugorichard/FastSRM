from fastsrm2.srm import projection, detsrm, probsrm
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
    seed = 2
    rng = np.random.RandomState(seed)
    Sigma = rng.dirichlet(np.ones(k), 1).flatten()
    S = np.sqrt(Sigma)[:, None] * rng.randn(k, n)
    W = np.array([projection(rng.randn(v, k)) for i in range(m)])
    sigmas = 0.1 * rng.rand(m)
    N = np.array([sigmas[i] * rng.randn(v, n) for i in range(m)])
    X = np.array([W[i].dot(S) + N[i] for i in range(m)])
    S_true = S

    def callback(source, gnorm, current_iter, current_time):
        return (
            float(reg_error(np.copy(S_true), np.copy(source))),
            float(current_time - t0),
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

    t0 = time()
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
        return np.array(S)


method = "prob"
seed = 1
iters = 50
do_expe(50, 1, "fastsrm_%s_%s_%i" % (method, "pca", k))
