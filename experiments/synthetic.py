import os
from time import time
from fastsrm.fastsrm import fastsrm
from fastsrm.srm import detsrm, probsrm, projection
from fastsrm.utils import reg_error
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

    if algo == "detsrm":
        S = detsrm(
            X, k, n_iter=it, random_state=rng, callback=callback, tol=-1,
        )[-1]
    if algo == "probsrm":
        S = probsrm(
            X, k, n_iter=it, random_state=rng, callback=callback, tol=-1,
        )[-1]
    if algo == "fastdet":
        S = fastsrm(
            [[x] for x in X],
            k,
            n_iter=it,
            random_state=rng,
            callback=callback,
            tol=-1,
            method="det",
        )[-1]
    if algo == "fastprob":
        S = fastsrm(
            [[x] for x in X],
            k,
            n_iter=it,
            random_state=rng,
            callback=callback,
            tol=-1,
            method="prob",
        )[-1]
    return np.array(S)


seeds = np.arange(30)
iters = 100
os.makedirs("./results", exist_ok=True)
# for algo in ["detsrm", "probsrm", "fastdet", "fastprob"]:
for algo in ["fastdet"]:
    res = Parallel(n_jobs=4, verbose=10)(
        delayed(do_expe)(iters, seed, algo) for seed in seeds
    )
    res = np.array(res)
    res = res.reshape((len(seeds), iters, -1))
    np.save("./results/synthetic_grad_%s.npy" % algo, res)
