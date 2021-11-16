from glob import glob
import matplotlib.pyplot as plt
import numpy as np
import os
from os.path import join
from fastsrm.utils import error_source
os.makedirs("../figures", exist_ok=True)

rc = {
    "pdf.fonttype": 42,
    "text.usetex": True,
    "font.size": 16,
    "xtick.labelsize": 16,
    "ytick.labelsize": 16,
    "text.latex.preview": True,
    "font.family": "serif",
}
plt.rcParams.update(rc)
shared_dir = (
    "/home/hugo/Research/drago/store3/work/hrichard/fastsrm/identifiability/shared/"
)

def standardize(X):
    X_ = X - np.mean(X, axis=1)
    X_ = X_ / np.std(X_, axis=1)
    return X

errors = []
for algo in ["brainiak", "fast"]:
    errors_ = []
    for n_repeat in range(9):
        X, Y = (
            join(
                shared_dir,
                "%i-%i-sherlock-10-%s.npy" % (n_repeat, split, algo),
            )
            for split in (0, 1)
        )
        X = np.load(X)
        Y = np.load(Y)
        errors_.append(np.mean(1 - error_source(X, Y)))
    errors.append(errors_)

errors = np.array(errors)
plt.figure(figsize=(3, 3))
plt.scatter(errors[0], errors[1])
plt.plot(np.arange(0, 2), np.arange(0, 2), color="black")
plt.xlim(np.min(errors) - 0.1, np.max(errors) + 0.1)
plt.ylim(np.min(errors) - 0.1, np.max(errors) + 0.1)
plt.xlabel("Stability index \n (General covariance)")
plt.ylabel("Stability index \n (Diagonal covariance)")
plt.savefig("../paper/figures/identifiability.pdf", bbox_inches="tight")
