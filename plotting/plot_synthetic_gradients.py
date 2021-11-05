import numpy as np
import matplotlib.ticker as plticker
import matplotlib.pyplot as plt
from matplotlib.cm import get_cmap
from parameters import NAMES
import os

os.makedirs("../figures", exist_ok=True)
loc = plticker.MultipleLocator(base=10)

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

dim = (50, 50, 50)
m, v, k, n = 10, np.prod(dim), 50, 1000

algos_det = ["detsrm", "fastdet"]
algos_prob = ["probsrm", "fastprob"]

NAMES = {}
NAMES["probsrm"] = "None"
NAMES["detsrm"] = "None"
NAMES["fastdet"] = "Optimal"
NAMES["fastprob"] = "Optimal"

seeds = np.arange(30)


vir = get_cmap("Set2", len(algos_det))
f, axes = plt.subplots(3, 2, sharex="col", sharey="row", figsize=(10, 9))
for j, algos_group in enumerate([algos_det, algos_prob]):
    for c, algo in enumerate(algos_group):
        res = np.load("../experiments/results/synthetic_grad_%s.npy" % algo)
        iters = np.arange(len(res[0]))
        times = res[:, :, 1].astype(float)
        errors = res[:, :, 0].astype(float)
        gnorms = res[:, :, 5].astype(float)
        linestyle="-"
        if "fast" in algo:
            marker = "x"
        else:
            marker = "o"

        axes[0, j].plot(
            iters,
            np.median(errors, axis=0),
            marker=marker,
            linestyle=linestyle,
            color=vir(c),
            markevery=10,
        )
        axes[1, j].plot(
            iters,
            np.median(times, axis=0),
            marker=marker,
            linestyle=linestyle,
            color=vir(c),
            label=NAMES[algo],
            markevery=10,
        )
        axes[2, j].plot(
            iters,
            np.median(gnorms, axis=0),
            marker=marker,
            linestyle=linestyle,
            color=vir(c),
            label=NAMES[algo],
            markevery=10,
        )
        axes[0, j].fill_between(
            iters,
            np.quantile(errors, 0.25, axis=0),
            np.quantile(errors, 0.75, axis=0),
            color=vir(c),
            alpha=0.1,
        )
        axes[1, j].fill_between(
            iters,
            np.quantile(times, 0.25, axis=0),
            np.quantile(times, 0.75, axis=0),
            color=vir(c),
            alpha=0.1,
        )
        axes[2, j].fill_between(
            iters,
            np.quantile(gnorms, 0.25, axis=0),
            np.quantile(gnorms, 0.75, axis=0),
            color=vir(c),
            alpha=0.1,
        )
axes[0, 0].set_yscale("log")
axes[1, 0].set_yscale("log")
axes[2, 0].set_yscale("log")
y_minor = plticker.LogLocator(base=10.0, subs=np.arange(1.0, 10.0) * 0.1, numticks = 10)
axes[1, 0].set_ylim([1e1,1e5])
axes[1, 0].set_yticks([1e1, 1e2, 1e3, 1e4, 1e5])
axes[0, 0].yaxis.set_minor_locator(y_minor)
axes[1, 0].yaxis.set_minor_locator(y_minor)
axes[0, 0].set_title("DetSRM")
axes[0, 1].set_title("ProbSRM")
axes[1, 0].set_ylabel("Time (in s)")
axes[2, 0].set_ylabel("Convergence measure")
axes[0, 0].set_ylabel("Mean squared error")
axes[2, 0].set_xlabel("Number of iterations")
axes[2, 1].set_xlabel("Number of iterations")
plt.tick_params(axis='y', which='minor')
plt.legend(loc="upper center", bbox_to_anchor=(-0.1, 4), ncol=5, title="ATLAS")
plt.savefig("../figures/synthetic_gradient.pdf", bbox_inches="tight")
