import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.cm import get_cmap

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

vir = get_cmap("Set2", 2)


algos_prob = ["prob", "fastprob"]
algos_det = ["det", "fastdet"]

NAMES = {}
NAMES["prob"] = "None"
NAMES["det"] = "None"
NAMES["fastprob"] = "Optimal"
NAMES["fastdet"] = "Optimal"

sessions = np.arange(5)
datasets = ["sherlock"]
n_components = [5, 10, 20, 50]
for dataset in datasets:
    vir = get_cmap("Set2", len(algos_det))
    fig, axes = plt.subplots(3, 2, sharex="col", sharey="row", figsize=(10, 9))
    plt.suptitle("Dataset: %s" % dataset.upper())
    for j, algos_group in enumerate([algos_det, algos_prob]):
        for c, algo in enumerate(algos_group):
            for i, path_name in enumerate(
                [
                    "cv_accuracy_timesegment_matching",
                    "fit_time_timesegmentmatching",
                    "memory_usage_timesegmentmatching",
                ]
            ):
                highs = []
                lows = []
                meds = []
                for k in n_components:
                    datas = []
                    for session in sessions:
                        path = "../experiments/results/%s-%i-%i-%s-%s.npy" % (
                            path_name,
                            session,
                            k,
                            algo,
                            dataset,
                        )
                        data = np.load(path)
                        datas.append(data)
                    datas = np.array(datas).flatten()
                    lows.append(np.quantile(datas, 0.25))
                    highs.append(np.quantile(datas, 0.75))
                    meds.append(np.median(datas))
                if "fast" in algo:
                    marker = "x"
                else:
                    marker = "o"
                axes[i, j].plot(
                    n_components, meds, color=vir(c), label=NAMES[algo], marker=marker
                )
                axes[i, j].fill_between(
                    n_components, lows, highs, color=vir(c), alpha=0.1
                )

    axes[0, 0].set_xscale("log")
    axes[0, 0].get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
    axes[0, 0].set_xticks([5, 10, 20, 50])
    axes[0, 1].set_xscale("log")
    axes[0, 1].get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
    axes[0, 1].set_xticks([5, 10, 20, 50])
    axes[1, 0].set_yscale("log")
    axes[2, 0].set_yscale("log")

    axes[0, 0].set_title("DetSRM")
    axes[0, 1].set_title("ProbSRM")
    axes[1, 0].set_ylabel("Time (in s)")
    axes[2, 0].set_ylabel("Memory usage (in Mo)")
    axes[0, 0].set_ylabel("Accuracy")
    axes[2, 0].set_xlabel("Number of components")
    axes[2, 1].set_xlabel("Number of components")
    plt.tick_params(axis='y', which='minor')
    plt.legend(loc="upper center", bbox_to_anchor=(-0.1, -0.3), ncol=5, title="ATLAS")
    plt.savefig("../figures/timesegment_matching_%s.pdf" % dataset , bbox_inches="tight")
