import os

# Split sherlock data in different sessions
from glob import glob

import numpy as np
from joblib import Parallel, delayed

from nilearn.image import index_img, load_img
from nilearn.input_data import NiftiMasker


def fetch_file(url, save_path):
    if not os.path.exists(save_path):
        print("Downloading from %s" % url)
        os.makedirs("%s" % os.path.dirname(save_path), exist_ok=True)
        os.system(
            "wget -q --show-progress --no-check-certificate -r '%s' -O '%s'"
            % (url, save_path)
        )
    return save_path


fetch_file(
    "http://cogspaces.github.io/assets/data/hcp_mask.nii.gz",
    "./data/hcp_mask.nii.gz",
)

fetch_file(
    "https://dataspace.princeton.edu/bitstream/88435/dsp01nz8062179/7/SherlockMovies_published.tgz",
    "./data/sherlock_data.tgz",
)

os.system("tar -xvzf ./data/sherlock_data.tgz -C ./data/")


mask_img = "./data/hcp_mask.nii.gz"
masker = NiftiMasker(
    mask_img=mask_img,
    detrend=True,
    standardize=True,
    low_pass=0.1,
    high_pass=0.01,
    t_r=1.5,
).fit()


os.makedirs("./data/masked_sherlock", exist_ok=True)


def do_mask_img(f, i):
    _, _, _, t = load_img(f).shape
    # Create 5 sessions
    sessions_indexes = np.array_split(np.arange(t), 5)
    # Loop across sessions
    for j, indexes in enumerate(sessions_indexes):
        img = index_img(f, slice(indexes[0], indexes[-1]))
        X = masker.transform(img).T
        np.save("./data/masked_sherlock/subject_%i_session_%i.npy" % (i, j), X)

        # Loop across subjects


Parallel(n_jobs=10, verbose=10)(
    delayed(do_mask_img)(f, i)
    for i, f in enumerate(glob("./data/movie_files/*"))
)
