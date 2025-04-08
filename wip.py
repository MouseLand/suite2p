import os
import json
import pprint as pp
from pathlib import Path
from tempfile import TemporaryDirectory
import matplotlib.pyplot as plt
import numpy as np
import suite2p

# Figure Style settings for notebook.
import matplotlib as mpl

mpl.rcParams.update(
    {
        "axes.spines.left": False,
        "axes.spines.bottom": False,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "legend.frameon": False,
        "figure.subplot.wspace": 0.01,
        "figure.subplot.hspace": 0.01,
        "figure.figsize": (18, 13),
        "ytick.major.left": False,
    }
)
jet = mpl.colormaps.get_cmap("jet")
jet.set_bad(color="k")
SESS_DIR = "/home/cyberaxolotl/Desktop/2022_01_25/1/"
OUT_DIR = "/home/cyberaxolotl/Desktop/processed_test/"


from natsort import natsorted
import glob

ops = suite2p.default_ops()
print(ops)

with open(os.path.join(SESS_DIR + "ops.json"), "r") as f:
    ops = json.load(f)
ops.update(
    {
        "multiplane_parallel": 0,
        "delete_bin": 0,
        "combined": 1,
        "data_path": [SESS_DIR],
        "save_path0": OUT_DIR,
        "keep_movie_raw": 1,
        "force_sktiff": True,
    }
)
pp.pprint(ops)


# list all the tiffs in the folder
tiffs = sorted([str(p) for p in Path(SESS_DIR).rglob("*.tif")])

db = {
    "data_path": ops["data_path"],
    "save_path0": ops["save_path0"],
    "tiff_list": tiffs,
}
ops.update(db)

output_ops = suite2p.run_s2p(ops=ops, db=db)