"""
Copyright © 2023 Howard Hughes Medical Institute, Authored by Carsen Stringer and Marius Pachitariu.
"""
import string
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.transforms as mtransforms
import numpy as np
from matplotlib import rcParams
from matplotlib.patches import Rectangle
from matplotlib.colors import ListedColormap

default_font = 12
rcParams["font.family"] = "Arial"
rcParams["savefig.dpi"] = 300
rcParams["axes.spines.top"] = False
rcParams["axes.spines.right"] = False
rcParams["axes.titleweight"] = "normal"
rcParams["font.size"] = default_font

alg_cols = ["g", [1, 0.5, 1], [0.5, 0, 1.0]]
alg_names = ['suite2p', 'caiman', 'fiola']

locmin = matplotlib.ticker.LogLocator(base=10.0,subs=np.arange(0.1, 1., 0.1), 
                                          numticks=10)
    
ltr = string.ascii_lowercase
fs_title = 16
weight_title = "normal"

def plot_label(ltr, il, ax, trans, fs_title=20):
    ax.text(
        0.0,
        1.0,
        ltr[il],
        transform=ax.transAxes + trans,
        va="bottom",
        fontsize=fs_title,
        fontweight="bold",
    )
    il += 1
    return il
