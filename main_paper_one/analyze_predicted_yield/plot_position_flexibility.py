import pandas as pd 
import numpy as np
import math
from sklearn import preprocessing
from functools import partial
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.colors

fig, ax = plt.subplots(1,1,figsize=[1.5, 0.3],dpi=300)
fig.subplots_adjust(bottom=0.5)

cmap = mpl.cm.spring
norm = mpl.colors.Normalize(vmin=0.5, vmax=9)

cbar=fig.colorbar(mpl.cm.ScalarMappable(norm=norm, cmap=cmap),
             cax=ax, orientation='horizontal')
cbar.ax.tick_params(labelsize=6)


plt.tight_layout(pad=0.2)
fig.savefig('./tolerance_heatmap.png')