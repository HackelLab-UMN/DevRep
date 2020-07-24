
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import ns_sampling_modules as sm
mpl.use('Agg')
import pandas as pd
import seaborn as sns
import numpy as np
import os
import sys
import tensorflow as tf
import ns_sampling_modules as sm
import ns_plot_modules as pm



#pm.violin_saved_dataset(sm.make_directory(Nb_loops=15000,Nb_steps=4))
ML_DEVELOPABILITY = '/Users/bryce.johnson/Desktop/ML/Developability'
#TODO:  add a globals file , with these file paths.
nb_steps=4
nb_loops=15000
dir_name=sm.make_directory(Nb_steps=nb_steps,Nb_loops=nb_loops)
src=ML_DEVELOPABILITY+'/sampling_data/'+dir_name
# times=pd.read_pickle(path=src+'/times.pkl')
# print(times)

#TODO: ideally reads from the stat file

stats=pd.read_pickle(path=os.path.join(src,'run_stats.pkl'))
loops_2_show=sm.convert2numpy(df=stats,field='Loops to show')[0]+1
#loops_2_show=np.hstack((loops_2_show[0:2].copy(),loops_2_show[3:8:2].copy()))
loops_2_show=loops_2_show[]
pm.violin_saved_dataset(nb_steps=nb_steps,nb_loops=nb_loops,loops_2_show=loops_2_show)