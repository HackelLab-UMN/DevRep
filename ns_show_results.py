import matplotlib as mpl

mpl.use('Agg')
import pandas as pd
import os
import ns_sampling_modules as sm
import ns_plot_modules as pm
import ns_data_modules as dm
from input_deck import inputs
import numpy as np
def violin_loop_plots(c,strings_2_show=5):


    stats = pd.read_pickle(path=dm.make_file_name(c=c,file_description='run_stats'))

    loops_2_show = sm.convert2numpy(df=stats, field='Loops to show')[0] + 1

    #TODO: solve the string to show problem , once you solve it then you can call this
    # script on the pbs script
    #l2s=loops_2_show.shape[0]
    #if l2s>strings_2_show:

    loops_2_show=np.array([1,1501,3001,4501,6001,7501,9001,10501])
    pm.violin_saved_dataset(c=c,loops_2_show=loops_2_show)



#loops_2_show=np.hstack((loops_2_show[0:2].copy(),loops_2_show[3:8:2].copy()))
c=inputs()

violin_loop_plots(c=c)


# TODO: make sure c params will be compatible for a smart sample,,,
#   so for repeated calls to nested_sample, you can still mess with the input deck.
# also include that on their too.... like I should just have one function call and then it
# gets my data not this craziness.
# im passing around things like an idiot is is so confusing...
#df=pd.read_pickle(path=dm.make_file_name(c=c,file_description='nb_mutations'))


