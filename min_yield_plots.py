import ns_data_modules as dm
import matplotlib.pyplot as plt
import matplotlib as mpl
mpl.use('Agg')
from input_deck import inputs
import ns_sampling_modules as sm
import pandas as pd
import numpy as np
from input_deck import names
n=names()




def min_yield(C,field2Show):
    '''

    :param C: list of runs
    :param field2Show: what thing to show ?
    :return: None

    '''

    for c in C:
        df=pd.read_pickle(path=dm.make_file_name(c=c,file_description=field2Show))
        stat=sm.convert2numpy(df=df,field=field2Show)

        if c.mutation_type=='dynamic':
            nm=''
        else :
            nm=' ,# mutations: %i'%c.nb_mutations
        plt.plot(np.arange(stat.shape[0]).tolist(),stat,label=c.mutation_type+' '+nm,)
    plt.title('%s vs nested sample loop'%field2Show)
    plt.ylabel(field2Show)
    plt.xlabel('loop number')
    plt.legend()
    plt.savefig(dm.make_file_name(c=C[0],file_description='%_plot',fileformat='png'))
    plt.close()



