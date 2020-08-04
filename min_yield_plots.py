


import ns_data_modules as dm
import matplotlib.pyplot as plt
import matplotlib as mpl
mpl.use('Agg')
from input_deck import inputs
import ns_sampling_modules as sm
import pandas as pd
import numpy as np
C=[inputs(Nb_sequences=1000,nb_loops=20000,nb_steps=5,mutation_type='dynamic',nb_mutations=10),
   inputs(Nb_sequences=1000,nb_loops=10000,nb_steps=5,mutation_type='static',nb_mutations=6),
  inputs(Nb_sequences=1000,nb_loops=15000,nb_steps=4,mutation_type='static',nb_mutations=1)]

Color=['r','b','g']
y='percent positive'
for c,color in zip(C,Color):
    df=pd.read_pickle(path=dm.make_file_name(c=c,file_description='percent_pos'))
    stat=sm.convert2numpy(df=df,field='percent_pos')

    if c.mutation_type=='dynamic':
        nm=''
    else :
        nm=' ,# mutations: %i'%c.nb_mutations

    # todo :  twin acess stuff, first thing tmr.
    #  plus the density of states.
    plt.plot(np.arange(stat.shape[0]).tolist(),stat,label=c.mutation_type+' '+nm,
             color=color)
plt.title('%s vs nested sample loop'%y)
plt.ylabel(y)
plt.xlabel('loop number')
plt.legend()
plt.savefig(dm.make_file_name(c=C[0],file_description='percent_pos_plot',fileformat='png'))
plt.close()



