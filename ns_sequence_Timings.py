import ns_nested_sampling as ns
from input_deck import inputs
import numpy as np
import ns_main_sampling as ms
import ns_data_modules as dm
import pandas as pd
import ns_sampling_modules as sm
import matplotlib.pyplot as plt
import matplotlib as mpl
mpl.use('Agg')


A=np.array([1000,5000,10000,20000,30000,40000,50000])
t=[]
t2=[]

for a in A:
    c=inputs(Nb_sequences=a,nb_loops=2,nb_steps=8,nb_snapshots=2)
    df=ms.driver(c=c)
    #df=pd.read_pickle(path=dm.make_file_name(c=c,file_description='times',fileformat='pkl'))
    times=sm.convert2numpy(df=df,field='1th loop')
    times2=sm.convert2numpy(df=df,field='2th loop')
    t.append(np.average(times))
    t2.append(np.average(times2))
# todo : include standard of deviation.
plt.plot(A.tolist(), t,label='loop 1')
plt.plot(A.tolist(),t2,label='loop 2')
plt.title('number of sequences vs runtime for a single step')
plt.ylabel('runtime for a single step (sec)')
plt.xlabel('number of sequences')
plt.savefig('./sampling_data/time_stats_improved_get_yield.png')
plt.close()