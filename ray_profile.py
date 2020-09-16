

import ns_sampling_modules as sm
import ray
import tensorflow as tf
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import time
import sys,os
import ns_walk as nw
import sys

import ns_data_modules as dm
nproc=np.arange(16,72,8)
sequence=np.arange(20000,200000,20000)

# nproc=np.arange(3,9,2)
# sequence=np.arange(5000,20000,5000)
cpus=64



times = pd.DataFrame()
times['sequence']=sm.convert2pandas(sequence)

if ray.is_initialized() is True:
    ray.shutdown()
ray.init(ignore_reinit_error=True)
seed_parent = int.from_bytes(os.urandom(4), sys.byteorder)
g_parent = tf.random.experimental.Generator.from_seed(seed_parent)
# find number of shared memory cores
nb_steps=15
nb_mutations=16
yield2optimize='Developability'


walkers = [nw.walk.remote(nb_steps=nb_steps, yield2optimize=yield2optimize,profile=True) for _ in range(cpus)]
for n in nproc:
    t=[]
    for s in sequence:
        print('sequences: %i, nproc: %i'%(s,n))
        # if ray.is_initialized() is True:
        #     ray.shutdown()
        df = pd.DataFrame()
        with dm.suppress_stdout():
            df['Ordinal'] = sm.make_sampling_data(generator=g_parent, Nb_sequences=s)
        inputs = sm.splitPandas(df=df, nb_splits=n)
        # for the walkers already initilized:  just change the dataframe

        ray.get([walker.set_df.remote(i) for walker,i in zip(walkers[0:n],inputs)])

        # for walkers not already initilzed. add new ones!!
        # find the initial yield, return the min yield from each worker
        # res=ray.get([walker.get_df.remote() for walker in walkers[0:n]])
        # print(res)
        res=ray.get([walker.init_yield.remote() for walker in walkers[0:n]])
        min_yield=[np.min(res)]
        start = time.time()

        with dm.suppress_stdout():
            res=ray.get([walker.walk.remote(min_yield[0],nb_mutations) for walker in walkers[0:n]])
        t.append((time.time()-start)/nb_steps)

        # [walker.reset() for walker in walkers]



    # save states of those processor numbers
    # include standard of deviation
    times.loc[:,'nproc: %i'%n] =t
    times.to_pickle(path='./sampling_data/comparisons/ray/%s_profile_maxs_%i_maxn_%i_stepavg_%i.pkl'%(sys.platform,np.max(nproc),np.max(sequence),nb_steps))
    plt.plot(sequence,t,label=n)
    dm.zip_directory(dir_name='comparisons/ray',zip_filename='ray')
plt.legend()
plt.title('profiling with ray:nb mutations %i,steps: %i'%(nb_mutations,nb_steps))
plt.xlabel('number of sequences')
plt.ylabel('number of seconds per step')
plt.savefig('./sampling_data/comparisons/ray/%s_profile_maxs_%i_maxn_%i_stepavg_%i.png'%(sys.platform,np.max(nproc),np.max(sequence),nb_steps))
dm.zip_directory(dir_name='comparisons/ray',zip_filename='ray')


print(times)

