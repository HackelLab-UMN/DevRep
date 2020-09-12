

import ns_sampling_modules as sm
import ray
import tensorflow as tf
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import time
import sys,os
import ns_walk as nw

import ns_data_modules as dm
# nproc=np.arange(16,72,8)
# sequence=np.arange(20000,200000,20000)

nproc=np.arange(3,9,2)
sequence=np.arange(5000,20000,5000)
times = pd.DataFrame()
for n in nproc:
    t = []
    for s in sequence:
        print('sequences: %i, nproc: %i'%(s,n))
        # if ray.is_initialized() is True:
        #     ray.shutdown()

        ray.init(ignore_reinit_error=True)
        seed_parent = int.from_bytes(os.urandom(4), sys.byteorder)
        g_parent = tf.random.experimental.Generator.from_seed(seed_parent)

        df = pd.DataFrame()
        df['Ordinal'] = sm.make_sampling_data(generator=g_parent, Nb_sequences=s)

        inputs = sm.splitPandas(df=df, nb_splits=n)

        walkers = [nw.walk.remote(i, 1, 'Developability') for i in inputs]


        # find the initial yield, return the min yield from each worker
        res=ray.get([walker.init_yield.remote() for walker in walkers])
        min_yield=[np.min(res)]
        start = time.time()

        res=ray.get([walker.walk.remote(min_yield[0],16) for walker in walkers])

        t.append(time.time()-start)

        # [walker.reset() for walker in walkers]



    # save states of those processor numbers

    times.loc[:,'nproc: %i'%n] =t
    times.to_pickle(path='./sampling_data/comparisons/ray/profile_maxs_%i_maxn_%i.pkl'%(np.max(nproc),np.max(sequence)))
    plt.plot(sequence,t,label=n)
    dm.zip_directory(dir_name='comparisons/ray',zip_filename='ray')
plt.legend()
plt.title('profiling with ray')
plt.xlabel('number of sequences')
plt.ylabel('number of processors')
plt.savefig('./sampling_data/comparisons/ray/profile_maxs_%i_maxn_%i.png'%(np.max(nproc),np.max(sequence)))
dm.zip_directory(dir_name='comparisons/ray',zip_filename='ray')




