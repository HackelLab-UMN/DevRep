import time
import submodels_module as mb
import pandas as pd
import numpy as np
import tensorflow as tf
import os
import sys
import ns_plot_modules as pm
import ns_sampling_modules as sm
# compiled optimizer
import matplotlib as mpl
mpl.use('Agg')
from input_deck import inputs
import ns_data_modules as dm
import ns_submodels_module as ns_mb
from joblib import wrap_non_picklable_objects
tf.config.optimizer.set_jit(True)
from input_deck import names
fn=names()
import ray
import multiprocessing
import ns_walk as nw

# @wrap_non_picklable_objects
class nested_sampling():
    # main method is to call is walk()
    def __init__(self, Nb_sequences=1000,df_filename=None, yield2optimize='Developability',
                 Nb_positions=16,nb_models=1):
        'nested sampling initilization for number of sequences and number of positions of ordinals'
        # initilize default model parameters
        # note: things may change between tensorflow versions
        seed_parent = int.from_bytes(os.urandom(4), sys.byteorder)
        self.g_parent = tf.random.experimental.Generator.from_seed(seed_parent)

        # make randomized data
        if df_filename is None:
            self.original_seq=pd.DataFrame()
            self.original_seq['Ordinal']= sm.make_sampling_data(generator=self.g_parent,Nb_sequences=Nb_sequences,Nb_positions=Nb_positions)
            self.original_seq[yield2optimize] = np.zeros(Nb_sequences)
        else:
            self.original_seq=pd.read_pickle(path=df_filename)

        self.nb_of_sequences = Nb_sequences
        # self.nb_of_sequences,_=np.shape(self.original_seq['Ordinal'])
        self.nb_models=nb_models

        # todo: should have for loop here for multiple s2a models, so that then your not constantly loading models

        # i'm putting zero here b/c it requires a parameter..

        self.times = pd.DataFrame()
        self.start_time = None
        self.min_yield = []
        # parent random number generator
        self.percent_pos = []
        self.dir_name=[]
        self.nb_mutations=[]
        self.run_stats=pd.DataFrame()
        self.yield2optimize=yield2optimize

    def nested_sample(self, c, loops_2_show=None):
        '''

        top function to call when making a nested sampling run
        will save run_stats to a /sampling_data/<new dir>
        where <new dir>  is the directory for the run.
        if running in parralel will make workers that are objects walk() of ns_walk.py
        utilizes the ray library

        :param c: inputs() object [describes the inputs]
        :param loops_2_show: ndarray of loops to show
        :return: the run times for the loops in loops to show as a pandas Dataframe
        '''
        'main method to call, does nested sampling'


        # lets design this right here.
        # first do some preprocessing
        self.nb_mutations.append(c.nb_mutations)
        self.dir_name= dm.make_directory(c=c)
        fileError=os.system('mkdir ./sampling_data/'+self.dir_name)
        dm.save_run_stats(c=c,loops_2_show=loops_2_show)
        loops_done=[]

        # then need to initilize ray , etc.
        nproc=multiprocessing.cpu_count()
        if ray.is_initialized() is True:
            ray.shutdown()
        ray.init()



        # preprocessing for ray. split the pandas dataframes, then get the length of each
        inputs = sm.splitPandas(df=self.original_seq, nb_splits=nproc)
        lengths=[]
        for i in inputs:
            lengths.append(len(i.index))


        # initilize workers
        walkers = [nw.walk.remote(i, c.nb_steps, self.yield2optimize) for i in inputs]


        # find the initial yield, return the min yield from each worker
        res=ray.get([walker.init_yield.remote() for walker in walkers])
        self.min_yield.append(np.min(res))
        self.percent_pos.append(1) # we accept everything on the first look at yield
        self.nb_mutations.append(c.nb_mutations) # set the number of mutations to what is specified by inputs() object
        # todo:  the above can be updated with a parameter for the number of mutations when calling the
        # function

        times=pd.DataFrame()

        for j in range(c.nb_loops):
            # maybe make a function called update sequence configuration, call it at start of walk every single time
            start=time.time()
            # walk the sequences on seperate cpus
            print('going to cpus')
            res=ray.get([walker.walk.remote(self.min_yield[-1],self.nb_mutations[-1]) for walker in walkers])
            # update min yield
            min_yield=[r[0] for r in res]
            pp=[r[1] for r in res]
            self.min_yield.append(np.min(min_yield))
            worker_idx=np.argmin(min_yield)
            # then find the actor that min yield and go update that sequence... that way less serialization
            # change the sequence with the lowest configuration to a random one
            ray.get(walkers[worker_idx].change_lowest_yield_sequence_configuration.remote()) # returns nothing

            # the average percent positve yield , this is then a weighted mean
            print(self.min_yield[-1])
            self.percent_pos.append(np.sum(np.array(pp)*np.array(lengths))/np.sum(lengths))

            self.update_nb_mutations(c=c)
            end=time.time()

            times.loc[j,'times loop']=end-start

            if j in loops_2_show:
                # join all dataframes together at very end
                # finally bring the sequences back togther if it is a loop2save, and take a snapshot.
                # make a heatmap. Etc. zip the data. do some post processing.
                res = ray.get([walker.get_df.remote() for walker in walkers])
                self.original_seq = pd.concat(res).copy()
                dm.take_snapshot(self=self, loop_nb=j, c=c,times=times)
                pm.make_heat_map(df=self.original_seq, c=c, loop_nb=j)
                dm.zip_data(c=c)

        pm.make_min_yield_plot(min_yield_lst=self.min_yield, c=c)
        pm.make_percent_positive_plot(c=c, percent_pos=self.percent_pos)
        # TODO: plot the rate of change of min yield as well ...
        return times

    def update_nb_mutations(self,c):
        '''
        when making multiple mutations , this determines how the number of mutations should change.
        right now if dynamic --- > if less than 20% loose a mutation
                                    if 20-30% then don't do anything
                                    if >30% , add a mutation until 16 mutations
        :return:appends to self.nb_mutations list
        '''
        if c.mutation_type is 'static':
            self.nb_mutations.append(self.nb_mutations[-1])
        elif c.mutation_type is 'dynamic':
            # find current percent positive and percent positive before that.
            last_pp = self.percent_pos[-1]
            if last_pp < 20 and self.nb_mutations[-1] > 1: # bug here this was zero before... dont make that mistake again...
                self.nb_mutations.append(self.nb_mutations[-1] - 1)
            elif (last_pp > 20 and last_pp < 30) or self.nb_mutations[-1]>=16: # shouldn't have more than 16 unique mutations
                self.nb_mutations.append(self.nb_mutations[-1])
            else:
                self.nb_mutations.append(self.nb_mutations[-1] + 1)





#TODO: write a bash script that opens an interactive job immedeatily after running
# todo: continue adding comments to code
# todo : have the update number of mutations be different based on the previous
#  slope rather than percentage , or a combo of both


