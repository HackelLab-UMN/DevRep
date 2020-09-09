import time
import submodels_module as mb
import pandas as pd
import numpy as np
import tensorflow as tf
import os
import sys
from abc import ABC, abstractmethod
import ns_plot_modules as pm
import ns_sampling_modules as sm
# compiled optimizer
import matplotlib as mpl
mpl.use('Agg')
from input_deck import  inputs
import ns_data_modules as dm
import ns_submodels_module as ns_mb
from joblib import wrap_non_picklable_objects
tf.config.optimizer.set_jit(True)
from input_deck import names
fn=names()


# @wrap_non_picklable_objects
class nested_sampling(ABC):
    # main method is to call is walk()
    def __init__(self, s2a_params=None, e2y_params=None, Nb_sequences=1000,Nb_positions=16,nb_models=1):
        'nested sampling initilization for number of sequences and number of positions of ordinals'
        # initilize default model parameters
        if e2y_params is None:
            e2y_params = ['svm', 1]
        if s2a_params is None:
            s2a_params = [[1, 8, 10], 'emb_cnn', 1]
        # note: things may change between tensorflow versions
        seed_parent = int.from_bytes(os.urandom(4), sys.byteorder)
        self.g_parent = tf.random.experimental.Generator.from_seed(seed_parent)

        self.original_seq=pd.DataFrame()

        self.original_seq['Ordinal']= sm.make_sampling_data(generator=self.g_parent,Nb_sequences=Nb_sequences,Nb_positions=Nb_positions)
        self.original_seq['Developability'] = np.zeros(Nb_sequences)

        self.nb_of_sequences = Nb_sequences
        self.test_seq = self.original_seq.copy()
        # self.nb_of_sequences,_=np.shape(self.original_seq['Ordinal'])
        self.nb_models=nb_models

        # todo: should have for loop here for multiple s2a models, so that then your not constantly loading models
        self.s2a = ns_mb.ns_seq_to_assay_model(s2a_params)
        self.s2a.init_sequence_embeddings()

        # i'm putting zero here b/c it requires a parameter...
        self.e2y=[]
        for i in np.arange(self.nb_models):
            self.e2y.append(ns_mb.ns_sequence_embeding_to_yield_model(s2a_params + [i], e2y_params))
            self.e2y[-1].init_e2y_model()

        self.times = pd.DataFrame()
        self.start_time = None
        self.min_yield = []
        # parent random number generator
        self.percent_pos = []
        self.dir_name=[]
        self.nb_mutations=[]
        self.mutation_type=[]
        self.run_stats=pd.DataFrame({'e2y'})

    def nested_sample(self, c, loops_2_show=None):
        '''
        top function to call when making a nested sampling run
        will save run_stats to a /sampling_data/<new dir>
        where <new dir>  is the directory for the run.

        :param c: inputs() object [describes the inputs]
        :param loops_2_show: ndarray of loops to show
        :return: the run times for the loops in loops to show as a pandas Dataframe
        '''
        'main method to call, does nested sampling'
        self.nb_mutations.append(c.nb_mutations)
        self.mutation_type.append(c.mutation_type)

        self.dir_name= dm.make_directory(c=c)
        fileError=os.system('mkdir ./sampling_data/'+self.dir_name)

        dm.save_run_stats(c=c,loops_2_show=loops_2_show)

        loops_done=[]
        for j in np.arange(c.nb_loops):
            print('LOOP %i of %i loops' % (j, c.nb_loops))
            #TODO: change the order of these so you are taking a snapshot and initing the walk at the right time
            if j == 0:
                self.original_seq = self.get_yield().copy()
                self.update_min_yield(self.original_seq)

            self.walk(min_yield=self.min_yield[-1], j=j,
                      loops_2_show=loops_2_show,c=c)
            _, idx = self.update_min_yield(self.original_seq)
            self.change_lowest_yield_sequence_configuration(idx)  # change to another sequence in the configuration
            self.update_nb_mutations()

            if j in loops_2_show:
                loops_done.append(j)
                dm.take_snapshot(self=self,loop_nb=j,c=c,loops_done=loops_done)
                pm.make_heat_map(df=self.original_seq, c=c, loop_nb=j)
                #todo: add making the UMAP here pm.make_UMAP
                dm.zip_data(c=c)



        # TODO: plot the rate of change of min yield as well ...
        pm.make_min_yield_plot(min_yield_lst=self.min_yield,c=c)
        pm.make_percent_positive_plot(c=c,percent_pos=self.percent_pos)

        return self.times

    def update_nb_mutations(self):
        '''
        when making multiple mutations , this determines how the number of mutations should change.
        right now if dynamic --- > if less than 20% loose a mutation
                                    if 20-30% then don't do anything
                                    if >30% , add a mutation until 16 mutations
        :return:appends to self.nb_mutations list
        '''
        if self.mutation_type[-1] is 'static':
            self.nb_mutations.append(self.nb_mutations[-1])
        elif self.mutation_type[-1] is 'dynamic':
            # find current percent positive and percent positive before that.
            last_pp = sum(self.percent_pos[-1]) / len(self.percent_pos[-1]) *100
            if last_pp < 20 and self.nb_mutations[-1] > 1: # bug here this was zero before... dont make that mistake again...
                self.nb_mutations.append(self.nb_mutations[-1] - 1)
            elif (last_pp > 20 and last_pp < 30) or self.nb_mutations[-1]>=16: # shouldn't have more than 16 unique mutations
                self.nb_mutations.append(self.nb_mutations[-1])
            else:
                self.nb_mutations.append(self.nb_mutations[-1] + 1)

    @abstractmethod
    def walk(self, min_yield,c, loops_2_show,j):
        'abstract method, must define in sublclass how to walk/mutate'
        pass

    # private methods
    def get_yield(self,df_only=None,yield2show=None):
        '''
        gets the predicted yield from a model
        uses the models specified in the constructor for s2a and e2y.
        e2y must have a seperate intilization for each model.
        :param df_only: if just passing in the data frame// not nested sampling
        :param yield2show: 2x1 numpy array of booleans of yields to return  [ iq yield, sh yield] if iq and sh yield are both
            true then it will return the sum of the two
        :return: updates self.test_seq if running nested_sampling() function. otherwise updates
        the dataframe with 'developability' column.
        '''
        if yield2show is None:
            yield2show=np.array([True , True ])
        if df_only is None :
            df=self.test_seq.copy()
        else :
            df=df_only

        df_with_embbeding = self.s2a.save_sequence_embeddings(df_list=df)

        predicted_yield_per_model = []
        for e2y in self.e2y:
            predicted_yield_per_model.append(e2y.save_predictions(input_df_description=df_with_embbeding,yield2show=yield2show))
        # determine which yield are optimizing wrt
        if np.count_nonzero(yield2show)>1:
            name='Developability'
        elif yield2show[0] :
            name='IQ_Average_bc'
        elif yield2show[1]:
            name='SH_Average_bc'
        else:
            raise AttributeError('wrong inputs for yield2show')

        df[name] = np.copy(np.average(predicted_yield_per_model, axis=0))


        if df_only is None:
            self.test_seq=df.copy()

        return df


    def update(self, min_yield):
        '''
        updates the sequences based on if they are higher than the last minimum yield
        :param min_yield: current threshold
        :return: will update original sequence based on if developability parameter found from self.test_seq was
        higher than thershold.

        '''
        print('updating the sequences based on last minimum yield')
        print('current minimum yield is  %0.2f' % min_yield)
        # convert the pandas columns to numpy arrays so no for loops  :/
        test_array = sm.convert2numpy(self.test_seq)
        orginal_array =sm.convert2numpy(self.original_seq)
        test_dev = sm.convert2numpy(self.test_seq,'Developability')
        org_dev = sm.convert2numpy(self.original_seq,'Developability')
        # accept changes that meet the min yield requirement
        mutatable_seq = min_yield < test_dev
        orginal_array[mutatable_seq, :] = np.copy(test_array[mutatable_seq, :])
        org_dev[mutatable_seq] = np.copy(test_dev[mutatable_seq])
        # update self.test_seq and self.original_seq
        # dangerous code below ; changing self parameters...

        self.save_testseq_2_original_seq(org_dev, orginal_array)
        # i really need to make some error checking statements
        # return percentage positive
        return np.count_nonzero(mutatable_seq) / mutatable_seq.shape[0]

    def save_testseq_2_original_seq(self, org_dev, orginal_array):
        print('Saving the updated sequence and developability of last sequence as well.')
        self.original_seq['Ordinal'] = sm.convert2pandas(orginal_array)
        self.original_seq['Developability'] = org_dev

        self.original_seq = self.original_seq[['Ordinal', 'Developability']]
        self.test_seq = self.original_seq.copy()
        self.test_seq = self.test_seq[['Ordinal']]

    def start_timer(self):
        print('starting timer')
        self.start_time = time.time()

    def stop_timer(self, loops_2_show, j, i=None ):
        print('stop timer')
        stop_time = time.time() - self.start_time
        if j in loops_2_show:
            self.times.loc[i, str(j+1) + 'th loop'] = stop_time

    def update_min_yield(self, seq):
        '''

        :param seq: pandas Dataframe containing the Developability column
        :return: the newest threshold value, the index of the lowest yield value to be changed
        '''
        print('update the minimum yield.. updating self.min_yield %0.2f' % np.min(
            seq['Developability'].to_numpy().tolist()))
        # consider making the update of the min yield...
        self.min_yield.append(np.min(seq['Developability'].to_numpy().tolist()))

        # return the last element in the sequence
        return self.min_yield[-1], np.argmin(seq['Developability'].to_numpy().tolist())

    def change_lowest_yield_sequence_configuration(self, idx):
        '''

        :param idx: index of sequence with lowest yield
        :return: updates self.original_seq['Ordinal']
        '''
        print('resampling sequence with lowest min yield, seq idx: %i' % idx)
        change_2_seq = idx
        # idk if any of those syntax is correct ...
        while change_2_seq == idx:
            change_2_seq = self.g_parent.uniform(shape=[1], minval=0, maxval=self.nb_of_sequences,  # [0,nb_of_sequences)
                                      dtype=tf.int64).numpy()[0]

        # just do the normal method here b/c this is being dumb.
        orginal_array = np.copy(np.array(self.original_seq['Ordinal'].to_numpy().tolist()))
        orginal_array[idx, :] = orginal_array[change_2_seq, :].copy()
        # TODO : optimize in pandas to change one sequence without changing everything
        self.original_seq['Ordinal'] = sm.convert2pandas(orginal_array)
        print('updated lowest min yield')
        # retrun the arg min for the lowest developability

    # def get_percent_pos_average(self):
    #     return np.sum(self.percent_pos,axis=0)/len(self.percent_pos[0])
class ns_random_sample(nested_sampling):
    # random sampling
    def __init__(self,Nb_sequences=1000,nb_models=1):
        '''

        :param Nb_sequences: number of sequences
        :param nb_models: number of models to average over, as of right no support for over 1
        '''
        super().__init__(Nb_sequences=Nb_sequences,nb_models=nb_models)
        # initilize generator
        seed = int.from_bytes(os.urandom(4), sys.byteorder)
        # note: things may change between tensorflow versions
        self.g = tf.random.experimental.Generator.from_seed(seed)
        self.rng = np.random.default_rng()
        #self.nb_mutations=nb_mutations
       # self.rng_times=pd.DataFrame()
    def walk(self, min_yield,j, loops_2_show,c):
        '''
        method to define how to do a random walk , here we also use rejection sampling.
        :param min_yield: the current threshold
        :param j: loop number
        :param loops_2_show: ndarray of loops to show , based on number of snapshots
        :param c: inputs() object
        :return: update current self.original_seq ordinal and developability column's
        '''

        # here make min_yield a local parameter,  it is required
        # N is the number of iterations , can update in the future to do an actual convergence algorithm
        # i and j represent the histogram to plot too. default is just a single walk.
        #TODO: easy parallelization using joblib library
        #TODO: make self.test_seq a local parameter... maybe idk yet
        percent_pos = []
        for i in np.arange(c.nb_steps):
            print('loop %i of %i, step %i of %i' % (j + 1, c.nb_loops, i + 1, c.nb_steps))
            self.start_timer()
            print('making %i mutations'%self.nb_mutations[-1])
            self.multiple_mutate(nb_mutations=self.nb_mutations[-1],j=j,i=i)
            self.test_seq = self.test_seq[['Ordinal']]
            print('getting yield')
            self.get_yield()
            pp = self.update(min_yield)
            percent_pos.append(pp)
            # self.check()
            self.stop_timer(j=j, i=i, loops_2_show=loops_2_show)
        self.percent_pos.append(percent_pos)

    def multiple_mutate(self,nb_mutations,j,i):
        '''
        the function for making multiple mutations, will just make continual calls to mutate
        :param nb_mutations: number of mutations to make
        :return: repetetive calls to self.mutate will cause changes to 'Ordinal' column of self.test_seq

        '''
       # start=time.time()
        S=np.tile(np.arange(16),(self.nb_of_sequences,1))

        for s in S:
            self.rng.shuffle(s)
            # if np.unique(s).shape[0] != 16:
            #     raise SyntaxError

        S=S[:,0:nb_mutations].copy()

        for random_AA_pos in S.T:
            self.mutate(random_AA_pos=random_AA_pos.copy())

       # total=time.time()-start

       # self.rng_times.loc[i,'%i loop'%(j+1)]=total

    def mutate(self,random_AA_pos=None):
        '''
        :param random_AA_pos: ndarray [Number of sequences x 1]  specifies which random positions
         to change for each sequence
        :return: make changes to self.test_seq['Ordinal'] based on a single mutation for each sequence
        '''
        # mutate every sequence of the original
        # for a mutation to occur ;
        # pseudo random number
        # generate a pseudo random number to define which AA to change [0-15]
        if random_AA_pos is None:
            random_AA_pos = self.g.uniform(shape=[self.nb_of_sequences], minval=0, maxval=16,
                                       dtype=tf.int64).numpy()  # [0,16)
        # generate a pseudo random number to define which AA to change to [0-20]
        # using the same generator might be problematic
        random_AA = self.g.uniform(shape=[self.nb_of_sequences], minval=0, maxval=21, dtype=tf.int64).numpy()
        # [0,21)
        # remove blanks from the sequence
        test_numpy_seq = sm.convert2numpy(df=self.test_seq, field='Ordinal')
        random_AA = sm.remove_blanks(generator=self.g, random_AA_pos=random_AA_pos, random_AA=random_AA,
                                     seq=test_numpy_seq)
        print('mutating test sequence')
        # converting to numpy for logical array manipulation
        # test_numpy_seq[:, random_AA_pos] = random_AA
        # there has to be a way to do this without a loop.
        test_list_seq = []
        for j, r_AA, r_AA_pos, i in zip(test_numpy_seq, random_AA, random_AA_pos, np.arange(test_numpy_seq.shape[0])):
            j[r_AA_pos] = r_AA
            test_list_seq.append((j))

        self.test_seq['Ordinal'] = test_list_seq

    def __reduce__(self):
        '''
        this function is used during serialization for parrelization
        https://docs.python.org/3/library/pickle.html
        :return: a tuple consisting of the class object and a nested tuple of the inputs to reconstruct the class
        '''
        return (self.__class__,(self.nb_of_sequences,self.nb_models))


#TODO: write a bash script that opens an interactive job immedeatily after running
# todo: continue adding comments to code

# class ns_percent_positve(ns_random_sample):
#
#     # todo : have the update number of mutations be different based on the previous slope rather than percentage , or a combo of both
#     def __init__(self,Nb_sequences,nb_models ):
#         super().__init__(Nb_sequences=Nb_sequences,nb_models=nb_models)
#     def update_nb_mutations(self):
#
#         print('hello')

