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
from global_consts import default_inputs
mpl.use('Agg')

tf.config.optimizer.set_jit(True)

class nested_sampling(ABC):
    # main method is to call is walk()
    def __init__(self, s2a_params=None, e2y_params=None, Nb_sequences=1000,Nb_positions=16):
        # TODO: check times for different number of sequences
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

        self.s2a = mb.seq_to_assay_model(*s2a_params)
        # i'm putting zero here b/c it requires a parameter...
        self.e2y = mb.sequence_embeding_to_yield_model(s2a_params + [0], *e2y_params)
        self.times = pd.DataFrame()
        self.start_time = None
        self.min_yield = []
        # parent random number generator
        self.percent_pos = []
        self.vp_step = []
        self.dir_name=[]
        self.nb_mutations=[]
        self.mutation_type=[]
        # TODO: make a run stats file save it to the directory
        self.run_stats=pd.DataFrame({'e2y'})

    def nested_sample(self, N_loops, N_steps,nb_mutations,mutation_type,steps_2_show=None, loops_2_show=None):
        'main method to call, does nested sampling'
        # TODO: describe what the inputs should be ...
        # this is the loop I would like to have done by the end of today. So that a driver script can just call this
        # method an all will be good.
        # write2pickle is a boolean flag to see where optimized sequences should be written too.
        # TODO: make sure to add nproc to this as well? maybe?
        self.nb_mutations.append(nb_mutations)
        self.mutation_type.append(mutation_type)

        self.dir_name= sm.make_directory(Nb_loops=N_loops,Nb_steps=N_steps,nb_sequences=self.nb_of_sequences,
                                         nb_mutations=self.nb_mutations[-1],mutation_type=self.mutation_type[-1])



        fileError=os.system('mkdir ./sampling_data/'+self.dir_name)
        #TODO: check for error in making the file, OS dependent for fileError


        if steps_2_show is None:
            # default is to show 3 steps
            steps_2_show = np.array([0, N_steps // 2, N_steps])
            steps_2_show = np.unique(steps_2_show).copy()
        if loops_2_show is None:
            # default is to show 3 loops
            loops_2_show = np.array([0, N_loops // 2, N_loops-1])
            loops_2_show =np.unique(loops_2_show).copy()


        sm.save_run_stats(self=self,nb_loops=N_loops,nb_steps=N_steps,steps_2_show=steps_2_show,
                          loops_2_show=loops_2_show)

        # self.init_step_plots(steps_2_show=steps_2_show)

        # TODO: figure out the orginal_seq and test_seq craziness... honestly test sequence
        #  should just be a local parameter to the walk. not a local to the class one... that would look much better ..

        # get yield should have an input parameter

        for j in np.arange(N_loops):
            print('LOOP %i of %i loops' % (j, N_loops))
            #TODO: change the order of these so you are taking a snapshot and initing the walk at the right time
            self.init_walk(j=j, steps_2_show=steps_2_show, loops_2_show=loops_2_show)
            #TODO: the heat map plots the last distribution before the last walk...
            self.walk(min_yield=self.min_yield[-1], steps_2_show=steps_2_show, j=j, N_steps=N_steps,
                      loops_2_show=loops_2_show, N_loops=N_loops)
            _, idx = self.update_min_yield(self.original_seq)
            self.change_lowest_yield_sequence_configuration(idx)  # change to another sequence in the configuration
            self.update_nb_mutations()

            if j in loops_2_show:
                sm.take_snapshot(self=self,loop_nb=j)

        # TODO: plot the rate of change of min yield as well ...
        pm.make_min_yield_plot(dir_name=self.dir_name,min_yield_lst=self.min_yield,N_loops=N_loops)

        pm.make_percent_positive_plot(dir_name=self.dir_name,N_steps=N_steps, N_loops=N_loops,
                                      percent_pos=self.percent_pos)

        self.times.to_pickle(path=pm.make_file_name(dir_name=self.dir_name,file_description='times',fileformat='pkl'))

        sm.zip_data(dir_name=self.dir_name)
        return self.times

    def update_nb_mutations(self):
        if self.mutation_type[-1] is 'static':
            self.nb_mutations.append(self.nb_mutations[-1])
        elif self.mutation_type[-1] is 'dynamic':
            # find current percent positive and percent positive before that.
            last_pp = sum(self.percent_pos[-1]) / len(self.percent_pos[-1]) *100
            if last_pp < 20 and self.nb_mutations[-1] > 0:
                self.nb_mutations.append(self.nb_mutations[-1] - 1)
            elif last_pp > 20 and last_pp < 30:
                self.nb_mutations.append(self.nb_mutations[-1])
            else:
                self.nb_mutations.append(self.nb_mutations[-1] + 1)

    @abstractmethod
    def walk(self, min_yield, steps_2_show, N_loops, loops_2_show, N_steps=10, j=1):
        'abstract method, must define in sublclass how to walk/mutate'
        pass

    def init_walk(self, j, steps_2_show, loops_2_show):
        # i should always be equal to zero here
        if j == 0:
            print('finding min yield of init sequences')
            self.original_seq = self.get_yield().copy()
            self.update_min_yield(self.original_seq)
        if j in loops_2_show and 0 in steps_2_show:
            pm.make_heat_map(df=self.original_seq,dir_name=self.dir_name,loop_nb=j)
    # private methods
    def get_yield(self):
        'gets the predicted yield from a model'
        df_with_embbeding = self.s2a.save_sequence_embeddings(df_list=[self.test_seq], is_ordinals_only=True)

        predicted_yield_per_model = []
        for i in np.arange(3):
            predicted_yield_per_model.append(
                self.e2y.save_predictions(df=df_with_embbeding, df_emb=True, sampling_nb=i))
        self.test_seq['Developability'] = np.copy(np.average(predicted_yield_per_model, axis=0))
        return self.test_seq

    def update(self, min_yield):
        'updates the sequences based on if they are higher than the last minimum yield'
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

    def stop_timer(self, loops_2_show, j, i=None, ):
        print('stop timer')
        stop_time = time.time() - self.start_time
        if j in loops_2_show:
            self.times.loc[i, str(j+1) + 'th loop'] = stop_time

    def update_min_yield(self, seq):
        print('update the minimum yield.. updating self.min_yield %0.2f' % np.min(
            seq['Developability'].to_numpy().tolist()))
        # consider making the update of the min yield...
        self.min_yield.append(np.min(seq['Developability'].to_numpy().tolist()))

        # return the last element in the sequence
        return self.min_yield[-1], np.argmin(seq['Developability'].to_numpy().tolist())

    def change_lowest_yield_sequence_configuration(self, idx):
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
    # nested sampling random sampling implementation.
    def __init__(self,Nb_sequences):
        super().__init__(Nb_sequences=Nb_sequences)
        # initilize generator
        seed = int.from_bytes(os.urandom(4), sys.byteorder)
        # note: things may change between tensorflow versions
        self.g = tf.random.experimental.Generator.from_seed(seed)
        #self.nb_mutations=nb_mutations


    def walk(self, min_yield, steps_2_show, loops_2_show, N_loops, N_steps=10, j=1):
        'this is a random walk that makes one mutation at a time'

        # here make min_yield a local parameter,  it is required
        # N is the number of iterations , can update in the future to do an actual convergence algorithm
        # i and j represent the histogram to plot too. default is just a single walk.
        #TODO: easy parallelization using joblib library
        #TODO: make self.test_seq a local parameter...
        percent_pos = []
        for i in np.arange(N_steps):
            print('loop %i of %i, step %i of %i' % (j + 1, N_loops, i + 1, N_steps))
            self.start_timer()
            print('making %i mutations'%self.nb_mutations[-1])
            for k in np.arange(self.nb_mutations[-1]):
                self.mutate()
            self.test_seq = self.test_seq[['Ordinal']]
            print('getting yield')
            self.get_yield()
            pp = self.update(min_yield)
            percent_pos.append(pp)
            # self.check()
            self.stop_timer(j=j, i=i, loops_2_show=loops_2_show)
        self.percent_pos.append(percent_pos)

    def mutate(self):
        'mutate the sequences where necessary, this is a random mutation'
        # mutate every sequence of the original
        # for a mutation to occur ;
        # pseudo random number
        # generate a pseudo random number to define which AA to change [0-15]
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



#TODO: smart sample which uses a combination of both
