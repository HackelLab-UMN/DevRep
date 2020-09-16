import ray
import ns_submodels_module as ns_mb
import os,sys
import numpy as np
import ns_nested_sampling as ns
import ns_sampling_modules as sm
import ns_data_modules as dm
import tensorflow as tf
@ray.remote
class walk():
    def __init__(self, df=None,nb_steps=5, yield2optimize='Developability',profile=False):
        '''

        walk class for doing random walks across multiple cpus
        :param df: input
        '''
        seed_parent = int.from_bytes(os.urandom(4), sys.byteorder)
        # seed_parent=8
        # tensorflow optimizer
        tf.config.optimizer.set_jit(True)
        self.g = tf.random.experimental.Generator.from_seed(seed_parent)
        self.rng = np.random.default_rng()
        e2y_params = ['svm', 1]
        s2a_params = [[1, 8, 10], 'emb_cnn', 1]

        # tensorflow model
        self.s2a = ns_mb.ns_seq_to_assay_model(s2a_params)
        self.s2a.init_sequence_embeddings()
        # todo:  is to put these two objects in the memory store
        self.e2y = ns_mb.ns_sequence_embeding_to_yield_model(s2a_params + [0], e2y_params)
        self.e2y.init_e2y_model()

        # set df to none if none specified
        self.df = df

        if self.df is None:
            if not profile:
                raise Exception('df is none but yet you didnt set profiling flag to true')
            self.nb_of_sequences=None
            self.test_seq=None
        else:
            self.nb_of_sequences = len(df.index)
            self.test_seq = df.copy()





        if yield2optimize == 'Developability':
            self.yield2show=np.array([True,True])
        elif yield2optimize== 'IQ_Average_bc':
            self.yield2show=np.array([True, False])
        elif yield2optimize== 'SH_Average_bc':
            self.yield2show=np.array([False,True])
        else:
            raise ValueError('Incorrect input for yield2optimize')
        self.yield2optimize=yield2optimize


        self.nb_steps=nb_steps
        self.idx=None

        # ray optimizer
       # os.environ["OMP_NUM_THREADS"] = str(os.cpu_count())


    def walk(self,min_yield,nb_mutations):
        percent_pos=[]
        for i in np.arange(self.nb_steps):
            # first make mutations
            self.multiple_mutate(nb_mutations=nb_mutations)
            self.get_yield()
            pp = self.update(min_yield)
            percent_pos.append(pp)

        # for starters join all the dataframes
        # i want to avoid as many long serializations as possible
        # return the minimum
        min_yield=np.min(sm.convert2numpy(df=self.df,field=self.yield2optimize))
        self.idx=np.argmin(sm.convert2numpy(df=self.df,field=self.yield2optimize))

        return min_yield,np.mean(percent_pos) # return the average pp for all the step

    def get_df(self):
        return self.df

    def set_df(self,df):
        self.df=df
        self.nb_of_sequences=len(df.index)
        self.test_seq = df.copy()


    def multiple_mutate(self, nb_mutations):
        '''
        the function for making multiple mutations, will just make continual calls to mutate
        :param nb_mutations: number of mutations to make
        :return: repetetive calls to self.mutate will cause changes to 'Ordinal' column of self.test_seq

        '''
        # start=time.time()
        # create a tiled array
        S = np.tile(np.arange(16), (self.nb_of_sequences, 1))

        for s in S:
            self.rng.shuffle(s)
            # if np.unique(s).shape[0] != 16:
            #     raise SyntaxError
        S = S[:, 0:nb_mutations].copy()

        for random_AA_pos in S.T:
            # make suggested mutations to self.test_seq
            self.mutate(random_AA_pos=random_AA_pos.copy())

    def mutate(self, random_AA_pos=None):
        '''
        mutate module responsible for multiple mutations
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
        with dm.suppress_stdout():
            random_AA = sm.remove_blanks(generator=self.g, random_AA_pos=random_AA_pos, random_AA=random_AA,
                                     seq=test_numpy_seq)
        # print('mutating test sequence')
        # converting to numpy for logical array manipulation
        # test_numpy_seq[:, random_AA_pos] = random_AA
        # there has to be a way to do this without a loop.
        test_list_seq = []
        for j, r_AA, r_AA_pos in zip(test_numpy_seq, random_AA, random_AA_pos):
            j[r_AA_pos] = r_AA
            test_list_seq.append((j))

        self.test_seq['Ordinal'] = test_list_seq

    def get_yield(self,df_only=None):
        '''
        gets the predicted yield from a model
        uses the models specified in the constructor for s2a and e2y.
        e2y must have a seperate intilization for each model.
        :param df_only: if just passing in the data frame// not nested sampling
        :return: updates self.test_seq if running nested_sampling() function. otherwise updates
        the dataframe with 'developability' column.
        '''
        if df_only is None :
            df=self.test_seq.copy()
        else :
            df=df_only

        df_with_embbeding = self.s2a.save_sequence_embeddings(df_list=df)
        # determine which yield are optimizing wrt
        df[self.yield2optimize]=self.e2y.save_predictions(input_df_description=df_with_embbeding, yield2show=self.yield2show)
        if df_only is None:
            self.test_seq=df.copy()

        return df

    def init_yield(self):

        self.df=self.get_yield(self.df).copy()

        return np.min(sm.convert2numpy(df=self.df,field=self.yield2optimize))

    def update(self, min_yield):
        '''
        updates the sequences based on if they are higher than the last minimum yield
        :param min_yield: current threshold
        :return: will update original sequence based on if developability parameter found from self.test_seq was
        higher than thershold.

        '''
        # print('updating the sequences based on last minimum yield')
        # print('current minimum yield is  %0.2f' % min_yield)
        # convert the pandas columns to numpy arrays so no for loops  :/
        test_array = sm.convert2numpy(self.test_seq)
        orginal_array =sm.convert2numpy(self.df)
        test_dev = sm.convert2numpy(self.test_seq,self.yield2optimize)
        org_dev = sm.convert2numpy(self.df,self.yield2optimize)
        # accept changes that meet the min yield requirement
        mutatable_seq = min_yield < test_dev

        orginal_array[mutatable_seq, :] = np.copy(test_array[mutatable_seq, :])
        org_dev[mutatable_seq] = np.copy(test_dev[mutatable_seq])

        # update self.test_seq and self.original_seq
        # dangerous code below ; changing self parameters...
        self.df['Ordinal'] = sm.convert2pandas(orginal_array)
        self.df[self.yield2optimize] = org_dev
        self.test_seq = self.df.copy()
        self.test_seq = self.test_seq[['Ordinal']]

        return np.count_nonzero(mutatable_seq) / mutatable_seq.shape[0]

    def change_lowest_yield_sequence_configuration(self):
        '''
        function to mutate the current sequences
        :param idx: index of sequence with lowest yield
        :return: updates self.original_seq['Ordinal']
        '''
        # print('sequence to change %i'%self.idx)
        change_2_seq = self.idx
        while change_2_seq == self.idx:
            change_2_seq = self.g.uniform(shape=[1], minval=0, maxval=self.nb_of_sequences,  # [0,nb_of_sequences)
                                      dtype=tf.int64).numpy()[0]
        # print('new idx  %i '%change_2_seq)
        orginal_array =sm.convert2numpy(df=self.df)
        orginal_array[self.idx, :] = orginal_array[change_2_seq, :].copy()
        # TODO : optimize in pandas to change one sequence without changing everything
        self.df['Ordinal'] = sm.convert2pandas(orginal_array)

        dev=sm.convert2numpy(df=self.df,field=self.yield2optimize)
        # print(dev[self.idx])
        dev[self.idx]=dev[change_2_seq]
        self.df[self.yield2optimize]=sm.convert2pandas(dev)


        # print(dev[self.idx])
        # print(dev[change_2_seq])


