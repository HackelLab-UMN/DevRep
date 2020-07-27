import numpy as np
import pickle
from functools import partial
from model_module import model
import load_format_data
import plot_model



class seq_to_x_model():
    'sets get_input_seq to ordinal or onehot sequence based upon model_architecture'
    def __init__(self, model_architecture):
        if 'emb' in model_architecture:
            self.get_input_seq=load_format_data.get_ordinal
        else:
            self.get_input_seq=load_format_data.get_onehot

class assay_to_x_model():
    'sets get_input_seq to assay scores of assays'
    def __init__(self, assays):
        self.get_input_seq=partial(load_format_data.get_assays,assays)

class control_to_x_model():
    'sets get_input_seq to nothing, not sure if needed'
    def __init__(self):
        self.get_input_seq=load_format_data.get_control

class sequence_embedding_to_x_model():
    'sets get_input_seq to load the sequence embedding from a saved seq-to-assay model'
    def __init__(self):
        self.get_input_seq=load_format_data.get_embedding

class x_to_yield_model(model):
    'sets model output to yield'
    def __init__(self, model_in, model_architecture, sample_fraction):
        super().__init__(model_in, 'yield', model_architecture, sample_fraction)
        self.get_output_and_explode=load_format_data.explode_yield
        self.plot_type=plot_model.x_to_yield_plot
        self.training_df=load_format_data.load_df('assay_to_dot_training_data')
        self.testing_df=load_format_data.load_df('seq_to_dot_test_data')
        self.lin_or_sig='linear'
        self.num_cv_splits=10
        self.num_cv_repeats=10
        self.num_test_repeats=10
        self.num_hyp_trials=50

    def change_sample_seed(self,seed):
        self.sample_seed=seed
        self.update_model_name('seed'+str(seed)+'_'+self.model_name)

    def save_predictions(self, input_df_description=None, df=None, df_emb=False, sampling_nb=None):
        'saves model predictions for the large dataset'
        # Inputs: input_df_description: input dataframe description
        #         df: a dataframe for monte carlo sampling
        #         df_emb: a boolean if df is just meant for monte carlo sampling
        #         sampling_nb:
        if input_df_description is None and df_emb is False:  # default for the non sampling case
            input_df_description = 'seq_to_assay_train_'+self.assay_str #only a certain number of these files exist, but more can be created
            df = load_format_data.load_df(input_df_description)  # will have to adjust if missing datapoints
        elif df_emb is False:  # we are not doing sampling
            df = load_format_data.load_df('predicted/' + input_df_description)  # for using predicted embeddings
            x_a = self.get_input_seq(df)
        else:  # this is for monte carlo sampling, df is a dataframe with attributes 'Ordinals' and
            # 'learned_sampling_'sampling_nb''
            x_a = self.get_input_seq(df, sampling_nb)

        OH_matrix = np.eye(2)
        matrix_col = ['IQ_Average_bc', 'SH_Average_bc']
        for z in range(1):  # no of models
            self.load_model(z)
            for i in range(2):
                cat_var = []
                for j in x_a:
                    cat_var.append(OH_matrix[i].tolist())
                x = load_format_data.mix_with_cat_var(x_a, cat_var)
                df_prediction = self._model.model.predict(x).squeeze().tolist()
                col_name = matrix_col[i]
                df.loc[:, col_name] = df_prediction
                col_name_std = matrix_col[i] + '_std'
                df.loc[:, col_name_std] = [0] * len(df_prediction)
            if df_emb:  # if this a monte carlo sampling call , return the averaged predictions for the model
                return self.avg_prediction(df)
            df.to_pickle('./datasets/predicted/' + input_df_description + '_' + self.model_name + '_' + str(z) + '.pkl')

    def avg_prediction(self, df):
        'load predictions and add the two cell types yield together'
        predicted_iq_yield = df['IQ_Average_bc'].to_numpy()
        predicted_sh_yield = df['SH_Average_bc'].to_numpy()
        predicted_added_yield = np.sum([predicted_iq_yield, predicted_sh_yield], axis=0)
        return predicted_added_yield

    def switch_train_test(self):
        regular_training_df=self.training_df
        extra_training_df,self.testing_df=load_format_data.get_random_split(self.testing_df)
        self.training_df=regular_training_df.append(extra_training_df)

    def limit_test_set(self,assays):
        #Limit test set for data that has all assay scores used in model
        sort_names=[]
        for i in assays:
            sort_names.append('Sort'+str(i)+'_mean_score')
        dataset=self.testing_df
        dataset=dataset[~dataset[sort_names].isna().any(axis=1)]
        self.testing_df=dataset

    def apply_predicted_assay_scores(self,seq_to_assay_model_prop):
        'uses saved predicted assay scores and saved assay-to-yield model to determine performance on test-set'
        seq_to_assay_model_name='seq_assay'+self.assay_str+'_'+str(seq_to_assay_model_prop[0])+'_'+str(seq_to_assay_model_prop[1])+'_'+str(seq_to_assay_model_prop[2])
        self.num_test_repeats=1
        self.testing_df=load_format_data.load_df('predicted/seq_to_dot_test_data_'+seq_to_assay_model_name)
        self.figure_file='./figures/'+self.model_name+'_'+seq_to_assay_model_name+'.png'
        self.stats_file='./model_stats/'+self.model_name+'_'+seq_to_assay_model_name+'.pkl'
        self.test_model()
        # self.plot()

class x_to_assay_model(model):
    'sets to assay_model'
    def __init__(self, model_in, assays, model_architecture, sample_fraction):
        assay_str=','.join([str(x) for x in assays])
        super().__init__(model_in, 'assay'+assay_str, model_architecture, sample_fraction)
        self.assays=assays
        self.get_output_and_explode=partial(load_format_data.explode_assays,assays)
        self.plot_type=plot_model.x_to_assay_plot
        self.training_df=load_format_data.load_df('seq_to_assay_train_1,8,10') #could adjust in future for sequences with predictive assays
        self.testing_df=load_format_data.load_df('assay_to_dot_training_data')
        self.lin_or_sig='sigmoid'
        self.num_cv_splits=3
        self.num_cv_repeats=3
        self.num_test_repeats=10
        self.num_hyp_trials=50


    def save_predictions(self):
        'save assay score predictions of test dataset to be used with assay-to-yield model'
        df=load_format_data.load_df('seq_to_dot_test_data') #will have to adjust if missing datapoints
        OH_matrix=np.eye(len(self.assays))
        x_a=self.get_input_seq(df)
        for z in range(3): #for each model
            for i in range(len(self.assays)): #for each assay
                cat_var=[]
                for j in x_a: #for each sequence add cat_var
                    cat_var.append(OH_matrix[i].tolist())
                x=load_format_data.mix_with_cat_var(x_a,cat_var)
                self._model.set_model(self.get_best_trial()['hyperparam'],xa_len=len(x[0])-len(cat_var[0]), cat_var_len=len(cat_var[0]),lin_or_sig=self.lin_or_sig) #need to build nn arch
                self.load_model(z) #load pkled sklearn model or weights of nn model
                df_prediction=self._model.model.predict(x).squeeze().tolist()
                df.loc[:,'Sort'+str(self.assays[i])+'_mean_score']=df_prediction
            df.to_pickle('./datasets/predicted/seq_to_dot_test_data_'+self.model_name+'_'+str(z)+'.pkl')

    def save_sequence_embeddings(self, df_list=None, is_ordinals_only=False):
        'save sequence embeddings of model'
        # df_list: must either be a list of strings to load dataframes
        #          of a list of dataframes
        # is_ordinals_only: True if it just ordinals [default: false]
        if not df_list:
            df_list = ['assay_to_dot_training_data', 'seq_to_dot_test_data','seq_to_assay_train_1,8,10']
        OH_matrix = np.eye(len(self.assays))
        for df_name in df_list:
            if is_ordinals_only:
                df = df_name
            else:
                df = load_format_data.load_df(df_name)
            x_a = self.get_input_seq(df)
            for z in range(3):  # for each model
                for i in range(1):  # only need to get cat var for one assay to get sequence embedding
                    cat_var = []
                    for j in x_a:  # for each sequence add cat_var
                        cat_var.append(OH_matrix[i].tolist())
                    x = load_format_data.mix_with_cat_var(x_a, cat_var)
                    self._model.set_model(self.get_best_trial()['hyperparam'], xa_len=len(x[0]) - len(cat_var[0]),
                                          cat_var_len=len(cat_var[0]),
                                          lin_or_sig=self.lin_or_sig)  # need to build nn arch
                    self.load_model(z)  # load pkled sklearn model or weights of nn model
                    seq_embedding_model = self._model.get_seq_embeding_layer_model()
                    df_prediction = seq_embedding_model.predict([x])
                    seq_emb_list = []
                    for i in df_prediction:
                        seq_emb_list.append([i])
                    if is_ordinals_only:
                        df.loc[:, 'learned_embedding_' + str(z)] = seq_emb_list
                    else:
                        df.loc[:, 'learned_embedding'] = seq_emb_list
                        df.to_pickle(
                            './datasets/predicted/learned_embedding_' + df_name + '_' + self.model_name + '_' + str(
                                z) + '.pkl')
            if is_ordinals_only:  # if is just ordinals return the dataframe with atrributes:
                # 'ordinals,learned_embedding_0,learned_embedding_1,learned_embedding_2'
                return df

class assay_to_yield_model(x_to_yield_model, assay_to_x_model):
    'assay to yield, provide which assays, limit test set to useable subset'
    def __init__(self, assays, model_architecture, sample_fraction):
        self.assay_str=','.join([str(x) for x in assays])
        super().__init__('assays'+self.assay_str, model_architecture, sample_fraction)
        assay_to_x_model.__init__(self,assays)

class weighted_assay_to_yield_model(x_to_yield_model, assay_to_x_model):
    'weight training data by average(log2(trials))'
    def __init__(self, assays, model_architecture, sample_fraction):
        self.assay_str=','.join([str(x) for x in assays])
        super().__init__('weighted_assays'+self.assay_str, model_architecture, sample_fraction)
        assay_to_x_model.__init__(self,assays)
        self.weightbycounts=True
        self.weightbycountsfxn=partial(load_format_data.weightbycounts,assays)

class twogate_assay_to_yield_model(x_to_yield_model, assay_to_x_model):
    'assay to yield, provide which assays and stringency, limit test set to useable subset'
    def __init__(self, assays, stringency, model_architecture, sample_fraction):
        self.assay_str=','.join([str(x) for x in assays])
        super().__init__('twogate'+stringency+'_assays'+self.assay_str, model_architecture, sample_fraction)
        assay_to_x_model.__init__(self,assays)
        self.training_df=load_format_data.load_df('assay_to_dot_training_data_twogate_'+stringency)
        self.testing_df=load_format_data.load_df('seq_to_dot_test_data_twogate_'+stringency)

class assay_count_to_yield_model(x_to_yield_model):
    'assay to yield including the number of observations in the input'
    def __init__(self, assays, model_architecture, sample_fraction):
        self.assay_str=','.join([str(x) for x in assays])
        super().__init__('assays_counts_'+self.assay_str, model_architecture, sample_fraction)
        self.get_input_seq=partial(load_format_data.get_assays_and_counts,assays)

class stassay_to_yield_model(x_to_yield_model):
    'assay to yield, provide which assays and which trial '
    def __init__(self, assays, trial, model_architecture, sample_fraction):
        self.assay_str=','.join([str(x) for x in assays])
        super().__init__('st'+str(trial)+'_assays'+self.assay_str, model_architecture, sample_fraction)
        self.get_input_seq=partial(load_format_data.get_stassays,assays,trial)

class ttassay_to_yield_model(x_to_yield_model):
    'assay to yield, provide which assays and which 2 trials'
    def __init__(self, assays, trials, model_architecture, sample_fraction):
        self.assay_str=','.join([str(x) for x in assays])
        super().__init__('tt'+str(trials[0])+','+str(trials[1])+'_assays'+self.assay_str, model_architecture, sample_fraction)
        self.get_input_seq=partial(load_format_data.get_ttassays,assays,trials)

class seq_to_yield_model(x_to_yield_model, seq_to_x_model):
    'seq to yield'
    def __init__(self, model_architecture, sample_fraction):
        super().__init__('seq', model_architecture, sample_fraction)
        seq_to_x_model.__init__(self,model_architecture)

class seqandassay_to_yield_model(x_to_yield_model):
    'combine sequence and assay scores for model input'
    def __init__(self,assays,model_architecture,sample_fraction):
        self.assay_str=','.join([str(x) for x in assays])
        super().__init__('seq_and_assays'+self.assay_str,model_architecture,sample_fraction)
        self.get_input_seq=partial(load_format_data.get_seq_and_assays,assays)

class seqandtwogateassay_to_yield_model(x_to_yield_model):
    'combine sequence and assay scores for model input'
    def __init__(self,assays, stringency, model_architecture,sample_fraction):
        self.assay_str=','.join([str(x) for x in assays])
        super().__init__('seq_and_twogate'+stringency+'_assays'+self.assay_str,model_architecture,sample_fraction)
        self.get_input_seq=partial(load_format_data.get_seq_and_assays,assays)
        self.training_df=load_format_data.load_df('assay_to_dot_training_data_twogate_'+stringency)
        self.testing_df=load_format_data.load_df('seq_to_dot_test_data_twogate_'+stringency)

class seqandweightedassay_to_yield_model(x_to_yield_model):
    'sequence and assay input, training weighted by observations'
    def __init__(self,assays,model_architecture,sample_fraction):
        self.assay_str=','.join([str(x) for x in assays])
        super().__init__('weighted_seq_and_assays'+self.assay_str,model_architecture,sample_fraction)
        self.get_input_seq=partial(load_format_data.get_seq_and_assays,assays)
        self.weightbycounts=True
        self.weightbycountsfxn=partial(load_format_data.weightbycounts,assays)

class seqandstassay_to_yield_model(x_to_yield_model):
    'seq and assay to yield, provide which assays and which trial '
    def __init__(self, assays, trial, model_architecture, sample_fraction):
        self.assay_str=','.join([str(x) for x in assays])
        super().__init__('seq_and_st'+str(trial)+'_assays'+self.assay_str, model_architecture, sample_fraction)
        self.get_input_seq=partial(load_format_data.get_seq_and_stassays,assays,trial)

class seqandttassay_to_yield_model(x_to_yield_model):
    'seq and assay to yield, provide which assays and which trials to average '
    def __init__(self, assays, trials, model_architecture, sample_fraction):
        self.assay_str=','.join([str(x) for x in assays])
        super().__init__('seq_and_tt'+str(trials[0])+','+str(trials[1])+'_assays'+self.assay_str, model_architecture, sample_fraction)
        self.get_input_seq=partial(load_format_data.get_seq_and_ttassays,assays,trials)

class seqandassay_count_to_yield_model(x_to_yield_model):
    'seq and assay (including counts) to yield'
    def __init__(self,assays,model_architecture,sample_fraction):
        self.assay_str=','.join([str(x) for x in assays])
        super().__init__('seq_and_assays_counts_'+self.assay_str,model_architecture,sample_fraction)
        self.get_input_seq=partial(load_format_data.get_seq_and_assays_and_counts,assays)

class final_seq_to_yield_model(seq_to_yield_model):
    'redoes training and testing divison for final comparison'
    def __init__(self,model_architecture,sample_fraction):
        super().__init__(model_architecture,sample_fraction)
        self.update_model_name('final'+self.model_name)
        self.switch_train_test()

class seq_to_pred_yield_model(x_to_yield_model,seq_to_x_model):
    'sequence to yield model using predicted yields from assay scores'
    def __init__(self, pred_yield_model_prop, seq_to_pred_yield_prop):
        super().__init__('seq',seq_to_pred_yield_prop[0],seq_to_pred_yield_prop[1])
        seq_to_x_model.__init__(self,seq_to_pred_yield_prop[0])
        self.assay_str=','.join([str(x) for x in pred_yield_model_prop[0]])
        pred_yield_model_name='seq_and_assays'+self.assay_str+'_yield_'+pred_yield_model_prop[1]+'_'+str(pred_yield_model_prop[2])+'_'+str(pred_yield_model_prop[3]) #change for seq and assay
        self.update_model_name(self.model_name+':'+pred_yield_model_name)
        # self.training_df=load_format_data.load_df('predicted/seq_to_assay_train_1,8,10_'+pred_yield_model_name)
        self.training_df=load_format_data.load_df('predicted/seq_to_assay_train_1,8,10_seq_and_assay_yield_forest_1_0')

        self.num_cv_splits=3
        self.num_cv_repeats=3
        self.num_test_repeats=1
        self.num_hyp_trials=50


class seq_to_assay_model(x_to_assay_model, seq_to_x_model):
    'seq to assay, provide assays'
    def __init__(self, assays, model_architecture, sample_fraction):
        super().__init__('seq',assays, model_architecture, sample_fraction)
        seq_to_x_model.__init__(self,model_architecture)

class control_to_assay_model(x_to_assay_model, control_to_x_model):
    'predict assay scores based upon average of assay score of training set'
    def __init__(self, assays, model_architecture, sample_fraction):
        super().__init__('control',assays, model_architecture, sample_fraction)
        control_to_x_model.__init__(self)

class control_to_yield_model(x_to_yield_model, control_to_x_model):
    'predict assay scores based upon average of assay score of training set'
    def __init__(self, model_architecture, sample_fraction):
        super().__init__('control', model_architecture, sample_fraction)
        control_to_x_model.__init__(self)

class sequence_embeding_to_yield_model(x_to_yield_model, sequence_embedding_to_x_model):
    'predict yield from sequence embedding trained by a seq-to-assay model'
    def __init__(self, seq_to_assay_model_prop, model_architecture, sample_fraction):
        self.assay_str=','.join([str(x) for x in seq_to_assay_model_prop[0]])
        seq_to_assay_model_name='seq_assay'+self.assay_str+'_'+str(seq_to_assay_model_prop[1])+'_'+str(seq_to_assay_model_prop[2])+'_'+str(seq_to_assay_model_prop[3])
        super().__init__('embedding_'+seq_to_assay_model_name, model_architecture, sample_fraction)
        sequence_embedding_to_x_model.__init__(self)
        self.num_test_repeats=1
        self.training_df=load_format_data.load_df('/predicted/learned_embedding_assay_to_dot_training_data_'+seq_to_assay_model_name)
        self.testing_df=load_format_data.load_df('/predicted/learned_embedding_seq_to_dot_test_data_'+seq_to_assay_model_name)

class final_sequence_embeding_to_yield_model(sequence_embeding_to_yield_model):
    'look at class name, but done with better train/test split'
    def __init__(self, seq_to_assay_model_prop, model_architecture, sample_fraction):
        super().__init__(seq_to_assay_model_prop, model_architecture, sample_fraction)
        self.update_model_name('final'+self.model_name)
        self.switch_train_test()