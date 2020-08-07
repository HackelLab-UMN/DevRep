import numpy as np
import pickle
from functools import partial
from model_module import model
import load_format_data
import plot_model

## The foolowing classes are considered LIST_B classes:
## seq_to_x_model , assay_to_x_model , control_to_x_model , sequence_embedding _to_x_model
class seq_to_x_model():
    'sets get_input_seq to ordinal or onehot sequence based upon model_architecture'
    ## Used for models in which sequence is used to build the prediction model
    ## This class requires a string relating to regression models class objects in the model_architecture.py script
    ## This class creates a class variable get_input_seq which if the object has an embedded regression then it uses the ordinalsequence
    ## or else it uses the one hot sequence 
    def __init__(self, model_architecture):
        if 'emb' in model_architecture:
            self.get_input_seq=load_format_data.get_ordinal
        else:
            self.get_input_seq=load_format_data.get_onehot

class assay_to_x_model():
    'sets get_input_seq to assay scores of assays'
    ## Used for models in which assay scores is used to build the prediction model
    ## This class requires a list containing non-repeating numbers between [1,10] for object instantiation
    ## This sets the get_input_seq class variable to the output of get_assay function fromm load_format_data for the input assays list.
    ## The get_assays function also requires a dataframe which will be specified in its child classes
    def __init__(self, assays):
        self.get_input_seq=partial(load_format_data.get_assays,assays)

class control_to_x_model():
    'sets get_input_seq to nothing, not sure if needed'
    ## Sets the get_input_seq class variable as an empty list of a certain length.
    ## This class doesn't require any input for object instantiation and the get_input_seq is linked to the get_control() function
    ## in the load_format_data.py script
    def __init__(self):
        self.get_input_seq=load_format_data.get_control 

class sequence_embedding_to_x_model():
    'sets get_input_seq to load the sequence embedding from a saved seq-to-assay model'
    ## Used for models in which sequence embedding is used to build the prediction model
    ## This class requires no input for object instantiation. and the get_input_seq is linked to the get_embedding() function
    ## in the load_format_data.py script
    def __init__(self):
        self.get_input_seq=load_format_data.get_embedding 

## The following two class objects will be refferred as LIST_C objects 
class x_to_yield_model(model):
    ## This class inherits from the model class in the model_module.py script
    ## Used to create a yield based model.
    'sets model output to yield'
    def __init__(self, model_in, model_architecture, sample_fraction):
        ## To instantiate the object two strings and a float is required. These are then used along with the yield string
        ## to create the class variables belonging to its parent class. 
        super().__init__(model_in, 'yield', model_architecture, sample_fraction)
        self.get_output_and_explode=load_format_data.explode_yield
        self.plot_type=plot_model.x_to_yield_plot
        ## A get_output_and_explode class vfunction is created and linked to the explode_yield() function of the load_format_data.py cript
        ## Similarly a plot_type class function is created and linked to the x_to_yield_plot object class in the plot_model.py script 
        self.training_df=load_format_data.load_df('assay_to_dot_training_data')
        self.testing_df=load_format_data.load_df('seq_to_dot_test_data') 
        ## Training and testing data is accessed using the load_df() function from the load_format_data.py script
        ## Training and testing data is accessed from the assay_to_dot_training_data and the seq_to_dot_test_data files 
        self.lin_or_sig='linear'
        self.num_cv_splits=10
        self.num_cv_repeats=10
        ## A linear regression model is suggested in the lin_or_sig string class variable, then the number of splits and repeats for the cross validatiion is 
        ## specified in the num_cv_splits and num_cv_repeats respectively 
        self.num_test_repeats=10
        self.num_hyp_trials=50
        ## Finally the number of test repetas and the nuber of hyperopt trials are also mentioned in the num_test_repeats and num_hyp_trials class variable
        ## respectively.
    
    def change_sample_seed(self,seed):
        ## A new class variable is created called sample_seed and this is linked to the input seed
        self.sample_seed=seed
        ## Then the update_model_name() function is run for the given input to change the model name
        self.update_model_name('seed'+str(seed)+'_'+self.model_name)

    def save_predictions(self,input_df_description=None):
        'saves model predictions for the large dataset'
        ## This function only works for child class that inherits from both this class and any of the LIST_B classes 
        ## The input_df_description is a string of the file containg the data we are going tp access
        ## If no value is entered for input_df_dicription then the default seq_to_assay_train_1,8,10 data is loaded.
        ## Or else the subsequent file in the predicted directory is loaded.
        if not input_df_description:
            input_df_description='seq_to_assay_train_1,8,10'
            df=load_format_data.load_df(input_df_description) #will have to adjust if missing datapoints
        else:
            df=load_format_data.load_df('predicted/'+input_df_description) #for using predicted embeddings 
        OH_matrix=np.eye(2)
        ## A 2zD identity matrix is created and assigned to to OH_matrix variable
        matrix_col=['IQ_Average_bc','SH_Average_bc']
        ## Another list created with the column heading for the previous OH_matrix as IQ_Average_bc and SH_Avegrage_bc respectively
        x_a=self.get_input_seq(df)
        ## Depending on which LIST_B class the child classes inherits from the get_input_seq function is linked to a certain a function and 
        ## returns a particular dataframe.
        for z in range(1): #no of models
            self.load_model(z)
            ## The load_model() function outlined in the model class is run which updates the model class variable of the 
            ## model_architecture.py class objects. 
            for i in range(2):
                cat_var=[]
                for j in x_a:
                    cat_var.append(OH_matrix[i].tolist())
                ## An empty list cat_var is created with one of its element repeating the same amount of time as the 
                ## x_a dataframe. 
                x=load_format_data.mix_with_cat_var(x_a,cat_var)
                ## Then the cat_var list along with the x_a dataframe is inputtted into the mix_with_cat_var() function which inturn
                ## reurns a concatanated list with the x_a and cat_var
                df_prediction=self._model.model.predict(x).squeeze().tolist()
                ## This accesses the model class variable for that particular model architecture then using the list created above
                ## a predicted model is created which is squeezed to remove single dimensional entries and then convert it into a list 
                col_name=matrix_col[i]
                ## A col_name list tracks the predictions for the IQ and SH average_bc.
                df.loc[:,col_name]=df_prediction
                ## Then the inital dataframe df, accessed using the input string, has its column corresponding to the IQ_Average_bc or 
                ## SH_average_bc to match the predictions generated and stored in the df_prediction list. 
                col_name_std=matrix_col[i]+'_std'
                df.loc[:,col_name_std]=[0]*len(df_prediction)
                ## Similarly the IC_Avergae_bc_std and SH_Average_bc_std columns are also editted to be a list containing [0]
            df.to_pickle('./datasets/predicted/'+input_df_description+'_'+self.model_name+'_'+str(z)+'.pkl')
            ## Finally the updated dataframe is stored in the predicted directory of the dataset directory under the input_df_description
            ## and the model_name (class variable) as a pickle file

    def switch_train_test(self):
        ## This function is used to edit the training_df class varirable 
        regular_training_df=self.training_df
        ## Initally a the training_df is assigned to another local variable regular_triaing_df
        extra_training_df,self.testing_df=load_format_data.get_random_split(self.testing_df)
        ## Then the get_random_split() function is run for the testing_df class dataframe which splits the testing data into training data
        ## testing data which is stored in the extra_training_df dataframe and self.testing_df dataframe respectively 
        self.training_df=regular_training_df.append(extra_training_df)
        ## The newly created extra_training_df is added to the initally created regular_training_df and this is saved in the training_df.

    def limit_test_set(self,assays):
        #Limit test set for data that has all assay scores used in model
        ## The input assays is a list of number ranging between 1-10 including the endpoints and none of the numbers repeat
        sort_names=[]
        ## A new empty list is created  and stored under the name of sort_names
        for i in assays:
            sort_names.append('Sort'+str(i)+'_mean_score')
        ## The column headings for each assay score are saved in the sort_names list
        dataset=self.testing_df
        ## the testing dataframe is then temporarily stored in the dataset local variable
        dataset=dataset[~dataset[sort_names].isna().any(axis=1)]
        ## The dataset is modified and the class dataframe testing_df is updated to that dataframe. 
        self.testing_df=dataset

    def apply_predicted_assay_scores(self,seq_to_assay_model_prop):
        ## This function takes an input called seq_to_assay_model_prop which is an rray or some other
        ## iterable object
        'uses saved predicted assay scores and saved assay-to-yield model to determine performance on test-set' 
        seq_to_assay_model_name='seq_assay'+self.assay_str+'_'+str(seq_to_assay_model_prop[0])+'_'+str(seq_to_assay_model_prop[1])+'_'+str(seq_to_assay_model_prop[2])
        ## First a local string variable is created by combining the 'seq_assay' with self.assay_str, class variable present in child classes
        ## listed in LIST_A, along with the first 3 elements of the seq_to_assay_model_prop list. 
        self.num_test_repeats=1
        ## The class variable num_test_repeats is changed to 1.
        self.testing_df=load_format_data.load_df('predicted/seq_to_dot_test_data_'+seq_to_assay_model_name)
        ## Similarly the testing_df dataframe is updated to another file in the predicted directory
        self.figure_file='./figures/'+self.model_name+'_'+seq_to_assay_model_name+'.png'
        self.stats_file='./model_stats/'+self.model_name+'_'+seq_to_assay_model_name+'.pkl'
        ## The strings attached to the figure_file and stats_file from the model parent class is updated, so as to
        ## access the proper files in the figures and model_stats directories. 
        self.test_model()
        ## The the given prediction is tested, using the test_model() function in the model parent class. 
        # self.plot()
        
class x_to_assay_model(model):
    ## This class inherits from the model class in the model_module.py script
    ## Used to create a assay based model.
    'sets to assay_model'
    def __init__(self, model_in, assays, model_architecture, sample_fraction):
        ## To instantiate the object two strings, an array and a float is required, the array is a list of non-repeating numbers between 1-10
        ## These are then used along with the yield string to create the class variables belonging to its parent class. 
        assay_str=','.join([str(x) for x in assays])
        ## The elemenst of the assay list are combined in a string. 
        super().__init__(model_in, 'assay'+assay_str, model_architecture, sample_fraction)
        ## This is then passed into the model class instantiaon and its respective class variables and functions are also created
        self.assays=assays
        self.get_output_and_explode=partial(load_format_data.explode_assays,assays)
        ## A class variable list called assays is assigned to the input list and the get_output_and_explode is linked to the explode_assays()
        ## function from load_data_format.py with the assays list input already in it.
        self.plot_type=plot_model.x_to_assay_plot
        self.training_df=load_format_data.load_df('seq_to_assay_train_1,8,10') #could adjust in future for sequences with predictive assays
        self.testing_df=load_format_data.load_df('assay_to_dot_training_data')
        ## Similarly a plot_type class function is created and linked to the x_to_assay_plot object class in the plot_model.py script
        ## Similarly two class dataframes called testing_df and training_df are also created by accessing the seq_to_assay_train_1,8,10
        ## and the assay_to_dot_training_data respectively. 
        self.lin_or_sig='sigmoid'
        self.num_cv_splits=3
        self.num_cv_repeats=3
        ## A sigmoid regression model is suggested in the lin_or_sig string class variable, then the number of splits and repeats for the cross validatiion is 
        ## specified in the num_cv_splits and num_cv_repeats respectively 
        self.num_test_repeats=10
        self.num_hyp_trials=50
        ## Finally the number of test repetas and the nuber of hyperopt trials are also mentioned in the num_test_repeats and num_hyp_trials class variable
        ## respectively.

    def save_predictions(self):
        'save assay score predictions of test dataset to be used with assay-to-yield model'
        ## This function requires no input and it saves assay score prediction
        df=load_format_data.load_df('seq_to_dot_test_data') #will have to adjust if missing datapoints
        ## Initally the seq_to_dot_test_data file is accessed and it is assigned to the df dataframe
        OH_matrix=np.eye(len(self.assays))
        ## Then an identity matrix is created at the same size as the number of assay used to build the prediction. 
        x_a=self.get_input_seq(df)
        ## Depending on which LIST_B class the child classes (LIST_A objects) inherits from the get_input_seq function is linked to a certain a function and 
        ## returns a particular dataframe.
        for z in range(3): #for each model
            for i in range(len(self.assays)): #for each assay
                ## For each assay in the assays list, a new list called cat_var is created then each element of the OH_matrix is appended
                ## to the cat_var the same nuber of times as the length of the x_a dataframe. 
                ## Then a dataframe x is created using the mix_with_cat_var() function which is used to run the regression model for the best_trial
                ## with the hyperparamters specified in the x_a ,x and cat_var dataframes. Then the load_model() function is run 
                ## then the predictions are made using the x dataframe which are then saved in the original dataframe under the assay_score_mean 
                ## FInally the dataframe is saved as a pickle file in the datasets directory in the same combined name of the 'seq_to_dot_test_data'
                ## along with the model_name. 
                cat_var=[]
                for j in x_a: #for each sequence add cat_var
                    cat_var.append(OH_matrix[i].tolist())
                x=load_format_data.mix_with_cat_var(x_a,cat_var)
                self._model.set_model(self.get_best_trial()['hyperparam'],xa_len=len(x[0])-len(cat_var[0]), cat_var_len=len(cat_var[0]),lin_or_sig=self.lin_or_sig) #need to build nn arch
                self.load_model(z) #load pkled sklearn model or weights of nn model
                df_prediction=self._model.model.predict(x).squeeze().tolist()
                df.loc[:,'Sort'+str(self.assays[i])+'_mean_score']=df_prediction
            df.to_pickle('./datasets/predicted/seq_to_dot_test_data_'+self.model_name+'_'+str(z)+'.pkl')
        

    def save_sequence_embeddings(self,df_list=None):
        'save sequence embeddings of model'
        ## For this function a list input is option. If an input is given then the temporary variable dataframe is set to 
        ## a list containing the name of the two dataframes. 
        if not df_list:
            df_list=['assay_to_dot_training_data','seq_to_dot_test_data']
        OH_matrix=np.eye(len(self.assays))
        ## An identity matrix is created the same length as the number of assys used to build the prediction model.
        for df_name in df_list:
            df=load_format_data.load_df(df_name)
            x_a=self.get_input_seq(df)
            ## For each name in the df_list the dataframe is accessed and the get_input_seq() function is run on it and stored in the x_a dataframe
            ## Depending on which LIST_B class the child classes (LIST_A objects) inherits from the get_input_seq function is linked to a certain a function and 
            ## returns a particular dataframe.
            for z in range(3): #for each model
                for i in range(1): #only need to get cat var for one assay to get sequence embedding 
                    ## For each assay in the assays list, a new list called cat_var is created then each element of the OH_matrix is appended
                ## to the cat_var the same nuber of times as the length of the x_a dataframe. 
                ## Then a dataframe x is created using the mix_with_cat_var() function which is used to run the regression model for the best_trial
                ## with the hyperparamters specified in the x_a ,x and cat_var dataframes. Then the load_model() function is run 
                ## then the predictions are made using the x dataframe which are then saved in the original dataframe under the assay_score_mean 
                ## FInally the dataframe is saved as a pickle file in the datasets directory in the same combined name of the 'seq_to_dot_test_data'
                ## along with the model_name. 
                    cat_var=[]
                    for j in x_a: #for each sequence add cat_var
                        cat_var.append(OH_matrix[i].tolist())
                    x=load_format_data.mix_with_cat_var(x_a,cat_var)
                    self._model.set_model(self.get_best_trial()['hyperparam'],xa_len=len(x[0])-len(cat_var[0]), cat_var_len=len(cat_var[0]),lin_or_sig=self.lin_or_sig) #need to build nn arch
                    self.load_model(z) #load pkled sklearn model or weights of nn model
                    seq_embedding_model=self._model.get_seq_embeding_layer_model()
                    df_prediction=seq_embedding_model.predict([x])
                    seq_emb_list=[]
                    for i in df_prediction:
                        seq_emb_list.append([i])
                    df.loc[:,'learned_embedding']=seq_emb_list
                df.to_pickle('./datasets/predicted/learned_embedding_'+df_name+'_'+self.model_name+'_'+str(z)+'.pkl')


## The following classes are considered LIST_A objects which are discribed in the model_module.py script.
## Some of these objects inherit from the LIST_B and also inherits from the LIST_C class objects, while others just inherit from LIST_B

class assay_to_yield_model(x_to_yield_model, assay_to_x_model):
    'assay to yield, provide which assays, limit test set to useable subset'
    ## To instantiate this class object an array, a string and a float between 0 and 1 are required.
    ## This is in turn used construct a class variable assay_str, then the x_to_yield_model
    ## object is instantiated followed by the assay_to_x_model thereby inheriting its respective class variables and funtions
    def __init__(self, assays, model_architecture, sample_fraction):
        self.assay_str=','.join([str(x) for x in assays])
        super().__init__('assays'+self.assay_str, model_architecture, sample_fraction)
        assay_to_x_model.__init__(self,assays)

class weighted_assay_to_yield_model(x_to_yield_model, assay_to_x_model):
    'weight training data by average(log2(trials))'
    ## To instantiate this class object an array, a string and a float between 0 and 1 are required.
    ## This is in turn used construct a class variable assay_str another boolean weightbycounts and the function 
    ## weightbycountsfxn is linked to the weightbycounts() function in load_format_data, then the x_to_yield_model
    ## object is instantiated followed by the assay_to_x_model thereby inheriting its respective class variables and funtions
    def __init__(self, assays, model_architecture, sample_fraction):
        self.assay_str=','.join([str(x) for x in assays])
        super().__init__('weighted_assays'+self.assay_str, model_architecture, sample_fraction)
        assay_to_x_model.__init__(self,assays)
        self.weightbycounts=True 
        self.weightbycountsfxn=partial(load_format_data.weightbycounts,assays)

class twogate_assay_to_yield_model(x_to_yield_model, assay_to_x_model):
    'assay to yield, provide which assays and stringency, limit test set to useable subset'
    ## To instantiate this class object an array, a string and a float between 0 and 1 are required.
    ## This is in turn used construct a class variable assay_str and two class dataframes called training_df
    ## and testing_df, then the x_to_yield_model object is instantiated followed by the assay_to_x_model thereby inheriting its respective class variables and funtions
    def __init__(self, assays, stringency, model_architecture, sample_fraction):
        self.assay_str=','.join([str(x) for x in assays])
        super().__init__('twogate'+stringency+'_assays'+self.assay_str, model_architecture, sample_fraction)
        assay_to_x_model.__init__(self,assays)    
        self.training_df=load_format_data.load_df('assay_to_dot_training_data_twogate_'+stringency)
        self.testing_df=load_format_data.load_df('seq_to_dot_test_data_twogate_'+stringency)

class assay_count_to_yield_model(x_to_yield_model):
    'assay to yield including the number of observations in the input'
    ## To instantiate this class object an array, a string and a float between 0 and 1 are required.
    ## This is in turn used construct a class variable assay_str then the get_input_seq is linked to the get_assays_and_counts 
    ## function in the load_format_data script, then the x_to_yield_model object is instantiated, thereby inheriting its class variables.
    def __init__(self, assays, model_architecture, sample_fraction):
        self.assay_str=','.join([str(x) for x in assays])
        super().__init__('assays_counts_'+self.assay_str, model_architecture, sample_fraction)
        self.get_input_seq=partial(load_format_data.get_assays_and_counts,assays)

class stassay_to_yield_model(x_to_yield_model):
    'assay to yield, provide which assays and which trial '
    ## To instantiate this class object two arrays, a string and a float between 0 and 1 are required.
    ## This is in turn used construct a class variable assay_str then the get_input_seq is linked to the get_stassays 
    ## function in the load_format_data script, then the x_to_yield_model object is instantiated, thereby inheriting its class variables.
    def __init__(self, assays, trial, model_architecture, sample_fraction):
        self.assay_str=','.join([str(x) for x in assays])
        super().__init__('st'+str(trial)+'_assays'+self.assay_str, model_architecture, sample_fraction)
        self.get_input_seq=partial(load_format_data.get_stassays,assays,trial)

class ttassay_to_yield_model(x_to_yield_model):
    ## To instantiate this class object two arrays, a string and a float between 0 and 1 are required.
    ## This is in turn used construct a class variable assay_str then the get_input_seq is linked to the get_ttassays 
    ## function in the load_format_data script, then the x_to_yield_model object is instantiated, thereby inheriting its class variables.
    'assay to yield, provide which assays and which 2 trials'
    def __init__(self, assays, trials, model_architecture, sample_fraction):
        self.assay_str=','.join([str(x) for x in assays])
        super().__init__('tt'+str(trials[0])+','+str(trials[1])+'_assays'+self.assay_str, model_architecture, sample_fraction)
        self.get_input_seq=partial(load_format_data.get_ttassays,assays,trials)

class seq_to_yield_model(x_to_yield_model, seq_to_x_model):
    'seq to yield'
    ## To instantiate this class object, a string and a float between 0 and 1 are required. First the x_to_yield_model
    ##  object is instantiated followed by the seq_to_x_model thereby inheriting its respective class variables and funtions
    def __init__(self, model_architecture, sample_fraction):
        super().__init__('seq', model_architecture, sample_fraction)
        seq_to_x_model.__init__(self,model_architecture)

class seqandassay_to_yield_model(x_to_yield_model):
    'combine sequence and assay scores for model input'
    ## To instantiate this class object an arrays, a string and a float between 0 and 1 are required.
    ## This is in turn used to construct a class variable assay_str then the get_input_seq is linked to the get_seq_and_assays() 
    ## function in the load_format_data script, then the x_to_yield_model object is instantiated,thereby inheriting its class variables.
    def __init__(self,assays,model_architecture,sample_fraction):
        self.assay_str=','.join([str(x) for x in assays])
        super().__init__('seq_and_assays'+self.assay_str,model_architecture,sample_fraction)
        self.get_input_seq=partial(load_format_data.get_seq_and_assays,assays)

class seqandtwogateassay_to_yield_model(x_to_yield_model):
    'combine sequence and assay scores for model input'
    ## To instantiate this class object an array, two strings and a float between 0 and 1 are required.
    ## This is in turn used to construct a class variable assay_str then the get_input_seq is linked to the get_seq_and_assays() 
    ## function in the load_format_data script, similarly two class dataframesd are also constructed testing_df and training_df which are
    ## connected to the assay_to_dot_training_data_twogate file and seq_to_dot_test_data_twogate file respectively.
    ## The x_to_yield_model object is instantiated ensuring it inherits its respective class variables and functions. 
    def __init__(self,assays, stringency, model_architecture,sample_fraction):
        self.assay_str=','.join([str(x) for x in assays])
        super().__init__('seq_and_twogate'+stringency+'_assays'+self.assay_str,model_architecture,sample_fraction)
        self.get_input_seq=partial(load_format_data.get_seq_and_assays,assays)   
        self.training_df=load_format_data.load_df('assay_to_dot_training_data_twogate_'+stringency)
        self.testing_df=load_format_data.load_df('seq_to_dot_test_data_twogate_'+stringency) 

class seqandweightedassay_to_yield_model(x_to_yield_model):
    'sequence and assay input, training weighted by observations'
    ## To instantiate this class object an array, a string and a float between 0 and 1 are required.
    ## This is in turn used to construct a class variable assay_str then the get_input_seq is linked to the get_seq_and_assays() 
    ## function, similarly a class boolean weightbycounts is marked true and the function weightbycountsfxn is linked to the 
    ## weightbycounts() functions in the lod_format_data.py script
    ## The x_to_yield_model object is instantiated ensuring it inherits its respective class variables and functions. 
    def __init__(self,assays,model_architecture,sample_fraction):
        self.assay_str=','.join([str(x) for x in assays])
        super().__init__('weighted_seq_and_assays'+self.assay_str,model_architecture,sample_fraction)
        self.get_input_seq=partial(load_format_data.get_seq_and_assays,assays)
        self.weightbycounts=True 
        self.weightbycountsfxn=partial(load_format_data.weightbycounts,assays)

class seqandstassay_to_yield_model(x_to_yield_model):
    'seq and assay to yield, provide which assays and which trial '
    ## To instantiate this class object two arrays, a string and a float between 0 and 1 are required.
    ## This is in turn used to construct a class variable assay_str then the get_input_seq is linked to the get_seq_and_stassays() 
    ## function in the load_format_data script, then the x_to_yield_model object is instantiated, thereby inheriting its class variables.
    def __init__(self, assays, trial, model_architecture, sample_fraction):
        self.assay_str=','.join([str(x) for x in assays])
        super().__init__('seq_and_st'+str(trial)+'_assays'+self.assay_str, model_architecture, sample_fraction)
        self.get_input_seq=partial(load_format_data.get_seq_and_stassays,assays,trial)

class seqandttassay_to_yield_model(x_to_yield_model):
    'seq and assay to yield, provide which assays and which trials to average '
    ## To instantiate this class object two arrays, a string and a float between 0 and 1 are required.
    ## This is in turn used to construct a class variable assay_str then the get_input_seq is linked to the get_seq_and_ttassays() 
    ## function in the load_format_data script, then the x_to_yield_model object is instantiated, thereby inheriting its class variables.
    def __init__(self, assays, trials, model_architecture, sample_fraction):
        self.assay_str=','.join([str(x) for x in assays])
        super().__init__('seq_and_tt'+str(trials[0])+','+str(trials[1])+'_assays'+self.assay_str, model_architecture, sample_fraction)
        self.get_input_seq=partial(load_format_data.get_seq_and_ttassays,assays,trials)

class seqandassay_count_to_yield_model(x_to_yield_model):
    'seq and assay (including counts) to yield'
    ## To instantiate this class object an array, a string and a float between 0 and 1 are required.
    ## This is in turn used to construct a class variable assay_str then the get_input_seq is linked to the get_seq_and_assays_and_counts() 
    ## function in the load_format_data script, then the x_to_yield_model object is instantiated, thereby inheriting its class variables.
    def __init__(self,assays,model_architecture,sample_fraction):
        self.assay_str=','.join([str(x) for x in assays])
        super().__init__('seq_and_assays_counts_'+self.assay_str,model_architecture,sample_fraction)
        self.get_input_seq=partial(load_format_data.get_seq_and_assays_and_counts,assays)

class final_seq_to_yield_model(seq_to_yield_model):
    ## This object inherits from the seq_to_yield_model() class (LIST_B). To instantiate this class object a string and a float between 0 and 1 are required
    ## Then the seq_to_yield_model class object is instantiated with the model_architecture and sample_fraction input, thereby inheriting its class variables
    ## Then the switch_train_test() function is run to edit the training and testing data.
    'redoes training and testing divison for final comparison'
    def __init__(self,model_architecture,sample_fraction):
        super().__init__(model_architecture,sample_fraction)
        self.update_model_name('final'+self.model_name)
        self.switch_train_test()

class seq_to_pred_yield_model(x_to_yield_model,seq_to_x_model):
    ## This objects inherits from both x_to_yield_model (LIST_C) and seq_to_x_model (LIST_B). To instantiate this class object two lists are rquired
    ## The x_to_yield_model is instantiated with the first two elements of seq_to_pred_yield_prop list, while the seq_to_x_model is instantiated with
    ## the first element of the seq_to_pred_yield_prop list. An assay_str class variable is built from the elements of the pred_yield_model_prop's first element
    ## The model name is updatated using the model function update_model_name() and the training_df class variable is updated to access data from
    ## seq_to_assay_train_1,8,10_seq_and_assay_yield_forest_1_0.pkl in the predicted directory. FInally the number of cross validation splits and repeats
    ## along with the number of test repeats and hyperparameter trials are also updated. 
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
    ## To instantiate this class object,a list, a string and a float between 0 and 1 are required. First the x_to_assay_model
    ##  object is instantiated followed by the seq_to_x_model thereby inheriting its respective class variables and funtions
    def __init__(self, assays, model_architecture, sample_fraction):
        super().__init__('seq',assays, model_architecture, sample_fraction)
        seq_to_x_model.__init__(self,model_architecture)

class control_to_assay_model(x_to_assay_model, control_to_x_model):
    'predict assay scores based upon average of assay score of training set'
    ## To instantiate this class object,a list, a string and a float between 0 and 1 are required. First the x_to_assay_model
    ##  object is instantiated followed by the control_to_x_model thereby inheriting its respective class variables and funtions
    def __init__(self, assays, model_architecture, sample_fraction):
        super().__init__('control',assays, model_architecture, sample_fraction)
        control_to_x_model.__init__(self)

class control_to_yield_model(x_to_yield_model, control_to_x_model):
    'predict assay scores based upon average of assay score of training set'
    ## To instantiate this class object,a list, a string and a float between 0 and 1 are required. First the x_to_yiel_model
    ##  object is instantiated followed by the control_to_x_model thereby inheriting its respective class variables and funtions
    def __init__(self, model_architecture, sample_fraction):
        super().__init__('control', model_architecture, sample_fraction)
        control_to_x_model.__init__(self)

class sequence_embeding_to_yield_model(x_to_yield_model, sequence_embedding_to_x_model):
    'predict yield from sequence embedding trained by a seq-to-assay model'
    ## To instantiate this class object, an array, a string and a flot between 0 and 1 are required. Initally a class variable assay_str
    ## is constructed using the seq_to_assay_model_prop list. Then a temporary string is created by using the 2nd,3rd,4th elements of
    ## the seq_to_assay_model_prop list. Following this the x_to_yield_model and sequence_embedding_to_x_model are instantiated thereby
    ## inheriting the respective class variables and functions. Following this the class variable for the number of test repeats and
    ## the training and testing dataframes are also updated. 
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
     ## This yields from the above mentioned class sequence_embedding_to_yield_model() (LIST_A object). To instantiate
     ## this object a list, a string and a float between 0 and 1 is required, these inputs are then passed into the sequence_embeddding_to_yield_model
     ## instantiator to inherit its variables and functions. Following this the model name is updated and the trainig and testing data set is switched.
    def __init__(self, seq_to_assay_model_prop, model_architecture, sample_fraction):
        super().__init__(seq_to_assay_model_prop, model_architecture, sample_fraction)
        self.update_model_name('final'+self.model_name)
        self.switch_train_test()