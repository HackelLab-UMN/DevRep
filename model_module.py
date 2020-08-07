import numpy as np
import pandas as pd
import pickle
from hyperopt import Trials,fmin,tpe,STATUS_OK
from sklearn.model_selection import RepeatedKFold
from sklearn.metrics import mean_squared_error as mse
import load_format_data
from model_architectures import get_model
from plot_model import x_to_yield_plot

class model:
    '''The model class will cross-validate the training set to determine hyperparameters
    then use the set hyperparameters to evaluate against a test set and save predictions''' 

    def __init__(self, model_in, model_out, model_architecture, sample_fraction):
        ## This creates an instance of an object of type model 
        ## Of the given inputs model_in, model_out and model_architecture are strings
        ## sample fractions is a float of value between 0.0-1.0
        self.sample_fraction=sample_fraction
        
        self.model_architecture=model_architecture
        ## Notice how the class variables sample_fraction and model_architecture
        ## are attributed to their respective inputs
        
        self.model_name=model_in+'_'+model_out+'_'+model_architecture+'_'+str(sample_fraction)
        ## model_name is a string combination of all the inputs. 
        
        self.trials_file='./trials/'+self.model_name+'.pkl'
        self.stats_file='./model_stats/'+self.model_name+'.pkl'
        self.plotpairs_file='./plotpairs/'+self.model_name+'.pkl'
        ## The above class variables are string pathways to respective pickle files needed to construct the model
        ## These class variables are also dependent on the previously mentioned variable model_name
        
        self.figure_file='./figures/'+self.model_name+'.png'
        ## figure_file is a string pathway to access the corresponding model image given the model_name
        
        self.model_loc='./models/'+self.model_name
        ## model_loc is a string pathway showing the location of the model.
        
        
        self._model = get_model(model_architecture) #ridge,forest,svm,nn(emb,)
        ## This sets a self._model to the regression typoe used to build the model, model_architercture is a string input
        
        ## If-else statement checks if the following model has already been run. 
        ## if it does exsist then it access them and assigns them to respective class variable
        ## else it sets them to a default value. 
        if(self.load_hyp()==True):
            self.model_stats = pickle.load(open(self.stats_file, "rb"))
            [self.plotpairs_cv,self.plotpairs_test]=pickle.load(open(self.plotpairs_file,'rb'))
            
        else:
            print('No previous trial data, model data or plot data available')
            self.model_stats= {'cv_avg_loss': np.inf,'cv_std_loss': [],'test_avg_loss': np.inf,'test_std_loss': []}
            [self.model_plotpairs_cv,self.model_plotpairs_test]=[[[],[],[]],[[],[],[]]]
        
        self.plot_type=None

    def parent_warning(self):
        print('im in parent class')

### Question: should we be amending the model_architecture and sample fraction
    def update_model_name(self,model_name):
        ## This is a setter function used to change the model_name, thereby the assay, model_architecture and corresponding
        ## trial data, model data and plot data. The function input is a string
        self.model_name=model_name
        self.trials_file='./trials/'+self.model_name+'.pkl'
        self.stats_file='./model_stats/'+self.model_name+'.pkl'
        self.plotpairs_file='./plotpairs/'+self.model_name+'.pkl'
        self.figure_file='./figures/'+self.model_name+'.png'
        self.model_loc='./models/'+self.model_name
        ## The above statements set the class variable established in the instantiation function to the new value
        self.load_hyp()
        self.load_model_stats()
        self.load_plotpairs()
        ## The above functions check if trial data, model data and plot data already exsist for the given model_name
        ## If it doesn exsist then it sets it respectively or else it sets it to a 

    def save_plotpairs(self):
        ## This function navigates to the plotpairs directory and saves 
        ## the plotpairs_cv and plotpairs_test data as a pickle file under the name of the model_name
        with open (self.plotpairs_file,'wb') as f:
            pickle.dump([self.plotpairs_cv,self.plotpairs_test],f)

    def load_plotpairs(self):
        ## This function checks whether plot data is available in the plotpairs directory
        ## if it is available then it sets the ploatpairs_cv and plotpairs_test to the data else it defualts them
        try:
            [self.plotpairs_cv,self.plotpairs_test]=pickle.load(open(self.plotpairs_file,'rb'))
        except:
            print('No plot data available')
            [self.model_plotpairs_cv,self.model_plotpairs_test]=[[[],[],[]],[[],[],[]]]

    def save_model_stats(self):
        ## This function navigates to the model_stats directory and saves the 
        ## self.model_stats data as a pickle file under the model_name
        with open (self.stats_file,'wb') as f:
            pickle.dump(self.model_stats,f)

    def load_model_stats(self):
        ## This function checks whether model data is available in the model_stats directory
        ## if it is available then it sets the model_stats variable to the data else it defaults the variable
        try: 
            self.model_stats = pickle.load(open(self.stats_file, "rb"))
        except:
            print('No previous model data saved')
            self.model_stats= {
            'cv_avg_loss': np.inf,
            'cv_std_loss': [],
            'test_avg_loss': np.inf,
            'test_std_loss': []
            }
            print(self.model_name)

    def build_architecture(self, model_architecture):
        ## This function is used to create a model_architecure model object and make it a protected class variable
        ## also acts aas a setter function
        'load architecture class which sets hyp space'
        self._model=get_model(model_architecture)

    def load_hyp(self):
        ## This function checks whether trial data is available in the trials directory
        ## if it is available then it sets the tpe_trails variable to the data else it defaults the variable
        'load hyperopt trials'
        try:  # try to load an already saved trials object
            self.tpe_trials = pickle.load(open(self.trials_file, "rb"))
            return True
        except:
            self.tpe_trials = Trials()
            return False

    def save_hyp(self):
        ## This function navigates to the trials directory and saves the
        ## self.tpe_trials  data as a pickle file under the model name
        'save hyperopt trials, refresh best trial'
        with open(self.trials_file, "wb") as f:
            pickle.dump(self.tpe_trials, f)
    
    def save_model(self,model_no):
        'save the trained model'
        ## Checks if the regression model for the data is based on neural network
        ## If it is saves all layer weights as a tensor flow in the models directors under the model name
        ## Else it saves the model as a pickle file in models directory under the model name
        if 'nn' in self.model_architecture:
            self._model.model.save_weights(self.model_loc+'_'+str(model_no)+'/')
        else:
            with open(self.model_loc+'_'+str(model_no)+'.pkl', "wb") as f:
                pickle.dump(self._model.model, f)

    def load_model(self,model_no):
        ## Check if the regression model for the data is based on neural network 
        ## If it is, it loads all the layer weights based on the network to the self._model class variable
        ## Else it loads the class variable self._model as usual
        if 'nn' in self.model_architecture:
            self._model.model.load_weights(self.model_loc+'_'+str(model_no)+'/').expect_partial()
        else:
            self._model.model=pickle.load(open(self.model_loc+'_'+str(model_no)+'.pkl', "rb"))

            
## Some functions only works for the objects defined in the submodels_module.py:
#assay_to_yield_model
#seq_to_yield_model
#seqandassay_to_yield_model
#final_seq_to_yield_model
#seq_to_pred_yield_model
#seq_to_assay_model
#control_to_assay_model
#control_to_yield_model
#sequence_embeding_to_yield_model
## These objects will be reffered as LIST_A objects. 


    
    def format_modelIO(self,df):
        ## df is a dataframe object
        'based upon model architecture and catagorical variables create the numpy input (x) and output (y) for the model'
        ## This function only works for the objects defined in the submodels_module.py which is specified above as LIST_A objects.
        ## Depending on the object the .get_output_and_explode() function accesses the .explode_yield or the .explode_assay function respectively
        ## Similarly depening on the object the .get_input_seq() function accesses the .get_ordinal() , .get_onehot() , .get_control(), .get_embedding(), .get_assays() or .get_seq_and_assay() function
        ## The function listed above that .get_output_and )explode and .get_input_seq functions acesses are available in the load_format_data.py script
        df_local,cat_var,y=self.get_output_and_explode(df) #set y, do output firest to explode cat variables
        ## Refer to the load_format_data functions of explode_yield and explode_assay to determine the value in df_local, cat_var and y
        x_a=self.get_input_seq(df_local) #set xa (OH seq, Ord seq, assay, control)
        x=load_format_data.mix_with_cat_var(x_a,cat_var) #mix xa with cat variables
        ## The function returns a tuple with x,y and cat_var
        return x,y,cat_var
    
    def make_cv_dataset(self):
        'create list of subtraining/validation by repeated cv of training data'
        ## This function only works for the objects defined in the submodels_module.py which is specified above as LIST_A objects. 
        ## Depending on the object the self.training_df dataframe class object the default training data is either the seq_to_assay_train_1,8,10.pkl file or the assay_to_dot_training_data.pkl file
        ## Similarly the self.num_cv_splits and the self.num_cv_repeats integer class objects are either 10 or 3 depending on the LIST_A object
        ## local_df using the sub_sample() function from load_format_data.py returns a dataframe of randomly selected data a fraction of the traininf_df data entered
        local_df=load_format_data.sub_sample(self.training_df,self.sample_fraction,self.sample_seed)
        ## RepeatedKFold splits data into test and train for cross validation. As mentioned above depending on the object calling this function.
        ## the number of flods are either 100 or 30 or 9. 
        kf = RepeatedKFold(n_splits=self.num_cv_splits,n_repeats=self.num_cv_repeats)
        train,validate=[],[]
        ## an array of the same length as the above local_df filled with zeros is created and then indices are generated to split into train and test data
        for train_index, test_index in kf.split(np.zeros(len(local_df))):
            train.append(local_df.iloc[train_index])
            validate.append(local_df.iloc[test_index])
            ## The data corresponding to the train and test indicies generated in local_df are placed in the array.
            ## The train and validate lists will both be the same length and each contain arrays with dataframe objects in them.
            ## The arrays that have the same indice in the train and validate lists are complementary training and testing data. 
        ## This creates a new class variable self.data_pairs which is a zip object which is tuple iterator, it takes two iteratable objects
        ## and pairs object in the same indice in one tuple.
        ## self.data_pairs is a tuple containing mutiple tuples each of length 2. The 2 elements of the inner tuples are the test and train complementary data
        self.data_pairs=zip(train,validate)

    def make_test_dataset(self):
        'create list of full training set/test set for repeated model performance evaluation'
        ## This function only works for the objects defined in the submodels_module.py which is specified above as LIST_A objects. 
        ## Depending on the object the self.training_df dataframe class object the default training data is either the seq_to_assay_train_1,8,10.pkl file or the assay_to_dot_training_data.pkl file
        ## Similarly the self.testing_df dataframe class object is an attribute for LIST_A objects. The defaults dataframe for self.testing_df is the seq_to_dot_test.pkl and assay_to_data_training.pkl file
        ## local_df using the sub_sample() function from load_format_data.py returns a dataframe of randomly selected data a fraction of the training_df data entered
        local_df=load_format_data.sub_sample(self.training_df,self.sample_fraction,self.sample_seed)
        train,test=[],[]
        ## Depending on object the self.num_test_repeats is an integer attribute defaulted to either 10 or 1.
        for i in range(self.num_test_repeats):
            train.append(local_df)
            test.append(self.testing_df)
            ## The training and the testing data are appended to the train and test list respectively
        ## self.data_pairs is a tuple containing mutiple tuples each of length 2. The 2 elements of the inner tuples are the test and train complementary data
        self.data_pairs=zip(train,test)

            
    def evaluate_model_common(self,space,save_model = False):
        ## The space input is a dictionary with the details regarding the parameter space 
        ## save_model is an input affirming that the model should be saved. 
        true_pred_pairs=[]
        model_no=0
        ## The data_pairs a zip attribute created in the make_test_dataset or the make_cv_dataset function is accessed
        for i in self.data_pairs:
            ## Some of the LIST_A object have the weightbycounts class boolean, if this exsits and is true 
            ## then the local_train_df variable is set to the output of the weighbycountfxn function with
            ## the input as a the first element of each of elements in the data_pairs. Or else it is set as the 
            ## the first element of each of elements in the data_pairs.
            if self.weightbycounts==True:
                local_train_df=self.weightbycountsfxn(i[0])
            else:
                local_train_df=i[0]
            train_x,train_y,train_cat_var=self.format_modelIO(local_train_df)
            ## creates input x and output y  for the training model, this uses the previously defined format_modelIO function
            test_x,test_y,test_cat_var=self.format_modelIO(i[1])
            ## creates input x and output y  for the testing model, this uses the previously defined format_modelIO function
            self._model.set_model(space,xa_len=len(train_x[0])-len(train_cat_var[0]), cat_var_len=len(train_cat_var[0]), lin_or_sig=self.lin_or_sig)
            ## Depending on the model architecture the specified regression is run, this function is defined in the model_architecture.py script
            self._model.fit(train_x,train_y)
            ## Takes the training data and according to the above run regression fits the data to gain a model
            if save_model:
                self.save_model(model_no)
            ## Runs the save_model() function previously defined if needed 
            test_prediction=self._model.model.predict(test_x).squeeze().tolist()
            ## test_predictions is a list containing the predicted values using a linear model on the test data
            true_pred_pairs.append([test_y,test_prediction,test_cat_var])
            ## Then the test predicted values, along with the test_y and test_cat_var values are added to the same index as tuple containing the test and train data used to construct the model
            model_no=model_no+1
            ## The number of models constructed is same as the length of the data_pairs attribute.
        ## Finally the test_y data along with the predicted test data are returned for each training testing data pair in the self.data_pairs attribute
        return true_pred_pairs

    def evaluate_model_cv(self,space,force_saveplots=False):
        'train the repeated kfold dataset. Caclulate average across splits of one dataset, then average across repeats of dataset'
        true_pred_pairs=self.evaluate_model_common(space,False)
        ## Initally the evalute_model_common is run and the value is assigned to the true_pred_pairs variable. 
        cv_mse=[] #average mse values for each repeat of the spliting
        for i in range(0,len(true_pred_pairs),self.num_cv_splits):
            split_mse=[] #mse values for each split of the data
            for j in range(i,i+self.num_cv_splits):
                if len(true_pred_pairs[j][0])>1:
                    split_mse.append(mse(true_pred_pairs[j][0],true_pred_pairs[j][1]))
                else:
                    split_mse.append(np.square(true_pred_pairs[j][0][0]-true_pred_pairs[j][1]))
                ## The test_y and test_prediction value from the evaluate_model_common function are passed into a mean squared error
                ## function which returns a float representing a mean squared error regression loss 
            cv_mse.append(np.average(split_mse))
            ## the split_mse list is appended to the empty cv_mse list
        cur_cv_mse=np.average(cv_mse)
        ## computes the average of the mean squared error regression loss. 
        if force_saveplots or (cur_cv_mse < self.model_stats['cv_avg_loss']):
            self.model_stats['cv_avg_loss']=cur_cv_mse
            self.model_stats['cv_std_loss']=np.std(cv_mse)
            ## This saves the model statistics in the model_dsats variable.
            self.plotpairs_cv=[[],[],[]] #if best CV loss, save the predictions for the first repeat across the splits 
            for i in range(0,self.num_cv_splits):
                self.plotpairs_cv[0]=self.plotpairs_cv[0]+true_pred_pairs[i][0]
                ## Adds each of the test_y values in the true_pred_pairs and stores it in first indices of the plotpairs_cv
                if len(true_pred_pairs[i][1])>1:
                    self.plotpairs_cv[1]=self.plotpairs_cv[1]+true_pred_pairs[i][1]
                else:
                    print('uhoh')
                    self.plotpairs_cv[1]=self.plotpairs_cv[1]+[true_pred_pairs[i][1]]
                ## Adds each of the test_prediction values in the true_pred_pairs and stores it in second indices of the plotpairs_cv
                self.plotpairs_cv[2]=self.plotpairs_cv[2]+true_pred_pairs[i][2]
                ## Adds each of the test_cat_var values in the true_pred_pairs and stores it in third indices of the plotpairs_cv
        ## Returns the average mean squared error regression loss 
        return cur_cv_mse
    
    def evaluate_model_test(self,space):
        'train the reapeated training data. Calculate average loss on test set to average out model randomness'
        true_pred_pairs=self.evaluate_model_common(space,True)
        ## Initally the evalute_model_common is run and the value is assigned to the true_pred_pairs variable.
        mse_list=[]
        for i in true_pred_pairs:
            mse_list.append(mse(i[0],i[1]))
            ## This calculates the mean squared error regression loss for each test and prediction pair and adds it to the mse_list
        cur_test_mse=np.average(mse_list)
        ## this calculates the average regression loss
        self.model_stats['test_avg_loss']=cur_test_mse
        self.model_stats['test_std_loss']=np.std(mse_list)
        self.plotpairs_test=[[],[],[]]
        self.plotpairs_test[0]=self.plotpairs_test[0]+true_pred_pairs[0][0]
        self.plotpairs_test[1]=self.plotpairs_test[1]+true_pred_pairs[0][1]
        self.plotpairs_test[2]=self.plotpairs_test[2]+true_pred_pairs[0][2]
        ## The above commands update the self.model_stats and self.plotpairs_test class variables similar to the evaluvate_model_cv() function
        ## This function returns the average mean squared error regression loss
        return cur_test_mse

    def print_tpe_trials(self):
        print(pd.DataFrame(list(self.tpe_trials.results)))

    def get_best_trial(self):
        'sort trials by loss, return best trial'
        if len(self.tpe_trials)>0:
            ## The length of the hyperopt trails are verified to be greater than 0
            if len(self.tpe_trials)<self.num_hyp_trials:
                ## self.num_hyp_trials is a class variable exsisting in the LIST_A objects. This number is default set to 50.
                ## If the hyperopt trials length is lesser than the default number of hyperopt trials a warning is displayed.
                print('Warning: Not fully tested hyperparameters: ' + str(len(self.tpe_trials)) + '<' + str(self.num_hyp_trials)+':'+self.model_name)
            sorted_trials = sorted(self.tpe_trials.results, key=lambda x: x['loss'], reverse=False)
            ## the hyperopt trial data is sorted in ascending order with respect to the mse regression loss 
            ## The lesser the regression loss the better the trials, therefore the first trial in the sorted_trials which has the least amount of loss
            ## thereby being the best trial.
            return sorted_trials[0]
        ## if the tpe_trials is zero that means no data was found during for this specific model.
        print('no trials found')


    def set_model_state(self,cv):
        'create list of paired dataframes and determine how to calculate loss based upon cross-validaiton or applying to test set'
        if cv:
            self.evaluate_model=self.evaluate_model_cv
            self.make_cv_dataset()
            ## If a cross validation test is to be done a class variable self.evaluate_model is created linking it to the
            ## previously defined evaluate_model_cv(). Then make_cv_dataset() function is run and the self.data_pairs test and train datasets sets are created
        else:
            self.evaluate_model=self.evaluate_model_test
            self.make_test_dataset()
            ## If a cross validation is not to be run, then the self.evaluate_model is created linking it to the evaluate_model_test() function
            ## and the data_pairs class variable is created which is a complementary test and train data.

    def hyperopt_obj(self,space):
        'for a given hyperparameter set, build model arch, evaluate model, return validation loss'
        self.set_model_state(cv=True)
        loss=self.evaluate_model(space)
        ## For a given hyperparameter space, the training and testing data are created then depending on the model archietecture
        ## a regression is run on the training and is used to predict the testing data behaviour. Then the regression loss is calculated
        ## for the prediction made and the regression loss is averaged over various regression models created. Following this the regression loss
        ## along with the hyperparameter spcae are returned in a dictionary. 
        return {'loss': loss, 'status': STATUS_OK ,'hyperparam':space}

    def cross_validate_model(self):
        'use hpyeropt to determine hyperparameters for self.tpe_trials'
        if len(self.tpe_trials)<self.num_hyp_trials:
            ## Happens if the hyperparameters arent fully tested. 
            if 'nn' in self.model_architecture:
                ## If the regression model architecture is a form of neural network model then
                for i in range(10):
                    max_evals=min(len(self.tpe_trials)+5,self.num_hyp_trials)
                    tpe_best=fmin(fn=self.hyperopt_obj,space=self._model.parameter_space,algo=tpe.suggest,trials=self.tpe_trials,max_evals=max_evals)
                    ## The hyperopt_obj function is mininmized with a downhill simplex algorithm and all the plots, model_stats and hyperopt trials are saved in theire respective datasets. 
                    self.save_hyp()
                    self.save_model_stats()
                    self.save_plotpairs()
            else:
                tpe_best=fmin(fn=self.hyperopt_obj,space=self._model.parameter_space,algo=tpe.suggest,trials=self.tpe_trials,max_evals=self.num_hyp_trials)
                ## The hyperopt_obj function is mininmized with a downhill simplex algorithm and all the plots, model_stats and hyperopt trials are saved in theire respective datasets.
                self.save_hyp()
                self.save_model_stats()
                self.save_plotpairs()
        else:
            print('Already done with cross-validation')
            self.set_model_state(cv=True)
            self.evaluate_model(self.get_best_trial()['hyperparam'],force_saveplots=True)
            ## First the set_model_state() function is run, which in turn runs the make_cv_dataset()
            ## function and then the average squared mean regression loss is calculated for the given  hyperparameter space.

    def test_model(self):
        'using the best hyperparameters, train using full training dataset and predict test set'
        self.set_model_state(cv=False)
        loss=self.evaluate_model(self.get_best_trial()['hyperparam'])
        ## For a given hyperparameter space, the training and testing data are created then depending on the model archietecture
        ## a regression is run on the training and is used to predict the testing data behaviour. Then the regression loss is calculated
        ## for the prediction made and the regression loss is averaged over various regression models created.
        self.save_model_stats()
        self.save_plotpairs()
        ## The model_stats and plot data is saved and the average regression loss is printed.
        print('test loss=',str(loss))

    def plot(self):
        figure=self.plot_type(self)
        figure.fig.savefig(self.figure_file)
        figure.fig.clf()
        ## The figure for the data is saved. 