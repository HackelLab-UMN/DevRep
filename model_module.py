import numpy as np
import pandas as pd
import pickle
from hyperopt import Trials,fmin,tpe,STATUS_OK
from sklearn.model_selection import RepeatedKFold
from sklearn.metrics import mean_squared_error as mse
import load_format_data
from model_architectures import get_model
from plot_model import model_plot

class model:
    '''The model class will cross-validate the training set to determine hyperparameters
    then use the set hyperparameters to evaluate against a test set and save predictions''' 

    def __init__(self, model_in, model_out, model_architecture, sample_fraction):
        self.sample_fraction=sample_fraction
        self.model_architecture=model_architecture
        self.model_name=model_in+'_'+model_out+'_'+model_architecture+'_'+str(sample_fraction)
        self.trials_file='./trials/'+self.model_name+'.pkl'
        self.stats_file='./model_stats/'+self.model_name+'.pkl'
        self.plotpairs_file='./plotpairs/'+self.model_name+'.pkl'
        self.figure_file='./figures/'+self.model_name+'.png'
        self.model_loc='./models/'+self.model_name


        self.build_architecture(model_architecture) #ridge,forest,svm,nn(emb,)
        self.load_hyp()
        self.load_model_stats()
        self.plotpairs_cv=[[],[],[]] #1st repeat CV (concat all splits) predictions [true y, pred y, cat var]
        self.plotpairs_test=[[],[],[]] #1st repeat test predictions 
        self.load_plotpairs()
        self.sample_seed=42 #default for unchanged seed value
        self.weightbycounts=False

        self.plot_type=None

    def parent_warning(self):
        print('im in parent class')

    def update_model_name(self,model_name):
        self.model_name=model_name
        self.trials_file='./trials/'+self.model_name+'.pkl'
        self.stats_file='./model_stats/'+self.model_name+'.pkl'
        self.plotpairs_file='./plotpairs/'+self.model_name+'.pkl'
        self.figure_file='./figures/'+self.model_name+'.png'
        self.model_loc='./models/'+self.model_name
        self.load_hyp()
        self.load_model_stats()
        self.load_plotpairs()

    def save_plotpairs(self):
        with open (self.plotpairs_file,'wb') as f:
            pickle.dump([self.plotpairs_cv,self.plotpairs_test],f)

    def load_plotpairs(self):
        try:
            [self.plotpairs_cv,self.plotpairs_test]=pickle.load(open(self.plotpairs_file,'rb'))
        except:
            print('No plot data available')
            [self.model_plotpairs_cv,self.model_plotpairs_test]=[[[],[],[]],[[],[],[]]]

    def save_model_stats(self):
        with open (self.stats_file,'wb') as f:
            pickle.dump(self.model_stats,f)

    def load_model_stats(self):
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
        'load architecture class which sets hyp space'
        self._model=get_model(model_architecture)

    def evaluate_model_common(self,space,save_model):
        true_pred_pairs=[]
        model_no=0
        for i in self.data_pairs:
            if self.weightbycounts==True:
                local_train_df=self.weightbycountsfxn(i[0])
            else:
                local_train_df=i[0]
            train_x,train_y,train_cat_var=self.format_modelIO(local_train_df)
            test_x,test_y,test_cat_var=self.format_modelIO(i[1])
            self._model.set_model(space,xa_len=len(train_x[0])-len(train_cat_var[0]), cat_var_len=len(train_cat_var[0]), lin_or_sig=self.lin_or_sig)
            self._model.fit(train_x,train_y)
            if save_model:
                self.save_model(model_no)
            test_prediction=self._model.model.predict(test_x).squeeze().tolist()
            true_pred_pairs.append([test_y,test_prediction,test_cat_var])
            model_no=model_no+1
        return true_pred_pairs

    def evaluate_model_cv(self,space,force_saveplots=False):
        'train the repeated kfold dataset. Caclulate average across splits of one dataset, then average across repeats of dataset'
        true_pred_pairs=self.evaluate_model_common(space,False)
        cv_mse=[] #average mse values for each repeat of the spliting
        for i in range(0,len(true_pred_pairs),self.num_cv_splits):
            split_mse=[] #mse values for each split of the data
            for j in range(i,i+self.num_cv_splits):
                if len(true_pred_pairs[j][0])>1:
                    split_mse.append(mse(true_pred_pairs[j][0],true_pred_pairs[j][1]))
                else:
                    split_mse.append(np.square(true_pred_pairs[j][0][0]-true_pred_pairs[j][1]))
            cv_mse.append(np.average(split_mse))
        cur_cv_mse=np.average(cv_mse)
        if force_saveplots or (cur_cv_mse < self.model_stats['cv_avg_loss']):
            self.model_stats['cv_avg_loss']=cur_cv_mse
            self.model_stats['cv_std_loss']=np.std(cv_mse)

            self.plotpairs_cv=[[],[],[]] #if best CV loss, save the predictions for the first repeat across the splits 
            for i in range(0,self.num_cv_splits):
                self.plotpairs_cv[0]=self.plotpairs_cv[0]+true_pred_pairs[i][0]
                if len(true_pred_pairs[i][1])>1:
                    self.plotpairs_cv[1]=self.plotpairs_cv[1]+true_pred_pairs[i][1]
                else:
                    print('uhoh')
                    self.plotpairs_cv[1]=self.plotpairs_cv[1]+[true_pred_pairs[i][1]]
                self.plotpairs_cv[2]=self.plotpairs_cv[2]+true_pred_pairs[i][2]
        return cur_cv_mse

    def evaluate_model_test(self,space):
        'train the reapeated training data. Calculate average loss on test set to average out model randomness'
        true_pred_pairs=self.evaluate_model_common(space,True)
        mse_list=[]
        for i in true_pred_pairs:
            mse_list.append(mse(i[0],i[1]))
        cur_test_mse=np.average(mse_list)
        self.model_stats['test_avg_loss']=cur_test_mse
        self.model_stats['test_std_loss']=np.std(mse_list)
        self.plotpairs_test=[[],[],[]]
        self.plotpairs_test[0]=self.plotpairs_test[0]+true_pred_pairs[0][0]
        self.plotpairs_test[1]=self.plotpairs_test[1]+true_pred_pairs[0][1]
        self.plotpairs_test[2]=self.plotpairs_test[2]+true_pred_pairs[0][2]
        return cur_test_mse

    def print_tpe_trials(self):
        print(pd.DataFrame(list(self.tpe_trials.results)))

    def get_best_trial(self):
        'sort trials by loss, return best trial'
        if len(self.tpe_trials)>0:
            if len(self.tpe_trials)<self.num_hyp_trials:
                print('Warning: Not fully tested hyperparameters: ' + str(len(self.tpe_trials)) + '<' + str(self.num_hyp_trials)+':'+self.model_name)
            sorted_trials = sorted(self.tpe_trials.results, key=lambda x: x['loss'], reverse=False)
            return sorted_trials[0]
        print('no trials found')

    def load_hyp(self):
        'load hyperopt trials'
        try:  # try to load an already saved trials object
            self.tpe_trials = pickle.load(open(self.trials_file, "rb"))
        except:
            self.tpe_trials = Trials()

    def save_hyp(self):
        'save hyperopt trials, refresh best trial'
        with open(self.trials_file, "wb") as f:
            pickle.dump(self.tpe_trials, f)

    def save_model(self,model_no):
        'save the trained model'
        if 'nn' in self.model_architecture:
            self._model.model.save_weights(self.model_loc+'_'+str(model_no)+'/')
        else:
            with open(self.model_loc+'_'+str(model_no)+'.pkl', "wb") as f:
                pickle.dump(self._model.model, f)

    def load_model(self,model_no):
        if 'nn' in self.model_architecture:
            self._model.model.load_weights(self.model_loc+'_'+str(model_no)+'/').expect_partial()
        else:
            self._model.model=pickle.load(open(self.model_loc+'_'+str(model_no)+'.pkl', "rb"))

    def format_modelIO(self,df):
        'based upon model architecture and catagorical variables create the numpy input (x) and output (y) for the model'
        df_local,cat_var,y=self.get_output_and_explode(df) #set y, do output firest to explode cat variables
        x_a=self.get_input_seq(df_local) #set xa (OH seq, Ord seq, assay, control)
        x=load_format_data.mix_with_cat_var(x_a,cat_var) #mix xa with cat variables
        return x,y,cat_var

    def make_cv_dataset(self):
        'create list of subtraining/validation by repeated cv of training data'
        local_df=load_format_data.sub_sample(self.training_df,self.sample_fraction,self.sample_seed)
        kf=RepeatedKFold(n_splits=self.num_cv_splits,n_repeats=self.num_cv_repeats)
        train,validate=[],[]
        for train_index, test_index in kf.split(np.zeros(len(local_df))):
            train.append(local_df.iloc[train_index])
            validate.append(local_df.iloc[test_index])
        self.data_pairs=zip(train,validate)

    def make_test_dataset(self):
        'create list of full training set/test set for repeated model performance evaluation'
        local_df=load_format_data.sub_sample(self.training_df,self.sample_fraction,self.sample_seed)
        train,test=[],[]
        for i in range(self.num_test_repeats):
            train.append(local_df)
            test.append(self.testing_df)
        self.data_pairs=zip(train,test)

    def set_model_state(self,cv):
        'create list of paired dataframes and determine how to calculate loss based upon cross-validaiton or applying to test set'
        if cv:
            self.evaluate_model=self.evaluate_model_cv
            self.make_cv_dataset() 
        else:
            self.evaluate_model=self.evaluate_model_test
            self.make_test_dataset()

    def hyperopt_obj(self,space):
        'for a given hyperparameter set, build model arch, evaluate model, return validation loss'
        self.set_model_state(cv=True)
        loss=self.evaluate_model(space)

        return {'loss': loss, 'status': STATUS_OK ,'hyperparam':space}

    def cross_validate_model(self):
        'use hpyeropt to determine hyperparameters for self.tpe_trials'
        if len(self.tpe_trials)<self.num_hyp_trials:
            if 'nn' in self.model_architecture:
                for i in range(10):
                    max_evals=min(len(self.tpe_trials)+5,self.num_hyp_trials)
                    tpe_best=fmin(fn=self.hyperopt_obj,space=self._model.parameter_space,algo=tpe.suggest,trials=self.tpe_trials,max_evals=max_evals)
                    self.save_hyp()
                    self.save_model_stats()
                    self.save_plotpairs()
            else:
                tpe_best=fmin(fn=self.hyperopt_obj,space=self._model.parameter_space,algo=tpe.suggest,trials=self.tpe_trials,max_evals=self.num_hyp_trials)
                self.save_hyp()
                self.save_model_stats()
                self.save_plotpairs()
        else:
            print('Already done with cross-validation')
            self.set_model_state(cv=True)
            self.evaluate_model(self.get_best_trial()['hyperparam'],force_saveplots=True)

    def test_model(self):
        'using the best hyperparameters, train using full training dataset and predict test set'
        self.set_model_state(cv=False)
        loss=self.evaluate_model(self.get_best_trial()['hyperparam'])
        self.save_model_stats()
        self.save_plotpairs()
        print('test loss=',str(loss))

    def plot(self):
        figure=self.plot_type(self)
        figure.fig.savefig(self.figure_file)
        figure.fig.clf()
