import submodels_module as modelbank
import numpy as np
from itertools import combinations
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import load_format_data 
import pandas as pd


class results:
    
    def __init__(self):
        ## This is used to instantiate a with class variables model_list and compare_test which are none and false respectively. The 
        ## class object needs no input.
        self.model_list=None
        self.compare_test=False

    def get_loss_list(self,model_list):
        ## The input for this function is a list containing various model objects, these may also be objects belonging to the child classes (submodels_module.py)
        ## of the model object (from model_module.py)
        model_loss_list,model_loss_std_list,model_name_list=[],[],[]
        ## Three empty lists are created above
        if self.compare_test:
            ## if the compare_test boolean is true, the testing data's is run through the get_variance and its variance is caluclated
            exp_var=self.get_variance(model_list[0].testing_df)
        else: 
            ## Else, the training data's is run through the get_variance and its varaiance is calculated
            exp_var=self.get_variance(model_list[0].training_df)
        ## Each object in model_list is accessed  
        for model in model_list:
            # model.get_best_trial()
            ## Then the model name for each model in the model_list is saved in the model_name_list
            model_name_list.append(model.model_name)
            ## Then if the caompare_test class variable is true the test_avg_loss and the test_std_loss
            ## is added to the model_loss_list and model_loss_std_list respectively.
            ## Else the cv_avg_loss and the cv_std_loss is added to the model_loss_list
            ## and model_loss_std_list respectively.
            if self.compare_test:
                model_loss_list.append(model.model_stats['test_avg_loss'])
                model_loss_std_list.append(model.model_stats['test_std_loss'])
            else:
                model_loss_list.append(model.model_stats['cv_avg_loss'])
                model_loss_std_list.append(model.model_stats['cv_std_loss'])
        ## Then the newly created lists are returned 
        return model_loss_list,model_loss_std_list,model_name_list,exp_var 


    def save_loss_list(self,model_name_list,model_loss_list,model_loss_std_list):
        ## The input for this function is the values returned from the get_loss_list function
        df=pd.DataFrame(np.transpose(np.array([model_name_list,model_loss_list,model_loss_std_list])))
        ## This is then compiled into a dataframe and saved as a csv file under the name aty_best_arch_cv.csv
        df.to_csv('./aty_best_arch_cv.csv')
    
    def get_assay_control(self):
        ## Initally a control_to_yield object is created from the submodels_module.py program,
        ## with the model architecture of ridge and sample fraction as 1 and saved as control_model
        control_model=modelbank.control_to_yield_model('ridge',1)
        if self.compare_test:
            ## If the compare_test class variable is true then a tuple containing the average loss and standard deviation loss of the testing data
            ## is returned
            return control_model.model_stats['test_avg_loss'],control_model.model_stats['test_std_loss']
        else:
            ## Else the average loss and standard deviation loss of the cross validation is returned in the tuple
            return control_model.model_stats['cv_avg_loss'],control_model.model_stats['cv_std_loss']
    
    def plot_distribution(self,set_name,model_loss_list):
        ## The function input requires the model_loss_list created in the get_loss_list() function and also a set_name string for the name
        ## of the plot image to be saved.
        control_model_loss,_=self.get_control()
        ## The get_control function is the get_assay_control function and this is run and depending on whether the 
        ## compare_test boolen is true the control_model_loss is the average loss for the testing data or the cross validated data.
        seq_model=self.get_best_seq_to_yield_simple()
        ## This checks which architecture model can best predict the seq_to_yield model and creates a seq_to_yield_model object
        ## with the corresponding model architecture and sample fraction of 1 and returns the object.
        seq_model_cv_loss=seq_model.model_stats['cv_avg_loss']
        ## Then the cross validated averagre regression loss is accessed and stored on the seq_model_cv_loss\
        fig,ax = plt.subplots(1,1,figsize=[2,2],dpi=300)
        ## A figure object and an Axis array is created, the figure created only has 1 plot
        ax.hist(model_loss_list,bins=20,color='black')
        ## A histogram is constructed with the frequency of each regression loss mapped out. 
        ax.set_xlabel('Cross Validation Model Loss',fontsize=6)
        ax.set_ylabel('# of Models',fontsize=6)
        ax.tick_params(axis='both', which='major', labelsize=6)
        ## The x-label and y-labels are set to cross validation model loss and # of Models respectively and the fontsize of the labels are set
        ax.axvline(x=control_model_loss,label='Cell Type Control',color='red')
        ax.axvline(x=seq_model_cv_loss,label='Sequence Model',color='blue')
        ## the regression loss from the control stored in control_model_loss and the regression loss of the best simple seq_to_yield
        ## model stored in the seq_model_cv_loss are marked in the histogram in red and blue respectively under the titles cell type control
        ## and sequence model respectively. 
        ax.legend(fontsize=6,bbox_to_anchor=(0.5, 1.2), loc='center')
        plt.tight_layout()
        ## Then a corresponding legend is placed and the figure is squeezed to fit in the given figure area
        ## and the figure is stored under the string name given in the input parameter set_name
        fig.savefig('./'+set_name+'.png')
        
    def get_best_model(self,model_list,save=False,plot=False):
        ## This function requires an input of a list and two boolean values defaulted to false. The list contains different types of models
        ## constructed, the booleans correspond on whether secondary data produced should be saved or not
        model_loss_list,model_loss_std_list,model_name_list,exp_var=self.get_loss_list(model_list)
        ## Initially the get_loss_list() function is run for the given model_list and this returns 4 lists containing the different model names
        ## their corresponding regression loss and the standard deviation for the regression loss
        if save:
            ## If the save boolean is true then the save_loss_list() function is run and the above generated lists are saved as a csv file
            self.save_loss_list(model_name_list,model_loss_list,model_loss_std_list)
        if plot:
            ## if the plot boolean is true then the plot_distribution function is run and a plot of the regression loss is saved.
            self.plot_distribution(plot,model_loss_list)
        best_index=np.argmin(np.array(model_loss_list))
        ## The model with the least regression loss is found and the corresponding model in the input list is then returned as the best_model
        return model_list[best_index]
    
    def get_best_seq_to_yield_simple(self):
        self.compare_test=False
        ## First sets the comapre_test class boolean to false, then creates a new class variable get_control to 
        ## get_control and links it to the function get_assay_control
        self.get_control=self.get_assay_control
        ## b_models is a list containing different simple regression models used to build the seq to yield correlation
        b_models=['ridge','forest','svm','fnn']
        model_list=[]
        for arch in b_models:
            ## for each different type of the regression model a seq_to_yield object specified in submodel_module.py program,
            ## is built with a sample fraction of 1 and this in turn is added to the temporary model_list list. 
            model_list.append(modelbank.seq_to_yield_model(arch,1))

        best_model=self.get_best_model(model_list)
        ## The model_list compiled is run through the get_best_model() function and the output from it is returned in this function.
        return best_model

    def plot_bar(self,model_list,set_name):
        ## This function requires that a list containing the various models be passed in along with a string correspondng to the name that the plot is to be saved under
        model_loss_list,model_loss_std_list,model_name_list,exp_var=self.get_loss_list(model_list)
        ## The get_loss_list() function is run for the input model_list and the output of the model name and 
        ## the corresponding regression loss and the standard deviation of the loss is accessed 
        control_model_loss,control_model_loss_std=self.get_control()
        ## Then the get_assay_control() function is run to get the regression loss and its standard deviation for either the testing data or
        ## the cross validated dataset
        fig,ax = plt.subplots(1,1,figsize=[6,3],dpi=300)
        ## A figure object and an Axis array is created, the figure created only has 1 plot
        ax.bar(list(range(len(model_list)+2)),[exp_var,control_model_loss]+model_loss_list,tick_label=['exp_variance','control']+model_name_list)
        ax.set_ylabel('Model Loss')
        ax.tick_params(axis='both', which='major', labelsize=6)
        ## A bar graph is created with each type of model in the model list along the x-axis and the y-axis tracks the regression loss of each model
        ## The variance and cell type control model loss is also tracked along with the rest of the models in the model list
        plt.tight_layout()
        ## Then a corresponding legend is placed and the figure is squeezed to fit in the given figure area
        ## and the figure is stored under the string name given in the input parameter set_name
        fig.savefig('./'+set_name+'.png')
        
### Talk to Alex about this!!!
    def assay_to_yield_best_arch(self):
        self.compare_test=False
        self.get_control=self.get_assay_control
        ## The class variables compare_test is set to false while the get_control variable is linked to the get_assay_control function
        a=[1,2,3,4,5,6,7,8,9,10]
        combin_list=[]
        ## a function is created to map wach assay
        for i in range(1,11):
            combin_list_temp=combinations(a,i)
            for j in combin_list_temp:
                combin_list.append(j)
        ## Each possible combination of the 10 assays are created, combinations ranging from only 1 assay to combination with all 10.
        b_models=['ridge','forest','svm','fnn']        
        # b_models=b_models[0:3]
        combin_list=combin_list[0:10]
        best_model_per_combin=[]
        for combin in combin_list:
            model_list=[]
            for arch in b_models:
                model_list.append(modelbank.assay_to_yield_model(combin,arch,1))
            best_model_per_combin.append(self.get_best_model(model_list)) 
        ## For each combination of assays, the combination with the particular regression model is stored in the best_model_per_combin list
        best_model=self.get_best_model(best_model_per_combin,save=True,plot='assay_to_yield_best_arch')
        ## Then the best model is selected form the ones compiled in the best_model_per_combin list using the get_best_model() function
        ## then the model is displayed as a print statement. 
        print(best_model.model_name)
        # self.plot_distribution(best_model_per_combin,'assay_to_yield_best_arch')


    def set_model_list(self,mode):
        mode_dict={
        'aty_best_arch':self.assay_to_yield_best_arch,
        }
        ## A dictionary is created linking the string key 'aty_best_arch' to the function assay_to_yield_best-arch
        self.make_model_list=mode_dict[mode]
        self.make_model_list()

    def get_variance(self,df):
        ## A dataframe object is needed for this function. 
        ## This access the explode_yield function from the load_format_data.py file. The explode_yield() function returns three dataframe objects 
        ## the first an exploded_df data frame 
        exploded_df,_,_=load_format_data.explode_yield(df)
        ## The above dataframe is turned into an array where each element is squared and then then the squared standard deviation is averaged. 
        ## This value is returned.
        return np.average(np.square(np.array(exploded_df['y_std'])))





def main():
    a=results()
    a.set_model_list('aty_best_arch')
    # a.set_model_list('aty_simple')


if __name__ == '__main__':
    main()