import sys
import submodels_module as modelbank
import numpy as np 
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
from scipy.stats import ttest_ind_from_stats as ttest
import load_format_data

def main():
    '''
    compare test performances when weighting the training dataset by the average log2 number of observations
    '''
    ## This program is run on the terminal with an integer input in the range of [0,8), and this integer
    ## input is assigned to a.
    a=int(sys.argv[1])
    arch_list=['ridge','svm','forest','fnn']
    ## A string list called arch_list is created which has strings corresponding to different regression models.
    if a<4:
        b=0
        mdl=modelbank.seqandweightedassay_to_yield_model([1,8,10],arch_list[a],1)
        ## If a is less than 4, then an integr b is created and set to 0. Then a seqandweightedassay_to_yield_model
        ## model defined in the submodels_module.py program is created. This is instantiated with an integer list
        ## corresponding to the assays to be used to build the model, a string from the 'a' position of 
        ## the arch_list and finally a float,1, to indicate the sample fraction. 
    elif a<8:
        a=a-4
        b=1
        mdl=modelbank.weighted_assay_to_yield_model([1,8,10],arch_list[a],1)
        ## If a is in the range [4,8)], then an integer b is created and set to 1 and a is reduced by 4, 
        ## Then a weighted_assay_to_yield_model object defined in the submodels_module.py program is created.
        ## This is instantiated with an integer list which respresent the assays to be used to build the regression model,
        ## a string from the 'a' position of the arch_list and finally a float,1, to indicate the sample fraction.
    else:
        print('incorrect toggle number')
        ## If an integer input is given outside the range, then an error message is printed. 

    mdl.cross_validate_model()
    mdl.limit_test_set([1,8,10])
    mdl.test_model()
    ## Initally the cross_validate_model() function of the parent model class is run
    ## This determines the hyperparameters for the regression model. Then the limit_test_set()
    ## function of the parent class x_to_yield_model is run to modify the testing_df class dataframe
    ## to reflect only the 1,8,10 assays scores. Finally the hyperparameters are used along with the 
    ## training dataset to train the regression model.


# if __name__ == '__main__':
#     main()

loss_per_model,std_per_model=[],[]
arch_list=['ridge','svm','forest','fnn']
## A string list called arch_list is created which has strings corresponding to different regression models.
## Two empty lists are created to store the regression loss and the standard deviation of the loss for each model. 
## These are loss_per_model and std_per_model respectively.
for i in range (4):
    ## An iterable object is created to iterate through the integrs of [0,4)
    cv_loss,test_loss,test_std=np.inf,np.inf,0
    ## Three new objects are created cv_loss and test_loss are set to infinity while test_std is set to 0. 
    for arch in arch_list:
        ## Then for each element in the arch_list is accessed via the arch iterable, the following if-elif-else block is executed. 
        if i==0:
            mdl=modelbank.assay_to_yield_model([1,8,10],arch,1)
            ## If the iterable i is equal to 1, then a assay_to_yield_model object defined in the submodels_modules.py 
            ## program is created and it is instantiated with an integer list which represents the assays to be used to build
            ## regression model, the arch iterable and a float,1, representing the sample fraction.
        elif i==1:
            mdl=modelbank.weighted_assay_to_yield_model([1,8,10],arch,1)
            ## If the iterable i is equal to 2, then a weighted_assay_to_yield_model object defined in the submodels_modules.py 
            ## program is created and it is instantiated with an integer list which represents the assays to be used to build
            ## regression model, the arch iterable and a float,1, representing the sample fraction
        elif i==2:
            mdl=modelbank.seqandassay_to_yield_model([1,8,10],arch,1)
            ## If the iterable i is equal to 3, then a seqandassay_to_yield_model object defined in the submodels_modules.py 
            ## program is created and it is instantiated with an integer list which represents the assays to be used to build
            ## regression model, the arch iterable and a float,1, representing the sample fraction
        else:
            mdl=modelbank.seqandweightedassay_to_yield_model([1,8,10],arch,1)
            ## If the iterable i is equal to 3, then a seqandweightedassay_to_yield_model object defined in the submodels_modules.py 
            ## program is created and it is instantiated with an integer list which represents the assays to be used to build
            ## regression model, the arch iterable and a float,1, representing the sample fraction.
        if mdl.model_stats['cv_avg_loss'] < cv_loss:
            cv_loss=mdl.model_stats['cv_avg_loss']
            test_loss=mdl.model_stats['test_avg_loss']
            test_std=mdl.model_stats['test_std_loss']
            ## Then the model_stats dataframe of the mdl object is accessed, its cv_avg_loss column value is compared to
            ## the cv_loss values created at the start, if the cv_avg_loss values is lesser than infinity. Then the cv_loss
            ## object is updated to reflect the cv_avg_loss value and similarly the test_loss and test_std object value are also
            ## updated to reflect the test_avg_loss and test_std_loss values in the model_stats dataframe of mdl.  
    loss_per_model.append(test_loss)
    std_per_model.append(test_std)
    ## The updated object test_loss and test_std are added to the loss_per_model and std_per_model lists created outside the for loops
    ## The for loop is used for training different models with different regression models and collecting each the regression loss and the
    ## standard deviation of each regression loss  for the different models that were built.
    
seq_model=modelbank.seq_to_yield_model('forest',1)
## A seq_to_yield_model object which is defined in the submodels_module.py is created with a Randomforest regression model and a sample 
## fraction of 1 and stored under the name seq_model
seq_loss=seq_model.model_stats['test_avg_loss']
seq_std=seq_model.model_stats['test_std_loss']
## The test_avg_loss and the test_std_loss columns in the model_stats dataframe of the seq_model object is accessed and stored
## in seq_loss and seq_std respectively.
x=[-0.3,0.8]
## A float list is created with x-values 
seq_plus=[seq_loss+seq_std]*2
seq_min=[seq_loss-seq_std]*2
## The loss range by adding and subtracting the standard deviation to and from the regression loss, these values are then stored twice
## in the seq_plus and seq_min list respectively. 
control_model=modelbank.control_to_yield_model('ridge',1)
## A control_to_yield_model object defined in submodels_module is created with a reidge regression model and a sample fraction of 1. 
control_loss=control_model.model_stats['test_avg_loss']
## The test_avg_loss columns in the model_stats column of the control_model is accessed and stored in the control_loss variable
control_model.limit_test_set([1,8,10])
## Following this the limit_test_set() function from the x_to_yield_model parent classis run which modifies the testing_df dataframe 
## to only reflect data from the 1,8 and 10 assays
exploded_df,_,_=load_format_data.explode_yield(control_model.testing_df)
## The explode_yield function from the load_format_data.py program and it is run on the testing_df dataframe of the control_model object
## The output is the exploded_yield 
exp_var=np.average(np.square(np.array(exploded_df['y_std'])))
## The mean square of the y_std column in the exploded_df dataframe is calculated and stored in the exp_var variable. 
fig,ax=plt.subplots(1,1,figsize=[2,2],dpi=300)
## A Figure and an Axis object is created, the plot created has only one subplot in it. 
xloc=[0,0.5]
## Another float list is created with x-values
ax.axhline(seq_loss,-0.5,4.5,color='green',linestyle='--',label='Sequence Model')
## A horizontal line is constructed at the seq_loss position and this line is labelled Sequence Model
ax.axhline(control_loss,-0.5,2.5,color='red',linestyle='--',label='Control')
## Another horizontal line is constructed at the control_loss position and labelled Control
ax.axhline(exp_var,-0.5,2.5,color='purple',linestyle='--',label='Experimental')
## Another horizonal line is constructed at the exp_var position and labelled Experimental.
ax.bar(np.subtract(xloc[0],0.1),loss_per_model[0],yerr=std_per_model[0],label='Non-Weighted',width=0.2,color='blue')
## Then a bar plot is constructed with the bar positioned at -0.1 and with the height of the bar corresponding to the first 
##  element of the loss_per_model list, the error bars of the height is determined by the first element of the std_per_model list.
## This bar is blue in color and is labelled 'Non_Weighted' 
ax.bar(np.add(xloc[0],0.1),loss_per_model[1],yerr=std_per_model[1],label='Weighted',width=0.2,color='blue',alpha=0.3)
## Another bar is positioned at 0.1 and with a height dictated by the second element of the loss_per_model list and the
## bar height error determined by the second element of the std_per_model. This bar is blue in color and labelled 'Weighted'
ax.bar(np.subtract(xloc[1],0.1),loss_per_model[2],yerr=std_per_model[2],width=0.2,color='orange')
## Another bar is positioned at 0.4 and its height is dictated by the third element of the loss_per_model list and the bar height 
## error is dictated by the third element of the std_per_model list. This bar is orange in color.
ax.bar(np.add(xloc[1],0.1),loss_per_model[3],yerr=std_per_model[3],width=0.2,color='orange',alpha=0.3)
## Another bar is positioned at 0.6 and its height is dictated by the fourth element of the loss_per_model list and the bar height 
## error is dictated by the fourth element of the std_per_model list. This bar is orange in color.
ax.fill_between(x,seq_plus,seq_min,alpha=0.2,color='green')
## The x list is plotted against the seq_min and seq_plus and the two resulting parrellel lines are filled green in color. 
ax.set_xticks([xloc[0]-0.1,xloc[0]+0.1,xloc[1]-0.1,xloc[1]+0.1])
ticklabels=['None','$Log_2$','None','$Log_2$']
ax.set_xticklabels(ticklabels)
## Then the x-axis ticks are set in the following positions [-0.1,0.1,0.4,0.6] and then each position is labelled 'None'
## '$Log_2$','None' and '$Log_2$' respectively.  
# ax.legend(fontsize=6)
ax.set_xlabel('Training Sample Weighting',fontsize=6)
ax.tick_params(axis='both', which='major', labelsize=6)
ax.set_ylabel('$Test^2$ Loss',fontsize=6)
ax.set_ylim([0.35,0.8])
ax.set_xlim([-0.3,0.8])
## Then the x and y axis are labelled 'Training Sample Weighting' and '$Test^2$ Loss' respectively. Then the ticks on both axis are
## set to a certain font size. Following this the y-axis and x-axis ranges are set to [0.35,0.8] and [-0.3,0.8] respectively.
fig.tight_layout()
fig.savefig('./Weighting_by_obs.png')
plt.close()
## Finally the plot is squeezed to fit within the figure size. Following this the figure is saved as a png file under the name 
## 'Weighting_by_ops'. Then the pyplot module is closed. 




