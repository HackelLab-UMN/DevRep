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
    compare test performances when reducing training sample size. This version is for first paper, predicting yield from assays and one-hot encoded sequence. 
    '''
    ## This program should be run on the terminal with an integer input. The integer input should be within the following range: [0,11]
    ## This number is then saved in integer a. 
    a=int(sys.argv[1])
    if a<4:
        b=0
        ## If a is less than 4, then the integer b is set 0.
    elif a<8:
        a=a-4
        b=1
        ## If a is in the range [4,7], then integer b is set to 1 and a is reduced by 4. 
    elif a<12:
        a=a-8
        b=2
        ## If a is in the range [8,11], then integer b is set to 2 and a is reduced by 8. 
    else:
        print('incorrect toggle number')
        ## If the integer input is outside the valid range then an error message is printed. 

    arch_list=['ridge','svm','forest','fnn']
    stringency_list= ['high','medium','low']
    ## Two string lists are created one contains the different types of regression models,arch_list, the other one
    ## contains various levels, stringency_list. 
    mdl=modelbank.twogate_assay_to_yield_model([1,8,10],stringency_list[b],arch_list[a],1)
    ## Then a twogate_assay_to_yield_model object defined in submodels_module is created and it is instantiated with
    ## an integer list to show the assays used to construct the model, the 'b' index of the stringency_list, the 'a' index
    ## of the arch_list for the regression model and thn float 1 for sample fraction. 
    mdl.cross_validate_model()
    mdl.limit_test_set([1,8,10])
    mdl.test_model()
    ## Initally the cross_validate_model() function of the parent model class is run
    ## This determines the hyperparameters for the regression model. Then the limit_test_set()
    ## function of the parent class x_to_yield_model is run to modify the testing_df class dataframe
    ## to reflect only the 1,8,10 assays scores. Finally the hyperparameters are used along with the 
    ## training dataset to train the regression model.
    mdl=modelbank.seqandtwogateassay_to_yield_model([1,8,10],stringency_list[b],arch_list[a],1)
    ## Then another submodel_module object called seqandtwogatesassay_to_yield_model is created and it is instantiated with an 
    ## integer list to show the assays used to construct the model, the b index of the stringency_list and the 'a' index of the
    ## arch_list to show regression model  and a float,1, to show the sample fraction. 
    mdl.cross_validate_model()
    mdl.limit_test_set([1,8,10])
    mdl.test_model()
    ## Then like above the cross_validate_model(), limit_test_set() and test_model() function are run. 

# if __name__ == '__main__':
#     main()

arch_list=['ridge','svm','forest','fnn']
stringency_list= ['high','medium','low','4-gate']
## Two string lists are created one contains the different types of regression models,arch_list, the other one
## contains various levels, stringency_list. 
loss_per_mdl,std_per_mdl=[],[]
## Two empty lists loss_per_mdl and std_per_mdl are created to track the regression loss and the 
## standard deviation of the loss for each kind of model respectively
for i in range (2):
    ## An iterable object i is created to run through the int range[0,1].
    loss_per_str,std_per_str=[],[]
    ## Another two empty lists are created loss_per_str and std_per_str to track the regression loss and its standard deviation for each
    ## different kind of str levels. 
    for stringency in stringency_list:
        ## Each object in the stringency_list is accessed via an iterbale object stringency.
        cv_loss,test_loss,test_std=np.inf,np.inf,0
        ## Three objects are created cv_loss, test_loss and test_std and are set to infinity, infinity and 0 respectively. This is to represent
        ## a worst case for regression loss in terms of cross-validated and test rgression loss and this with a test standard deviation of 0.
        for arch in arch_list:
            ## Each object in the arch_list is accessed via an iterbale object arch. 
            if i==0:
                ## If the i iterbale is equal to 0, then the following if-else block is executed. 
                if stringency=='4-gate':
                    mdl=modelbank.assay_to_yield_model([1,8,10],arch,1)
                    ## If the stringency iterbale is equal to 4-gate then a assay_to_yield_model object defined in the submodels_module.py
                    ## is created. It is instanittaed with an integer list to show the different assays used to build the model, the arch iterable
                    ## to show the regression model used and a float used to indicate the sample fraction.
                else:
                    mdl=modelbank.twogate_assay_to_yield_model([1,8,10],stringency,arch,1)
                    ## If the stringency iterbale is any other element other than 4-gate, then a twogate_assay_to_yield_model object
                    ##  defined in the submodels_module.py is created. It is instanittaed with an integer list to show the different assays used to build the model, 
                    ## the arch iterable to show the regression model used, the stringency iterable and a float used to indicate the sample fraction.
            else:
                ## If the i iterbale is equal to 1, then the following if-else block is executed.
                if stringency=='4-gate':
                    mdl=modelbank.seqandassay_to_yield_model([1,8,10],arch,1)
                    ## If the stringency iterbale is equal to 4-gate then a seqandassay_to_yield_model object defined in the submodels_module.py
                    ## is created. It is instanittaed with an integer list to show the different assays used to build the model, the arch iterable
                    ## to show the regression model used and a float used to indicate the sample fraction.
                else:
                    mdl=modelbank.seqandtwogateassay_to_yield_model([1,8,10],stringency,arch,1)
                    ## If the stringency iterbale is any other element other than 4-gate, then a seqandtwogate_assay_to_yield_model object
                    ## defined in the submodels_module.py is created. It is instanittaed with an integer list to show the different assays used to build the model, 
                    ## the arch iterable to show the regression model used, the stringency iterable and a float used to indicate the sample fraction.
            ## Once the above if-else block is executed the code enters the following if-block.
            if mdl.model_stats['cv_avg_loss'] < cv_loss:
                ## If the cv_avg_loss values in the model_stats class dataframe is lesser than infinity then the following code is executed.
                cv_loss=mdl.model_stats['cv_avg_loss']
                test_loss=mdl.model_stats['test_avg_loss']
                test_std=mdl.model_stats['test_std_loss']
                ## The cv_avg_loss, test_avg_loss and test_std_loss values in the model_stats class dataframe is assigned to the cv_loss, 
                ## test_loss and test_std objects. 
        loss_per_str.append(test_loss)
        std_per_str.append(test_std)
        ## The test_loss and test_std values reassigned above is then added to the loss_per_str and std_per_str lists respectively.
    loss_per_mdl.append(loss_per_str)
    std_per_mdl.append(std_per_str)
    ## Then the loss_per_str and the std_per_str lists modified above are then appended to the loss_per_mdl and the std_per_mdl lists
    ## created outside of the for-loop. 
seq_model=modelbank.seq_to_yield_model('forest',1)
## A seq_to_yield model defined in the submodels_module.py is created and it is instantiated with a 'forest' string and a float 1. 
## the string indicates the model regression to be used, Randomforest regression and the float is the sample fraction. 
seq_loss=seq_model.model_stats['test_avg_loss']
seq_std=seq_model.model_stats['test_std_loss']
## The average test regression loss and the standard deviation of the regression loss of the seq_model is accessed and stored in the 
## seq_loss and seq_std variables respectively. 
x=[-0.3,2.3]
## A list with x-values to be sued in the plot is created
seq_plus=[seq_loss+seq_std]*2
seq_min=[seq_loss-seq_std]*2
## The high and low of the regression loss is calculated by adding and subtraction the std to and from the loss, these values are then added
## twice to each list in seq_plus and seq_min respectively. 

control_model=modelbank.control_to_yield_model('ridge',1)
## A control_to_yield_model object defined in the submodel_module.py is created and it is instantiated with a 'ridge' string and float,1,.
## The model is build with a ridge regression and a smaple fraction of 1. It is stored in a variable named control_model
control_loss=control_model.model_stats['test_avg_loss']
## The average test loss of the control model is accessed and this is stored in the control_loss variable.
control_model.limit_test_set([1,8,10])
## Then the limit_test_set() function from the x_to_yield_model parent class is run which modifies the testing_df dataframe to only reflect
## data from assays 1,8 and 10. 
exploded_df,_,_=load_format_data.explode_yield(control_model.testing_df)
## The updated testing_df of the control_model object is run through the explode_yield function defined in the load_format_data.py file
## The output dataframe is stored in the exploded_df dataframe
exp_var=np.average(np.square(np.array(exploded_df['y_std'])))
## Then the mean square of the y_std column in the exploded_df dataframe is calculated and stored under the variable exp_var.

fig,ax=plt.subplots(1,1,figsize=[2,2],dpi=300)
## A Figure and An Axis objbect is created, the plot created has only 1 subplot in it.
xloc=[1.2,0.6,0,2]
## xloc list contains the x positions for the bars in the bar graph that is to be constructed. 
ax.axhline(seq_loss,-0.5,4.5,color='green',linestyle='--',label='Sequence Model')
ax.axhline(control_loss,-0.5,2.5,color='red',linestyle='--',label='Control')
ax.axhline(exp_var,-0.5,2.5,color='purple',linestyle='--',label='Experimental')
## Three horizontal lines are constructed at the seq_loss, control_loss and exp_var positions, these are green,red and purple in color
## and they are labelled 'Sequence Model', 'Control' and 'Experimental' respectively. 
ax.bar(np.subtract(xloc,0.075),loss_per_mdl[0],yerr=std_per_mdl[0],label='Assay',width=0.15,color='blue')
ax.bar(np.add(xloc,0.075),loss_per_mdl[1],yerr=std_per_mdl[1],label='Seq and Assay',width=0.15,color='orange')
## six bar ares constructed one set is at [1.125,0.525,0.125] and their bar heights and error is determined by the first element of the loss_per_mdl
## and the std_per_mdl list respectively, these bars are blue in color and labelled 'Assay'. The other set of bars are at the following positions
## [1.275,0.675,0.275] and the bar height and eror is determined by the second element of the loss_per_mdl and std_per_mdl lists respectively
## These bares are orange in color and labelled 'Seq and Assay'
ax.fill_between(x,seq_plus,seq_min,alpha=0.2,color='green')
## The x-list defined above is plotted against the seq_plus and seq_min lists giving two parrallel lines, and the area in between is colored green.
ax.set_xticks(xloc)
## The ticks in the x-axis are set to those defined in the xloc list, thereby making it inbetween two adjacent bars. 
ticklabels=['High','Medium','Low','4-Gate']
ax.set_xlabel('Sort Stringency',fontsize=6)
ax.set_xticklabels(ticklabels)
## Then the x-axis is labelled 'Sort Stringency' and the ticks along the x-axis are labelled 'High','Medium','Low','4-Gate' from left-to right
# ax.legend(fontsize=6)
ax.tick_params(axis='both', which='major', labelsize=6)
ax.set_ylabel('$Test^2$ Loss',fontsize=6)
ax.set_ylim([0.35,0.8])
ax.set_xlim([-0.3,2.3])
## The fontsize of ticks on both the x-axis and the y-axis is changed. The y-axis is labelled '$Test^2$ Loss'. Then the x and y axis range in 
## the plot is set to [-0.3,2.3] and [0.35,0.8] respectively. 
fig.tight_layout()
fig.savefig('./two_gate.png')
plt.close()
## Then the plot is squeezed to fit within the given Figure dimensions. The figure is then saved as a png file under the name 'two_gate.png'
## Then the pyplot module is closed.


