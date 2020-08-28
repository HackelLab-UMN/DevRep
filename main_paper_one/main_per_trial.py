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
    
    a=int(sys.argv[1])
    ## This program is run of the terminal command prompt and needs a integer input in the range of
    ## [0,15]. Anything above this gives an error message
    if a<4:
        b=0
        ## If a is less than 4, then the b values is set to 0
    elif a<8:
        a=a-4
        b=1
        ## If the a value is between 4 and 8, then the a value is reduced by 4 and the b value is set
        ## to 1
    elif a<12:
        a=a-8
        b=2
        ## If the a value is between 8 and 12, then a is reduced by 8 and b is set to 2
    elif a<16:
        a=a-12
        b=3
        ## If the a value is between 12 and 16,then a is reduced by 12 and b is set to 3
    else:
        print('incorrect toggle number')
        ## This is the error message shown if the input is outside the given range 

    arch_list=['ridge','svm','forest','fnn']
    ## A string list arch_list is created with different strings corresponding to the different
    ## regression classes outlined in the model_architecture.py program. 
    if b<2:
        ## If the b value calculated above is less than 2 
        for trial in range(1,4):
            ## The iterable object trial is in the range [1,4). 
            if b==0:
                ## If b equals to 0, then a seqandstassay_to_yield_model is created and it is instantiated with
                ## a integer list to indicate the assays used to build the model, the 'a' element of the arch_list
                ## to indicate the regression model, the final integer is the sample fraction and the trial iterable is
                ## used to indicate the number of trials that are to be used
                mdl=modelbank.seqandstassay_to_yield_model([1,8,10],trial,arch_list[a],1)
            elif b==1:
                ## If b equals to 1, then a stassay_to_yield_model is created and it is instantiated with
                ## a integer list to indicate the assays used to build the model, the 'a' element of the arch_list
                ## to indicate the regression model, the final integer is the sample fraction and the trial iterable is
                ## used to indicate the number of trials that are to be used
                mdl=modelbank.stassay_to_yield_model([1,8,10],trial,arch_list[a],1)

            mdl.cross_validate_model()
            mdl.limit_test_set([1,8,10])
            mdl.test_model()
            ## Once the object is created the cross_validate_model() function of the model parent class is run which
            ## determined the hyperparameters for this model, then the limit_test_set() function from the x_to_yield_model
            ## parent class is run which modifies the class testing dataframe to reflect data only from the 1,8,10 assays
            ## Finally the test_model() function from the model parent class is run to train the model using the hyperparameters
            ## and the training class dataframe
    else:
        trials_list=[[1,2],[1,3],[2,3]]
        ## A trials_list is created with the different 2-combinations of 1,2,3
        for trials in trials_list:
            ## For each element in the trials_list 
            if b==2:
                mdl=modelbank.seqandttassay_to_yield_model([1,8,10],trials,arch_list[a],1)
                ## If b is equal to 2, then a seqandttassay_to_yield_model object from the submodels_module class is created
                ## this is instantiated with a integer list indicating the assays used to build the prediction, the iterable trials
                ## object is used to link the trials used. The a index of the arch_list tells us about the regression model and a integr 1
                ## to indicate the sample fraction. 
            elif b==3:
                mdl=modelbank.ttassay_to_yield_model([1,8,10],trials,arch_list[a],1)
                ## If b is equal to 3, then a ttassay_to_yield_model object from the submodels_module class is created
                ## this is instantiated with a integer list indicating the assays used to build the prediction, the iterable trials
                ## object is used to link the trials used. The a index of the arch_list tells us about the regression model and a integr 1
                ## to indicate the sample fraction.

            mdl.cross_validate_model()
            mdl.limit_test_set([1,8,10])
            mdl.test_model()
            ## Once the object is created the cross_validate_model() function of the model parent class is run which
            ## determined the hyperparameters for this model, then the limit_test_set() function from the x_to_yield_model
            ## parent class is run which modifies the class testing dataframe to reflect data only from the 1,8,10 assays
            ## Finally the test_model() function from the model parent class is run to train the model using the hyperparameters
            ## and the training class dataframe


# if __name__ == '__main__':
#     main()

trials_list=[[1],[2],[3],[1,2],[1,3],[2,3],[1,2,3]]
## A trials_list is created which has all the possible combinations of 1,2 and 3.
arch_list=['ridge','svm','forest','fnn']
## A string list, arch_list, has different types of regression models stored in it. 
loss_per_model,std_per_model=[],[]
## Two empty lists are created loss_per_model and std_per_model, to track the loss and standard deviation
## of each regression model.
for i in range(2):
    ## an iterable object i is created which runs through the following list [0,1]
    loss_per_trial,std_per_trial=[],[]
    ## Two new empty lists are created to track the loss and standard deviation for each trial
    for trials in trials_list:
        ## Then each element in the trials_list is accessed and three elements are created where
        ## two are infinty and one is zero, low_cv, low_mse and low_std respectively
        low_cv,low_mse,low_std=np.inf,np.inf,0
        for arch in arch_list:
            ## Then each element in the arch_list is accessed. 
            if i==0:
                ## if the i iterable is equal to 0, then the following segment is executed
                if len(trials)==1:
                    mdl=modelbank.seqandstassay_to_yield_model([1,8,10],trials[0],arch,1)
                    ## If the length of the trial iterable is equal to 1, then a seqandstassay_to_yield_model
                    ## object from the submodels_module program. The object is instantiated with an integer list representing
                    ## the assay scores, along with the first element of the trials iterable for the number of trials
                    ## used to build the model, the arch iterable used to determine the regression model and the integer 1
                    ## to indicate the sample fraction
                elif len(trials)==2:
                    mdl=modelbank.seqandttassay_to_yield_model([1,8,10],trials,arch,1)
                    ## If the length of the trial iterable is equal to 2, then a seqandttassay_to_yield_model
                    ## object from the submodels_module program. The object is instantiated with an integer list representing
                    ## the assay scores, along with the trials iterable for the number of trials used to build the model, the arch iterable 
                    ## used to determine the regression model and the integer 1 to indicate the sample fraction
                else:
                    mdl=modelbank.seqandassay_to_yield_model([1,8,10],arch,1)
                    ## If the length of the trial iterable is equal to 3, then a seqandassay_to_yield_model
                    ## object from the submodels_module program. The object is instantiated with an integer list representing
                    ## the assay scores, along with the arch iterable used to determine the regression model 
                    ## and the integer 1 to indicate the sample fraction
            else:
                ## If the i iterable is equal t0 1, then the following segment is executed
                if len(trials)==1:
                    mdl=modelbank.stassay_to_yield_model([1,8,10],trials[0],arch,1)
                    ## If the length of the trial iterable is equal to 1, then a stassay_to_yield_model
                    ## object from the submodels_module program. The object is instantiated with an integer list representing
                    ## the assay scores, along with the first element of the trials iterable for the number of trials
                    ## used to build the model, the arch iterable used to determine the regression model and the integer 1
                    ## to indicate the sample fraction
                elif len(trials)==2:
                    mdl=modelbank.ttassay_to_yield_model([1,8,10],trials,arch,1)
                    ## If the length of the trial iterable is equal to 2, then a ttassay_to_yield_model
                    ## object from the submodels_module program. The object is instantiated with an integer list representing
                    ## the assay scores, along with the trials iterable for the number of trials used to build the model, the arch iterable 
                    ## used to determine the regression model and the integer 1 to indicate the sample fraction
                else:
                    mdl=modelbank.assay_to_yield_model([1,8,10],arch,1)
                    ## If the length of the trial iterable is equal to 3, then a assay_to_yield_model
                    ## object from the submodels_module program. The object is instantiated with an integer list representing
                    ## the assay scores, along with the arch iterable used to determine the regression model 
                    ## and the integer 1 to indicate the sample fraction

            if mdl.model_stats['cv_avg_loss']<low_cv:
                ## The cv_avg_loss column of the model_stats class dataframe is accessed and if this
                ## less than the low_cv object established.
                low_cv=mdl.model_stats['cv_avg_loss']
                low_mse=mdl.model_stats['test_avg_loss']
                low_std=mdl.model_stats['test_std_loss']
                ## The cv_avg_loss , test_avg_loss and test_std_loss of the model_stats dataframe is accessed
                ## and assigned to the low_cv, low_mse and low_std variable respectively
        loss_per_trial.append(low_mse)
        std_per_trial.append(low_std)
        ## The low_mse and low_std are then appended to the loss_per_trial and std_per_trial lists made at the start respectively 
    loss_per_model.append(loss_per_trial)
    std_per_model.append(std_per_trial)
    ## Then the loss_per_trial and std_per_trial list is then added to the loss_per_model and std_per_model list respectively

seq_model=modelbank.seq_to_yield_model('forest',1)
## Then a seq_to_yield_model object from the submodels_module program is created and it is instantiated with a 'forest' string
## to indicate the regression model and a sample fraction of 1. This object is sotred in the seq_model variable
# seq_model.limit_test_set([1,8,10])
# seq_model.test_model()
seq_loss=seq_model.model_stats['test_avg_loss']
seq_std=seq_model.model_stats['test_std_loss']
## Then the test_avg_loss and test_std_loss columns of the models_stats dataframe of the seq_model object is accessed
## and then it was stored in the seq_loss and seq_std variables respectively. 

x=[-0.5,6.5]
seq_plus=[seq_loss+seq_std]*2
seq_min=[seq_loss-seq_std]*2
## A high and low limit of the loss with the standard deviation is created and added to the seq_plus and seq_min list respectively

control_model=modelbank.control_to_yield_model('ridge',1)
## A control_to_yield_model object from the submodels_module program is created and it is instantiated with
## a string indicating the model regression and a float to indicate the sample fraction
control_loss=control_model.model_stats['test_avg_loss']
## The test_avg_loss column of the model_stats dataframe is assigned to the control_loss variable
control_model.limit_test_set([1,8,10])
## The limit_test_set() function of the x_to_yield_model parent class, which modifies the class dataframe testing_df to reflect the
## the assays to be used.
exploded_df,_,_=load_format_data.explode_yield(control_model.testing_df)
## Then the explode_yield function from the load_format_data program is run on the testing_df class dataframe of the
## control_model object. Then only the exploded_df dataframe output from the function is assigned to exploded_df
exp_var=np.average(np.square(np.array(exploded_df['y_std'])))
## Then the y_std column of the exploded_df dataframe is turned into an array and then the values are squared
##  and the the average of the values were taken and this number is assigned to the exp_var variable

fig,ax=plt.subplots(1,1,figsize=[2,2],dpi=300)
## A Figure and an Axis objects are created are created, the following plot only has one plot

# # xloc=[0,0.5,1,2,2.5,3,4]
# # ax.axhline(seq_loss,-0.5,4.5,color='green',linestyle='--',label='Sequence')
# # ax.bar(np.subtract(xloc,0.075),loss_per_model[1],yerr=std_per_model[1],label='Assays',width=0.15)
# # ax.bar(np.add(xloc,0.075),loss_per_model[0],yerr=std_per_model[0],label='Sequence and Assays',width=0.15)
# # ax.fill_between(x,seq_plus,seq_min,alpha=0.2,color='green')
# # ax.set_xticks(xloc)
# # ticklabels=[str(x) for x in trials_list]
# # ax.set_xticklabels(ticklabels)

# # ax.legend(fontsize=6)
# # ax.tick_params(axis='both', which='major', labelsize=6)
# # ax.set_ylabel('$Test^2$ Loss',fontsize=6)
# # ax.set_xlabel('Trials',fontsize=6)
# # ax.set_ylim([0.55,0.7])
# # ax.set_xlim([-0.5,4.5])

# # fig.tight_layout()
# # fig.savefig('./changing_trials.png')
# # plt.close()

# # print('seq and assays')
# # for i in range(len(loss_per_model[0])-1):
# #     print(ttest(loss_per_model[0][i],std_per_model[0][i],10,loss_per_model[0][-1],std_per_model[0][-1],10))

# # print('assays')
# # for i in range(len(loss_per_model[1])-1):
# #     print(ttest(loss_per_model[1][i],std_per_model[1][i],10,loss_per_model[1][-1],std_per_model[1][-1],10))


ax.axhline(seq_loss,-0.5,2.5,color='green',linestyle='--',label='Sequence')
## A horizontal line is constructed on the axis at the seq_loss position and labelled 'Sequence'
ax.fill_between(x,seq_plus,seq_min,alpha=0.2,color='green')
## The x array constructed on top is used as the x-cordinates of the nodes while the seq_plus and seq_min
## lists are used as the high and low y-coordinates and the area in between is filled in green color
ax.axhline(control_loss,-0.5,2.5,color='red',linestyle='--',label='Control')
ax.axhline(exp_var,-0.5,2.5,color='purple',linestyle='--',label='Experimental')
## Two horizontal lines are constructed on the axis at the control_loss and exp_var position labelled
## 'Control' and 'Experimental' respectively.
ax.errorbar([-0.05,0,0.05],loss_per_model[1][0:3],yerr=std_per_model[1][0:3],marker='o',color='blue',ls='none',alpha=0.2)
single_trial_average=np.average(loss_per_model[1][0:3])
ax.plot([-0.25,0.25],[single_trial_average]*2,linestyle='-',color='blue')
## The [-0.05,0,-0.05] list is plotted as the x axis against the first three elements of the second element of the loss_per_model
## and the y-axis error determined by the first three elements of the second element of the std_per_model.
## Then the average of the plotted y-values is taken and placed in a list twice and stored in the variable
## single_trial_average. Following this the [-0.25,0.25] list is plotted against the single_trial_list, however this time without
## an errorbar. 
ax.errorbar([0.95,1,1.05],loss_per_model[1][3:6],yerr=std_per_model[1][3:6],marker='o',color='blue',ls='none',alpha=0.2)
two_trial_average=np.average(loss_per_model[1][3:6])
ax.plot([0.75,1.25],[two_trial_average]*2,linestyle='-',color='blue')
## The [0.95,1,1.05] list is plotted as the x axis against the next three elements of the second element of the loss_per_model
## and the y-axis error determined by the next three elements of the second element of the std_per_model.
## Then the average of the plotted y-values is taken and placed in a list twice and stored in the variable
## two_trial_average. Following this the [0.75,1.25] list is plotted against the two_trial_list, however this time without
## an errorbar.
ax.errorbar(2,loss_per_model[1][6],yerr=std_per_model[1][6],marker='o',color='blue',ls='none',alpha=0.2)
ax.plot([1.75,2.25],[loss_per_model[1][6]]*2,linestyle='-',color='blue',label='Assay')
## The integer 2 is plotted as the x axis against the seventh element of the second element of the loss_per_model
## and the y-axis error determined by the seventh element of the second element of the std_per_model.
## Following this the [1.75,2.25] list is plotted against seventh element of the second element of the loss_per_model twice
## however this time without an errorbar and it is labelled 'Assay'
## All the above lines and errorbars are done in blue color with a 'o' marker

ax.errorbar([-0.05,0,0.05],loss_per_model[0][0:3],yerr=std_per_model[0][0:3],marker='o',color='orange',ls='none',alpha=0.2)
single_trial_average=np.average(loss_per_model[0][0:3])
ax.plot([-0.25,0.25],[single_trial_average]*2,linestyle='-',color='orange')
## The [-0.05,0,-0.05] list is plotted as the x axis against the first three elements of the first element of the loss_per_model
## and the y-axis error determined by the first three elements of the first element of the std_per_model.
## Then the average of the plotted y-values is taken and placed in a list twice and stored in the variable
## single_trial_average. Following this the [-0.25,0.25] list is plotted against the single_trial_list, however this time without
## an errorbar. 
ax.errorbar([0.95,1,1.05],loss_per_model[0][3:6],yerr=std_per_model[0][3:6],marker='o',color='orange',ls='none',alpha=0.2)
two_trial_average=np.average(loss_per_model[0][3:6])
ax.plot([0.75,1.25],[two_trial_average]*2,linestyle='-',color='orange')
## The [0.95,1,1.05] list is plotted as the x axis against the next three elements of the first element of the loss_per_model
## and the y-axis error determined by the next three elements of the first element of the std_per_model.
## Then the average of the plotted y-values is taken and placed in a list twice and stored in the variable
## two_trial_average. Following this the [0.75,1.25] list is plotted against the two_trial_list, however this time without
## an errorbar.
ax.errorbar(2,loss_per_model[0][6],yerr=std_per_model[0][6],marker='o',color='orange',ls='none',alpha=0.2)
ax.plot([1.75,2.25],[loss_per_model[0][6]]*2,linestyle='-',color='orange',label='Seq and Assay')
## The integer 2 is plotted as the x axis against the seventh element of the first element of the loss_per_model
## and the y-axis error determined by the seventh element of the first element of the std_per_model.
## Following this the [1.75,2.25] list is plotted against seventh element of the first element of the loss_per_model twice
## however this time without an errorbar and it is labelled 'Seq and Assay'
## All the above lines and errorbars are done in orange color with a 'o' marker

# ax.legend(fontsize=6)
ax.set_xticks([0,1,2])
ax.set_xticklabels(['1','2','3'])
## The x-axis ticks are set to the number 0,1,2 and they are labelled with the strings '1','2','3' respectively
ax.set_xlim([-0.5,2.5])
ax.set_ylim([0.35,0.8])
## Then the x-axis and y-axis limits are set to [-0.5,2,5] and [0.35,0.8] respectively
ax.set_xlabel('Number of Trials',fontsize=6)
ax.tick_params(axis='both', which='major', labelsize=6)
ax.set_ylabel('$Test^2$ Loss',fontsize=6)
## A x-axis and y-axis label is set to 'Number of Trials' and '$Test^2$ Loss' respectively, then the sizes
## of the ticks on both the axes are changed to reflect the font size of 6.
fig.tight_layout()
fig.savefig('./changing_trials.png')
plt.close()
## Following this the figure is squeezed so that plot fits within the figure area, then the figure is saved
## as a png file under the name 'changing_trials.png'. Finally the pyplot module is closed. 
